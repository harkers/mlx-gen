"""Qwen3-VL vision-side contracts used by MiniMax-H3 first-frame conditioning."""

import numpy as np
import PIL.Image
import pytest

from mflux.models.minimax_h3.model.h3_text_encoder.qwen3_vl_model import rope_index
from mflux.models.minimax_h3.model.h3_text_encoder.qwen3_vl_vision_model import (
    position_embedding_taps,
    preprocess_image,
    smart_resize,
    vision_position_ids,
)


@pytest.mark.fast
def test_preprocess_keeps_canvas_sized_images_and_orders_patches_by_merge_block():
    image = PIL.Image.fromarray((np.random.default_rng(0).random((544, 960, 3)) * 255).astype(np.uint8))
    patches, grid = preprocess_image(image)
    assert grid == (1, 34, 60) and patches.shape == (34 * 60, 3 * 2 * 16 * 16)
    pixels = (np.asarray(image, dtype=np.float32) / 255.0 - 0.5) / 0.5
    # first patch = top-left 16x16 block, second patch = the block to its right (same merge window), duplicated in time
    first = patches[0].reshape(3, 2, 16, 16)
    assert np.allclose(first[:, 0], pixels[:16, :16].transpose(2, 0, 1)) and np.array_equal(first[:, 0], first[:, 1])
    second = patches[1].reshape(3, 2, 16, 16)
    assert np.allclose(second[:, 0], pixels[:16, 16:32].transpose(2, 0, 1))
    third = patches[2].reshape(3, 2, 16, 16)
    assert np.allclose(third[:, 0], pixels[16:32, :16].transpose(2, 0, 1))
    assert smart_resize(544, 960, 32, 65536, 16777216) == (544, 960)
    assert smart_resize(100, 100, 32, 65536, 16777216) == (256, 256)


@pytest.mark.fast
def test_vision_position_ids_and_taps_follow_merge_block_order():
    positions = vision_position_ids((1, 4, 6), 2)
    assert positions.shape == (24, 2)
    assert positions[:4].tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]
    indices, weights = position_embedding_taps((1, 4, 6), side=48, merge_size=2)
    assert indices.shape == (24, 4) and np.allclose(weights.sum(axis=1), 1.0)


@pytest.mark.fast
def test_rope_index_lays_out_text_then_image_grid():
    ids = np.array([5, 6, 152] + [150] * 6 + [153, 7])
    positions = rope_index(ids, 150, [(1, 4, 6)], 2)
    assert positions[:, :3].tolist() == [[0, 1, 2]] * 3
    assert positions[0, 3:9].tolist() == [3] * 6  # temporal axis constant
    assert positions[1, 3:9].tolist() == [3, 3, 3, 4, 4, 4]  # rows
    assert positions[2, 3:9].tolist() == [3, 4, 5, 3, 4, 5]  # columns
    assert positions[:, 9:].tolist() == [[6, 7]] * 3  # text resumes after max(h, w) // merge


@pytest.mark.fast
def test_vision_attention_never_materializes_the_attention_matrix():
    """Qwen3-VL's 72-wide heads fall off MLX's fused attention kernel on older MLX, which then builds the full
    heads x N x N matrix: 25 GiB for a 2048-pixel Ref2VA reference. The output must not change either way."""
    import mlx.core as mx

    from mflux.models.minimax_h3.model.h3_text_encoder.qwen3_vl_vision_model import Qwen3VLVisionAttention

    num_heads, head_dim = 16, 72
    attention = Qwen3VLVisionAttention(num_heads * head_dim, num_heads)
    mx.eval(attention.parameters())

    def inputs(seq_len):
        x = mx.random.normal((seq_len, num_heads * head_dim), key=mx.random.key(0)).astype(mx.bfloat16)
        angles = mx.random.uniform(shape=(seq_len, head_dim), key=mx.random.key(1))
        mx.eval(x, angles)
        return x, mx.cos(angles), mx.sin(angles)

    def transient(seq_len):
        x, cos, sin = inputs(seq_len)
        mx.eval(cos, sin)
        mx.clear_cache()
        mx.reset_peak_memory()
        base = mx.get_active_memory()
        mx.eval(attention(x, cos, sin))
        return mx.get_peak_memory() - base

    # A materialized matrix quadruples when the sequence doubles; fused attention's working set only doubles.
    small, large = transient(2048), transient(4096)
    assert large / small < 3, f"{small / 2**20:.0f} MiB at 2048 rows, {large / 2**20:.0f} MiB at 4096"

    seq_len = 1024
    x, cos, sin = inputs(seq_len)
    out = attention(x, cos, sin)
    # Same result as plain softmax attention, computed in float32.
    qkv = attention.qkv(x).reshape(seq_len, 3, num_heads, head_dim).astype(mx.float32)
    query, key, value = qkv[:, 0], qkv[:, 1], qkv[:, 2]
    rotate = lambda t: mx.concatenate([-t[..., head_dim // 2 :], t[..., : head_dim // 2]], axis=-1)  # noqa: E731
    query = query * cos[:, None] + rotate(query) * sin[:, None]
    key = key * cos[:, None] + rotate(key) * sin[:, None]
    q, k, v = (t.transpose(1, 0, 2) for t in (query, key, value))
    weights = mx.softmax((q @ k.transpose(0, 2, 1)) * head_dim**-0.5, axis=-1)
    expected = attention.proj((weights @ v).transpose(1, 0, 2).reshape(seq_len, -1).astype(mx.bfloat16))
    assert mx.allclose(out.astype(mx.float32), expected.astype(mx.float32), atol=5e-2, rtol=5e-2)
