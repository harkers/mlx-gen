"""Fast contracts of the MiniMax-H3 scheduler, packed layout and prompt composition (no weights)."""

import mlx.core as mx
import numpy as np
import pytest

from mflux.models.minimax_h3.latent_creator.h3_layout import (
    AUDIO_TAG,
    TEXT_TAG,
    VIDEO_TAG,
    H3ReferenceBlock,
    align_num_frames,
    audio_latent_num_frames,
    build_packed_sequence,
    build_ref2va_packed_sequence,
    build_row_timesteps,
    pack_audio_rows,
    patchify_video_latents,
    resolve_canvas_size,
    unpack_audio_rows,
    unpatchify_video_rows,
    video_latent_num_frames,
)
from mflux.models.minimax_h3.scheduler.minimax_h3_scheduler import MiniMaxH3Scheduler
from mflux.models.minimax_h3.variants.minimax_h3 import compose_prompt

# torch.linspace + MiniMaxH3Scheduler (diffusers 0.40) grids, float32, verified bit-exact on 600 grids.
TORCH_SIGMAS_N9_SHIFT12 = [
    1.0,
    0.9882352948188782,
    0.9729729890823364,
    0.9523809552192688,
    0.9230769276618958,
    0.8780487775802612,
    0.800000011920929,
    0.6315789222717285,
    0.0,
]
TORCH_SIGMAS_N9_SHIFT3 = [
    1.0,
    0.9545454382896423,
    0.8999999761581421,
    0.8333333134651184,
    0.75,
    0.6428571343421936,
    0.5,
    0.30000001192092896,
    0.0,
]


@pytest.mark.fast
@pytest.mark.parametrize("shift, expected", [(12.0, TORCH_SIGMAS_N9_SHIFT12), (3.0, TORCH_SIGMAS_N9_SHIFT3)])
def test_sigma_grid_matches_torch_bit_for_bit(shift, expected):
    scheduler = MiniMaxH3Scheduler(shift=shift)
    scheduler.set_timesteps(9)
    assert scheduler.sigmas.dtype == np.float32
    assert scheduler.sigmas.tolist() == expected
    # N grid points drive N - 1 transformer evaluations; the timestep the model sees is 1 - sigma.
    assert len(scheduler.timesteps) == 8
    assert np.array_equal(scheduler.timesteps, (np.float32(1.0) - scheduler.sigmas[:-1]).astype(np.float32))


@pytest.mark.fast
def test_step_follows_the_rectified_flow_line_exactly():
    scheduler = MiniMaxH3Scheduler(shift=12.0)
    scheduler.set_timesteps(9)
    x0 = mx.random.normal((6, 4))
    noise = mx.random.normal((6, 4))
    sigma = float(scheduler.sigmas[3])
    x_t = scheduler.scale_noise(x0, 1.0 - sigma, noise)
    velocity = x0 - noise  # the exact data-ward velocity of a straight path
    x_next = scheduler.step(velocity, 3, x_t)
    expected = scheduler.scale_noise(x0, 1.0 - float(scheduler.sigmas[4]), noise)
    assert float(mx.abs(x_next - expected).max()) < 1e-6


@pytest.mark.fast
def test_layout_partitions_rows_and_tags_each_stream():
    num_text, frames, height, width, audio = 5, 7, 4, 6, 8
    layout = build_packed_sequence(np.full((num_text,), TEXT_TAG, dtype=np.int32), frames, height, width, audio)
    video_rows = frames * (height // 2) * (width // 2)
    assert layout.sequence_length == num_text + audio * 2 + video_rows
    indices = np.concatenate(
        [np.array(layout.text_indices), np.array(layout.audio_indices), np.array(layout.video_indices)]
    )
    assert sorted(indices.tolist()) == list(range(layout.sequence_length))
    tags = np.array(layout.token_tags)
    assert (tags[np.array(layout.text_indices)] == TEXT_TAG).all()
    assert (tags[np.array(layout.audio_indices)] == AUDIO_TAG).all()
    assert (tags[np.array(layout.video_indices)] == VIDEO_TAG).all()
    assert layout.num_condition_video_rows == 0 and layout.num_condition_audio_rows == 0
    assert layout.position_ids.shape == (layout.sequence_length, 3)

    unique, inverse = build_row_timesteps(layout, 0.2, 0.5, 0.999, 1.0)
    rows = np.array(unique)[np.array(inverse)]
    assert unique.shape == (2,) and np.allclose(rows[np.array(layout.video_indices)], 0.2)
    assert np.allclose(rows[np.array(layout.audio_indices)], 0.5) and np.allclose(
        rows[np.array(layout.text_indices)], 0.2
    )


@pytest.mark.fast
def test_patchify_and_audio_packing_round_trip():
    latents = mx.random.normal((1, 24, 7, 4, 6))
    rows = patchify_video_latents(latents, (1, 2, 2))
    assert rows.shape == (7 * 2 * 3, 24 * 4)
    assert mx.array_equal(unpatchify_video_rows(rows, 24, 7, 4, 6, (1, 2, 2)), latents)
    audio = mx.random.normal((2, 32, 8))
    packed = pack_audio_rows(audio)
    assert packed.shape == (16, 32)
    assert mx.array_equal(unpack_audio_rows(packed, 2, 8), audio)


@pytest.mark.fast
def test_frame_and_canvas_contracts():
    assert align_num_frames(120) == 124 and align_num_frames(124) == 124 and align_num_frames(125) == 141
    assert video_latent_num_frames(124) == 37 and audio_latent_num_frames(124) == 207
    assert resolve_canvas_size(16, 9, 32) == (768, 1344)
    assert resolve_canvas_size(9, 16, 32) == (1344, 768)
    assert resolve_canvas_size(1, 1, 32) == (768, 768)


@pytest.mark.fast
def test_compose_prompt_wraps_plain_text_and_keeps_structured_prompts():
    assert compose_prompt("A fox", "wind", "piano") == (
        "integrated_multimodal_description: A fox\n\noverall_soundscape: wind\n\nnon_diegetic_music: piano"
    )
    structured = "integrated_multimodal_description: [Shot 1] A fox.\n\noverall_soundscape: Wind."
    assert compose_prompt(structured) == structured
    assert compose_prompt("  A fox  ") == "integrated_multimodal_description: A fox"


@pytest.mark.fast
def test_compose_prompt_rejects_a_section_the_prompt_already_carries():
    """A structured prompt plus the matching flag would send the model two of the same section."""
    from mflux.models.minimax_h3.variants.minimax_h3 import compose_prompt, prompt_sections_present

    structured = "integrated_multimodal_description: A fox.\n\noverall_soundscape: Wind.\n\nnon_diegetic_music: None."
    assert prompt_sections_present(structured) == (
        "integrated_multimodal_description",
        "overall_soundscape",
        "non_diegetic_music",
    )
    for option, kwargs in (("--soundscape", {"soundscape": "birds"}), ("--music", {"music": "piano"})):
        with pytest.raises(ValueError, match=option):
            compose_prompt(structured, **kwargs)

    # A partial structured prompt still accepts the sections it does not carry.
    partial = "integrated_multimodal_description: A fox."
    assert prompt_sections_present(partial) == ("integrated_multimodal_description",)
    composed = compose_prompt(partial, soundscape="birds", music="piano")
    assert composed.count("overall_soundscape:") == 1 and composed.count("non_diegetic_music:") == 1

    # A label mentioned mid-sentence is prose, not a section header: the words stay in the description
    # and the flag still adds the one real section.
    prose = "A documentary about how an overall_soundscape: is written."
    assert prompt_sections_present(prose) == ()
    composed = compose_prompt(prose, soundscape="birds")
    assert prompt_sections_present(composed) == ("integrated_multimodal_description", "overall_soundscape")
    assert composed.endswith("overall_soundscape: birds")


@pytest.mark.fast
def test_ref2va_layout_packs_references_in_request_order_and_advances_the_clock():
    tags_in = np.full((5,), TEXT_TAG, dtype=np.int32)
    image = H3ReferenceBlock(kind="image", latent_frames=1, latent_height=8, latent_width=4)  # 4 * 2 = 8 rows
    voice = H3ReferenceBlock(kind="audio", num_audio_latents=3)  # 3 * 2 channels = 6 rows
    layout = build_ref2va_packed_sequence(tags_in, [image, voice], 7, 4, 6, 8)

    target_video_rows = 7 * 2 * 3
    assert layout.sequence_length == 5 + 8 + 6 + 16 + target_video_rows
    assert layout.num_condition_video_rows == 8 and layout.num_condition_audio_rows == 6
    video_idx, audio_idx = np.array(layout.video_indices), np.array(layout.audio_indices)
    # [text 0-4 | image 5-12 | voice 13-18 | target audio 19-34 | target video 35-]
    assert video_idx[:8].tolist() == list(range(5, 13))
    assert audio_idx[:6].tolist() == list(range(13, 19))
    assert audio_idx[6:].tolist() == list(range(19, 35))
    assert video_idx[8:].tolist() == list(range(35, layout.sequence_length))

    t = np.array(layout.position_ids)[:, 0]
    assert np.allclose(t[5:13], 5.0)  # an image takes one rotary slot at the text length
    assert np.allclose(t[13:16], [6, 7, 8]) and np.allclose(t[16:19], [6, 7, 8])  # voice, per channel
    assert np.allclose(t[19:27], 9 + np.arange(8))  # target audio starts where the references leave the clock
    assert np.isclose(t[35], 9.0)  # target video shares that origin
    tags = np.array(layout.token_tags)
    assert (tags[video_idx] == VIDEO_TAG).all() and (tags[audio_idx] == AUDIO_TAG).all()
    assert (tags[:5] == TEXT_TAG).all()

    # Order is semantic: audio first moves the voice to the front of the clock.
    swapped = build_ref2va_packed_sequence(tags_in, [voice, image], 7, 4, 6, 8)
    t_swapped = np.array(swapped.position_ids)[:, 0]
    assert np.allclose(t_swapped[5:8], [5, 6, 7]) and np.allclose(t_swapped[11:19], 8.0)
    assert np.isclose(t_swapped[np.array(swapped.audio_indices)[6]], 9.0)


@pytest.mark.fast
def test_ref2va_row_timesteps_pin_each_reference_kind():
    image = H3ReferenceBlock(kind="image", latent_frames=1, latent_height=8, latent_width=4)
    voice = H3ReferenceBlock(kind="audio", num_audio_latents=3)
    layout = build_ref2va_packed_sequence(np.full((5,), TEXT_TAG, dtype=np.int32), [image, voice], 7, 4, 6, 8)
    unique, inverse = build_row_timesteps(layout, 0.2, 0.5, 0.999, 1.0)
    rows = np.array(unique)[np.array(inverse)]
    video_idx, audio_idx = np.array(layout.video_indices), np.array(layout.audio_indices)
    assert np.allclose(rows[video_idx[:8]], 0.999) and np.allclose(rows[video_idx[8:]], 0.2)
    assert np.allclose(rows[audio_idx[:6]], 1.0) and np.allclose(rows[audio_idx[6:]], 0.5)


@pytest.mark.fast
def test_ref2va_layout_without_references_matches_the_text_only_layout():
    tags_in = np.full((5,), TEXT_TAG, dtype=np.int32)
    text_only = build_packed_sequence(tags_in, 7, 4, 6, 8)
    no_refs = build_ref2va_packed_sequence(tags_in, [], 7, 4, 6, 8)
    for field in ("position_ids", "token_tags", "video_indices", "audio_indices", "text_indices"):
        assert np.array_equal(np.array(getattr(text_only, field)), np.array(getattr(no_refs, field))), field
    positions, audio_idx = np.array(text_only.position_ids), np.array(text_only.audio_indices)
    assert np.allclose(positions[audio_idx[:8], 0], 5 + np.arange(8))
    assert np.allclose(positions[audio_idx[8:], 0], 5 + np.arange(8))
    assert len(set(positions[audio_idx[:8], 2].tolist())) == 1
    assert positions[audio_idx[0], 2] < positions[audio_idx[8], 2]


@pytest.mark.fast
def test_ref2va_layout_rejects_unknown_reference_kinds():
    with pytest.raises(ValueError, match="'image' or an 'audio'"):
        build_ref2va_packed_sequence(
            np.full((5,), TEXT_TAG, dtype=np.int32), [H3ReferenceBlock(kind="video")], 7, 4, 6, 8
        )
