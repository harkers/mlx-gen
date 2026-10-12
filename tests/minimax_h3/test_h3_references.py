"""Ref2VA reference media: validation, the 2048-pixel image fit, and 32 kHz stereo audio loading."""

import json
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import PIL.Image
import pytest

from mflux.models.minimax_h3.variants.h3_references import (
    H3Reference,
    fit_reference_image,
    load_reference_audio,
    validate_references,
)
from tests.minimax_h3.media_fixtures import write_wav


@pytest.mark.fast
def test_reference_image_is_fitted_to_its_own_2048_short_edge():
    # Not the target canvas: 960x544 goes up to a 2048 short edge on the 32-pixel grid.
    assert fit_reference_image(PIL.Image.new("RGB", (960, 544))).size == (3616, 2048)
    assert fit_reference_image(PIL.Image.new("RGB", (100, 100))).size == (2048, 2048)
    already = PIL.Image.new("RGB", (2048, 2048))
    assert fit_reference_image(already) is already
    with pytest.raises(ValueError, match="1:4"):
        fit_reference_image(PIL.Image.new("RGB", (900, 200)))


@pytest.mark.fast
def test_mono_reference_is_resampled_and_duplicated_at_unity(tmp_path):
    path = tmp_path / "mono.wav"
    write_wav(path, seconds=3.0, rate=16000, channels=1)
    waveform = load_reference_audio(str(path), 32000, max_seconds=10.0)
    assert waveform.shape[0] == 2 and waveform.dtype == np.float32
    assert abs(waveform.shape[1] - 96000) <= 64
    assert np.array_equal(waveform[0], waveform[1])
    assert 0.45 < float(np.abs(waveform).max()) < 0.55  # a unity copy, not a -3 dB pan


@pytest.mark.fast
def test_reference_audio_is_truncated_to_the_generated_duration(tmp_path):
    path = tmp_path / "stereo.wav"
    write_wav(path, seconds=6.0, rate=32000, channels=2)
    waveform = load_reference_audio(str(path), 32000, max_seconds=124 / 24)
    assert waveform.shape == (2, int(124 / 24 * 32000))


@pytest.mark.fast
def test_reference_audio_shorter_than_two_seconds_is_rejected(tmp_path):
    path = tmp_path / "short.wav"
    write_wav(path, seconds=1.5, rate=32000, channels=1)
    with pytest.raises(ValueError, match="at least 2"):
        load_reference_audio(str(path), 32000, max_seconds=10.0)


@pytest.mark.fast
def test_reference_list_contract():
    validate_references([H3Reference("image", "a.png"), H3Reference("audio", "v.wav")])
    with pytest.raises(ValueError, match="at least one reference"):
        validate_references([])
    with pytest.raises(ValueError, match="paired with at least one image"):
        validate_references([H3Reference("audio", "v.wav")])
    with pytest.raises(ValueError, match="'image' or 'audio'"):
        validate_references([H3Reference("video", "c.mp4")])
    with pytest.raises(ValueError, match="at most 9 image"):
        validate_references([H3Reference("image", "a.png")] * 10)
    with pytest.raises(ValueError, match="at most 3 audio"):
        validate_references([H3Reference("image", "a.png")] + [H3Reference("audio", "v.wav")] * 4)


@pytest.mark.slow
def test_released_audio_vae_round_trips_a_speech_reference():
    """The audio VAE *encoder* was never exercised before Ref2VA; check it against the released weights.

    Needs only the 0.56 GiB `audio_vae/`, not the Ref2VA transformer.
    """
    root, sample = os.environ.get("MFLUX_H3_ROOT"), os.environ.get("MFLUX_H3_SPEECH_SAMPLE")
    if not root or not sample:
        pytest.skip("needs MFLUX_H3_ROOT (an H3 snapshot) and MFLUX_H3_SPEECH_SAMPLE (a speech clip)")
    from mflux.models.common.weights.loading.streaming_weight_loader import StreamingWeightLoader
    from mflux.models.minimax_h3.model.h3_audio_vae.h3_audio_vae import H3AudioVAE
    from mflux.models.minimax_h3.weights.h3_audio_vae_weights import audio_vae_kwargs
    from mflux.models.minimax_h3.weights.h3_weight_definition import MiniMaxH3WeightDefinition
    from mflux.models.minimax_h3.weights.h3_weight_mapping import MiniMaxH3WeightMapping

    root = Path(root)
    vae = H3AudioVAE(**audio_vae_kwargs(json.loads((root / "audio_vae" / "config.json").read_text())))
    component = next(c for c in MiniMaxH3WeightDefinition().get_components() if c.name == "audio_vae")
    StreamingWeightLoader.load_into(
        model=vae,
        component_root=root,
        component=component,
        quantize_arg=None,
        quantization_predicate=MiniMaxH3WeightDefinition.quantization_predicate,
        post_transform=MiniMaxH3WeightMapping.convert_audio_vae_state,
    )
    waveform = load_reference_audio(sample, vae.sampling_rate, max_seconds=10.0)
    latents = vae.encode(mx.array(waveform)[:, None, :])
    assert latents.shape == (2, vae.latent_channels, -(-waveform.shape[1] // vae.hop_length))
    decoded = np.array(vae.decode(latents))[:, 0, : waveform.shape[1]]

    def log_spectrogram(x: np.ndarray) -> np.ndarray:
        frames = np.lib.stride_tricks.sliding_window_view(x, 1024)[::256] * np.hanning(1024)
        return np.log1p(np.abs(np.fft.rfft(frames, axis=-1)))

    correlation = np.corrcoef(log_spectrogram(waveform[0]).ravel(), log_spectrogram(decoded[0]).ravel())[0, 1]
    print(f"audio VAE round-trip log-spectrogram correlation: {correlation:.4f}")
    assert correlation > 0.85
