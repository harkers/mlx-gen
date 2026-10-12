"""`ref2va` references for MiniMax-H3: an ordered list of images and audio clips.

Normalisation follows the diffusers `modular_pipelines/minimax_h3` setup step (Apache-2.0, The MiniMax and
HuggingFace Teams): an image is resized with LANCZOS to its own 2048-pixel short edge on the 32-pixel grid, never to
the target canvas; an audio clip becomes 32 kHz stereo (a mono clip is copied onto both channels) truncated to the
generated duration. Limits are MiniMax's: at most 9 images, 3 audio clips and 12 references, and an audio clip of
2 to 15 seconds that rides with at least one image. Video references are not supported yet.
"""

from dataclasses import dataclass

import av
import numpy as np
import PIL.Image

AUDIO_REFERENCE = "audio"
REFERENCE_IMAGE_SHORT_EDGE = 2048
REFERENCE_MULTIPLE = 32
MIN_REFERENCE_AUDIO_SECONDS = 2.0
MAX_IMAGE_REFERENCES = 9
MAX_AUDIO_REFERENCES = 3
MAX_REFERENCES = 12


@dataclass(frozen=True)
class H3Reference:
    kind: str  # "image" | "audio"
    path: str


def validate_references(references: list[H3Reference]) -> None:
    if not references:
        raise ValueError("`ref2va` needs at least one reference.")
    kinds = [reference.kind for reference in references]
    unknown = sorted(set(kinds) - {"image", "audio"})
    if unknown:
        raise ValueError(f"A reference must be 'image' or 'audio' (video references are not supported), got {unknown}.")
    if kinds.count("image") > MAX_IMAGE_REFERENCES:
        raise ValueError(
            f"MiniMax-H3 accepts at most {MAX_IMAGE_REFERENCES} image references, got {kinds.count('image')}."
        )
    if kinds.count("audio") > MAX_AUDIO_REFERENCES:
        raise ValueError(
            f"MiniMax-H3 accepts at most {MAX_AUDIO_REFERENCES} audio references, got {kinds.count('audio')}."
        )
    if len(kinds) > MAX_REFERENCES:
        raise ValueError(f"MiniMax-H3 accepts at most {MAX_REFERENCES} references in total, got {len(kinds)}.")
    if "image" not in kinds:
        raise ValueError("An audio reference has to be paired with at least one image reference.")


def fit_reference_image(image: PIL.Image.Image, short_edge: int = REFERENCE_IMAGE_SHORT_EDGE) -> PIL.Image.Image:
    """Resize (LANCZOS, upscaling included, no area cap) to `short_edge` on the 32-pixel grid."""
    width, height = image.size
    if width > 4 * height or height > 4 * width:
        raise ValueError(f"A reference image must be within 1:4 and 4:1, got {width}x{height}.")
    scale = short_edge / min(width, height)
    target = (
        max(REFERENCE_MULTIPLE, round(width * scale / REFERENCE_MULTIPLE) * REFERENCE_MULTIPLE),
        max(REFERENCE_MULTIPLE, round(height * scale / REFERENCE_MULTIPLE) * REFERENCE_MULTIPLE),
    )
    return image if image.size == target else image.resize(target, PIL.Image.Resampling.LANCZOS)


def load_reference_audio(path: str, sample_rate: int, max_seconds: float) -> np.ndarray:
    """`(2, samples)` float32 at `sample_rate`, truncated to `max_seconds`.

    Resampled per channel with the clip's own layout, then a mono clip is copied onto both channels at unity, as
    the reference does; a stereo downmix through the resampler would pan it at -3 dB instead.
    """
    chunks = []
    with av.open(path) as container:
        if not container.streams.audio:
            raise ValueError(f"{path} has no audio stream.")
        stream = container.streams.audio[0]
        layout = "mono" if stream.channels == 1 else "stereo"
        resampler = av.AudioResampler(format="fltp", layout=layout, rate=sample_rate)
        for frame in container.decode(stream):
            chunks.extend(out.to_ndarray() for out in resampler.resample(frame))
        chunks.extend(out.to_ndarray() for out in resampler.resample(None))
    waveform = np.concatenate(chunks, axis=1)[:, : int(max_seconds * sample_rate)].astype(np.float32)
    if waveform.shape[1] < MIN_REFERENCE_AUDIO_SECONDS * sample_rate:
        raise ValueError(
            f"A reference audio clip must be at least {MIN_REFERENCE_AUDIO_SECONDS:g} seconds, "
            f"got {waveform.shape[1] / sample_rate:.2f} s from {path}."
        )
    return np.repeat(waveform, 2, axis=0) if waveform.shape[0] == 1 else waveform
