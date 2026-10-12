"""Small media files for MiniMax-H3 tests, written with the standard library only."""

import wave

import numpy as np


def write_wav(path, seconds: float, rate: int, channels: int, amplitude: float = 0.5) -> None:
    """A 220 Hz int16 tone of `seconds` length, identical on every channel."""
    t = np.arange(int(seconds * rate)) / rate
    tone = (amplitude * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)
    frames = np.repeat(tone[:, None], channels, axis=1)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames.tobytes())
