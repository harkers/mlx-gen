"""The MiniMax-H3 backend CLI on Ref2VA requests, with a recording stand-in for the multi-GB model."""

import sys

import PIL.Image
import pytest

from mflux.models.minimax_h3.cli import minimax_h3_generate
from mflux.models.minimax_h3.variants.h3_references import H3Reference
from mflux.models.minimax_h3.variants.minimax_h3 import MiniMaxH3
from tests.minimax_h3.media_fixtures import write_wav


class _Stop(Exception):
    """Raised by the stand-in once it has seen the generate call, so no real work runs."""


class _RecordingH3:
    _check_route = staticmethod(MiniMaxH3._check_route)
    constructed: list[dict] = []
    generated: list[dict] = []

    def __init__(self, **kwargs):
        _RecordingH3.constructed.append(kwargs)

    def generate_video(self, **kwargs):
        _RecordingH3.generated.append(kwargs)
        raise _Stop


@pytest.fixture
def recording(monkeypatch):
    _RecordingH3.constructed, _RecordingH3.generated = [], []
    monkeypatch.setattr(minimax_h3_generate, "MiniMaxH3", _RecordingH3)
    return _RecordingH3


@pytest.fixture
def media(tmp_path):
    image, voice = tmp_path / "lucien.png", tmp_path / "burton.wav"
    PIL.Image.new("RGB", (96, 64)).save(image)
    write_wav(voice, seconds=3.0, rate=32000, channels=1)
    return str(image), str(voice)


def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["mlxgen-generate-minimax-h3", *argv, "--seed", "1", "--no-progress"])
    minimax_h3_generate.main()


@pytest.mark.fast
def test_cli_passes_references_to_the_ref2va_model(monkeypatch, recording, media):
    image, voice = media
    with pytest.raises(_Stop):
        _run(
            monkeypatch,
            "--model",
            "minimax-h3-ref2va",
            "--reference-image",
            image,
            "--reference-audio",
            voice,
            "--prompt",
            "subject_definitions: x",
            "--frames",
            "124",
        )
    (constructed,) = recording.constructed
    assert constructed["model_config"].aliases[0] == "minimax-h3-ref2va"
    (generated,) = recording.generated
    assert generated["references"] == [H3Reference("image", image), H3Reference("audio", voice)]
    assert generated["image_path"] is None and generated["num_frames"] == 124


@pytest.mark.fast
def test_cli_orders_images_before_audio(monkeypatch, recording, media):
    image, voice = media
    with pytest.raises(_Stop):
        _run(
            monkeypatch,
            "--model",
            "minimax-h3-ref2va",
            "--reference-audio",
            voice,
            "--reference-image",
            image,
            "--prompt",
            "x",
        )
    assert [reference.kind for reference in recording.generated[0]["references"]] == ["image", "audio"]


@pytest.mark.fast
def test_cli_rejects_bad_requests_before_loading(monkeypatch, recording, media, capsys):
    image, voice = media
    bad_requests = [
        ("--model", "minimax-h3-ref2va", "--reference-image", image + ".missing", "--prompt", "x"),
        ("--model", "minimax-h3-ref2va", "--reference-image", image, "--soundscape", "wind", "--prompt", "x"),
        ("--model", "minimax-h3-ref2va", "--reference-audio", voice, "--prompt", "x"),
        ("--model", "minimax-h3-ref2va", "--prompt", "x"),
        ("--model", "minimax-h3", "--reference-image", image, "--prompt", "x"),
    ]
    for argv in bad_requests:
        with pytest.raises(SystemExit) as exit_info:
            _run(monkeypatch, *argv)
        assert exit_info.value.code == 1, argv
    captured = capsys.readouterr()
    errors = captured.out + captured.err
    assert "not found" in errors and "--soundscape" in errors and "paired with at least one image" in errors
    assert recording.constructed == []  # every refusal came before the model load

    with pytest.raises(SystemExit) as exit_info:  # argparse: references and a keyframe are exclusive
        _run(
            monkeypatch,
            "--model",
            "minimax-h3-ref2va",
            "--reference-image",
            image,
            "--image-path",
            image,
            "--prompt",
            "x",
        )
    assert exit_info.value.code == 2 and recording.constructed == []


@pytest.mark.fast
def test_cli_forwards_the_reference_resize_policy(monkeypatch, recording, media):
    image, voice = media
    with pytest.raises(_Stop):
        _run(
            monkeypatch,
            "--model",
            "minimax-h3-ref2va-turbo",
            "--reference-image",
            image,
            "--reference-audio",
            voice,
            "--reference-resize",
            "match",
            "--prompt",
            "x",
        )
    assert recording.generated[0]["reference_resize"] == "match"
    assert recording.constructed[0]["model_config"].aliases[0] == "minimax-h3-ref2va-turbo"
