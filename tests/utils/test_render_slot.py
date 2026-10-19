import os
import time

import pytest

from mflux.utils import render_slot
from mflux.utils.render_slot import acquire_render_slot


@pytest.fixture(autouse=True)
def _fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("MFLUX_RENDER_SLOT_DIR", str(tmp_path))
    monkeypatch.delenv("MFLUX_MAX_RENDERS", raising=False)
    render_slot._HELD = None
    yield
    if render_slot._HELD is not None:
        render_slot._HELD.release()


def test_default_is_one_slot_held_for_the_process(tmp_path):
    slot = acquire_render_slot()
    assert slot.index == 0
    assert (tmp_path / "slot0.lock").read_text().strip() == str(os.getpid())
    assert acquire_render_slot() is slot, "a second call in the same process reuses the held slot"


def test_a_held_slot_blocks_another_holder_until_released(tmp_path):
    first = acquire_render_slot()
    render_slot._HELD = None  # pretend to be a second process: its own flock must conflict with the first
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        acquire_render_slot(poll_s=0.05, timeout_s=0.3)
    assert time.monotonic() - t0 >= 0.3
    first.release()
    assert acquire_render_slot(poll_s=0.05, timeout_s=0.3).index == 0


def test_more_slots_admit_more_renders(tmp_path):
    acquire_render_slot(max_slots=2)
    render_slot._HELD = None
    second = acquire_render_slot(max_slots=2, poll_s=0.05, timeout_s=0.3)
    assert second.index == 1
    second.release()


def test_env_configures_slots_and_zero_disables(monkeypatch):
    monkeypatch.setenv("MFLUX_MAX_RENDERS", "0")
    assert acquire_render_slot() is None
    monkeypatch.setenv("MFLUX_MAX_RENDERS", "3")
    acquire_render_slot()
    render_slot._HELD = None
    render_slot._HELD = None
    assert acquire_render_slot(poll_s=0.05, timeout_s=0.3).index == 1
