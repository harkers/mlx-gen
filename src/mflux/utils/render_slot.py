"""Machine-wide admission for memory-hungry renders.

A MiniMax-H3 render holds ~60 GB resident. Two at once on a 128 GB Mac swap tens of GB; three wedged the machine
hard enough for the hardware watchdog to reset it (2026-10-19). Every render therefore takes a slot before it loads
the model and keeps it until the process exits -- the lock is an ``flock`` on a per-slot file, so a killed or
orphaned render releases it with no cleanup of its own.

    MFLUX_MAX_RENDERS      slots on this machine (default 1; 0 disables the gate)
    MFLUX_RENDER_SLOT_DIR  lock directory (default ~/.cache/mflux/render-slots)

Stdlib only: the queue shell scripts rely on this import staying cheap.
"""

from __future__ import annotations

import fcntl
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_HELD: RenderSlot | None = None


@dataclass
class RenderSlot:
    index: int
    path: Path
    _fd: int

    def release(self) -> None:
        global _HELD
        if self._fd >= 0:
            os.close(self._fd)  # closing drops the flock
            self._fd = -1
        if _HELD is self:
            _HELD = None


def acquire_render_slot(
    max_slots: int | None = None,
    lock_dir: str | os.PathLike | None = None,
    poll_s: float = 10.0,
    timeout_s: float | None = None,
) -> RenderSlot | None:
    """Block until one of ``max_slots`` slots is free, then hold it for the life of this process.

    Idempotent per process: a second call returns the slot already held. Returns None when the gate is disabled.
    Raises TimeoutError when ``timeout_s`` elapses first.
    """
    global _HELD
    if _HELD is not None:
        return _HELD
    slots = int(os.environ.get("MFLUX_MAX_RENDERS", "1")) if max_slots is None else max_slots
    if slots <= 0:
        return None
    directory = Path(lock_dir or os.environ.get("MFLUX_RENDER_SLOT_DIR") or Path.home() / ".cache/mflux/render-slots")
    directory.mkdir(parents=True, exist_ok=True)
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    warned = False
    while True:
        for index in range(slots):
            path = directory / f"slot{index}.lock"
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(fd)
                continue
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()}\n".encode())
            _HELD = RenderSlot(index=index, path=path, _fd=fd)
            if warned:
                print(f"render slot {index} acquired after waiting", file=sys.stderr, flush=True)
            return _HELD
        if not warned:
            holders = ", ".join(_holder(directory / f"slot{i}.lock") for i in range(slots))
            print(f"all {slots} render slot(s) held (pid {holders}); waiting", file=sys.stderr, flush=True)
            warned = True
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"no render slot free within {timeout_s}s ({slots} slot(s) in {directory})")
        time.sleep(poll_s)


def _holder(path: Path) -> str:
    try:
        return path.read_text().strip() or "?"
    except OSError:
        return "?"
