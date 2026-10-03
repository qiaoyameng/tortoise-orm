"""Sandbox shim: replace asyncio's socketpair-based self-pipe with os.pipe.

Some restricted execution sandboxes run under a seccomp profile that makes
``send()`` on a ``socketpair`` end fail with EPERM, which breaks the event
loop's cross-thread wakeup (aiosqlite's worker thread relies on it). Anonymous
pipes are still allowed, so teach the selector event loop to use a pipe for
``_write_to_self`` in that case.

The replacement is only installed when the environment is actually broken,
so normal platforms (including Windows, where this POSIX-only shim would not
apply) keep the standard socketpair implementation.
"""

from __future__ import annotations

import os
import socket

from asyncio import selector_events


class _PipeEnd:
    def __init__(self, fd: int) -> None:
        self._fd = fd
        os.set_blocking(fd, False)

    def fileno(self) -> int:
        return self._fd

    def setblocking(self, flag: bool) -> None:
        os.set_blocking(self._fd, flag)

    def close(self) -> None:
        os.close(self._fd)

    def send(self, data: bytes) -> int:
        return os.write(self._fd, data)

    def recv(self, size: int) -> bytes:
        return os.read(self._fd, size)


def _socketpair_send_blocked() -> bool:
    try:
        r, w = socket.socketpair()
        try:
            w.send(b"\0")
            r.recv(1)
        finally:
            r.close()
            w.close()
    except OSError:
        return True
    return False


def _make_self_pipe(self) -> None:  # type: ignore[no-untyped-def]
    rfd, wfd = os.pipe()
    self._ssock = _PipeEnd(rfd)
    self._csock = _PipeEnd(wfd)
    self._internal_fds += 1
    self._add_reader(rfd, self._read_from_self)


if _socketpair_send_blocked():
    selector_events.BaseSelectorEventLoop._make_self_pipe = _make_self_pipe  # type: ignore[method-assign]
