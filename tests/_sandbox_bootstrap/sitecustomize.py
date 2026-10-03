"""Sandbox-only bootstrap inherited by test subprocesses.

Restricted execution environments may block the socket ``send()`` syscall
(EPERM). asyncio uses it to wake the event loop from worker threads, which
deadlocks aiosqlite. Use a raw ``write()`` on the self-pipe fd instead.
"""

import os
import sys

if sys.platform != "win32":
    try:
        import asyncio.unix_events as _unix_events

        def _write_to_self(self) -> None:
            csock = self._csock
            if csock is None:
                return
            try:
                os.write(csock.fileno(), b"\x00")
            except OSError:
                try:
                    csock.send(b"\x00")
                except OSError:
                    pass

        _unix_events._UnixSelectorEventLoop._write_to_self = _write_to_self
    except (ImportError, AttributeError):  # pragma: no cover
        pass
