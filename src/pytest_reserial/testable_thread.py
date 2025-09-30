"""
Override the `Thread` class.

This will buble any exception so that pytest can report them when running tests.
"""

from __future__ import annotations

from threading import Thread


class TestableThread(Thread):
    """
    Wrapper around `threading.Thread` that propagates exceptions.

    REF: https://gist.github.com/sbrugman/59b3535ebcd5aa0e2598293cfa58b6ab
    """

    def __init__(self, *args, **kwargs) -> None: # noqa: ANN002, ANN003
        """Override the `Thread.__init__` method."""
        super().__init__(*args, **kwargs)
        self.exc = None

    def run(self) -> None:
        """Override the `Thread.run` method."""
        try:
            super().run()
        except BaseException as e: # noqa: BLE001
            self.exc = e

    def join(self, timeout: float | None =None) -> None:
        """Override the `Thread.join` method."""
        super().join(timeout)
        if self.exc:
            raise self.exc
