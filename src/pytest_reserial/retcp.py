"""Record or replay tcp traffic when running tests."""

from __future__ import annotations

import socket
import socketserver
import threading
from collections.abc import Buffer, Generator
import time
from typing import Any, Callable

import pytest

from pytest_reserial.common import Mode, get_log_files, get_mode_from_config
from pytest_reserial.reserial import (
    TrafficLog,
    TrafficLogStats,
    get_traffic_log,
    get_traffic_log_stats,
    write_log,
)
from pytest_reserial.testable_thread import TestableThread

PatchMethods = tuple[
    Callable[[socket.socket, int, int], bytes], # recv
    Callable[[socket.socket, Buffer, int], int ], # send
    Callable[[socket.socket, Buffer, int], None], # sendall
    Callable[[socket.socket], None], # close
]


class TestTCPHandler(socketserver.BaseRequestHandler):
    """
    TCP server handler class.

    Handles incoming TCP connections, verifies messages,
    and responds based on predefined expectations.
    """

    log: TrafficLog
    log_stats: TrafficLogStats
    mode: Mode
    shutdown_evt: threading.Event

    def handle(self: TestTCPHandler) -> None:
        """Handle incoming TCP connection request."""
        # NOTE: handle_replay() will return False only if
        #       there are more messages recorded, but the client will disconnect.
        #       This means that a new connection should be made to consume the rest of
        #       the messages. In any other case, shutdown the server.
        shutdown = True
        try:
            if self.mode != Mode.REPLAY:
                msg = "Only --replay is supported"
                pytest.fail(msg)
            shutdown = self.handle_replay()
        finally:
            if shutdown:
                self.shutdown_evt.set()


    def handle_replay(self: TestTCPHandler) -> bool:
        """Iterate over expected data and respond the the client accordingly.

        Returns
        -------
        bool: True if there is no more expected data, or False if we expect more data.
              If we return True, the TCPServer will shutdown.
        """
        # For better type hints
        _socket: socket.socket = self.request

        while len(self.log_stats) > 0:
            stat_chunk = self.log_stats.pop(0)
            if stat_chunk[0] == "c":
                # If there are more messages, we do not shutdown
                return len(self.log_stats) == 0

            if stat_chunk[0] != "w":
                pytest.fail("First message not Write (client side)\n")

            data = bytes(_socket.recv(stat_chunk[1]))
            size = len(data)

            expected_data = self.log["tx"][:size]
            self.log["tx"] = self.log["tx"][size:]

            if data != expected_data:
                pytest.fail(f"Expected {expected_data}, got {data}")

            if len(self.log_stats) > 0 and self.log_stats[0][0] == "t":
                stat = self.log_stats.pop(0)
                time.sleep(stat[1])

            while len(self.log_stats) > 0 and \
                    (self.log_stats[0][0] == "r" or self.log_stats[0][0] == "t"):

                stat = self.log_stats.pop(0)

                if stat[0] == "t":
                    time.sleep(stat[1])
                    if len(self.log_stats) == 0:
                        break
                    stat = self.log_stats.pop(0)

                size = stat[1]

                to_send = self.log["rx"][:size]
                self.log["rx"] = self.log["rx"][size:]

                _socket.sendall(to_send)

        return True


@pytest.fixture
def retcp(monkeypatch: pytest.MonkeyPatch,
          request: pytest.FixtureRequest) -> Generator[str, Any, Any]:
    """Record or replay tcp traffic.

    Raises
    ------
    _pytest.outcomes.Failed
        If less data than expected was read or written during replay.
    """
    address_param = str(request.param).split(":")
    addr: str = address_param[0]
    port: int = int(address_param[1])

    mode = get_mode_from_config(request.config)

    log_path, log_stats_path = get_log_files(request.path)
    test_name = request.node.name
    log = get_traffic_log(mode, log_path, test_name)
    log_stats = get_traffic_log_stats(mode, log_stats_path, test_name)

    tests_thread: TestableThread | None = None
    watcher_thread: threading.Thread | None = None

    if mode == Mode.REPLAY:
        server = socketserver.TCPServer(("localhost", 0), TestTCPHandler)

        shutdown_evt = threading.Event()
        def stop_server_worker() -> None:
            shutdown_evt.wait()
            server.shutdown()

        if len(log_stats) == 0:
            pytest.fail(f"No log_stats recorded for test '{test_name}'\n")

        TestTCPHandler.log = log
        TestTCPHandler.log_stats = log_stats
        TestTCPHandler.mode = mode
        TestTCPHandler.shutdown_evt = shutdown_evt

        addr = "localhost"
        port = server.server_address[1]

        watcher_thread = threading.Thread(target=stop_server_worker)
        watcher_thread.start()

        # Run the server in a background thread
        tests_thread = TestableThread(
            target=server.serve_forever)

        tests_thread.start()
    else:
        (
            recv_patch,
            send_patch,
            sendall_patch,
            close_patch,
        ) = get_patched_methods(mode, log, log_stats)
        monkeypatch.setattr(socket.socket, "recv", recv_patch)
        monkeypatch.setattr(socket.socket, "send", send_patch)
        monkeypatch.setattr(socket.socket, "sendall", sendall_patch)
        monkeypatch.setattr(socket.socket, "close", close_patch)

    yield f"{addr}:{port}"

    if mode == Mode.RECORD:
        write_log(log, log_stats, log_path, log_stats_path, test_name)
        return

    if mode == Mode.REPLAY and tests_thread and watcher_thread:
        watcher_thread.join()
        tests_thread.join()

        if tests_thread.exc:
            pytest.fail(str(tests_thread.exc))

    if log["rx"] or log["tx"] or len(log_stats) > 0:
        msg = (
            "Some messages where not replayed\n"
            f"Remaining RX:    {len(log['rx'])}\n"
            f"Remaining TX:    {len(log['tx'])}\n"
            f"Remaining Stats: {len(log_stats)}\n"
        )
        pytest.fail(msg)


def get_patched_methods(mode: Mode, log: TrafficLog,
                        log_stats: TrafficLogStats) -> PatchMethods:
    """Return patched recv, send, close, etc methods.

    The methods should be monkeypatched over the corresponding `socket.socket` methods.

    Parameters
    ----------
    mode: Mode
        The requested mode of operation, i.e. `REPLAY`, `RECORD`, or `DONT_PATCH`.
    log: dict[str, list[int]]
        Dictionary holding logged traffic (replay) / where traffic will be logged to
        (record). If mode is `DONT_PATCH`, this parameter is ignored.
    log_stats: list[Tuple[Literal["r", "w", "c"], int]]
        List of read, write and close events and the size of data that was exchanged.

    Returns
    -------
    record_recv: Callable[[socket.socket, int, int], bytes]
        Monkeypatch this over `socket.socket.recv`.
    record_send: Callable[[socket.socket, Buffer, int], int ]
        Monkeypatch this over `socket.socket.send`.
    record_sendall: Callable[[socket.socket, Buffer, int], None ]
        Monkeypatch this over `socket.socket.sendall`.
    record_close: Callable[[socket.socket], None]
        Monkeypatch this over `socket.socket.close`.
    """
    if mode == Mode.RECORD:
        return get_record_methods(log, log_stats)
    return (
        socket.socket.recv,
        socket.socket.send,
        socket.socket.sendall,
        socket.socket.close,
    )


def get_record_methods(log: TrafficLog, log_stats: TrafficLogStats) -> PatchMethods:
    """Return patched recv, send, close, etc methods for recording traffic.

    Parameters
    ----------
    log: dict[str, list[int]]
        Dictionary where recorded traffic will be logged.

    log_stats: list[Tuple[Literal["r", "w", "c"], int]]
        List of read, write and close events and the size of data that was exchanged.

    Returns
    -------
    record_recv: Callable[[socket.socket, int, int], bytes]
        Logs RX data read from the socket.
    record_send: Callable[[socket.socket, Buffer, int], int ]
        Logs TX data before writing it to the socket.
    record_sendall: Callable[[socket.socket, Buffer, int], None ]
        Logs TX data before writing it to the socket.
    record_close: Callable[[socket.socket], None]
        Logs when close is called to record the event.
    """
    real_recv = socket.socket.recv
    real_send = socket.socket.send
    real_sendall = socket.socket.sendall
    real_close = socket.socket.close

    def record_send(self: socket.socket, data: Buffer, flags: int = 0) -> int:
        """Record TX data before writing to the socket.

        Monkeypatch this method over socket.send to record traffic. Parameters and
        return values are identical to socket.send.
        """
        size = memoryview(data).nbytes
        log["tx"] += data
        log_stats.append(("w", size))
        written: int = real_send(self, data, flags)
        return written

    def record_sendall(self: socket.socket, data: Buffer, flags: int = 0) -> None:
        """Record TX data before writing to the socket.

        Monkeypatch this method over socket.sendall to record traffic. Parameters and
        return values are identical to socket.sendall.
        """
        size = memoryview(data).nbytes
        log["tx"] += data
        log_stats.append(("w", size))
        real_sendall(self, data, flags)

    def record_recv(self: socket.socket, bufsize: int, flags: int = 0) -> bytes:
        """Record RX data after reading from the socket.

        Monkeypatch this method over socket.recv to record traffic. Parameters and
        return values are identical to socket.recv.
        """
        data: bytes = real_recv(self, bufsize, flags)
        log["rx"] += data
        log_stats.append(("r", len(data)))
        return data

    def record_close(self: socket.socket) -> None:
        """Record the close connection call.

        Monkeypatch this method over socket.close to record close event. Parameters and
        return values are identical to socket.close.
        """
        log_stats.append(("c", -1))
        real_close(self)

    return (
        record_recv,
        record_send,
        record_sendall,
        record_close,
    )
