from enum import IntEnum
from pathlib import Path
from typing import Tuple

from pytest import Config


class Mode(IntEnum):
    """Mode of operation, selected by the 'replay' and 'record' flags to pytest."""

    DONT_PATCH = 0
    REPLAY = 1
    RECORD = 2
    INVALID = 3


def get_log_files(path: Path) -> Tuple[Path, Path]:
    log_path = Path(path).parent / (Path(path).stem + ".jsonl")
    log_stats_path = Path(path).parent / (Path(path).stem + ".log.jsonl")

    return log_path, log_stats_path


def get_mode_from_config(config: Config) -> Mode:
    record = bool(config.getoption("--record"))
    replay = bool(config.getoption("--replay"))
    return Mode(replay | record << 1)
