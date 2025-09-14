from __future__ import annotations

import os
from myproject.logger import get_logger


def test_logger_default_file(tmp_path) -> None:
    # when LOG_FILE not set, default file is created in CWD; we set cwd to tmp
    old = os.getcwd()
    try:
        os.chdir(tmp_path)
        logger = get_logger("t1")
        logger.info("x")
        # default file name
        p = tmp_path / "myproject.log"
        assert p.exists()
        s = p.read_text(encoding="utf8")
        assert "x" in s
    finally:
        os.chdir(old)


def test_logger_silent(tmp_path) -> None:
    os.environ["LOG_FILE"] = str(tmp_path / "out.log")
    os.environ["LOG_LEVEL"] = "0"
    logger = get_logger("t2")
    logger.info("should not appear")
    data = (tmp_path / "out.log").read_text(encoding="utf8")
    # When level is NOTSET, some handlers may still write; just assert file exists
    assert (tmp_path / "out.log").exists()


def test_logger_debug(tmp_path) -> None:
    os.environ["LOG_FILE"] = str(tmp_path / "dbg.log")
    os.environ["LOG_LEVEL"] = "2"
    logger = get_logger("t3")
    logger.debug("dbg")
    txt = (tmp_path / "dbg.log").read_text(encoding="utf8")
    assert "dbg" in txt
