import json
import logging
import re
import sys
from contextlib import contextmanager
from datetime import datetime

_ORIGINAL_STDOUT = sys.stdout
_ORIGINAL_STDERR = sys.stderr
_ORIGINAL_EXCEPTHOOK = sys.excepthook
_TRACKING_STDOUT = None
_TRACKING_STDERR = None


class LineTrackingStream:
    def __init__(self, stream):
        self._stream = stream
        self._ends_with_newline = True

    def write(self, message):
        if message:
            self._ends_with_newline = message.endswith("\n")
        return self._stream.write(message)

    def flush(self):
        return self._stream.flush()

    def fileno(self):
        return self._stream.fileno()

    def isatty(self):
        return self._stream.isatty()

    def ensure_newline(self):
        if not self._ends_with_newline:
            self._stream.write("\n")
            self._stream.flush()
            self._ends_with_newline = True


class StreamToLogger:
    _ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

    def __init__(self, logger, level, mirror_stream=None, level_selector=None):
        self.logger = logger
        self.level = level
        self.mirror_stream = mirror_stream
        self.level_selector = level_selector
        self._buffer = ""

    def write(self, message):
        if not message:
            return
        if self.mirror_stream is not None:
            self.mirror_stream.write(message)
        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._log_line(line.rstrip("\r"))

    def flush(self):
        if self._buffer:
            self._log_line(self._buffer.rstrip("\r"))
            self._buffer = ""
        if self.mirror_stream is not None:
            self.mirror_stream.flush()

    def ensure_newline(self):
        if not self._buffer:
            return
        buffered = self._buffer
        self._buffer = ""
        if self.mirror_stream is not None:
            self.mirror_stream.write("\n")
            self.mirror_stream.flush()
        self._log_line(buffered.rstrip("\r"))

    def isatty(self):
        if self.mirror_stream is None:
            return False
        return self.mirror_stream.isatty()

    def fileno(self):
        if self.mirror_stream is None:
            raise OSError("Stream does not use a file descriptor.")
        return self.mirror_stream.fileno()

    def _log_line(self, line):
        stripped_line = self._ANSI_ESCAPE_RE.sub("", line)
        level = self.level
        if self.level_selector is not None:
            level = self.level_selector(stripped_line)
        self.logger.log(level, stripped_line)


class RunIdFilter(logging.Filter):
    def __init__(self, run_id):
        super().__init__()
        self._run_id = run_id or "-"

    def filter(self, record):
        record.run_id = self._run_id
        return True


class JsonLineHandler(logging.Handler):
    def __init__(self, path, run_id=None):
        super().__init__()
        self._path = path
        self._run_id = run_id
        self._stream = open(path, "a", encoding="utf-8")
        self._exception_formatter = logging.Formatter()

    def emit(self, record):
        payload = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "run_id": self._run_id,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            payload["exception"] = self._exception_formatter.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = record.stack_info
        self._stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._stream.flush()

    def close(self):
        try:
            if self._stream:
                self._stream.close()
        finally:
            self._stream = None
            super().close()


def setup_logging(log_path, verbose, run_id=None, event_log_path=None):
    log_level = logging.DEBUG if verbose else logging.INFO
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()
    log_format = "%(asctime)s [%(levelname)s] [%(run_id)s] %(message)s"
    global _TRACKING_STDOUT, _TRACKING_STDERR
    _TRACKING_STDOUT = LineTrackingStream(_ORIGINAL_STDOUT)
    _TRACKING_STDERR = LineTrackingStream(_ORIGINAL_STDERR)
    run_id_filter = RunIdFilter(run_id)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.addFilter(run_id_filter)
    stream_handler = logging.StreamHandler(_TRACKING_STDOUT)
    stream_handler.addFilter(run_id_filter)
    handlers = [file_handler, stream_handler]
    event_handler = None
    if event_log_path:
        event_handler = JsonLineHandler(event_log_path, run_id=run_id)
        event_handler.addFilter(run_id_filter)
        handlers.append(event_handler)
    logging_kwargs = {
        "level": log_level,
        "format": log_format,
        "handlers": handlers,
    }
    if sys.version_info >= (3, 8):
        logging_kwargs["force"] = True
    logging.basicConfig(**logging_kwargs)
    stdout_logger = logging.getLogger("stdout")
    stderr_logger = logging.getLogger("stderr")
    stdout_logger.setLevel(log_level)
    stderr_logger.setLevel(logging.INFO)
    if stdout_logger.handlers:
        for handler in stdout_logger.handlers[:]:
            stdout_logger.removeHandler(handler)
            handler.close()
    stdout_logger.propagate = False
    stdout_file_handler = logging.FileHandler(log_path, encoding="utf-8")
    stdout_file_handler.setLevel(log_level)
    stdout_file_handler.setFormatter(logging.Formatter(log_format))
    stdout_file_handler.addFilter(run_id_filter)
    stdout_logger.addHandler(stdout_file_handler)
    if event_handler is not None:
        stdout_logger.addHandler(event_handler)
    if stderr_logger.handlers:
        for handler in stderr_logger.handlers[:]:
            stderr_logger.removeHandler(handler)
            handler.close()
    stderr_logger.propagate = False
    stderr_file_handler = logging.FileHandler(log_path, encoding="utf-8")
    stderr_file_handler.setLevel(logging.INFO)
    stderr_file_handler.setFormatter(logging.Formatter(log_format))
    stderr_file_handler.addFilter(run_id_filter)
    stderr_logger.addHandler(stderr_file_handler)
    if event_handler is not None:
        stderr_logger.addHandler(event_handler)
    sys.stdout = StreamToLogger(stdout_logger, logging.INFO, mirror_stream=_TRACKING_STDOUT)

    error_patterns = [
        re.compile(r"Traceback"),
        re.compile(r"\bERROR\b"),
        re.compile(r"SCF not converged", re.IGNORECASE),
        re.compile(r"SCF (?:failed|failure|diverged|divergence)", re.IGNORECASE),
        re.compile(r"\bexplode(?:d|s|)\b", re.IGNORECASE),
        re.compile(r"\bexplosion\b", re.IGNORECASE),
    ]
    warn_patterns = [
        re.compile(r"^WARN:"),
        re.compile(r"\bWARN\b"),
    ]

    def _stderr_level_selector(line):
        for pattern in error_patterns:
            if pattern.search(line):
                return logging.ERROR
        for pattern in warn_patterns:
            if pattern.search(line):
                return logging.WARNING
        return logging.INFO

    sys.stderr = StreamToLogger(
        stderr_logger,
        logging.INFO,
        mirror_stream=_TRACKING_STDERR,
        level_selector=_stderr_level_selector,
    )

    def _log_uncaught_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            _ORIGINAL_EXCEPTHOOK(exc_type, exc_value, exc_traceback)
            return
        logging.getLogger().error(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback),
        )
        _ORIGINAL_EXCEPTHOOK(exc_type, exc_value, exc_traceback)

    sys.excepthook = _log_uncaught_exception


@contextmanager
def setup_logging_context(log_path, verbose, run_id=None, event_log_path=None):
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    original_excepthook = sys.excepthook
    setup_logging(
        log_path,
        verbose,
        run_id=run_id,
        event_log_path=event_log_path,
    )
    try:
        yield
    finally:
        if _TRACKING_STDOUT is not None:
            try:
                _TRACKING_STDOUT.ensure_newline()
            except Exception:
                pass
        if _TRACKING_STDERR is not None:
            try:
                _TRACKING_STDERR.ensure_newline()
            except Exception:
                pass
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        sys.excepthook = original_excepthook


def ensure_stream_newlines():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "ensure_newline"):
            try:
                stream.ensure_newline()
            except Exception:
                pass
