import logging
import os

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs")
LOG_FILE = os.path.join(LOG_DIR, "hygraph.log")

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR, exist_ok=True)

_logger = logging.getLogger("hygraph-global-logger")
_logger.setLevel(logging.INFO)

if not _logger.handlers:
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(filename)s:%(lineno)d: %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    _logger.addHandler(file_handler)
    _logger.addHandler(stream_handler)
    _logger.propagate = False
_noisy_loggers = [
    "httpx",
    "httpcore",
    "numexpr",
    "numexpr.utils",
]

for name in _noisy_loggers:
    lg = logging.getLogger(name)
    lg.setLevel(logging.WARNING)
    lg.propagate = False
def get_global_logger():
    """
    Get global logger.
    Usage example:
        from hygraph.Logger.index import get_global_logger
        logger = get_global_logger()
        logger.info("Hello, logger!")
    """
    return _logger

