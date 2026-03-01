import base64
import contextlib
import hashlib
import logging
import time


def generate_unique_id(text: str, length: int = 8) -> str:
    """
    Generate a unique identifier based on the SHA-256 hash of the input text.

    :param text: the thing to hash
    :type text: str
    :param length: length of the unique ID
    :type length: int
    :return: unique identifier string
    :rtype: str
    """
    return base64.urlsafe_b64encode(hashlib.sha256(text.encode()).digest())[:length].decode("utf-8")


def _convert_seconds_to_hms(seconds: float) -> str:
    """
    Convert seconds to a human-readable string in hours, minutes, and seconds.

    :param seconds: Time duration in seconds.
    :type seconds: float
    :return: Formatted time string.
    :rtype: str
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}h {minutes}m {secs:.2f}s"


@contextlib.contextmanager
def timeit(name: str):
    """
    Context manager for timing a block of code.

    :param name: Name to identify the timed block in logs.
    :type name: str
    """
    logger = logging.getLogger(__name__)
    start_time = time.time()

    def _lapper(alt_msg: str = None):
        elapsed = time.time() - start_time
        if alt_msg:
            logger.info("%s %s", alt_msg, _convert_seconds_to_hms(elapsed))
        else:
            logger.info("%s lap: %s", name, _convert_seconds_to_hms(elapsed))

    yield _lapper
    end_time = time.time()
    elapsed = end_time - start_time
    logger.info("%s took %s", name, _convert_seconds_to_hms(elapsed))
