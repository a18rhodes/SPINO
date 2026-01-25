import base64
import hashlib


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
