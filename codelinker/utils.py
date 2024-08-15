import tiktoken
from typing import Tuple

encoding = tiktoken.encoding_for_model("gpt-4o")  # TODO: this is not good

def get_token_nums(text: str) -> int:
    """
    Calculate the number of tokens in the given text.

    Args:
        text (str): The text whose tokens need to be counted.

    Returns:
        int: The number of tokens in the text.
    """
    return len(encoding.encode(str(text)))


def clip_text(text: str, max_tokens: int = None, clip_end=False) -> Tuple[str, int]:
    """
    Truncate the given text to the specified number of tokens.
    If the original text and the clipped text are not of the same length, '`wrapped`' is added to the beginning or the end of the clipped text.

    Args:
        text (str): The text to be clipped.
        max_tokens (int, optional): Maximum number of tokens. The text will be clipped to contain not more than this number of tokens.
        clip_end (bool, optional): If True, text will be clipped from the end. If False, text will be clipped from the beginning.

    Returns:
        str, int: The clipped text, and the total number of tokens in the original text.
    """
    text = str(text)
    encoded = encoding.encode(text)
    if max_tokens is not None and max_tokens <= 0:
        return '', 0
    decoded = encoding.decode(
        encoded[:max_tokens] if clip_end else encoded[-max_tokens:])
    if len(decoded) != len(text):
        decoded = decoded + '`wrapped`' if clip_end else '`wrapped`' + decoded
    return decoded, len(encoded)
