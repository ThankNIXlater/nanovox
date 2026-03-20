"""
NanoVox text tokenizer.

Converts raw text to phoneme-like token sequences suitable for the TTS model.
Uses a simple character-level tokenizer with basic text normalization.
For production use, swap in a full phonemizer (e.g. phonemizer + espeak).
"""

import re
import unicodedata
from typing import List, Optional


# Character vocabulary
_PUNCTUATION = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_DIGITS = "0123456789"
_SPECIAL = " \t\n"

PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

_VOCAB = SPECIAL_TOKENS + list(_LETTERS + _DIGITS + _PUNCTUATION + _SPECIAL)

CHAR_TO_ID = {c: i for i, c in enumerate(_VOCAB)}
ID_TO_CHAR = {i: c for i, c in enumerate(_VOCAB)}

PAD_ID = CHAR_TO_ID[PAD_TOKEN]
BOS_ID = CHAR_TO_ID[BOS_TOKEN]
EOS_ID = CHAR_TO_ID[EOS_TOKEN]
UNK_ID = CHAR_TO_ID[UNK_TOKEN]

VOCAB_SIZE = len(_VOCAB)


# Number-to-words conversion (minimal)
_ONES = [
    "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _int_to_words(n: int) -> str:
    if n == 0:
        return "zero"
    if n < 0:
        return "negative " + _int_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else "-" + _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + _int_to_words(n % 100))
    if n < 1_000_000:
        return _int_to_words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + _int_to_words(n % 1000))
    return str(n)  # fallback for large numbers


def normalize_text(text: str) -> str:
    """Basic text normalization."""
    # Unicode normalize
    text = unicodedata.normalize("NFKC", text)

    # Expand numbers
    text = re.sub(r"\b(\d+)\b", lambda m: _int_to_words(int(m.group(1))), text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Lowercase
    text = text.lower()

    return text


class CharTokenizer:
    """
    Simple character-level tokenizer with text normalization.

    For production-grade phoneme accuracy, replace with a
    full grapheme-to-phoneme model.
    """

    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self.vocab_size = VOCAB_SIZE
        self.pad_id = PAD_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID
        self.unk_id = UNK_ID

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> List[int]:
        """Encode text to token IDs."""
        if self.normalize:
            text = normalize_text(text)

        ids = []
        if add_bos:
            ids.append(BOS_ID)

        for ch in text:
            ids.append(CHAR_TO_ID.get(ch, UNK_ID))

        if add_eos:
            ids.append(EOS_ID)

        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text."""
        chars = []
        for i in ids:
            ch = ID_TO_CHAR.get(i, "")
            if ch in SPECIAL_TOKENS:
                continue
            chars.append(ch)
        return "".join(chars)

    def pad(
        self,
        sequences: List[List[int]],
        max_len: Optional[int] = None,
    ) -> tuple:
        """Pad a batch of sequences. Returns (padded, lengths)."""
        if max_len is None:
            max_len = max(len(s) for s in sequences)

        padded = []
        lengths = []
        for seq in sequences:
            length = min(len(seq), max_len)
            lengths.append(length)
            padded.append(seq[:length] + [PAD_ID] * (max_len - length))

        return padded, lengths

    @property
    def vocab(self):
        return CHAR_TO_ID


# Module-level default tokenizer
_default_tokenizer: Optional[CharTokenizer] = None


def get_tokenizer() -> CharTokenizer:
    global _default_tokenizer
    if _default_tokenizer is None:
        _default_tokenizer = CharTokenizer()
    return _default_tokenizer
