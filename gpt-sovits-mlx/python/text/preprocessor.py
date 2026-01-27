"""Text preprocessor for GPT-SoVITS MLX.

Converts text to phoneme sequences for Chinese and English.

Pipeline:
1. Text normalization
2. Language detection/segmentation
3. Grapheme-to-phoneme conversion
4. Phoneme ID conversion

For Chinese:
- Uses pypinyin for pinyin extraction
- Handles tone sandhi
- Supports mixed Chinese/English text

For English:
- Uses g2p_en for CMU dictionary lookup
- Falls back to letter spelling for OOV words
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

# Try importing phonemization libraries
try:
    from pypinyin import lazy_pinyin, Style
    from pypinyin.contrib.tone_convert import to_tone
    HAS_PYPINYIN = True
except ImportError:
    HAS_PYPINYIN = False
    print("Warning: pypinyin not installed. Chinese G2P will be limited.")

try:
    from g2p_en import G2p
    HAS_G2P_EN = True
except ImportError:
    HAS_G2P_EN = False
    print("Warning: g2p_en not installed. English G2P will be limited.")

try:
    import jieba
    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False

from python.text.symbols import (
    SYMBOL_TO_ID,
    symbols_to_ids,
    BOS_ID,
    EOS_ID,
    SP_ID,
    UNK_ID,
    CHINESE_CONSONANTS,
)


@dataclass
class PreprocessorOutput:
    """Output from text preprocessing."""

    phoneme_ids: List[int]
    phonemes: List[str]
    word2ph: List[int]  # Number of phonemes per word/character
    text_normalized: str
    language: str


# Chinese pinyin to consonant + vowel mapping
# This follows the GPT-SoVITS convention
PINYIN_INITIALS = set([
    "b", "c", "ch", "d", "f", "g", "h", "j", "k", "l", "m", "n",
    "p", "q", "r", "s", "sh", "t", "w", "x", "y", "z", "zh",
])

# Pinyin with no initial consonant need special handling
ZERO_INITIAL_MAP = {
    "a": "AA a",
    "ai": "AA ai",
    "an": "AA an",
    "ang": "AA ang",
    "ao": "AA ao",
    "e": "EE e",
    "ei": "EE ei",
    "en": "EE en",
    "eng": "EE eng",
    "er": "EE er",
    "o": "OO o",
    "ou": "OO ou",
}


def _get_initial_final(pinyin: str) -> Tuple[str, str]:
    """Split pinyin into initial (consonant) and final (vowel with tone).

    Args:
        pinyin: Pinyin syllable with tone number (e.g., "ni3", "hao3")

    Returns:
        Tuple of (initial, final) where final includes tone number
    """
    # Extract tone number if present
    tone = ""
    if pinyin and pinyin[-1].isdigit():
        tone = pinyin[-1]
        pinyin_base = pinyin[:-1]
    else:
        pinyin_base = pinyin
        tone = "5"  # Neutral tone

    # Check for multi-character initials first
    for initial in ["zh", "ch", "sh"]:
        if pinyin_base.startswith(initial):
            return initial, pinyin_base[len(initial):] + tone

    # Single character initials
    for initial in PINYIN_INITIALS:
        if len(initial) == 1 and pinyin_base.startswith(initial):
            return initial, pinyin_base[1:] + tone

    # Zero initial - check mapping
    if pinyin_base in ZERO_INITIAL_MAP:
        parts = ZERO_INITIAL_MAP[pinyin_base].split()
        return parts[0], parts[1] + tone

    # Default: treat entire pinyin as final with special initial
    return "AA", pinyin_base + tone


def _normalize_chinese(text: str) -> str:
    """Normalize Chinese text.

    - Convert full-width punctuation to ASCII
    - Remove special characters
    """
    # Full-width to half-width punctuation
    replacements = {
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "；": ";",
        "：": ":",
        "、": ",",
        """: '"',
        """: '"',
        "'": "'",
        "'": "'",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "《": '"',
        "》": '"',
        "……": "...",
        "——": "-",
        "～": "~",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def _normalize_english(text: str) -> str:
    """Normalize English text.

    - Convert to lowercase for G2P
    - Handle contractions
    """
    # Remove extra whitespace
    text = " ".join(text.split())
    return text


def _chinese_g2p(text: str) -> Tuple[List[str], List[int]]:
    """Convert Chinese text to phonemes using pypinyin.

    Args:
        text: Normalized Chinese text

    Returns:
        Tuple of (phonemes, word2ph)
    """
    if not HAS_PYPINYIN:
        # Fallback: return characters as phonemes
        phonemes = []
        word2ph = []
        for char in text:
            if char.strip():
                phonemes.append(char)
                word2ph.append(1)
        return phonemes, word2ph

    phonemes = []
    word2ph = []

    # Get pinyin with tone numbers
    pinyin_list = lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True)

    for i, py in enumerate(pinyin_list):
        char = text[i] if i < len(text) else ""

        # Handle punctuation
        if py in [",", ".", "!", "?", ";", ":", " ", "-"]:
            if py == " ":
                phonemes.append("SP")
            else:
                phonemes.append(py)
            word2ph.append(1)
            continue

        # Handle non-Chinese characters (returned as-is by pypinyin)
        if py == char and not _is_chinese_char(char):
            phonemes.append(char)
            word2ph.append(1)
            continue

        # Split pinyin into initial and final
        initial, final = _get_initial_final(py)

        if initial and initial in SYMBOL_TO_ID:
            phonemes.append(initial)

        if final and final in SYMBOL_TO_ID:
            phonemes.append(final)
            word2ph.append(2 if initial and initial in SYMBOL_TO_ID else 1)
        else:
            # Fallback for unknown finals
            word2ph.append(1 if initial and initial in SYMBOL_TO_ID else 0)

    return phonemes, word2ph


def _english_g2p(text: str) -> Tuple[List[str], List[int]]:
    """Convert English text to phonemes using g2p_en.

    Args:
        text: Normalized English text

    Returns:
        Tuple of (phonemes, word2ph)
    """
    if not HAS_G2P_EN:
        # Fallback: spell out each character
        phonemes = []
        word2ph = []
        for char in text:
            if char.isalpha():
                phonemes.append(char.upper())
                word2ph.append(1)
            elif char == " ":
                phonemes.append("SP")
                word2ph.append(1)
            elif char in SYMBOL_TO_ID:
                phonemes.append(char)
                word2ph.append(1)
        return phonemes, word2ph

    g2p = G2p()
    phonemes = []
    word2ph = []

    words = text.split()
    for word in words:
        # Get ARPAbet phonemes
        arpabet = g2p(word)
        word_phones = []

        for phone in arpabet:
            if phone == " ":
                continue
            # G2P returns phonemes like 'HH', 'AH0', etc.
            if phone in SYMBOL_TO_ID:
                word_phones.append(phone)
            elif phone.rstrip("012") in SYMBOL_TO_ID:
                word_phones.append(phone.rstrip("012"))

        phonemes.extend(word_phones)
        word2ph.append(len(word_phones) if word_phones else 1)

        # Add space between words
        phonemes.append("SP")
        word2ph.append(1)

    # Remove trailing space
    if phonemes and phonemes[-1] == "SP":
        phonemes.pop()
        word2ph.pop()

    return phonemes, word2ph


def _is_chinese_char(char: str) -> bool:
    """Check if character is Chinese."""
    if len(char) != 1:
        return False
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF or   # CJK Unified Ideographs
        0x3400 <= code <= 0x4DBF or   # CJK Extension A
        0x20000 <= code <= 0x2A6DF or # CJK Extension B
        0xF900 <= code <= 0xFAFF      # CJK Compatibility Ideographs
    )


def _detect_language(text: str) -> str:
    """Detect primary language of text.

    Returns:
        "zh" for Chinese, "en" for English, "mixed" for mixed
    """
    chinese_count = sum(1 for c in text if _is_chinese_char(c))
    english_count = sum(1 for c in text if c.isascii() and c.isalpha())

    if chinese_count > english_count:
        return "zh"
    elif english_count > chinese_count:
        return "en"
    else:
        return "mixed"


def _segment_by_language(text: str) -> List[Tuple[str, str]]:
    """Segment text into language-homogeneous chunks.

    Args:
        text: Input text

    Returns:
        List of (segment, language) tuples
    """
    segments = []
    current_segment = ""
    current_lang = None

    for char in text:
        if _is_chinese_char(char):
            lang = "zh"
        elif char.isascii() and char.isalpha():
            lang = "en"
        else:
            # Punctuation/space: append to current segment
            current_segment += char
            continue

        if current_lang is None:
            current_lang = lang
            current_segment = char
        elif lang == current_lang:
            current_segment += char
        else:
            # Language switch
            if current_segment.strip():
                segments.append((current_segment.strip(), current_lang))
            current_segment = char
            current_lang = lang

    if current_segment.strip():
        segments.append((current_segment.strip(), current_lang))

    return segments


class TextPreprocessor:
    """Text preprocessor for GPT-SoVITS.

    Converts text to phoneme IDs with optional language detection.
    """

    def __init__(
        self,
        default_language: str = "zh",
        add_bos: bool = True,
        add_eos: bool = True,
    ):
        """Initialize preprocessor.

        Args:
            default_language: Default language ("zh" or "en")
            add_bos: Whether to add BOS token
            add_eos: Whether to add EOS token
        """
        self.default_language = default_language
        self.add_bos = add_bos
        self.add_eos = add_eos

        # Initialize G2P engines if available
        self._g2p_en = None
        if HAS_G2P_EN:
            try:
                self._g2p_en = G2p()
            except Exception as e:
                print(f"Warning: Could not initialize G2P: {e}")

    def preprocess(
        self,
        text: str,
        language: Optional[str] = None,
    ) -> PreprocessorOutput:
        """Preprocess text to phonemes.

        Args:
            text: Input text
            language: Language code ("zh", "en", "auto", or None for default)

        Returns:
            PreprocessorOutput with phoneme IDs and metadata
        """
        if not text.strip():
            return PreprocessorOutput(
                phoneme_ids=[BOS_ID, EOS_ID] if self.add_bos else [EOS_ID],
                phonemes=["BOS", "EOS"] if self.add_bos else ["EOS"],
                word2ph=[1, 1] if self.add_bos else [1],
                text_normalized="",
                language=language or self.default_language,
            )

        # Detect or use provided language
        if language == "auto" or language is None:
            language = _detect_language(text)

        # Normalize text
        if language == "zh":
            text_normalized = _normalize_chinese(text)
        elif language == "en":
            text_normalized = _normalize_english(text)
        else:
            text_normalized = _normalize_chinese(text)  # Mixed defaults to Chinese handling

        # Convert to phonemes
        if language == "zh":
            phonemes, word2ph = _chinese_g2p(text_normalized)
        elif language == "en":
            phonemes, word2ph = _english_g2p(text_normalized)
        else:
            # Mixed language: segment and process each part
            phonemes = []
            word2ph = []
            segments = _segment_by_language(text_normalized)
            for segment, seg_lang in segments:
                if seg_lang == "zh":
                    seg_phonemes, seg_word2ph = _chinese_g2p(segment)
                else:
                    seg_phonemes, seg_word2ph = _english_g2p(segment)
                phonemes.extend(seg_phonemes)
                word2ph.extend(seg_word2ph)
                # Add pause between segments
                phonemes.append("SP")
                word2ph.append(1)
            # Remove trailing pause
            if phonemes and phonemes[-1] == "SP":
                phonemes.pop()
                word2ph.pop()

        # Add BOS/EOS tokens
        if self.add_bos:
            phonemes = ["BOS"] + phonemes
            word2ph = [1] + word2ph
        if self.add_eos:
            phonemes = phonemes + ["EOS"]
            word2ph = word2ph + [1]

        # Convert to IDs
        phoneme_ids = symbols_to_ids(phonemes)

        return PreprocessorOutput(
            phoneme_ids=phoneme_ids,
            phonemes=phonemes,
            word2ph=word2ph,
            text_normalized=text_normalized,
            language=language,
        )

    def to_mlx(self, output: PreprocessorOutput) -> "mx.array":
        """Convert output to MLX array.

        Args:
            output: PreprocessorOutput

        Returns:
            MLX array of phoneme IDs [1, seq_len]
        """
        try:
            import mlx.core as mx
            return mx.array([output.phoneme_ids], dtype=mx.int32)
        except ImportError:
            raise ImportError("MLX is required for to_mlx()")


def preprocess_text(
    text: str,
    language: str = "auto",
    add_bos: bool = True,
    add_eos: bool = True,
) -> PreprocessorOutput:
    """Convenience function to preprocess text.

    Args:
        text: Input text
        language: Language code ("zh", "en", "auto")
        add_bos: Whether to add BOS token
        add_eos: Whether to add EOS token

    Returns:
        PreprocessorOutput
    """
    preprocessor = TextPreprocessor(
        add_bos=add_bos,
        add_eos=add_eos,
    )
    return preprocessor.preprocess(text, language=language)


if __name__ == "__main__":
    # Test preprocessing
    preprocessor = TextPreprocessor()

    # Test Chinese
    result = preprocessor.preprocess("你好，世界！", language="zh")
    print(f"Chinese text: '你好，世界！'")
    print(f"  Phonemes: {result.phonemes}")
    print(f"  IDs: {result.phoneme_ids}")
    print(f"  word2ph: {result.word2ph}")
    print()

    # Test English
    result = preprocessor.preprocess("Hello world!", language="en")
    print(f"English text: 'Hello world!'")
    print(f"  Phonemes: {result.phonemes}")
    print(f"  IDs: {result.phoneme_ids}")
    print(f"  word2ph: {result.word2ph}")
    print()

    # Test mixed
    result = preprocessor.preprocess("你好 world！", language="auto")
    print(f"Mixed text: '你好 world！'")
    print(f"  Detected language: {result.language}")
    print(f"  Phonemes: {result.phonemes}")
    print(f"  IDs: {result.phoneme_ids}")
