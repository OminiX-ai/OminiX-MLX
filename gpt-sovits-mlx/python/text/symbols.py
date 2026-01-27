"""Phoneme symbols for GPT-SoVITS.

Based on the GPT-SoVITS phoneme vocabulary, combining:
- Chinese pinyin consonants and vowels with tones
- English ARPAbet phonemes
- Punctuation and special tokens
"""

from typing import Dict, List

# Special tokens
PAD = "_"
UNK = "UNK"
BOS = "BOS"  # Beginning of sequence
EOS = "EOS"  # End of sequence
SP = "SP"    # Short pause
SP2 = "SP2"  # Medium pause
SP3 = "SP3"  # Long pause

# Chinese consonants (initials)
CHINESE_CONSONANTS = [
    "b", "c", "ch", "d", "f", "g", "h", "j", "k", "l", "m", "n",
    "p", "q", "r", "s", "sh", "t", "w", "x", "y", "z", "zh",
    "AA", "EE", "OO",  # Special vowel-only initials
]

# Chinese vowels (finals) with tones 1-5
CHINESE_VOWELS_BASE = [
    "E", "En", "a", "ai", "an", "ang", "ao", "e", "ei", "en", "eng", "er",
    "i", "ia", "ian", "iang", "iao", "ie", "in", "ing", "iong", "ir", "iu",
    "o", "ong", "ou", "u", "ua", "uai", "uan", "uang", "ui", "un", "uo",
    "v", "van", "ve", "vn",
]

# Generate vowels with tones
CHINESE_VOWELS = []
for vowel in CHINESE_VOWELS_BASE:
    for tone in range(1, 6):  # Tones 1-5
        CHINESE_VOWELS.append(f"{vowel}{tone}")

# English ARPAbet phonemes
ENGLISH_VOWELS = [
    "AA0", "AA1", "AA2",
    "AE0", "AE1", "AE2",
    "AH0", "AH1", "AH2",
    "AO0", "AO1", "AO2",
    "AW0", "AW1", "AW2",
    "AY0", "AY1", "AY2",
    "EH0", "EH1", "EH2",
    "ER0", "ER1", "ER2",
    "EY0", "EY1", "EY2",
    "IH0", "IH1", "IH2",
    "IY0", "IY1", "IY2",
    "OW0", "OW1", "OW2",
    "OY0", "OY1", "OY2",
    "UH0", "UH1", "UH2",
    "UW0", "UW1", "UW2",
]

ENGLISH_CONSONANTS = [
    "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N",
    "NG", "P", "R", "S", "SH", "T", "TH", "V", "W", "Y", "Z", "ZH",
]

# Punctuation
PUNCTUATION = ["!", "?", ".", ",", ";", ":", "-", "'", '"', "(", ")", " "]

# Build full symbol list
SPECIAL_TOKENS = [PAD, UNK, BOS, EOS, SP, SP2, SP3]
ALL_PHONEMES = (
    CHINESE_CONSONANTS +
    CHINESE_VOWELS +
    ENGLISH_VOWELS +
    ENGLISH_CONSONANTS +
    PUNCTUATION
)

# Full symbol list: special tokens first, then phonemes
SYMBOLS: List[str] = SPECIAL_TOKENS + ALL_PHONEMES

# Create mappings
SYMBOL_TO_ID: Dict[str, int] = {s: i for i, s in enumerate(SYMBOLS)}
ID_TO_SYMBOL: Dict[int, str] = {i: s for i, s in enumerate(SYMBOLS)}

# Special token IDs
PAD_ID = SYMBOL_TO_ID[PAD]
UNK_ID = SYMBOL_TO_ID[UNK]
BOS_ID = SYMBOL_TO_ID[BOS]
EOS_ID = SYMBOL_TO_ID[EOS]
SP_ID = SYMBOL_TO_ID[SP]

# Total vocabulary size
VOCAB_SIZE = len(SYMBOLS)


def symbol_to_id(symbol: str) -> int:
    """Convert symbol to ID, with fallback to UNK."""
    return SYMBOL_TO_ID.get(symbol, UNK_ID)


def id_to_symbol(idx: int) -> str:
    """Convert ID to symbol, with fallback to UNK."""
    return ID_TO_SYMBOL.get(idx, UNK)


def symbols_to_ids(symbols: List[str]) -> List[int]:
    """Convert list of symbols to IDs."""
    return [symbol_to_id(s) for s in symbols]


def ids_to_symbols(ids: List[int]) -> List[str]:
    """Convert list of IDs to symbols."""
    return [id_to_symbol(i) for i in ids]


# For compatibility with original GPT-SoVITS phoneme vocab (732 tokens)
# We expand to cover all needed symbols
assert VOCAB_SIZE <= 732, f"Vocabulary size {VOCAB_SIZE} exceeds expected 732"


if __name__ == "__main__":
    print(f"Total symbols: {VOCAB_SIZE}")
    print(f"Special tokens: {SPECIAL_TOKENS}")
    print(f"Chinese consonants: {len(CHINESE_CONSONANTS)}")
    print(f"Chinese vowels: {len(CHINESE_VOWELS)}")
    print(f"English vowels: {len(ENGLISH_VOWELS)}")
    print(f"English consonants: {len(ENGLISH_CONSONANTS)}")
    print(f"Punctuation: {len(PUNCTUATION)}")
