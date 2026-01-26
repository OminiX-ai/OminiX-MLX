#!/usr/bin/env python3
"""
Compare Python GPT-SoVITS processing of "合并，并将" to debug pronunciation issues.

This script manually shows what the expected output should be based on
the GPT-SoVITS phone symbol table.
"""

# GPT-SoVITS v2 symbol table (extracted from symbols.py)
# This matches what Rust uses
symbols_v2 = [
    "!", ",", "-", ".", "?", "AA", "AA0", "AA1", "AA2", "AE0", "AE1", "AE2",
    "AH0", "AH1", "AH2", "AO0", "AO1", "AO2", "AW0", "AW1", "AW2", "AY0",
    "AY1", "AY2", "B", "CH", "D", "DH", "E1", "E2", "E3", "E4", "E5", "EE",
    "EH0", "EH1", "EH2", "ER", "ER0", "ER1", "ER2", "EY0", "EY1", "EY2",
    "En1", "En2", "En3", "En4", "En5", "F", "G", "HH", "I", "IH", "IH0",
    "IH1", "IH2", "IY0", "IY1", "IY2", "JH", "K", "L", "M", "N", "NG", "OO",
    "OW0", "OW1", "OW2", "OY0", "OY1", "OY2", "P", "R", "S", "SH", "SP",
    "SP2", "SP3", "T", "TH", "U", "UH0", "UH1", "UH2", "UNK", "UW0", "UW1",
    "UW2", "V", "W", "Y", "Z", "ZH", "_", "a", "a1", "a2", "a3", "a4", "a5",
    "ai1", "ai2", "ai3", "ai4", "ai5", "an1", "an2", "an3", "an4", "an5",
    "ang1", "ang2", "ang3", "ang4", "ang5", "ao1", "ao2", "ao3", "ao4",
    "ao5", "b", "by", "c", "ch", "cl", "d", "dy", "e", "e1", "e2", "e3",
    "e4", "e5", "ei1", "ei2", "ei3", "ei4", "ei5", "en1", "en2", "en3",
    "en4", "en5", "eng1", "eng2", "eng3", "eng4", "eng5", "er1", "er2",
    "er3", "er4", "er5", "f", "g", "gy", "h", "hy", "i", "i01", "i02",
    "i03", "i04", "i05", "i1", "i2", "i3", "i4", "i5", "ia1", "ia2", "ia3",
    "ia4", "ia5", "ian1", "ian2", "ian3", "ian4", "ian5", "iang1", "iang2",
    "iang3", "iang4", "iang5", "iao1", "iao2", "iao3", "iao4", "iao5",
    "ie1", "ie2", "ie3", "ie4", "ie5", "in1", "in2", "in3", "in4", "in5",
    "ing1", "ing2", "ing3", "ing4", "ing5", "iong1", "iong2", "iong3",
    "iong4", "iong5", "ir1", "ir2", "ir3", "ir4", "ir5", "iu1", "iu2",
    "iu3", "iu4", "iu5", "j", "k", "ky", "l", "m", "my", "n", "ny", "o",
    "o1", "o2", "o3", "o4", "o5", "ong1", "ong2", "ong3", "ong4", "ong5",
    "ou1", "ou2", "ou3", "ou4", "ou5", "p", "py", "q", "r", "ry", "s", "sh",
    "t", "ts", "u", "u1", "u2", "u3", "u4", "u5", "ua1", "ua2", "ua3",
    "ua4", "ua5", "uai1", "uai2", "uai3", "uai4", "uai5", "uan1", "uan2",
    "uan3", "uan4", "uan5", "uang1", "uang2", "uang3", "uang4", "uang5",
    "ui1", "ui2", "ui3", "ui4", "ui5", "un1", "un2", "un3", "un4", "un5",
    "uo1", "uo2", "uo3", "uo4", "uo5", "v", "v1", "v2", "v3", "v4", "v5",
    "van1", "van2", "van3", "van4", "van5", "ve1", "ve2", "ve3", "ve4",
    "ve5", "vn1", "vn2", "vn3", "vn4", "vn5", "w", "x", "y", "z", "zh", "…"
]

symbol_to_id = {s: i for i, s in enumerate(symbols_v2)}

print("=" * 60)
print("PYTHON EXPECTED OUTPUT (manual)")
print("=" * 60)

# Based on GPT-SoVITS G2P rules:
# 合 (hé) -> h + e2 (tone 2)
# 并 (bìng) -> b + ing4 (tone 4)
# ， (comma) -> SP (pause)
# 将 (jiāng) -> j + iang1 (tone 1)

test_cases = [
    ("合并", ["h", "e2", "b", "ing4"], [2, 2]),
    ("合并，并将", ["h", "e2", "b", "ing4", "SP", "b", "ing4", "j", "iang1"], [2, 2, 1, 2, 2]),
    ("合并两个文件", ["h", "e2", "b", "ing4", "l", "iang3", "g", "e5", "w", "en2", "j", "ian4"], [2, 2, 2, 2, 2, 2]),
]

for text, expected_phones, expected_word2ph in test_cases:
    print(f"\n{'=' * 60}")
    print(f"Text: {text}")
    print("=" * 60)

    print(f"\nExpected phones: {expected_phones}")
    print(f"Expected word2ph: {expected_word2ph}")

    phone_ids = [symbol_to_id.get(p, -1) for p in expected_phones]
    print(f"Phone IDs: {phone_ids}")

    # Show specific IDs for comparison
    print(f"\nKey symbol IDs:")
    print(f"  e1 = {symbol_to_id.get('e1', 'N/A')}")
    print(f"  e2 = {symbol_to_id.get('e2', 'N/A')}")
    print(f"  h = {symbol_to_id.get('h', 'N/A')}")
    print(f"  b = {symbol_to_id.get('b', 'N/A')}")
    print(f"  ing4 = {symbol_to_id.get('ing4', 'N/A')}")
    print(f"  SP = {symbol_to_id.get('SP', 'N/A')}")

print("\n" + "=" * 60)
print("COMPARISON WITH RUST OUTPUT")
print("=" * 60)
print("""
If Rust produces these exact same phone IDs and word2ph, then
the G2P layer is working correctly.

The issue must be in one of these areas:
1. BERT feature extraction/alignment
2. T2S model inference (semantic token generation)
3. VITS vocoder (audio synthesis)

Since phonemes are correct but audio sounds wrong, the most
likely culprit is the T2S model generating wrong semantic
tokens, which then causes VITS to produce wrong pronunciation.

To debug further:
- Compare BERT hidden states between Python and Rust
- Compare T2S logits at each generation step
- Compare final semantic token sequences
""")

# Print e1 vs e2 distinction
print("\n" + "=" * 60)
print("TONE DISTINCTION: e1 vs e2")
print("=" * 60)
print(f"""
In the symbol table:
- e1 (tone 1, level) is at index {symbol_to_id.get('e1', 'N/A')}
- e2 (tone 2, rising) is at index {symbol_to_id.get('e2', 'N/A')}

For 合 (hé), the correct phoneme is 'e2' (tone 2, rising).
For 合 pronounced as 'hē' (tone 1, level), the phoneme would be 'e1'.

If the T2S model outputs semantic tokens for 'e1' when given 'e2',
the resulting audio will sound like tone 1 instead of tone 2.

This could happen if:
1. The T2S model has learned to conflate similar phonemes
2. The BERT context is confusing the model
3. There's a bug in how phoneme IDs are processed

The fact that '合并两个' works but '合并，并将' doesn't suggests
the model is confused by the repetition of '并' or the pause context.
""")
