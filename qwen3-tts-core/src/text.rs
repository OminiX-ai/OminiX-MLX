//! Text processing utilities for TTS — sentence splitting, token estimation.

/// Split text into sentences for streaming TTS.
///
/// Splits on sentence-ending punctuation while preserving the punctuation
/// with the preceding sentence.
pub fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        current.push(ch);
        // Split on sentence-ending punctuation
        if matches!(ch, '.' | '!' | '?' | '。' | '！' | '？' | '；' | '\n') {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current.clear();
        }
    }

    // Don't forget trailing text without punctuation
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }

    sentences
}

/// Estimate max tokens needed for text based on character/word count.
///
/// Rough heuristic: ~1 codec frame per 0.08s of audio,
/// ~150 words per minute → ~2.5 words/sec → ~31 frames per word.
/// We use a generous multiplier to avoid truncation.
pub fn max_tokens_for_text(text: &str, base_max: i32) -> i32 {
    let word_count = text.split_whitespace().count();
    let char_count = text.chars().count();

    // Use character count for CJK (1 char ≈ 1 word), word count for Latin
    let effective_words = if has_cjk(text) {
        char_count
    } else {
        word_count
    };

    // ~30 frames per word, with 2x safety margin
    let estimated = (effective_words as i32) * 60;
    estimated.max(256).min(base_max)
}

/// Check if text contains CJK characters.
fn has_cjk(text: &str) -> bool {
    text.chars().any(|c| {
        matches!(c,
            '\u{4E00}'..='\u{9FFF}'   // CJK Unified Ideographs
            | '\u{3400}'..='\u{4DBF}' // CJK Extension A
            | '\u{3000}'..='\u{303F}' // CJK Symbols
            | '\u{3040}'..='\u{309F}' // Hiragana
            | '\u{30A0}'..='\u{30FF}' // Katakana
            | '\u{AC00}'..='\u{D7AF}' // Hangul
        )
    })
}

/// Split a long sentence into chunks that fit within max_tokens.
pub fn split_long_sentence(text: &str, max_chars: usize) -> Vec<String> {
    if text.len() <= max_chars {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut current = String::new();

    // Try splitting on comma, semicolon, or space
    for word in text.split_inclusive(|c: char| c == ',' || c == '，' || c == ';' || c == ' ') {
        if current.len() + word.len() > max_chars && !current.is_empty() {
            chunks.push(current.trim().to_string());
            current.clear();
        }
        current.push_str(word);
    }

    if !current.trim().is_empty() {
        chunks.push(current.trim().to_string());
    }

    chunks
}
