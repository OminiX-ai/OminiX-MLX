//! Language Segmentation (LangSegment)
//!
//! Port of Python's LangSegment library for detecting and segmenting
//! text by language (Chinese, English, Japanese, Korean).

use std::collections::HashSet;

/// Detected language type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Lang {
    Chinese,
    English,
    Japanese,
    Korean,
    Unknown,
}

impl Lang {
    pub fn as_str(&self) -> &'static str {
        match self {
            Lang::Chinese => "zh",
            Lang::English => "en",
            Lang::Japanese => "ja",
            Lang::Korean => "ko",
            Lang::Unknown => "unknown",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "zh" | "chinese" | "all_zh" => Lang::Chinese,
            "en" | "english" | "all_en" => Lang::English,
            "ja" | "japanese" | "all_ja" => Lang::Japanese,
            "ko" | "korean" | "all_ko" => Lang::Korean,
            _ => Lang::Unknown,
        }
    }
}

/// A segment of text with its detected language
#[derive(Debug, Clone)]
pub struct LangText {
    pub text: String,
    pub lang: Lang,
}

impl LangText {
    pub fn new(text: String, lang: Lang) -> Self {
        Self { text, lang }
    }
}

/// Language Segment detector and splitter
pub struct LangSegment {
    /// Active language filters
    filters: HashSet<Lang>,
}

impl Default for LangSegment {
    fn default() -> Self {
        Self::new()
    }
}

impl LangSegment {
    pub fn new() -> Self {
        Self {
            filters: [Lang::Chinese, Lang::English, Lang::Japanese, Lang::Korean]
                .into_iter()
                .collect(),
        }
    }

    /// Set language filters
    /// Only languages in the filter will be detected, others treated as Unknown
    pub fn set_filters(&mut self, langs: &[&str]) {
        self.filters.clear();
        for lang_str in langs {
            self.filters.insert(Lang::from_str(lang_str));
        }
    }

    /// Detect the language of a single character
    pub fn detect_char_lang(&self, c: char) -> Lang {
        let code = c as u32;

        // English/ASCII letters
        if c.is_ascii_alphabetic() {
            return if self.filters.contains(&Lang::English) {
                Lang::English
            } else {
                Lang::Unknown
            };
        }

        // CJK character ranges - need to distinguish Chinese, Japanese, Korean

        // Korean Hangul
        // Hangul Jamo: U+1100-U+11FF
        // Hangul Syllables: U+AC00-U+D7AF
        // Hangul Compatibility Jamo: U+3130-U+318F
        // Hangul Jamo Extended-A: U+A960-U+A97F
        // Hangul Jamo Extended-B: U+D7B0-U+D7FF
        if (0x1100..=0x11FF).contains(&code)
            || (0xAC00..=0xD7AF).contains(&code)
            || (0x3130..=0x318F).contains(&code)
            || (0xA960..=0xA97F).contains(&code)
            || (0xD7B0..=0xD7FF).contains(&code)
        {
            return if self.filters.contains(&Lang::Korean) {
                Lang::Korean
            } else {
                Lang::Unknown
            };
        }

        // Japanese-specific characters
        // Hiragana: U+3040-U+309F
        // Katakana: U+30A0-U+30FF
        // Katakana Phonetic Extensions: U+31F0-U+31FF
        // Half-width Katakana: U+FF65-U+FF9F
        if (0x3040..=0x309F).contains(&code)
            || (0x30A0..=0x30FF).contains(&code)
            || (0x31F0..=0x31FF).contains(&code)
            || (0xFF65..=0xFF9F).contains(&code)
        {
            return if self.filters.contains(&Lang::Japanese) {
                Lang::Japanese
            } else {
                Lang::Unknown
            };
        }

        // CJK Unified Ideographs (shared by Chinese, Japanese, Korean)
        // But we'll default to Chinese if no other signals
        // CJK Unified Ideographs: U+4E00-U+9FFF
        // CJK Extension A: U+3400-U+4DBF
        // CJK Extension B: U+20000-U+2A6DF
        // CJK Compatibility Ideographs: U+F900-U+FAFF
        // CJK Radicals Supplement: U+2E80-U+2EFF
        // Kangxi Radicals: U+2F00-U+2FDF
        if (0x4E00..=0x9FFF).contains(&code)
            || (0x3400..=0x4DBF).contains(&code)
            || (0x20000..=0x2A6DF).contains(&code)
            || (0xF900..=0xFAFF).contains(&code)
            || (0x2E80..=0x2EFF).contains(&code)
            || (0x2F00..=0x2FDF).contains(&code)
        {
            // CJK ideographs - default to Chinese (most common)
            // In real applications, you'd need context or ML model to distinguish
            return if self.filters.contains(&Lang::Chinese) {
                Lang::Chinese
            } else if self.filters.contains(&Lang::Japanese) {
                Lang::Japanese
            } else if self.filters.contains(&Lang::Korean) {
                Lang::Korean
            } else {
                Lang::Unknown
            };
        }

        Lang::Unknown
    }

    /// Get texts segmented by language
    /// Returns a vector of LangText with text and detected language
    pub fn get_texts(&self, text: &str) -> Vec<LangText> {
        if text.is_empty() {
            return vec![];
        }

        let mut result = Vec::new();
        let mut current_text = String::new();
        let mut current_lang: Option<Lang> = None;

        for c in text.chars() {
            let char_lang = self.detect_char_lang(c);

            // Determine if this is a meaningful language char
            let is_lang_char = char_lang != Lang::Unknown;
            let is_punct_or_space = c.is_whitespace()
                || c.is_ascii_punctuation()
                || is_cjk_punctuation(c);

            if is_lang_char {
                // We have a language character
                if let Some(curr) = current_lang {
                    if curr == char_lang {
                        // Same language, continue
                        current_text.push(c);
                    } else {
                        // Language changed, save current and start new
                        if !current_text.is_empty() {
                            result.push(LangText::new(current_text.clone(), curr));
                            current_text.clear();
                        }
                        current_text.push(c);
                        current_lang = Some(char_lang);
                    }
                } else {
                    // First language character
                    current_text.push(c);
                    current_lang = Some(char_lang);
                }
            } else if is_punct_or_space {
                // Punctuation/space belongs to current segment
                current_text.push(c);
            } else {
                // Other characters (digits, etc.) - keep with current segment
                current_text.push(c);
            }
        }

        // Save final segment
        if !current_text.is_empty() {
            let lang = current_lang.unwrap_or(Lang::Chinese);
            result.push(LangText::new(current_text, lang));
        }

        // Post-process: merge adjacent segments with same language
        merge_same_lang_segments(result)
    }

    /// Detect primary language of entire text
    pub fn detect_primary_lang(&self, text: &str) -> Lang {
        let mut counts = std::collections::HashMap::new();

        for c in text.chars() {
            let lang = self.detect_char_lang(c);
            if lang != Lang::Unknown {
                *counts.entry(lang).or_insert(0) += 1;
            }
        }

        counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(lang, _)| lang)
            .unwrap_or(Lang::Chinese)
    }
}

/// Check if character is CJK punctuation
fn is_cjk_punctuation(c: char) -> bool {
    matches!(c,
        '。' | '，' | '、' | '；' | '：' | '？' | '！' |
        '\u{201C}' | '\u{201D}' | '\u{2018}' | '\u{2019}' | '（' | '）' | '【' | '】' |
        '《' | '》' | '「' | '」' | '『' | '』' | '〈' | '〉' |
        '—' | '…' | '·'
    )
}

/// Merge adjacent segments with the same language
fn merge_same_lang_segments(segments: Vec<LangText>) -> Vec<LangText> {
    if segments.is_empty() {
        return segments;
    }

    let mut result = Vec::new();
    let mut iter = segments.into_iter();
    let mut current = iter.next().unwrap();

    for seg in iter {
        if seg.lang == current.lang {
            current.text.push_str(&seg.text);
        } else {
            result.push(current);
            current = seg;
        }
    }
    result.push(current);

    result
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Check if a character is Chinese
pub fn is_chinese_char(c: char) -> bool {
    let code = c as u32;
    (0x4E00..=0x9FFF).contains(&code)      // CJK Unified Ideographs
        || (0x3400..=0x4DBF).contains(&code)   // CJK Extension A
        || (0x20000..=0x2A6DF).contains(&code) // CJK Extension B
        || (0xF900..=0xFAFF).contains(&code)   // CJK Compatibility Ideographs
}

/// Check if a character is Japanese-specific (Hiragana/Katakana)
pub fn is_japanese_char(c: char) -> bool {
    let code = c as u32;
    (0x3040..=0x309F).contains(&code)      // Hiragana
        || (0x30A0..=0x30FF).contains(&code)   // Katakana
        || (0x31F0..=0x31FF).contains(&code)   // Katakana Phonetic Extensions
}

/// Check if a character is Korean (Hangul)
pub fn is_korean_char(c: char) -> bool {
    let code = c as u32;
    (0x1100..=0x11FF).contains(&code)      // Hangul Jamo
        || (0xAC00..=0xD7AF).contains(&code)   // Hangul Syllables
        || (0x3130..=0x318F).contains(&code)   // Hangul Compatibility Jamo
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_char_lang() {
        let seg = LangSegment::new();

        assert_eq!(seg.detect_char_lang('a'), Lang::English);
        assert_eq!(seg.detect_char_lang('Z'), Lang::English);
        assert_eq!(seg.detect_char_lang('你'), Lang::Chinese);
        assert_eq!(seg.detect_char_lang('好'), Lang::Chinese);
        assert_eq!(seg.detect_char_lang('あ'), Lang::Japanese);
        assert_eq!(seg.detect_char_lang('ア'), Lang::Japanese);
        assert_eq!(seg.detect_char_lang('한'), Lang::Korean);
        assert_eq!(seg.detect_char_lang('1'), Lang::Unknown);
        assert_eq!(seg.detect_char_lang(' '), Lang::Unknown);
    }

    #[test]
    fn test_get_texts_pure_chinese() {
        let seg = LangSegment::new();
        let result = seg.get_texts("你好世界");

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].text, "你好世界");
        assert_eq!(result[0].lang, Lang::Chinese);
    }

    #[test]
    fn test_get_texts_pure_english() {
        let seg = LangSegment::new();
        let result = seg.get_texts("Hello World");

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].text, "Hello World");
        assert_eq!(result[0].lang, Lang::English);
    }

    #[test]
    fn test_get_texts_mixed() {
        let seg = LangSegment::new();
        let result = seg.get_texts("Hello世界");

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].text, "Hello");
        assert_eq!(result[0].lang, Lang::English);
        assert_eq!(result[1].text, "世界");
        assert_eq!(result[1].lang, Lang::Chinese);
    }

    #[test]
    fn test_get_texts_mixed_with_punctuation() {
        let seg = LangSegment::new();
        let result = seg.get_texts("Hello, 世界！");

        // Punctuation should stay with its segment
        assert!(result.len() >= 2);
    }

    #[test]
    fn test_detect_primary_lang() {
        let seg = LangSegment::new();

        assert_eq!(seg.detect_primary_lang("你好世界"), Lang::Chinese);
        assert_eq!(seg.detect_primary_lang("Hello World"), Lang::English);
        // "你好Hello" has 2 Chinese chars and 5 English chars, so English wins by count
        assert_eq!(seg.detect_primary_lang("你好Hello"), Lang::English);
        // When Chinese has more characters
        assert_eq!(seg.detect_primary_lang("你好世界Hi"), Lang::Chinese);
    }

    #[test]
    fn test_set_filters() {
        let mut seg = LangSegment::new();
        seg.set_filters(&["en"]);

        // Now only English should be detected
        assert_eq!(seg.detect_char_lang('a'), Lang::English);
        assert_eq!(seg.detect_char_lang('你'), Lang::Unknown);
    }

    #[test]
    fn test_japanese_detection() {
        let seg = LangSegment::new();
        let result = seg.get_texts("こんにちは");

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].lang, Lang::Japanese);
    }

    #[test]
    fn test_korean_detection() {
        let seg = LangSegment::new();
        let result = seg.get_texts("안녕하세요");

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].lang, Lang::Korean);
    }

    #[test]
    fn test_helper_functions() {
        assert!(is_chinese_char('你'));
        assert!(is_chinese_char('好'));
        assert!(!is_chinese_char('a'));

        assert!(is_japanese_char('あ'));
        assert!(is_japanese_char('ア'));
        assert!(!is_japanese_char('你'));

        assert!(is_korean_char('한'));
        assert!(is_korean_char('글'));
        assert!(!is_korean_char('你'));
    }
}
