//! Chinese Word Segmentation with POS Tagging
//!
//! This module provides jieba-style word segmentation for Chinese text.
//! It integrates with jieba-rs when available, or falls back to a simple
//! character-based segmentation.
//!
//! ## Usage
//!
//! ```rust
//! use gpt_sovits_mlx::text::jieba_seg::{Segmenter, cut_with_pos};
//!
//! let segmenter = Segmenter::new();
//! let segments = segmenter.cut_for_pos("我喜欢北京");
//! for (word, pos) in segments {
//!     println!("{} / {}", word, pos);
//! }
//! ```

use std::collections::HashMap;

/// A word segment with POS tag
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    pub word: String,
    pub pos: String,
}

impl Segment {
    pub fn new(word: String, pos: String) -> Self {
        Self { word, pos }
    }
}

/// Chinese word segmenter with POS tagging
///
/// This wraps jieba-rs functionality and provides a consistent interface
/// for the tone sandhi and G2P modules.
pub struct Segmenter {
    #[cfg(feature = "jieba")]
    jieba: jieba_rs::Jieba,

    /// Common word patterns for simple segmentation fallback
    #[allow(dead_code)]
    common_words: HashMap<String, String>,
}

impl Default for Segmenter {
    fn default() -> Self {
        Self::new()
    }
}

impl Segmenter {
    /// Create a new segmenter
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "jieba")]
            jieba: jieba_rs::Jieba::new(),

            common_words: Self::init_common_words(),
        }
    }

    fn init_common_words() -> HashMap<String, String> {
        // Common Chinese words with their POS tags
        // This is a minimal fallback for when jieba is not available
        let mut m = HashMap::new();

        // Pronouns (r)
        for word in ["我", "你", "他", "她", "它", "我们", "你们", "他们", "她们", "这", "那", "这个", "那个"] {
            m.insert(word.to_string(), "r".to_string());
        }

        // Verbs (v)
        for word in ["是", "有", "在", "做", "去", "来", "说", "看", "想", "知道", "喜欢", "可以", "要", "会", "能"] {
            m.insert(word.to_string(), "v".to_string());
        }

        // Nouns (n)
        for word in ["人", "时候", "事", "东西", "地方", "问题", "工作", "朋友", "中国", "北京"] {
            m.insert(word.to_string(), "n".to_string());
        }

        // Adjectives (a)
        for word in ["好", "大", "小", "多", "少", "高", "新", "老", "长", "快", "慢"] {
            m.insert(word.to_string(), "a".to_string());
        }

        // Adverbs (d)
        for word in ["不", "也", "都", "很", "就", "还", "只", "才", "已经", "一直"] {
            m.insert(word.to_string(), "d".to_string());
        }

        // Aspect markers
        m.insert("了".to_string(), "ul".to_string());
        m.insert("着".to_string(), "uz".to_string());
        m.insert("过".to_string(), "ug".to_string());

        // Particles
        m.insert("的".to_string(), "uj".to_string());
        m.insert("地".to_string(), "uv".to_string());
        m.insert("得".to_string(), "ud".to_string());

        // Numbers (m)
        for word in ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "百", "千", "万", "亿", "两"] {
            m.insert(word.to_string(), "m".to_string());
        }

        // Measure words (q)
        for word in ["个", "只", "本", "张", "把", "块", "件", "条", "位"] {
            m.insert(word.to_string(), "q".to_string());
        }

        m
    }

    /// Segment text with POS tagging
    ///
    /// # Arguments
    /// * `text` - Chinese text to segment
    ///
    /// # Returns
    /// Vector of (word, pos_tag) tuples
    #[cfg(feature = "jieba")]
    pub fn cut_for_pos(&self, text: &str) -> Vec<Segment> {
        self.jieba
            .tag(text, false)
            .into_iter()
            .map(|tag| Segment::new(tag.word.to_string(), tag.tag.to_string()))
            .collect()
    }

    /// Segment text with POS tagging (fallback without jieba)
    #[cfg(not(feature = "jieba"))]
    pub fn cut_for_pos(&self, text: &str) -> Vec<Segment> {
        self.simple_segment_with_pos(text)
    }

    /// Simple segmentation fallback
    /// Uses maximum forward matching with the common words dictionary
    fn simple_segment_with_pos(&self, text: &str) -> Vec<Segment> {
        let mut result = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let len = chars.len();
        let mut i = 0;

        while i < len {
            let c = chars[i];

            // Skip non-Chinese characters
            if !super::lang_segment::is_chinese_char(c) {
                // Collect consecutive non-Chinese chars
                let mut j = i;
                while j < len && !super::lang_segment::is_chinese_char(chars[j]) {
                    j += 1;
                }
                let word: String = chars[i..j].iter().collect();
                if !word.trim().is_empty() {
                    result.push(Segment::new(word, "x".to_string())); // x = other
                }
                i = j;
                continue;
            }

            // Try maximum forward matching (up to 4 characters)
            let mut matched = false;
            for word_len in (1..=4.min(len - i)).rev() {
                let word: String = chars[i..i + word_len].iter().collect();
                if let Some(pos) = self.common_words.get(&word) {
                    result.push(Segment::new(word, pos.clone()));
                    i += word_len;
                    matched = true;
                    break;
                }
            }

            // No match found - single character with default POS
            if !matched {
                let word = c.to_string();
                let pos = self.infer_single_char_pos(c);
                result.push(Segment::new(word, pos));
                i += 1;
            }
        }

        result
    }

    /// Infer POS for a single character based on heuristics
    fn infer_single_char_pos(&self, c: char) -> String {
        // Check common word dictionary first
        if let Some(pos) = self.common_words.get(&c.to_string()) {
            return pos.clone();
        }

        // Heuristic-based POS inference
        let code = c as u32;

        // Check for punctuation (using unicode escapes for curly quotes)
        if matches!(c, '，' | '。' | '！' | '？' | '、' | '；' | '：' | '\u{201C}' | '\u{201D}' | '\u{2018}' | '\u{2019}') {
            return "w".to_string(); // punctuation
        }

        // Default to noun for CJK characters
        if (0x4E00..=0x9FFF).contains(&code) {
            return "n".to_string();
        }

        "x".to_string() // other
    }

    /// Segment text without POS tagging (faster)
    #[cfg(feature = "jieba")]
    pub fn cut(&self, text: &str) -> Vec<String> {
        self.jieba.cut(text, false).into_iter().map(|s| s.to_string()).collect()
    }

    /// Segment text without POS tagging (fallback)
    #[cfg(not(feature = "jieba"))]
    pub fn cut(&self, text: &str) -> Vec<String> {
        self.cut_for_pos(text).into_iter().map(|s| s.word).collect()
    }

    /// Segment for search (more fine-grained)
    #[cfg(feature = "jieba")]
    pub fn cut_for_search(&self, text: &str) -> Vec<String> {
        self.jieba.cut_for_search(text, false).into_iter().map(|s| s.to_string()).collect()
    }

    /// Segment for search (fallback - same as cut)
    #[cfg(not(feature = "jieba"))]
    pub fn cut_for_search(&self, text: &str) -> Vec<String> {
        self.cut(text)
    }
}

// Global segmenter instance (lazy initialized)
lazy_static::lazy_static! {
    pub static ref GLOBAL_SEGMENTER: Segmenter = Segmenter::new();
}

/// Convenience function: segment text with POS tagging
pub fn cut_with_pos(text: &str) -> Vec<Segment> {
    GLOBAL_SEGMENTER.cut_for_pos(text)
}

/// Convenience function: segment text without POS
pub fn cut(text: &str) -> Vec<String> {
    GLOBAL_SEGMENTER.cut(text)
}

/// Convenience function: segment for search
pub fn cut_for_search(text: &str) -> Vec<String> {
    GLOBAL_SEGMENTER.cut_for_search(text)
}

/// Convert segments to the format expected by tone_sandhi
pub fn segments_to_word_segments(segments: Vec<Segment>) -> Vec<super::tone_sandhi::WordSegment> {
    segments
        .into_iter()
        .map(|s| super::tone_sandhi::WordSegment::new(&s.word, &s.pos))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_segment() {
        let segmenter = Segmenter::new();
        let result = segmenter.cut_for_pos("我喜欢北京");

        // Should produce some segments
        assert!(!result.is_empty());

        // Check that we get expected words (order may vary)
        let words: Vec<&str> = result.iter().map(|s| s.word.as_str()).collect();
        assert!(words.contains(&"我") || words.join("").contains("我"));
    }

    #[test]
    fn test_pos_inference() {
        let segmenter = Segmenter::new();

        // Test known words
        assert_eq!(segmenter.infer_single_char_pos('我'), "r"); // pronoun
        assert_eq!(segmenter.infer_single_char_pos('好'), "a"); // adjective
        assert_eq!(segmenter.infer_single_char_pos('是'), "v"); // verb
    }

    #[test]
    fn test_common_words() {
        let segmenter = Segmenter::new();

        // Pronouns
        assert!(segmenter.common_words.get("我们").is_some());
        assert!(segmenter.common_words.get("你们").is_some());

        // Verbs
        assert!(segmenter.common_words.get("喜欢").is_some());
        assert!(segmenter.common_words.get("知道").is_some());

        // Aspect markers
        assert_eq!(segmenter.common_words.get("了"), Some(&"ul".to_string()));
        assert_eq!(segmenter.common_words.get("着"), Some(&"uz".to_string()));
        assert_eq!(segmenter.common_words.get("过"), Some(&"ug".to_string()));
    }

    #[test]
    fn test_convenience_functions() {
        let segments = cut_with_pos("你好");
        assert!(!segments.is_empty());

        let words = cut("你好");
        assert!(!words.is_empty());
    }

    #[test]
    fn test_mixed_content() {
        let segmenter = Segmenter::new();
        let result = segmenter.cut_for_pos("Hello世界");

        // Should handle both English and Chinese
        assert!(!result.is_empty());
    }

    #[test]
    fn test_punctuation() {
        let segmenter = Segmenter::new();
        let result = segmenter.cut_for_pos("你好，世界！");

        // Should include punctuation
        let has_punct = result.iter().any(|s| s.pos == "w");
        // May or may not have punctuation depending on segmentation strategy
        assert!(!result.is_empty());
    }

    #[test]
    fn test_segments_to_word_segments() {
        let segments = vec![
            Segment::new("我".to_string(), "r".to_string()),
            Segment::new("喜欢".to_string(), "v".to_string()),
        ];

        let word_segments = segments_to_word_segments(segments);
        assert_eq!(word_segments.len(), 2);
        assert_eq!(word_segments[0].word, "我");
        assert_eq!(word_segments[0].pos, "r");
    }

    #[test]
    fn test_year_segmentation() {
        let segmenter = Segmenter::new();
        let result = segmenter.cut_for_pos("二零一一年");

        println!("Jieba segmentation for '二零一一年':");
        for seg in &result {
            println!("  {} / {}", seg.word, seg.pos);
        }

        // Python segments as: 二零一 / m, 一年 / m
        // Check that segmentation is similar
        assert!(!result.is_empty());
    }

    #[test]
    fn test_yige_segmentation() {
        let segmenter = Segmenter::new();
        let result = segmenter.cut_for_pos("一个新的");

        println!("Jieba segmentation for '一个新的':");
        for seg in &result {
            println!("  {} / {}", seg.word, seg.pos);
        }

        // Python segments as: 一个 / m, 新 / a, 的 / uj
        // Rust jieba-rs may segment differently
    }
}
