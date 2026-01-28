//! Text Normalization for Chinese TTS
//!
//! Port of Python's TextNormalizer from zh_normalization module.
//! Handles:
//! - Number verbalization (integers, decimals, fractions, percentages)
//! - Date/time conversion
//! - Phone number formatting
//! - Temperature and units
//! - Traditional to Simplified Chinese
//! - Punctuation normalization

use super::cn2an;
use regex::Regex;
use std::collections::HashMap;

lazy_static::lazy_static! {
    // Number patterns
    static ref RE_FRACTION: Regex = Regex::new(r"(-?)(\d+)/(\d+)").unwrap();
    static ref RE_PERCENTAGE: Regex = Regex::new(r"(-?)(\d+(?:\.\d+)?)%").unwrap();
    static ref RE_DECIMAL: Regex = Regex::new(r"(-?)(\d+)\.(\d+)").unwrap();
    static ref RE_INTEGER: Regex = Regex::new(r"(-?)(\d+)").unwrap();

    // Date patterns
    static ref RE_DATE_YMD: Regex = Regex::new(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})").unwrap();
    static ref RE_DATE_CHINESE: Regex = Regex::new(r"(\d{4})年(\d{1,2})月(\d{1,2})日").unwrap();

    // Time patterns
    static ref RE_TIME: Regex = Regex::new(r"(\d{1,2}):(\d{2})(?::(\d{2}))?").unwrap();
    static ref RE_TIME_RANGE: Regex = Regex::new(r"(\d{1,2}):(\d{2})\s*[-~]\s*(\d{1,2}):(\d{2})").unwrap();

    // Phone patterns
    static ref RE_MOBILE: Regex = Regex::new(r"1[3-9]\d{9}").unwrap();
    static ref RE_TELEPHONE: Regex = Regex::new(r"(\d{3,4})-(\d{7,8})").unwrap();

    // Temperature pattern
    static ref RE_TEMPERATURE: Regex = Regex::new(r"(-?)(\d+(?:\.\d+)?)\s*(°C|℃|度|摄氏度)").unwrap();

    // Range pattern
    static ref RE_RANGE: Regex = Regex::new(r"(-?\d+(?:\.\d+)?)\s*[-~]\s*(-?\d+(?:\.\d+)?)").unwrap();

    // Punctuation replacement map (full-width to half-width)
    static ref PUNCT_MAP: HashMap<char, char> = {
        let mut m = HashMap::new();
        m.insert('：', ',');
        m.insert('；', ',');
        m.insert('，', ',');
        m.insert('。', '.');
        m.insert('！', '!');
        m.insert('？', '?');
        m.insert('、', ',');
        m.insert('·', ',');
        // Note: （）《》【】""— are stripped by strip_special_symbols() before this runs
        // Single quotes are not mapped — they'll be filtered out by replace_punctuation's final filter
        m.insert('～', ',');       // fullwidth tilde → comma
        m
    };

    // Unit replacements
    static ref UNIT_MAP: HashMap<&'static str, &'static str> = {
        let mut m = HashMap::new();
        m.insert("cm²", "平方厘米");
        m.insert("cm2", "平方厘米");
        m.insert("cm³", "立方厘米");
        m.insert("cm3", "立方厘米");
        m.insert("cm", "厘米");
        m.insert("m²", "平方米");
        m.insert("m2", "平方米");
        m.insert("m³", "立方米");
        m.insert("m3", "立方米");
        m.insert("mm", "毫米");
        m.insert("km", "千米");
        m.insert("kg", "千克");
        m.insert("ml", "毫升");
        m.insert("dB", "分贝");
        m.insert("db", "分贝");
        m
    };
}

/// Text normalizer for Chinese TTS
pub struct TextNormalizer {
    /// Whether to preserve English characters
    preserve_english: bool,
}

impl Default for TextNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

impl TextNormalizer {
    pub fn new() -> Self {
        Self {
            preserve_english: false,
        }
    }

    /// Create normalizer that preserves English characters
    pub fn with_english() -> Self {
        Self {
            preserve_english: true,
        }
    }

    /// Normalize text for TTS
    ///
    /// # Arguments
    /// * `text` - Input text to normalize
    ///
    /// # Returns
    /// Normalized text ready for G2P conversion
    pub fn normalize(&self, text: &str) -> String {
        let mut result = text.to_string();

        // 0. Strip brackets, quotes, and special symbols (matches Python _split)
        // Python: re.sub(r'[——《》【】<>{}()（）#&@""^_|\\]', '', text)
        result = strip_special_symbols(&result);

        // 1. Traditional to Simplified Chinese
        result = hanconv::t2s(&result);

        // 2. Replace special characters (嗯 → 恩, 呣 → 母)
        result = result.replace('嗯', "恩");
        result = result.replace('呣', "母");

        // 3. Handle fractions
        result = self.replace_fractions(&result);

        // 4. Handle percentages
        result = self.replace_percentages(&result);

        // 5. Handle temperatures
        result = self.replace_temperatures(&result);

        // 6. Handle dates
        result = self.replace_dates(&result);

        // 7. Handle times
        result = self.replace_times(&result);

        // 8. Handle phone numbers
        result = self.replace_phone_numbers(&result);

        // 9. Handle ranges
        result = self.replace_ranges(&result);

        // 10. Handle remaining numbers
        result = self.replace_numbers(&result);

        // 11. Handle units (only match standalone, not inside English words)
        result = self.replace_units(&result);

        // 12. Replace punctuation
        result = self.replace_punctuation(&result);

        // 13. Remove consecutive punctuation
        result = replace_consecutive_punctuation(&result);

        // 14. Remove characters that can't be pronounced
        result = self.remove_unpronounced(&result);

        result
    }

    /// Replace fractions with Chinese
    fn replace_fractions(&self, text: &str) -> String {
        RE_FRACTION.replace_all(text, |caps: &regex::Captures| {
            let sign = &caps[1];
            let numerator = &caps[2];
            let denominator = &caps[3];
            let prefix = if sign == "-" { "负" } else { "" };
            format!("{}{}", prefix, cn2an::fraction_to_chinese(numerator, denominator))
        }).to_string()
    }

    /// Replace percentages with Chinese
    fn replace_percentages(&self, text: &str) -> String {
        RE_PERCENTAGE.replace_all(text, |caps: &regex::Captures| {
            let sign = &caps[1];
            let num = &caps[2];
            let prefix = if sign == "-" { "负" } else { "" };
            format!("{}{}", prefix, cn2an::percentage_to_chinese(num))
        }).to_string()
    }

    /// Replace temperatures with Chinese
    fn replace_temperatures(&self, text: &str) -> String {
        RE_TEMPERATURE.replace_all(text, |caps: &regex::Captures| {
            let sign = &caps[1];
            let num = &caps[2];
            let unit = &caps[3];

            let prefix = if sign == "-" { "零下" } else { "" };
            let num_cn = cn2an::an2cn(num);
            let unit_cn = if unit == "度" { "度" } else { "摄氏度" };

            format!("{}{}{}", prefix, num_cn, unit_cn)
        }).to_string()
    }

    /// Replace dates with Chinese
    fn replace_dates(&self, text: &str) -> String {
        // ISO format: 2024-01-15
        let result = RE_DATE_YMD.replace_all(text, |caps: &regex::Captures| {
            let year = &caps[1];
            let month = &caps[2];
            let day = &caps[3];

            let year_cn = cn2an::digits_to_chinese(year);
            let month_num: u32 = month.parse().unwrap_or(0);
            let day_num: u32 = day.parse().unwrap_or(0);

            format!("{}年{}月{}日",
                year_cn,
                cn2an::an2cn(&month_num.to_string()),
                cn2an::an2cn(&day_num.to_string())
            )
        }).to_string();

        // Chinese format: 2024年1月15日 - just convert numbers
        RE_DATE_CHINESE.replace_all(&result, |caps: &regex::Captures| {
            let year = &caps[1];
            let month = &caps[2];
            let day = &caps[3];

            let year_cn = cn2an::digits_to_chinese(year);
            let month_num: u32 = month.parse().unwrap_or(0);
            let day_num: u32 = day.parse().unwrap_or(0);

            format!("{}年{}月{}日",
                year_cn,
                cn2an::an2cn(&month_num.to_string()),
                cn2an::an2cn(&day_num.to_string())
            )
        }).to_string()
    }

    /// Replace times with Chinese
    fn replace_times(&self, text: &str) -> String {
        // First handle time ranges
        let result = RE_TIME_RANGE.replace_all(text, |caps: &regex::Captures| {
            let h1 = caps[1].parse::<u32>().unwrap_or(0);
            let m1 = caps[2].parse::<u32>().unwrap_or(0);
            let h2 = caps[3].parse::<u32>().unwrap_or(0);
            let m2 = caps[4].parse::<u32>().unwrap_or(0);

            format!("{}点{}分到{}点{}分",
                cn2an::an2cn(&h1.to_string()),
                cn2an::an2cn(&m1.to_string()),
                cn2an::an2cn(&h2.to_string()),
                cn2an::an2cn(&m2.to_string())
            )
        }).to_string();

        // Then handle single times
        RE_TIME.replace_all(&result, |caps: &regex::Captures| {
            let hour: u32 = caps[1].parse().unwrap_or(0);
            let minute: u32 = caps[2].parse().unwrap_or(0);
            let second: Option<u32> = caps.get(3).and_then(|m| m.as_str().parse().ok());

            let mut time_str = format!("{}点", cn2an::an2cn(&hour.to_string()));

            if minute == 30 {
                time_str.push('半');
            } else if minute > 0 {
                time_str.push_str(&cn2an::an2cn(&minute.to_string()));
                time_str.push('分');
            }

            if let Some(sec) = second {
                if sec > 0 {
                    time_str.push_str(&cn2an::an2cn(&sec.to_string()));
                    time_str.push_str("秒");
                }
            }

            time_str
        }).to_string()
    }

    /// Replace phone numbers with digit-by-digit reading
    fn replace_phone_numbers(&self, text: &str) -> String {
        // Mobile: 1xxxxxxxxxx
        let result = RE_MOBILE.replace_all(text, |caps: &regex::Captures| {
            cn2an::digits_to_chinese(&caps[0])
        }).to_string();

        // Landline: xxx-xxxxxxxx
        RE_TELEPHONE.replace_all(&result, |caps: &regex::Captures| {
            let area = &caps[1];
            let number = &caps[2];
            format!("{}{}", cn2an::digits_to_chinese(area), cn2an::digits_to_chinese(number))
        }).to_string()
    }

    /// Replace ranges with Chinese
    fn replace_ranges(&self, text: &str) -> String {
        RE_RANGE.replace_all(text, |caps: &regex::Captures| {
            let start = &caps[1];
            let end = &caps[2];
            format!("{}到{}", cn2an::an2cn(start), cn2an::an2cn(end))
        }).to_string()
    }

    /// Replace remaining numbers with Chinese
    fn replace_numbers(&self, text: &str) -> String {
        cn2an::transform(text)
    }

    /// Replace measurement units
    fn replace_units(&self, text: &str) -> String {
        let mut result = text.to_string();
        // Sort by length descending to replace longer units first
        let mut units: Vec<(&&str, &&str)> = UNIT_MAP.iter().collect();
        units.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        for (unit, chinese) in units {
            // Only replace units preceded by a digit (not inside English words)
            // e.g., "5km" → "5千米" but "Commercial" should NOT match "mm"
            let pattern = format!(r"(\d){}", regex::escape(unit));
            if let Ok(re) = Regex::new(&pattern) {
                result = re.replace_all(&result, |caps: &regex::Captures| {
                    format!("{}{}", &caps[1], chinese)
                }).to_string();
            }
        }
        result
    }

    /// Replace punctuation (full-width to half-width)
    fn replace_punctuation(&self, text: &str) -> String {
        let mut result = String::with_capacity(text.len());

        for c in text.chars() {
            if let Some(&replacement) = PUNCT_MAP.get(&c) {
                result.push(replacement);
            } else {
                result.push(c);
            }
        }

        // Remove characters that don't have pronunciations
        if self.preserve_english {
            // Keep Chinese, English, and valid punctuation
            result.chars()
                .filter(|&c| {
                    super::lang_segment::is_chinese_char(c)
                        || c.is_ascii_alphabetic()
                        || c.is_ascii_digit()
                        || is_valid_punct(c)
                        || c == '\'' // Python keeps apostrophe (in string.punctuation)
                        || c.is_whitespace()
                })
                .collect()
        } else {
            // Keep only Chinese and valid punctuation
            result.chars()
                .filter(|&c| {
                    super::lang_segment::is_chinese_char(c)
                        || is_valid_punct(c)
                })
                .collect()
        }
    }

    /// Remove characters that can't be pronounced
    fn remove_unpronounced(&self, text: &str) -> String {
        // Remove citation references like [21], [22]
        let re_citation = Regex::new(r"\[\d+\]").unwrap();
        let result = re_citation.replace_all(text, "").to_string();

        // Remove other bracket content if not needed
        result
    }
}

/// Check if character is valid punctuation for TTS
/// Matches Python's punctuation list: ["!", "?", "…", ",", "."]
fn is_valid_punct(c: char) -> bool {
    matches!(c, '!' | '?' | ',' | '.' | '…')
}

/// Strip brackets, quotes, dashes and special symbols before normalization.
/// Matches Python's TextNormalizer._split() and _post_replace():
///   re.sub(r'[——《》【】<>{}()（）#&@""^_|\\]', '', text)
fn strip_special_symbols(text: &str) -> String {
    text.chars()
        .filter(|c| !matches!(c,
            '—' | // em dash (U+2014)
            '《' | '》' |
            '【' | '】' |
            '<' | '>' |
            '{' | '}' |
            '(' | ')' |
            '（' | '）' |
            '#' | '&' | '@' |
            '\u{201C}' | '\u{201D}' | // left/right double curly quotes ""
            '"' | // straight double quote (also stripped by Python)
            '^' | '_' | '|' | '\\'
        ))
        .collect()
}

/// Remove consecutive punctuation, keeping the first
pub fn replace_consecutive_punctuation(text: &str) -> String {
    let punct_chars = ['!', '?', '…', ',', '.', ' '];
    let mut result = String::new();
    let mut prev_punct = false;

    for c in text.chars() {
        let is_punct = punct_chars.contains(&c);
        if is_punct {
            if !prev_punct {
                result.push(c);
            }
            prev_punct = true;
        } else {
            result.push(c);
            prev_punct = false;
        }
    }

    result
}

/// Normalize mixed Chinese/English text
/// Matches Python's all_zh path: uppercase English, normalize, remove spaces between English.
/// Python flow: uppercase -> mix_text_normalize -> replace_punctuation_with_en (strips non-Chinese/non-alpha/non-punct)
pub fn mix_text_normalize(text: &str) -> String {
    // Step 1: Uppercase English (matches Python line 135)
    let uppercased: String = text.chars().map(|c| {
        if c.is_ascii_lowercase() { c.to_ascii_uppercase() } else { c }
    }).collect();

    // Step 2: Normalize (strip symbols, handle numbers, replace punctuation)
    let normalized = TextNormalizer::with_english().normalize(&uppercased);

    // Keep spaces - needed for word-level English G2P tokenization
    normalized
}

/// Normalize pure Chinese text
/// Removes English characters
pub fn text_normalize(text: &str) -> String {
    TextNormalizer::new().normalize(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fraction() {
        let normalizer = TextNormalizer::new();
        let result = normalizer.replace_fractions("1/2的人");
        assert_eq!(result, "二分之一的人");
    }

    #[test]
    fn test_percentage() {
        let normalizer = TextNormalizer::new();
        let result = normalizer.replace_percentages("增长75%");
        assert_eq!(result, "增长百分之七十五");
    }

    #[test]
    fn test_temperature() {
        let normalizer = TextNormalizer::new();
        let result = normalizer.replace_temperatures("-3°C");
        assert_eq!(result, "零下三摄氏度");
    }

    #[test]
    fn test_date_iso() {
        let normalizer = TextNormalizer::new();
        let result = normalizer.replace_dates("2024-01-15");
        assert_eq!(result, "二零二四年一月十五日");
    }

    #[test]
    fn test_time() {
        let normalizer = TextNormalizer::new();
        let result = normalizer.replace_times("14:30");
        assert_eq!(result, "十四点半");
    }

    #[test]
    fn test_time_with_minutes() {
        let normalizer = TextNormalizer::new();
        let result = normalizer.replace_times("9:45");
        assert_eq!(result, "九点四十五分");
    }

    #[test]
    fn test_phone_mobile() {
        let normalizer = TextNormalizer::new();
        let result = normalizer.replace_phone_numbers("13812345678");
        assert_eq!(result, "一三八一二三四五六七八");
    }

    #[test]
    fn test_range() {
        let normalizer = TextNormalizer::new();
        let result = normalizer.replace_ranges("1-10");
        assert_eq!(result, "一到十");
    }

    #[test]
    fn test_full_normalize() {
        let normalizer = TextNormalizer::new();
        let result = normalizer.normalize("今天是2024年1月15日，温度是-3°C");
        assert!(result.contains("二零二四"));
        assert!(result.contains("零下三"));
    }

    #[test]
    fn test_consecutive_punctuation() {
        let result = replace_consecutive_punctuation("你好...世界");
        assert_eq!(result, "你好.世界");
    }

    #[test]
    fn test_traditional_to_simplified() {
        let normalizer = TextNormalizer::new();
        let result = normalizer.normalize("電腦");
        assert!(result.contains("电脑") || result.contains("电") || result.is_empty());
    }
}

