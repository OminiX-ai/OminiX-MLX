//! Chinese Number Conversion (cn2an)
//!
//! Port of Python's cn2an library for converting Arabic numerals to Chinese.
//! Handles integers, decimals, fractions, percentages, and special formats.

use std::collections::HashMap;

lazy_static::lazy_static! {
    /// Arabic digit to Chinese character mapping
    static ref DIGIT_MAP: HashMap<char, char> = {
        let mut m = HashMap::new();
        m.insert('0', '零');
        m.insert('1', '一');
        m.insert('2', '二');
        m.insert('3', '三');
        m.insert('4', '四');
        m.insert('5', '五');
        m.insert('6', '六');
        m.insert('7', '七');
        m.insert('8', '八');
        m.insert('9', '九');
        m
    };

    /// Unit characters for place values
    static ref UNITS: [&'static str; 4] = ["", "十", "百", "千"];

    /// Large unit characters
    static ref LARGE_UNITS: [&'static str; 5] = ["", "万", "亿", "万亿", "亿亿"];
}

/// Convert an Arabic numeral string to Chinese
///
/// # Arguments
/// * `num_str` - The number as a string (e.g., "123", "45.67", "-89")
///
/// # Returns
/// Chinese representation (e.g., "一百二十三", "四十五点六七", "负八十九")
pub fn an2cn(num_str: &str) -> String {
    let num_str = num_str.trim();

    if num_str.is_empty() {
        return String::new();
    }

    // Handle negative numbers
    if let Some(stripped) = num_str.strip_prefix('-') {
        return format!("负{}", an2cn(stripped));
    }

    // Handle decimals
    if let Some(dot_pos) = num_str.find('.') {
        let integer_part = &num_str[..dot_pos];
        let decimal_part = &num_str[dot_pos + 1..];

        let integer_cn = if integer_part.is_empty() || integer_part == "0" {
            "零".to_string()
        } else {
            integer_to_chinese(integer_part)
        };

        let decimal_cn = digits_to_chinese(decimal_part);

        return format!("{}点{}", integer_cn, decimal_cn);
    }

    // Pure integer
    integer_to_chinese(num_str)
}

/// Convert integer string to Chinese with proper place values
fn integer_to_chinese(num_str: &str) -> String {
    // Handle zero
    if num_str.chars().all(|c| c == '0') {
        return "零".to_string();
    }

    // Remove leading zeros
    let num_str = num_str.trim_start_matches('0');
    if num_str.is_empty() {
        return "零".to_string();
    }

    let len = num_str.len();

    // For very long numbers or numbers starting with 0, read digit by digit
    if len > 12 {
        return digits_to_chinese(num_str);
    }

    let digits: Vec<u32> = num_str
        .chars()
        .filter_map(|c| c.to_digit(10))
        .collect();

    // Special case for numbers 10-19 (omit leading 一)
    if len == 2 && digits[0] == 1 {
        let second = DIGIT_MAP.get(&num_str.chars().nth(1).unwrap()).unwrap_or(&'零');
        if digits[1] == 0 {
            return "十".to_string();
        }
        return format!("十{}", second);
    }

    // Process in groups of 4 digits (万 = 10^4, 亿 = 10^8)
    let mut result = String::new();
    let mut prev_zero = false;
    let mut has_content = false;

    // Split into groups of 4 from the right
    let padded_len = ((len + 3) / 4) * 4;
    let padding = padded_len - len;
    let padded: String = "0".repeat(padding) + num_str;

    let groups: Vec<&str> = padded
        .as_bytes()
        .chunks(4)
        .map(|chunk| std::str::from_utf8(chunk).unwrap())
        .collect();

    let num_groups = groups.len();

    for (group_idx, group) in groups.iter().enumerate() {
        let group_value: u32 = group.parse().unwrap_or(0);

        if group_value == 0 {
            prev_zero = true;
            continue;
        }

        // Convert this group (0-9999)
        let (group_cn, has_leading_zero) = four_digit_to_chinese(group);

        // Add zero placeholder if needed:
        // 1. Previous group was all zeros (prev_zero)
        // 2. This group has leading zeros (e.g., 10001 → 万 + 零 + 一)
        if has_content && (prev_zero || has_leading_zero) {
            result.push('零');
        }

        result.push_str(&group_cn);

        // Add large unit (万, 亿, etc.)
        let unit_idx = num_groups - group_idx - 1;
        if unit_idx > 0 && unit_idx < LARGE_UNITS.len() {
            result.push_str(LARGE_UNITS[unit_idx]);
        }

        prev_zero = false;
        has_content = true;
    }

    if result.is_empty() {
        "零".to_string()
    } else {
        result
    }
}

/// Convert a 4-digit group to Chinese (0-9999)
/// Returns (chinese_string, needs_leading_zero)
/// needs_leading_zero is true if the group has leading zeros (e.g., 0001)
fn four_digit_to_chinese(num_str: &str) -> (String, bool) {
    let digits: Vec<u32> = num_str
        .chars()
        .filter_map(|c| c.to_digit(10))
        .collect();

    if digits.iter().all(|&d| d == 0) {
        return (String::new(), false);
    }

    let mut result = String::new();
    let mut prev_zero = false;
    let mut has_leading_zero = false;
    let len = digits.len();

    for (i, &d) in digits.iter().enumerate() {
        let pos = len - 1 - i; // Position from right (0=ones, 1=tens, etc.)

        if d == 0 {
            prev_zero = true;
            if result.is_empty() {
                has_leading_zero = true;
            }
        } else {
            if prev_zero && !result.is_empty() {
                result.push('零');
            }

            // Get Chinese digit
            let digit_char = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九'][d as usize];

            // Special case: 10-19 in the thousands group doesn't need 一
            // But in general, we include the digit
            result.push(digit_char);

            // Add unit
            if pos > 0 && pos < UNITS.len() {
                result.push_str(UNITS[pos]);
            }

            prev_zero = false;
        }
    }

    (result, has_leading_zero)
}

/// Convert digits to Chinese one by one (for decimals or years)
pub fn digits_to_chinese(num_str: &str) -> String {
    num_str
        .chars()
        .filter_map(|c| DIGIT_MAP.get(&c))
        .collect()
}

/// Convert year to Chinese (digit by digit)
/// e.g., "2024" → "二零二四"
pub fn year_to_chinese(year_str: &str) -> String {
    digits_to_chinese(year_str)
}

/// Convert fraction to Chinese
/// e.g., "1/2" → "二分之一"
pub fn fraction_to_chinese(numerator: &str, denominator: &str) -> String {
    let num_cn = an2cn(numerator);
    let den_cn = an2cn(denominator);
    format!("{}分之{}", den_cn, num_cn)
}

/// Convert percentage to Chinese
/// e.g., "75" → "百分之七十五"
pub fn percentage_to_chinese(num_str: &str) -> String {
    let cn = an2cn(num_str);
    format!("百分之{}", cn)
}

/// Transform text containing numbers to Chinese
/// Main entry point matching Python's cn2an.transform(text, "an2cn")
pub fn transform(text: &str) -> String {
    let mut result = String::new();
    let mut num_buffer = String::new();
    let mut is_negative = false;

    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        let c = chars[i];

        // Check for negative sign
        if c == '-' && i + 1 < len && chars[i + 1].is_ascii_digit() {
            if !num_buffer.is_empty() {
                result.push_str(&process_number(&num_buffer, is_negative, None));
                num_buffer.clear();
            }
            is_negative = true;
            i += 1;
            continue;
        }

        if c.is_ascii_digit() {
            num_buffer.push(c);
        } else if c == '.' && !num_buffer.is_empty() && i + 1 < len && chars[i + 1].is_ascii_digit() {
            // Decimal point within number
            num_buffer.push(c);
        } else {
            if !num_buffer.is_empty() {
                // Check context for special handling
                let next_char = if i < len { Some(chars[i]) } else { None };
                result.push_str(&process_number(&num_buffer, is_negative, next_char));
                num_buffer.clear();
                is_negative = false;
            }
            result.push(c);
        }
        i += 1;
    }

    // Handle trailing number
    if !num_buffer.is_empty() {
        result.push_str(&process_number(&num_buffer, is_negative, None));
    }

    result
}

/// Process a number with context awareness
fn process_number(num_str: &str, is_negative: bool, next_char: Option<char>) -> String {
    let prefix = if is_negative { "负" } else { "" };

    // Check if followed by 年 (year marker) - use digit-by-digit
    if next_char == Some('年') && num_str.len() == 4 && !num_str.contains('.') {
        return format!("{}{}", prefix, year_to_chinese(num_str));
    }

    format!("{}{}", prefix, an2cn(num_str))
}

/// Replace fractions in text
/// e.g., "1/2" → "二分之一"
pub fn replace_fractions(text: &str) -> String {
    let re = regex::Regex::new(r"(-?)(\d+)/(\d+)").unwrap();
    re.replace_all(text, |caps: &regex::Captures| {
        let sign = &caps[1];
        let numerator = &caps[2];
        let denominator = &caps[3];
        let prefix = if sign == "-" { "负" } else { "" };
        format!("{}{}", prefix, fraction_to_chinese(numerator, denominator))
    })
    .to_string()
}

/// Replace percentages in text
/// e.g., "75%" → "百分之七十五"
pub fn replace_percentages(text: &str) -> String {
    let re = regex::Regex::new(r"(-?)(\d+(?:\.\d+)?)%").unwrap();
    re.replace_all(text, |caps: &regex::Captures| {
        let sign = &caps[1];
        let num = &caps[2];
        let prefix = if sign == "-" { "负" } else { "" };
        format!("{}{}", prefix, percentage_to_chinese(num))
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_digits() {
        assert_eq!(an2cn("0"), "零");
        assert_eq!(an2cn("1"), "一");
        assert_eq!(an2cn("5"), "五");
        assert_eq!(an2cn("9"), "九");
    }

    #[test]
    fn test_tens() {
        assert_eq!(an2cn("10"), "十");
        assert_eq!(an2cn("11"), "十一");
        assert_eq!(an2cn("15"), "十五");
        assert_eq!(an2cn("20"), "二十");
        assert_eq!(an2cn("21"), "二十一");
        assert_eq!(an2cn("99"), "九十九");
    }

    #[test]
    fn test_hundreds() {
        assert_eq!(an2cn("100"), "一百");
        assert_eq!(an2cn("101"), "一百零一");
        assert_eq!(an2cn("110"), "一百一十");
        assert_eq!(an2cn("111"), "一百一十一");
        assert_eq!(an2cn("999"), "九百九十九");
    }

    #[test]
    fn test_thousands() {
        assert_eq!(an2cn("1000"), "一千");
        assert_eq!(an2cn("1001"), "一千零一");
        assert_eq!(an2cn("1010"), "一千零一十");
        assert_eq!(an2cn("1100"), "一千一百");
        assert_eq!(an2cn("9999"), "九千九百九十九");
    }

    #[test]
    fn test_large_numbers() {
        assert_eq!(an2cn("10000"), "一万");
        assert_eq!(an2cn("10001"), "一万零一");
        assert_eq!(an2cn("100000000"), "一亿");
    }

    #[test]
    fn test_decimals() {
        assert_eq!(an2cn("0.5"), "零点五");
        assert_eq!(an2cn("3.14"), "三点一四");
        assert_eq!(an2cn("163.6"), "一百六十三点六");
        assert_eq!(an2cn("0.123"), "零点一二三");
    }

    #[test]
    fn test_negative() {
        assert_eq!(an2cn("-1"), "负一");
        assert_eq!(an2cn("-100"), "负一百");
        assert_eq!(an2cn("-3.14"), "负三点一四");
    }

    #[test]
    fn test_year() {
        assert_eq!(year_to_chinese("2024"), "二零二四");
        assert_eq!(year_to_chinese("1999"), "一九九九");
        assert_eq!(year_to_chinese("2000"), "二零零零");
    }

    #[test]
    fn test_fraction() {
        assert_eq!(fraction_to_chinese("1", "2"), "二分之一");
        assert_eq!(fraction_to_chinese("3", "4"), "四分之三");
    }

    #[test]
    fn test_percentage() {
        assert_eq!(percentage_to_chinese("50"), "百分之五十");
        assert_eq!(percentage_to_chinese("75.5"), "百分之七十五点五");
    }

    #[test]
    fn test_replace_fractions() {
        assert_eq!(replace_fractions("1/2的人"), "二分之一的人");
        assert_eq!(replace_fractions("-3/4"), "负四分之三");
    }

    #[test]
    fn test_replace_percentages() {
        assert_eq!(replace_percentages("增长75%"), "增长百分之七十五");
        assert_eq!(replace_percentages("-10.5%"), "负百分之十点五");
    }

    #[test]
    fn test_transform() {
        assert_eq!(transform("我有100元"), "我有一百元");
        assert_eq!(transform("2024年"), "二零二四年");
        assert_eq!(transform("温度是-10度"), "温度是负十度");
    }

    #[test]
    fn test_special_cases() {
        // Numbers with internal zeros
        assert_eq!(an2cn("1001"), "一千零一");
        assert_eq!(an2cn("10001"), "一万零一");
        assert_eq!(an2cn("10010"), "一万零一十");
        assert_eq!(an2cn("10100"), "一万零一百");
    }
}
