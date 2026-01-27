//! Erhua (儿化) handling for Mandarin Chinese
//!
//! Port of Python's _merge_erhua() function.
//! Handles the retroflex 儿 suffix that modifies the pronunciation
//! of the preceding syllable.

use std::collections::HashSet;

lazy_static::lazy_static! {
    /// Words that MUST have erhua applied
    pub static ref MUST_ERHUA: HashSet<&'static str> = {
        [
            "小院儿", "胡同儿", "范儿", "老汉儿", "撒欢儿", "寻老礼儿", "妥妥儿", "媳妇儿",
        ].into_iter().collect()
    };

    /// Words that must NOT have erhua applied
    pub static ref NOT_ERHUA: HashSet<&'static str> = {
        [
            "虐儿", "为儿", "护儿", "瞒儿", "救儿", "替儿", "有儿", "一儿", "我儿", "俺儿",
            "妻儿", "拐儿", "聋儿", "乞儿", "患儿", "幼儿", "孤儿", "婴儿", "婴幼儿", "连体儿",
            "脑瘫儿", "流浪儿", "体弱儿", "混血儿", "蜜雪儿", "舫儿", "祖儿", "美儿", "应采儿",
            "可儿", "侄儿", "孙儿", "侄孙儿", "女儿", "男儿", "红孩儿", "花儿", "虫儿", "马儿",
            "鸟儿", "猪儿", "猫儿", "狗儿", "少儿",
        ].into_iter().collect()
    };
}

/// Merge erhua (儿化) into the preceding syllable
///
/// # Arguments
/// * `initials` - Mutable vector of initials
/// * `finals` - Mutable vector of finals with tone numbers
/// * `word` - The word being processed
/// * `pos` - Part-of-speech tag
///
/// # Returns
/// Modified (initials, finals) with erhua merged
pub fn merge_erhua(
    initials: &mut Vec<String>,
    finals: &mut Vec<String>,
    word: &str,
    pos: &str,
) {
    let chars: Vec<char> = word.chars().collect();
    let word_len = chars.len();

    if finals.is_empty() || word_len == 0 {
        return;
    }

    // Fix er1 to er2 at word end (standalone 儿)
    for (i, final_str) in finals.iter_mut().enumerate() {
        if i == word_len - 1 && chars.get(i) == Some(&'儿') && final_str == "er1" {
            *final_str = "er2".to_string();
        }
    }

    // Check if erhua should be applied
    // Skip if in not_erhua list or wrong POS (adjective, abbreviation, proper noun)
    if !MUST_ERHUA.contains(word) && (NOT_ERHUA.contains(word) || matches!(pos, "a" | "j" | "nr")) {
        return;
    }

    // Handle length mismatch (e.g., "……" etc.)
    if finals.len() != word_len {
        return;
    }

    // Process erhua merging
    let mut new_initials = Vec::with_capacity(initials.len());
    let mut new_finals = Vec::with_capacity(finals.len());

    for (i, (init, fin)) in initials.iter().zip(finals.iter()).enumerate() {
        let mut new_fin = fin.clone();

        // Check if this is 儿 at the end that should be merged
        if i == word_len - 1
            && chars[i] == '儿'
            && (fin == "er2" || fin == "er5")
        {
            // Check if the last two characters are in not_erhua
            let last_two: String = if word_len >= 2 {
                chars[word_len - 2..].iter().collect()
            } else {
                String::new()
            };

            if !NOT_ERHUA.contains(last_two.as_str()) && !new_finals.is_empty() {
                // Merge with previous syllable: inherit the tone from previous final
                let prev_tone = new_finals.last()
                    .and_then(|f: &String| f.chars().last())
                    .filter(|c| c.is_ascii_digit())
                    .unwrap_or('2');
                new_fin = format!("er{}", prev_tone);
            }
        }

        new_initials.push(init.clone());
        new_finals.push(new_fin);
    }

    *initials = new_initials;
    *finals = new_finals;
}

/// Apply erhua to a pinyin by appending 'r' to the final
/// Used for post-processing when we want the combined form
///
/// # Arguments
/// * `pinyin` - The pinyin with tone (e.g., "yuan4")
///
/// # Returns
/// Pinyin with erhua applied (e.g., "yuan4r" or "yuanr4")
pub fn apply_erhua_to_pinyin(pinyin: &str) -> String {
    if pinyin.is_empty() {
        return pinyin.to_string();
    }

    // Check if already has 'r' suffix
    let chars: Vec<char> = pinyin.chars().collect();
    if chars.len() >= 2 {
        let second_last = chars[chars.len() - 2];
        if second_last == 'r' {
            return pinyin.to_string();
        }
    }

    // Extract tone number if present
    if let Some(last) = chars.last() {
        if last.is_ascii_digit() {
            // Insert 'r' before tone: yuan4 → yuanr4
            let base = &pinyin[..pinyin.len() - 1];
            return format!("{}r{}", base, last);
        }
    }

    // No tone number, just append 'r'
    format!("{}r", pinyin)
}

/// Check if a word ends with 儿 that could be erhua
pub fn has_potential_erhua(word: &str) -> bool {
    word.ends_with('儿') && !NOT_ERHUA.contains(word)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_erhua_basic() {
        let mut initials = vec!["x".to_string(), "y".to_string(), "".to_string()];
        let mut finals = vec!["iao3".to_string(), "uan4".to_string(), "er2".to_string()];

        merge_erhua(&mut initials, &mut finals, "小院儿", "n");

        // 儿 should inherit tone from previous (yuan4 → er4)
        assert_eq!(finals[2], "er4");
    }

    #[test]
    fn test_no_merge_erhua_in_not_list() {
        let mut initials = vec!["n".to_string(), "".to_string()];
        let mut finals = vec!["v3".to_string(), "er2".to_string()];

        merge_erhua(&mut initials, &mut finals, "女儿", "n");

        // 女儿 is in not_erhua list, should not merge
        assert_eq!(finals[1], "er2");
    }

    #[test]
    fn test_fix_er1_to_er2() {
        let mut initials = vec!["".to_string()];
        let mut finals = vec!["er1".to_string()];

        merge_erhua(&mut initials, &mut finals, "儿", "n");

        // Standalone 儿 with er1 should become er2
        assert_eq!(finals[0], "er2");
    }

    #[test]
    fn test_apply_erhua_to_pinyin() {
        assert_eq!(apply_erhua_to_pinyin("yuan4"), "yuanr4");
        assert_eq!(apply_erhua_to_pinyin("hua1"), "huar1");
        assert_eq!(apply_erhua_to_pinyin("tong"), "tongr");
    }

    #[test]
    fn test_has_potential_erhua() {
        assert!(has_potential_erhua("小院儿"));
        assert!(has_potential_erhua("胡同儿"));
        assert!(!has_potential_erhua("女儿")); // In not_erhua list
        assert!(!has_potential_erhua("你好")); // Doesn't end with 儿
    }
}
