//! Text preprocessor for GPT-SoVITS
//!
//! Converts text to phoneme sequences for Chinese and English.
//!
//! Pipeline:
//! 1. Text normalization
//! 2. Language detection
//! 3. Grapheme-to-phoneme conversion
//! 4. Phoneme ID conversion

use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;

use pinyin::ToPinyin;

use super::erhua::{MUST_ERHUA, NOT_ERHUA};
use super::g2pw::get_pinyin_with_g2pw;
use super::symbols::{self, bos_id, eos_id, has_symbol, symbol_to_id};
use super::tone_sandhi::{ToneSandhi, WordSegment, pre_merge_for_modify};
use super::jieba_seg::cut_with_pos;
use super::text_normalizer::mix_text_normalize;

/// Detected language
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    Chinese,
    English,
    Mixed,
}

impl Language {
    pub fn as_str(&self) -> &'static str {
        match self {
            Language::Chinese => "zh",
            Language::English => "en",
            Language::Mixed => "mixed",
        }
    }
}

/// Output from text preprocessing
#[derive(Debug, Clone)]
pub struct PreprocessorOutput {
    /// Phoneme IDs
    pub phoneme_ids: Vec<i32>,
    /// Phoneme strings
    pub phonemes: Vec<String>,
    /// Number of phonemes per word/character
    pub word2ph: Vec<i32>,
    /// Normalized text
    pub text_normalized: String,
    /// Detected/specified language
    pub language: Language,
}

/// Pinyin initials (consonants)
const PINYIN_INITIALS: &[&str] = &[
    "b", "c", "ch", "d", "f", "g", "h", "j", "k", "l", "m", "n",
    "p", "q", "r", "s", "sh", "t", "w", "x", "y", "z", "zh",
];

/// Multi-character initials (check these first)
const MULTI_CHAR_INITIALS: &[&str] = &["zh", "ch", "sh"];

/// Zero-initial vowel mapping
/// Note: "er" is special - it maps directly to "er" + tone without initial
fn zero_initial_map() -> HashMap<&'static str, (&'static str, &'static str)> {
    let mut map = HashMap::new();
    map.insert("a", ("AA", "a"));
    map.insert("ai", ("AA", "ai"));
    map.insert("an", ("AA", "an"));
    map.insert("ang", ("AA", "ang"));
    map.insert("ao", ("AA", "ao"));
    map.insert("e", ("EE", "e"));
    map.insert("ei", ("EE", "ei"));
    map.insert("en", ("EE", "en"));
    map.insert("eng", ("EE", "eng"));
    // "er" uses direct phoneme: er1, er2, er3, er4, er5 (no initial needed)
    map.insert("o", ("OO", "o"));
    map.insert("ou", ("OO", "ou"));
    map
}

/// Words where the last character should have neutral tone (tone 5)
/// Copied from Python's tone_sandhi.py must_neural_tone_words
fn must_neutral_tone_words() -> std::collections::HashSet<&'static str> {
    [
        "麻烦", "麻利", "鸳鸯", "高粱", "骨头", "骆驼", "马虎", "首饰", "馒头", "馄饨",
        "风筝", "难为", "队伍", "阔气", "闺女", "门道", "锄头", "铺盖", "铃铛", "铁匠",
        "钥匙", "里脊", "里头", "部分", "那么", "道士", "造化", "迷糊", "连累", "这么",
        "这个", "运气", "过去", "软和", "转悠", "踏实", "跳蚤", "跟头", "趔趄", "财主",
        "豆腐", "讲究", "记性", "记号", "认识", "规矩", "见识", "裁缝", "补丁", "衣裳",
        "衣服", "衙门", "街坊", "行李", "行当", "蛤蟆", "蘑菇", "薄荷", "葫芦", "葡萄",
        "萝卜", "荸荠", "苗条", "苗头", "苍蝇", "芝麻", "舒服", "舒坦", "舌头", "自在",
        "膏药", "脾气", "脑袋", "脊梁", "能耐", "胳膊", "胭脂", "胡萝", "胡琴", "胡同",
        "聪明", "耽误", "耽搁", "耷拉", "耳朵", "老爷", "老实", "老婆", "老头", "老太",
        "翻腾", "罗嗦", "罐头", "编辑", "结实", "红火", "累赘", "糨糊", "糊涂", "精神",
        "粮食", "簸箕", "篱笆", "算计", "算盘", "答应", "笤帚", "笑语", "笑话", "窟窿",
        "窝囊", "窗户", "稳当", "稀罕", "称呼", "秧歌", "秀气", "秀才", "福气", "祖宗",
        "砚台", "码头", "石榴", "石头", "石匠", "知识", "眼睛", "眯缝", "眨巴", "眉毛",
        "相声", "盘算", "白净", "痢疾", "痛快", "疟疾", "疙瘩", "疏忽", "畜生", "生意",
        "甘蔗", "琵琶", "琢磨", "琉璃", "玻璃", "玫瑰", "玄乎", "狐狸", "状元", "特务",
        "牲口", "牙碜", "牌楼", "爽快", "爱人", "热闹", "烧饼", "烟筒", "烂糊", "点心",
        "炊帚", "灯笼", "火候", "漂亮", "滑溜", "溜达", "温和", "清楚", "消息", "浪头",
        "活泼", "比方", "正经", "欺负", "模糊", "槟榔", "棺材", "棒槌", "棉花", "核桃",
        "栅栏", "柴火", "架势", "枕头", "枇杷", "机灵", "本事", "木头", "木匠", "朋友",
        "月饼", "月亮", "暖和", "明白", "时候", "新鲜", "故事", "收拾", "收成", "提防",
        "挖苦", "挑剔", "指甲", "指头", "拾掇", "拳头", "拨弄", "招牌", "招呼", "抬举",
        "护士", "折腾", "扫帚", "打量", "打算", "打点", "打扮", "打听", "打发", "扎实",
        "扁担", "戒指", "懒得", "意识", "意思", "情形", "悟性", "怪物", "思量", "怎么",
        "念头", "念叨", "快活", "忙活", "志气", "心思", "得罪", "张罗", "弟兄", "开通",
        "应酬", "庄稼", "干事", "帮手", "帐篷", "希罕", "师父", "师傅", "巴结", "巴掌",
        "差事", "工夫", "岁数", "屁股", "尾巴", "少爷", "小气", "小伙", "将就", "对头",
        "对付", "寡妇", "家伙", "客气", "实在", "官司", "学问", "学生", "字号", "嫁妆",
        "媳妇", "媒人", "婆家", "娘家", "委屈", "姑娘", "姐夫", "妯娌", "妥当", "妖精",
        "奴才", "女婿", "头发", "太阳", "大爷", "大方", "大意", "大夫", "多少", "多么",
        "外甥", "壮实", "地道", "地方", "在乎", "困难", "嘴巴", "嘱咐", "嘟囔", "嘀咕",
        "喜欢", "喇嘛", "喇叭", "商量", "唾沫", "哑巴", "哈欠", "哆嗦", "咳嗽", "和尚",
        "告诉", "告示", "含糊", "吓唬", "后头", "名字", "名堂", "合同", "吆喝", "叫唤",
        "口袋", "厚道", "厉害", "千斤", "包袱", "包涵", "匀称", "勤快", "动静", "动弹",
        "功夫", "力气", "前头", "刺猬", "刺激", "别扭", "利落", "利索", "利害", "分析",
        "出息", "凑合", "凉快", "冷战", "冤枉", "冒失", "养活", "关系", "先生", "兄弟",
        "便宜", "使唤", "佩服", "作坊", "体面", "位置", "似的", "伙计", "休息", "什么",
        "人家", "亲戚", "亲家", "交情", "云彩", "事情", "买卖", "主意", "丫头", "丧气",
        "两口", "东西", "东家", "世故", "不由", "不在", "下水", "下巴", "上头", "上司",
        "丈夫", "丈人", "一辈", "那个", "菩萨", "父亲", "母亲", "咕噜", "邋遢", "费用",
        "冤家", "甜头", "介绍", "荒唐", "大人", "泥鳅", "幸福", "熟悉", "计划", "扑腾",
    ].into_iter().collect()
}

/// Get polyphonic correction for a character based on context
/// Returns corrected pinyin if a rule applies, None otherwise
fn get_polyphonic_correction(prev_char: Option<char>, curr_char: char) -> Option<&'static str> {
    // 应 is ying4 when preceded by certain characters
    if curr_char == '应' {
        if let Some(prev) = prev_char {
            match prev {
                '回' | '反' | '适' | '效' | '响' | '相' | '对' | '供' => return Some("ying4"),
                _ => {}
            }
        }
    }
    None
}

/// Pinyin to phoneme mapping based on opencpop-strict.txt
/// Maps pinyin (without tone) to (initial, final)
fn pinyin_to_phoneme_map() -> HashMap<&'static str, (&'static str, &'static str)> {
    let mut map = HashMap::new();
    // j, q, x with ü vowels -> v
    map.insert("ju", ("j", "v"));
    map.insert("jv", ("j", "v"));
    map.insert("juan", ("j", "van"));
    map.insert("jvan", ("j", "van"));
    map.insert("jue", ("j", "ve"));
    map.insert("jve", ("j", "ve"));
    map.insert("jun", ("j", "vn"));
    map.insert("jvn", ("j", "vn"));
    map.insert("qu", ("q", "v"));
    map.insert("qv", ("q", "v"));
    map.insert("quan", ("q", "van"));
    map.insert("qvan", ("q", "van"));
    map.insert("que", ("q", "ve"));
    map.insert("qve", ("q", "ve"));
    map.insert("qun", ("q", "vn"));
    map.insert("qvn", ("q", "vn"));
    map.insert("xu", ("x", "v"));
    map.insert("xv", ("x", "v"));
    map.insert("xuan", ("x", "van"));
    map.insert("xvan", ("x", "van"));
    map.insert("xue", ("x", "ve"));
    map.insert("xve", ("x", "ve"));
    map.insert("xun", ("x", "vn"));
    map.insert("xvn", ("x", "vn"));
    // y with ü vowels -> v
    map.insert("yu", ("y", "v"));
    map.insert("yv", ("y", "v"));
    map.insert("yuan", ("y", "van"));
    map.insert("yvan", ("y", "van"));
    map.insert("yue", ("y", "ve"));
    map.insert("yve", ("y", "ve"));
    map.insert("yun", ("y", "vn"));
    map.insert("yvn", ("y", "vn"));
    // l, n with ü vowels -> v
    map.insert("lv", ("l", "v"));
    map.insert("lve", ("l", "ve"));
    map.insert("nv", ("n", "v"));
    map.insert("nve", ("n", "ve"));
    // Apical vowels: z, c, s + i -> i0 (different from zh, ch, sh, r + i -> ir)
    map.insert("zi", ("z", "i0"));
    map.insert("ci", ("c", "i0"));
    map.insert("si", ("s", "i0"));
    // Retroflex apicals: zh, ch, sh, r + i -> ir
    map.insert("zhi", ("zh", "ir"));
    map.insert("chi", ("ch", "ir"));
    map.insert("shi", ("sh", "ir"));
    map.insert("ri", ("r", "ir"));
    // Special y finals
    map.insert("yan", ("y", "En"));
    map.insert("ye", ("y", "E"));
    map
}

/// Full-width to half-width punctuation mapping
fn fullwidth_to_halfwidth() -> HashMap<char, char> {
    // Matches Python's rep_map from chinese.py
    let mut map = HashMap::new();
    map.insert('，', ',');
    // Note: '。' → '.' is deferred until after number conversion
    // to avoid '44.2011' being parsed as a decimal number
    // map.insert('。', '.');
    map.insert('！', '!');
    map.insert('？', '?');
    map.insert('；', ',');  // Python: "；" → "," (not ';')
    map.insert('：', ',');  // Python: "：" → "," (not ':')
    map.insert('、', ',');
    map.insert('"', '"');
    map.insert('"', '"');
    map.insert('\u{2018}', '\'');  // Left single quote
    map.insert('\u{2019}', '\'');  // Right single quote
    map.insert('（', '(');
    map.insert('）', ')');
    map.insert('【', '[');
    map.insert('】', ']');
    map.insert('《', '"');
    map.insert('》', '"');
    map.insert('～', '…');  // Python: "～" → "…"
    map.insert('~', '…');   // Python: "~" → "…"
    map.insert('·', ',');   // Python: "·" → ","
    map.insert('—', '-');   // Python: "—" → "-"
    map.insert('$', '.');   // Python: "$" → "."
    map.insert('/', ',');   // Python: "/" → ","
    map
}

/// Check if character is Chinese
pub fn is_chinese_char(c: char) -> bool {
    let code = c as u32;
    (0x4E00..=0x9FFF).contains(&code)      // CJK Unified Ideographs
        || (0x3400..=0x4DBF).contains(&code)   // CJK Extension A
        || (0x20000..=0x2A6DF).contains(&code) // CJK Extension B
        || (0xF900..=0xFAFF).contains(&code)   // CJK Compatibility Ideographs
}

/// Detect primary language of text
///
/// Returns `Mixed` if both Chinese and English characters are present,
/// regardless of which has more. This ensures proper phoneme conversion
/// for code-switching text like "Hello世界".
pub fn detect_language(text: &str) -> Language {
    let chinese_count = text.chars().filter(|&c| is_chinese_char(c)).count();
    let english_count = text.chars().filter(|&c| c.is_ascii_alphabetic()).count();

    // If both Chinese and English are present, treat as mixed
    if chinese_count > 0 && english_count > 0 {
        Language::Mixed
    } else if chinese_count > 0 {
        Language::Chinese
    } else if english_count > 0 {
        Language::English
    } else {
        // No letters found, default to Chinese (handles punctuation-only)
        Language::Chinese
    }
}

/// Normalize Chinese text (traditional → simplified, full-width → half-width)
pub fn normalize_chinese(text: &str) -> String {
    // First convert traditional Chinese to simplified (for BERT compatibility)
    let text = hanconv::t2s(text);

    let map = fullwidth_to_halfwidth();
    // Convert fullwidth punctuation to halfwidth
    let text: String = text.chars()
        .map(|c| *map.get(&c).unwrap_or(&c))
        .collect();
    // Convert measurement units to Chinese BEFORE removing English
    // Python quantifier.py: "s" → "秒", "m" → "米", etc.
    let text = replace_measure_units(&text);

    // CRITICAL: Remove ALL English letters (matching Python's re.sub("[a-zA-Z]+", ""))
    let re_english = regex::Regex::new(r"[a-zA-Z]+").unwrap();
    let text = re_english.replace_all(&text, "").to_string();

    // Clean up excess whitespace left after English removal
    let re_spaces = regex::Regex::new(r"\s+").unwrap();
    let text = re_spaces.replace_all(&text, "").to_string();

    // Convert numbers to Chinese BEFORE removing brackets/special chars
    // Python's TextNormalizer converts numbers while [brackets] and ％ are still present,
    // so they act as separators (e.g., "500％[47]" → "五零零％[四十七]", not "50047")
    let text = normalize_numbers_to_chinese(&text);

    // Now remove brackets (numbers inside already converted to Chinese)
    // e.g., [四十七] → 四十七
    let re_bracket = regex::Regex::new(r"[\[\]]").unwrap();
    let text = re_bracket.replace_all(&text, "").to_string();

    // Remove special characters (matching Python's replace_punctuation)
    let text: String = text.chars()
        .filter(|&c| !matches!(c,
            '"' | '\'' | '(' | ')' | ':' | ';' | '·' | '•' |
            '—' | '–' | '～' | '-' |
            '《' | '》' | '【' | '】' | '<' | '>' | '{' | '}' |
            '（' | '）' | '#' | '&' | '@' | '^' | '_' | '|' | '\\' |
            '％'  // Fullwidth percent sign - Python strips it
        ))
        .collect();

    // Now convert 。→ . (deferred from fullwidth conversion to avoid decimal parsing issues)
    let text = text.replace('。', ".");
    // Remove consecutive punctuation (matching Python's replace_consecutive_punctuation)
    replace_consecutive_punctuation(&text)
}

/// Deduplicate consecutive punctuation, keeping the first one
/// Matches Python's replace_consecutive_punctuation: pattern = f'([{punctuations}])([{punctuations}])+'
/// Example: "..." → ".", "!!" → "!", ",," → ","
/// Single punctuation marks are preserved as-is.
fn replace_consecutive_punctuation(text: &str) -> String {
    // GPT-SoVITS punctuation set: ! ? … , . -
    let punct_chars = ['!', '?', '…', ',', '.', '-'];
    let mut result = String::new();
    let mut prev_punct: Option<char> = None;

    for c in text.chars() {
        let is_punct = punct_chars.contains(&c);
        if is_punct {
            if prev_punct.is_none() {
                // First punctuation in a sequence - keep it
                result.push(c);
                prev_punct = Some(c);
            }
            // Otherwise skip (deduplicate)
        } else {
            result.push(c);
            prev_punct = None;
        }
    }
    result
}

/// Convert numbers in text to Chinese spoken form
/// Pipeline order (matching Python):
/// 1. Fractions
/// 2. Percentages
/// 3. Remaining decimals and integers (including negative)
///
/// Note: Date/time/range conversions are disabled to avoid false positives.
/// Enable them selectively if needed for specific use cases.
fn normalize_numbers_to_chinese(text: &str) -> String {
    // Step 1: Handle fractions (1/2 → 二分之一)
    let text = replace_fraction(text);

    // Step 2: Handle percentages (70% → 百分之七十)
    let text = replace_percentage(&text);

    // Step 3: Handle remaining decimals and integers (including negative)
    let mut result = String::new();
    let mut num_buffer = String::new();
    let mut is_negative = false;

    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];

        // Check for negative sign before a digit
        if c == '-' && i + 1 < chars.len() && chars[i + 1].is_ascii_digit() {
            // Flush any existing buffer first
            if !num_buffer.is_empty() {
                let next_char = Some(c);
                flush_number_buffer(&mut result, &mut num_buffer, is_negative, next_char);
            }
            is_negative = true;
            i += 1;
            continue;
        }

        if c.is_ascii_digit() {
            num_buffer.push(c);
        } else if c == '.' && !num_buffer.is_empty() && i + 1 < chars.len() && chars[i + 1].is_ascii_digit() {
            // Decimal point (only if followed by digit)
            num_buffer.push(c);
        } else {
            if !num_buffer.is_empty() {
                let next_char = Some(c);
                flush_number_buffer(&mut result, &mut num_buffer, is_negative, next_char);
                is_negative = false;
            }
            result.push(c);
        }
        i += 1;
    }

    // Handle trailing number
    if !num_buffer.is_empty() {
        flush_number_buffer(&mut result, &mut num_buffer, is_negative, None);
    }

    result
}

/// Helper to flush number buffer with optional negative prefix
/// Matches Python's TextNormalizer number conversion rules:
/// - 4-digit + 年: digit-by-digit with 一 (year mode)
/// - followed by 万/亿: semantic (一千一百万)
/// - standalone 3+ digits: digit-by-digit with 幺 (direct mode)
/// - 1-2 digits: semantic (四十四)
fn flush_number_buffer(result: &mut String, num_buffer: &mut String, is_negative: bool, next_char: Option<char>) {
    if num_buffer.ends_with('.') {
        num_buffer.pop();
        if is_negative {
            result.push_str("负");
        }
        result.push_str(&number_to_chinese_with_decimal(num_buffer));
        result.push('.');
    } else {
        if is_negative {
            result.push_str("负");
        }
        let len = num_buffer.len();
        let is_year = len == 4 && next_char == Some('年');
        let is_unit = matches!(next_char, Some('万') | Some('亿'));
        if is_year {
            // Year: digit-by-digit with 一
            result.push_str(&number_to_chinese_digits_year(num_buffer));
        } else if is_unit {
            // Followed by 万/亿: semantic
            result.push_str(&number_to_chinese(num_buffer));
        } else if len >= 3 {
            // Standalone 3+ digits: digit-by-digit with 幺
            result.push_str(&number_to_chinese_digits(num_buffer));
        } else {
            // 1-2 digits: semantic
            result.push_str(&number_to_chinese_with_decimal(num_buffer));
        }
    }
    num_buffer.clear();
}

/// Normalize Chinese text for BERT (removes English characters, keeps Chinese and punctuation)
/// This matches Python's replace_punctuation behavior
pub fn normalize_chinese_for_bert(text: &str) -> String {
    let map = fullwidth_to_halfwidth();
    text.chars()
        .filter_map(|c| {
            // Convert full-width punctuation first
            let c = *map.get(&c).unwrap_or(&c);
            // Keep only Chinese characters and basic punctuation
            if is_chinese_char(c) || is_punctuation(c) {
                Some(c)
            } else {
                None
            }
        })
        .collect()
}

/// Check if character is punctuation (matching Python's punctuation set)
fn is_punctuation(c: char) -> bool {
    matches!(c,
        // ASCII punctuation
        '!' | '"' | '#' | '$' | '%' | '&' | '\'' | '(' | ')' | '*' |
        '+' | ',' | '-' | '.' | '/' | ':' | ';' | '<' | '=' | '>' |
        '?' | '@' | '[' | '\\' | ']' | '^' | '_' | '`' | '{' | '|' |
        '}' | '~' | ' ' |
        // Chinese punctuation
        '，' | '。' | '！' | '？' | '、' | '；' | '：' |
        '\u{201C}' | '\u{201D}' |  // " " (curly double quotes)
        '\u{2018}' | '\u{2019}' |  // ' ' (curly single quotes)
        '（' | '）' | '【' | '】' | '《' | '》' | '—' |
        '…' | '·' | '「' | '」' | '『' | '』' | '〈' | '〉'
    )
}

/// Normalize English text
pub fn normalize_english(text: &str) -> String {
    // Remove extra whitespace
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Split pinyin into initial (consonant) and final (vowel with tone)
///
/// # Arguments
/// * `pinyin` - Pinyin syllable with tone number (e.g., "ni3", "hao3")
///
/// # Returns
/// Tuple of (initial, final) where final includes tone number
///
/// Uses the opencpop-strict mapping for special cases like:
/// - j/q/x + u/uan/ue/un → v/van/ve/vn (ü vowels)
/// - z/c/s + i → i0 (apical vowel)
/// - zh/ch/sh/r + i → ir (retroflex apical)
pub fn get_initial_final(pinyin: &str) -> (Option<&'static str>, String) {
    // Extract tone number if present
    let (pinyin_base, tone) = if pinyin.chars().last().map(|c| c.is_ascii_digit()).unwrap_or(false) {
        let tone = pinyin.chars().last().unwrap();
        (&pinyin[..pinyin.len()-1], tone)
    } else {
        (pinyin, '5') // Neutral tone
    };

    // First check the special pinyin mapping table
    let special_map = pinyin_to_phoneme_map();
    if let Some(&(init, vowel)) = special_map.get(pinyin_base) {
        return (Some(init), format!("{}{}", vowel, tone));
    }

    // Check for multi-character initials first (zh, ch, sh)
    for &initial in MULTI_CHAR_INITIALS {
        if pinyin_base.starts_with(initial) {
            let final_part = &pinyin_base[initial.len()..];
            return (Some(initial), format!("{}{}", final_part, tone));
        }
    }

    // Single character initials
    for &initial in PINYIN_INITIALS {
        if initial.len() == 1 && pinyin_base.starts_with(initial) {
            let final_part = &pinyin_base[1..];
            return (Some(initial), format!("{}{}", final_part, tone));
        }
    }

    // Special case: "er" has its own phoneme (er1, er2, er3, er4, er5)
    // But it still needs the EE glottal stop like other vowel-initial words
    if pinyin_base == "er" {
        return (Some("EE"), format!("er{}", tone));
    }

    // Zero initial - check mapping
    let zero_map = zero_initial_map();
    if let Some(&(init, vowel)) = zero_map.get(pinyin_base) {
        return (Some(init), format!("{}{}", vowel, tone));
    }

    // Default: treat entire pinyin as final with special initial
    (Some("AA"), format!("{}{}", pinyin_base, tone))
}

/// Convert Chinese character to pinyin using the pinyin crate
/// Pinyin correction map for characters with wrong tones in the pinyin crate
fn pinyin_corrections() -> HashMap<char, &'static str> {
    let mut map = HashMap::new();
    // Fix common tone errors
    map.insert('总', "zong3");  // 总 should be tone 3
    map.insert('统', "tong3");  // 统 should be tone 3
    map.insert('说', "shuo1");  // 说 should be tone 1 (not tone 4)
    map.insert('合', "he2");    // 合 should be tone 2 (merge/combine)
    // Fix 儿 - pinyin crate returns ren2 but should be er2
    map.insert('儿', "er2");    // 儿 (child/erhua suffix) is er2, not ren2
    map
}

/// Polyphone dictionary: word → (char_index, correct_pinyin)
/// These are words where context determines the pronunciation
fn polyphone_words() -> Vec<(&'static str, usize, &'static str)> {
    vec![
        // 行: háng (hang2) vs xíng (xing2) - Simplified
        ("银行", 1, "hang2"),    // bank
        ("行业", 0, "hang2"),    // industry
        ("行列", 0, "hang2"),    // ranks
        ("行情", 0, "hang2"),    // market conditions
        ("央行", 1, "hang2"),    // central bank
        ("商行", 1, "hang2"),    // trading company
        ("分行", 1, "hang2"),    // branch (bank)
        ("支行", 1, "hang2"),    // sub-branch
        ("总行", 1, "hang2"),    // headquarters (bank)
        ("行长", 0, "hang2"),    // bank president
        ("同行", 1, "hang2"),    // same profession (when noun)
        ("内行", 1, "hang2"),    // expert
        ("外行", 1, "hang2"),    // layman
        // 行: háng (hang2) vs xíng (xing2) - Traditional
        ("銀行", 1, "hang2"),    // bank (traditional)
        ("行業", 0, "hang2"),    // industry (traditional)
        ("總行", 1, "hang2"),    // headquarters (traditional)
        ("分行", 1, "hang2"),    // branch (traditional - same char)
        ("支行", 1, "hang2"),    // sub-branch (traditional - same char)
        // 长: cháng (chang2) vs zhǎng (zhang3)
        ("成长", 1, "zhang3"),   // grow up
        ("生长", 1, "zhang3"),   // grow
        ("增长", 1, "zhang3"),   // increase
        ("长大", 0, "zhang3"),   // grow up
        ("长辈", 0, "zhang3"),   // elder
        ("部长", 1, "zhang3"),   // minister
        ("市长", 1, "zhang3"),   // mayor
        ("校长", 1, "zhang3"),   // principal
        ("厂长", 1, "zhang3"),   // factory director
        ("董事长", 2, "zhang3"), // chairman
        ("家长", 1, "zhang3"),   // parent
        // 乐: lè (le4) vs yuè (yue4)
        ("音乐", 1, "yue4"),     // music
        ("乐器", 0, "yue4"),     // musical instrument
        ("乐队", 0, "yue4"),     // band
        ("乐曲", 0, "yue4"),     // musical composition
        // 数: shù (shu4) vs shǔ (shu3)
        ("数据", 0, "shu4"),     // data
        ("数字", 0, "shu4"),     // number/digit
        ("数量", 0, "shu4"),     // quantity
        ("数学", 0, "shu4"),     // mathematics
        // 重: zhòng (zhong4) vs chóng (chong2)
        ("重复", 0, "chong2"),   // repeat
        ("重新", 0, "chong2"),   // again
        // 着: zhe (zhe5) vs zháo (zhao2) vs zhuó (zhuo2)
        ("着急", 0, "zhao2"),    // anxious
        ("着火", 0, "zhao2"),    // catch fire
        ("着凉", 0, "zhao2"),    // catch cold
        // 的: dì (di4) for 目的, otherwise de5
        ("目的", 1, "di4"),      // purpose
        // 合: hé (he2) vs hè (he4) - Simplified
        ("合并", 0, "he2"),      // merge/combine
        ("合作", 0, "he2"),      // cooperate
        ("合适", 0, "he2"),      // suitable
        ("联合", 1, "he2"),      // unite
        ("结合", 1, "he2"),      // combine
        ("综合", 1, "he2"),      // comprehensive
        ("配合", 1, "he2"),      // coordinate
        ("符合", 1, "he2"),      // conform
        ("集合", 1, "he2"),      // gather
        // 合: hé (he2) vs hè (he4) - Traditional
        ("合併", 0, "he2"),      // merge/combine (traditional)
        ("聯合", 1, "he2"),      // unite (traditional)
        ("結合", 1, "he2"),      // combine (traditional)
        ("綜合", 1, "he2"),      // comprehensive (traditional)
        // 为: wèi (wei4) vs wéi (wei2)
        ("改为", 1, "wei2"),     // change to
        ("成为", 1, "wei2"),     // become
        ("作为", 1, "wei2"),     // as/being
        ("认为", 1, "wei2"),     // think/consider
        ("以为", 1, "wei2"),     // think/believe
        ("称为", 1, "wei2"),     // called as
        ("因为", 1, "wei2"),     // because
        ("为了", 0, "wei4"),     // for the purpose of
        ("为什么", 0, "wei4"),   // why
        ("行为", 1, "wei2"),     // behavior
        ("人为", 1, "wei2"),     // man-made
        ("视为", 1, "wei2"),     // regard as
    ]
}

/// Global polyphonic dictionary loaded from external files
/// Format: word -> Vec<pinyin> (one pinyin per character)
static POLYPHONIC_DICT: OnceLock<HashMap<String, Vec<String>>> = OnceLock::new();

/// Load polyphonic dictionary from external files
fn load_polyphonic_dict() -> HashMap<String, Vec<String>> {
    let mut dict = HashMap::new();

    // Try to load from common locations
    let dict_paths = [
        "/Users/yuechen/home/mofa-studio/node-hub/dora-primespeech/dora_primespeech/moyoyo_tts/text/g2pw/polyphonic.rep",
        "/Users/yuechen/home/mofa-studio/node-hub/dora-primespeech/dora_primespeech/moyoyo_tts/text/g2pw/polyphonic-fix.rep",
    ];

    for path in dict_paths {
        if Path::new(path).exists() {
            if let Ok(content) = std::fs::read_to_string(path) {
                for line in content.lines() {
                    if let Some((word, pinyins_str)) = line.split_once(':') {
                        // Parse format: word: ['pinyin1', 'pinyin2', ...]
                        let word = word.trim().to_string();
                        let pinyins_str = pinyins_str.trim();

                        // Extract pinyin values from ['pinyin1', 'pinyin2'] format
                        if pinyins_str.starts_with('[') && pinyins_str.ends_with(']') {
                            let inner = &pinyins_str[1..pinyins_str.len()-1];
                            let pinyins: Vec<String> = inner
                                .split(',')
                                .map(|s| s.trim().trim_matches('\'').trim_matches('"').to_string())
                                .filter(|s| !s.is_empty())
                                .collect();

                            if !pinyins.is_empty() {
                                dict.insert(word, pinyins);
                            }
                        }
                    }
                }
            }
        }
    }

    dict
}

/// Get the global polyphonic dictionary
fn get_polyphonic_dict() -> &'static HashMap<String, Vec<String>> {
    POLYPHONIC_DICT.get_or_init(|| {
        let dict = load_polyphonic_dict();
        if !dict.is_empty() {
            eprintln!("Polyphonic dict: Loaded {} entries", dict.len());
        }
        dict
    })
}

/// Apply polyphone corrections based on word context
fn apply_polyphone_corrections(chars: &[char], pinyins: &mut [Option<String>]) {
    let _text: String = chars.iter().collect();

    // First, apply corrections from the external polyphonic dictionary
    let poly_dict = get_polyphonic_dict();
    for (word, word_pinyins) in poly_dict.iter() {
        let word_chars: Vec<char> = word.chars().collect();
        let word_len = word_chars.len();

        // Only process if pinyin count matches character count
        if word_pinyins.len() != word_len {
            continue;
        }

        // Find all occurrences of this word in the text
        for start_idx in 0..chars.len().saturating_sub(word_len - 1) {
            let matches = chars[start_idx..start_idx + word_len]
                .iter()
                .zip(word_chars.iter())
                .all(|(a, b)| a == b);

            if matches {
                // Apply all pinyin corrections for this word
                for (i, pinyin) in word_pinyins.iter().enumerate() {
                    let target_pos = start_idx + i;
                    if target_pos < pinyins.len() && pinyins[target_pos].is_some() {
                        pinyins[target_pos] = Some(pinyin.clone());
                    }
                }
            }
        }
    }

    // Then, apply corrections from the hardcoded polyphone_words list
    for (word, char_idx, correct_pinyin) in polyphone_words() {
        let word_chars: Vec<char> = word.chars().collect();
        let word_len = word_chars.len();

        // Find all occurrences by sliding window over character indices
        for start_char_idx in 0..chars.len().saturating_sub(word_len - 1) {
            // Check if word matches at this character position
            let matches = chars[start_char_idx..start_char_idx + word_len]
                .iter()
                .zip(word_chars.iter())
                .all(|(a, b)| a == b);

            if matches {
                let target_pos = start_char_idx + char_idx;
                // Make sure we're within bounds and it's a Chinese character
                if target_pos < pinyins.len() && pinyins[target_pos].is_some() {
                    pinyins[target_pos] = Some(correct_pinyin.to_string());
                }
            }
        }
    }
}

fn get_pinyin_for_char(c: char) -> Option<String> {
    // First check correction map for known errors
    let corrections = pinyin_corrections();
    if let Some(&corrected) = corrections.get(&c) {
        return Some(corrected.to_string());
    }

    // Use the pinyin crate for full Chinese character coverage
    // ToPinyin trait works on &str slices
    let char_str = c.to_string();
    let char_slice: &str = &char_str;
    for pinyin_result in char_slice.to_pinyin() {
        if let Some(pinyin) = pinyin_result {
            // Use with_tone_num_end() for format like "ni3"
            let mut result = pinyin.with_tone_num_end().to_string();

            // Convert 'ü' to 'v' for GPT-SoVITS symbol table compatibility
            result = result.replace('ü', "v");

            // Ensure tone number is present (add neutral tone 5 if missing)
            if !result.chars().last().map(|c| c.is_ascii_digit()).unwrap_or(false) {
                result.push('5');
            }

            return Some(result);
        }
    }
    None
}

/// Convert Chinese character to phonemes
fn char_to_phonemes(c: char) -> Vec<String> {
    if let Some(pinyin) = get_pinyin_for_char(c) {
        let (initial, final_part) = get_initial_final(&pinyin);
        let mut phonemes = Vec::new();
        if let Some(init) = initial {
            if has_symbol(init) {
                phonemes.push(init.to_string());
            }
        }
        if has_symbol(&final_part) {
            phonemes.push(final_part);
        }
        if phonemes.is_empty() {
            // Fallback: return unknown
            phonemes.push(symbols::UNK.to_string());
        }
        phonemes
    } else if is_chinese_char(c) {
        // Unknown Chinese character (shouldn't happen with pinyin crate)
        vec![symbols::UNK.to_string()]
    } else {
        // Non-Chinese character
        vec![]
    }
}

/// Convert a number to Chinese digit-by-digit for years
/// e.g., 2025 -> "二零二五" (uses 一 for 1)
fn number_to_chinese_digits_year(num_str: &str) -> String {
    let digits = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九'];
    num_str.chars()
        .filter_map(|c| c.to_digit(10))
        .map(|d| digits[d as usize])
        .collect()
}

/// Convert a number to Chinese digit-by-digit for standalone numbers
/// e.g., 100 -> "幺零零", 500 -> "五零零"
/// Matches Python's cn2an 'direct' mode: uses 幺 for 1
fn number_to_chinese_digits(num_str: &str) -> String {
    let digits = ['零', '幺', '二', '三', '四', '五', '六', '七', '八', '九'];
    num_str.chars()
        .filter_map(|c| c.to_digit(10))
        .map(|d| digits[d as usize])
        .collect()
}

/// Convert a number to Chinese spoken form
/// e.g., 23 -> "二十三", 100 -> "一百", 2024 -> "二零二四"
fn number_to_chinese(num_str: &str) -> String {
    let digits = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九'];

    // For very long numbers or special cases, just read digits
    if num_str.len() > 4 || num_str.starts_with('0') {
        return num_str.chars()
            .filter_map(|c| c.to_digit(10))
            .map(|d| digits[d as usize])
            .collect();
    }

    let num: u64 = match num_str.parse() {
        Ok(n) => n,
        Err(_) => return num_str.chars()
            .filter_map(|c| c.to_digit(10))
            .map(|d| digits[d as usize])
            .collect(),
    };

    if num == 0 {
        return "零".to_string();
    }

    let mut result = String::new();
    let units = ["", "十", "百", "千"];
    let num_digits: Vec<u64> = num_str.chars()
        .filter_map(|c| c.to_digit(10))
        .map(|d| d as u64)
        .collect();

    let len = num_digits.len();
    let mut prev_zero = false;

    for (i, &d) in num_digits.iter().enumerate() {
        let pos = len - 1 - i;
        if d == 0 {
            prev_zero = true;
        } else {
            if prev_zero && !result.is_empty() {
                result.push('零');
            }
            // Special case: 十 at the beginning (10-19) doesn't need 一
            if !(d == 1 && pos == 1 && i == 0) {
                result.push(digits[d as usize]);
            }
            if pos > 0 {
                result.push_str(units[pos]);
            }
            prev_zero = false;
        }
    }

    result
}

/// Convert number string (including decimals) to Chinese
/// e.g., "163.6" → "一百六十三点六"
fn number_to_chinese_with_decimal(num_str: &str) -> String {
    let digits = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九'];

    if let Some(dot_pos) = num_str.find('.') {
        let integer_part = &num_str[..dot_pos];
        let decimal_part = &num_str[dot_pos + 1..];

        let integer_chinese = if integer_part.is_empty() || integer_part == "0" {
            "零".to_string()
        } else {
            number_to_chinese(integer_part)
        };

        // Decimal digits are read individually: "6" → "六"
        let decimal_chinese: String = decimal_part
            .chars()
            .filter_map(|c| c.to_digit(10))
            .map(|d| digits[d as usize])
            .collect();

        format!("{}点{}", integer_chinese, decimal_chinese)
    } else {
        number_to_chinese(num_str)
    }
}

/// Convert percentages to Chinese
/// e.g., "70%" → "百分之七十", "163.6%" → "百分之一百六十三点六"
fn replace_percentage(text: &str) -> String {
    let re = regex::Regex::new(r"(-?)(\d+(?:\.\d+)?)%").unwrap();
    re.replace_all(text, |caps: &regex::Captures| {
        let sign = &caps[1];
        let num = &caps[2];
        let prefix = if sign == "-" { "负" } else { "" };
        format!("{}百分之{}", prefix, number_to_chinese_with_decimal(num))
    })
    .to_string()
}

/// Convert fractions to Chinese
/// e.g., "1/2" → "二分之一", "-3/4" → "负四分之三"
fn replace_fraction(text: &str) -> String {
    let re = regex::Regex::new(r"(-?)(\d+)/(\d+)").unwrap();
    re.replace_all(text, |caps: &regex::Captures| {
        let sign = &caps[1];
        let numerator = &caps[2];
        let denominator = &caps[3];
        let prefix = if sign == "-" { "负" } else { "" };
        // Chinese order: denominator 分之 numerator
        format!(
            "{}{}分之{}",
            prefix,
            number_to_chinese(denominator),
            number_to_chinese(numerator)
        )
    })
    .to_string()
}

/// Convert date formats to Chinese
/// e.g., "2024年1月15日" stays as-is (numbers converted)
/// e.g., "2024-01-15" → "二零二四年一月十五日"
#[allow(dead_code)]
fn replace_date(text: &str) -> String {
    // ISO format: 2024-01-15 or 2024/01/15
    let re = regex::Regex::new(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})").unwrap();
    re.replace_all(text, |caps: &regex::Captures| {
        let year = &caps[1];
        let month = &caps[2];
        let day = &caps[3];
        // Year: digit by digit, month/day: cardinal
        let year_chinese: String = year
            .chars()
            .filter_map(|c| c.to_digit(10))
            .map(|d| ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九'][d as usize])
            .collect();
        let month_num: u32 = month.parse().unwrap_or(0);
        let day_num: u32 = day.parse().unwrap_or(0);
        format!(
            "{}年{}月{}日",
            year_chinese,
            number_to_chinese(&month_num.to_string()),
            number_to_chinese(&day_num.to_string())
        )
    })
    .to_string()
}

/// Convert time formats to Chinese
/// e.g., "14:30" → "十四点三十分", "14:30:00" → "十四点三十分"
#[allow(dead_code)]
fn replace_time(text: &str) -> String {
    let re = regex::Regex::new(r"(\d{1,2}):(\d{2})(?::(\d{2}))?").unwrap();
    re.replace_all(text, |caps: &regex::Captures| {
        let hour: u32 = caps[1].parse().unwrap_or(0);
        let minute: u32 = caps[2].parse().unwrap_or(0);
        let second: Option<u32> = caps.get(3).and_then(|m| m.as_str().parse().ok());

        let mut result = format!("{}点", number_to_chinese(&hour.to_string()));

        if minute == 30 {
            result.push('半');
        } else if minute > 0 {
            result.push_str(&number_to_chinese(&minute.to_string()));
            result.push('分');
        }

        if let Some(sec) = second {
            if sec > 0 {
                result.push_str(&number_to_chinese(&sec.to_string()));
                result.push_str("秒");
            }
        }

        result
    })
    .to_string()
}

/// Convert numeric ranges to Chinese
/// e.g., "1-10" → "一到十", "0.5~1.5" → "零点五到一点五"
#[allow(dead_code)]
fn replace_range(text: &str) -> String {
    let re = regex::Regex::new(r"(-?\d+(?:\.\d+)?)\s*[-~]\s*(-?\d+(?:\.\d+)?)").unwrap();
    re.replace_all(text, |caps: &regex::Captures| {
        let start = &caps[1];
        let end = &caps[2];
        format!(
            "{}到{}",
            number_to_chinese_with_decimal(start),
            number_to_chinese_with_decimal(end)
        )
    })
    .to_string()
}

/// Convert temperature to Chinese
/// e.g., "-3°C" → "零下三摄氏度", "25℃" → "二十五度"
#[allow(dead_code)]
fn replace_temperature(text: &str) -> String {
    let re = regex::Regex::new(r"(-?)(\d+(?:\.\d+)?)\s*(°C|℃|度|摄氏度)").unwrap();
    re.replace_all(text, |caps: &regex::Captures| {
        let sign = &caps[1];
        let num = &caps[2];
        let unit = &caps[3];
        let prefix = if sign == "-" { "零下" } else { "" };
        let unit_text = if unit == "度" { "度" } else { "摄氏度" };
        format!("{}{}{}", prefix, number_to_chinese_with_decimal(num), unit_text)
    })
    .to_string()
}

/// Convert measurement units to Chinese
/// e.g., "10cm" → "10厘米", "5kg" → "5千克"
#[allow(dead_code)]
fn replace_units(text: &str) -> String {
    let replacements = [
        ("cm²", "平方厘米"),
        ("cm2", "平方厘米"),
        ("cm³", "立方厘米"),
        ("cm3", "立方厘米"),
        ("cm", "厘米"),
        ("m²", "平方米"),
        ("m2", "平方米"),
        ("m³", "立方米"),
        ("m3", "立方米"),
        ("mm", "毫米"),
        ("km", "千米"),
        ("kg", "千克"),
        ("ml", "毫升"),
        ("db", "分贝"),
        ("dB", "分贝"),
    ];

    let mut result = text.to_string();
    for (unit, chinese) in replacements {
        result = result.replace(unit, chinese);
    }
    result
}

/// Convert measurement units to Chinese - Python-compatible version
/// Matches Python's quantifier.py exactly, including single-letter conversions
/// Note: This does simple string replacement, affecting English words too
/// e.g., "Bankers" → "Banker秒", "mm" → "米米" (matches Python behavior)
fn replace_measure_units(text: &str) -> String {
    // CRITICAL: Order must match Python's dict iteration order exactly!
    // Python processes 'm' before 'mm', so 'mm' becomes '米米' not '毫米'
    // This is Python's measure_dict key order from quantifier.py
    let replacements = [
        ("cm2", "平方厘米"),
        ("cm²", "平方厘米"),
        ("cm3", "立方厘米"),
        ("cm³", "立方厘米"),
        ("cm", "厘米"),
        ("db", "分贝"),
        ("ds", "毫秒"),
        ("kg", "千克"),
        ("km", "千米"),
        ("m2", "平方米"),
        ("m²", "平方米"),
        ("m³", "立方米"),
        ("m3", "立方米"),
        ("ml", "毫升"),
        ("m", "米"),   // Before 'mm' - so 'mm' → '米米' not '毫米'
        ("mm", "毫米"), // This will never match since 'm' is replaced first
        ("s", "秒"),   // Single letter - affects English words like "Bankers" → "Banker秒"
    ];

    let mut result = text.to_string();
    for (unit, chinese) in replacements {
        result = result.replace(unit, chinese);
    }
    result
}

/// Convert circled numbers to Chinese
/// e.g., "①" → "一", "②" → "二"
#[allow(dead_code)]
fn replace_circled_numbers(text: &str) -> String {
    let replacements = [
        ('①', '一'), ('②', '二'), ('③', '三'), ('④', '四'), ('⑤', '五'),
        ('⑥', '六'), ('⑦', '七'), ('⑧', '八'), ('⑨', '九'), ('⑩', '十'),
    ];

    let mut result = text.to_string();
    for (circled, chinese) in replacements {
        result = result.replace(circled, &chinese.to_string());
    }
    result
}

/// Convert Greek letters to Chinese pronunciation
#[allow(dead_code)]
fn replace_greek_letters(text: &str) -> String {
    let replacements = [
        ('α', "阿尔法"), ('β', "贝塔"), ('γ', "伽玛"), ('δ', "德尔塔"),
        ('ε', "艾普西龙"), ('ζ', "捷塔"), ('η', "依塔"), ('θ', "西塔"),
        ('ι', "艾欧塔"), ('κ', "喀帕"), ('λ', "拉姆达"), ('μ', "缪"),
        ('ν', "拗"), ('ξ', "克西"), ('ο', "欧米克伦"), ('π', "派"),
        ('ρ', "肉"), ('σ', "西格玛"), ('ς', "西格玛"), ('τ', "套"),
        ('υ', "宇普西龙"), ('φ', "服艾"), ('χ', "器"), ('ψ', "普赛"), ('ω', "欧米伽"),
    ];

    let mut result = text.to_string();
    for (greek, chinese) in replacements {
        result = result.replace(greek, chinese);
    }
    result
}

/// Convert math operators to Chinese
/// e.g., "+" → "加", "=" → "等于"
#[allow(dead_code)]
fn replace_math_operators(text: &str) -> String {
    // Only replace standalone operators, not in numeric context
    let mut result = text.to_string();
    result = result.replace("×", "乘");
    result = result.replace("÷", "除以");
    result = result.replace("＝", "等于");
    result = result.replace("≈", "约等于");
    result = result.replace("≠", "不等于");
    result = result.replace("≤", "小于等于");
    result = result.replace("≥", "大于等于");
    result = result.replace("＜", "小于");
    result = result.replace("＞", "大于");
    result
}

/// Replace slash with 每 (per)
/// e.g., "km/h" → "千米每小时"
#[allow(dead_code)]
fn replace_slash(text: &str) -> String {
    // Replace / with 每 in unit contexts
    let re = regex::Regex::new(r"(\p{Han}+)/(\p{Han}+)").unwrap();
    re.replace_all(text, |caps: &regex::Captures| {
        format!("{}每{}", &caps[1], &caps[2])
    })
    .to_string()
}

/// Apply word-level tone sandhi using jieba segmentation (Python-compatible)
/// This is the preferred method as it matches the Python implementation
fn apply_word_level_tone_sandhi(text: &str, pinyins: &mut [Option<String>]) {
    let sandhi = ToneSandhi::new();

    // 1. Use jieba segmentation to get words with POS tags
    let segments = cut_with_pos(text);

    // 2. Convert to WordSegment format and apply pre-merge
    let word_segments: Vec<WordSegment> = segments
        .into_iter()
        .map(|seg| WordSegment::new(&seg.word, &seg.pos))
        .collect();
    let merged_segments = pre_merge_for_modify(word_segments);

    // 3. Process each word
    let mut char_idx = 0;
    for segment in merged_segments {
        let word = &segment.word;
        let pos = &segment.pos;
        let word_chars: Vec<char> = word.chars().collect();
        let word_len = word_chars.len();

        // Skip non-Chinese segments
        if word_len == 0 || !word_chars.iter().any(|c| is_chinese_char(*c)) {
            // Still advance char_idx for non-Chinese chars
            for _c in word.chars() {
                if char_idx < pinyins.len() {
                    char_idx += 1;
                }
            }
            continue;
        }

        // 4. Extract finals from pinyins for this word
        let mut word_finals: Vec<String> = Vec::with_capacity(word_len);
        let start_idx = char_idx;

        for _ in 0..word_len {
            if char_idx < pinyins.len() {
                if let Some(ref pinyin) = pinyins[char_idx] {
                    // Extract final from pinyin (e.g., "ni3" -> "i3", "hao3" -> "ao3")
                    let final_part = extract_final_from_pinyin(pinyin);
                    word_finals.push(final_part);
                } else {
                    word_finals.push("5".to_string()); // neutral tone for non-pinyin
                }
                char_idx += 1;
            }
        }

        // 5. Apply tone sandhi rules
        if !word_finals.is_empty() {
            sandhi.modified_tone(word, pos, &mut word_finals);

            // 5.5 Apply erhua tone inheritance
            // If word ends with 儿 and is in MUST_ERHUA or not in NOT_ERHUA,
            // the 儿 inherits the tone from the previous syllable
            if word.ends_with('儿') && word_len >= 2 {
                let word_str: &str = word;
                let should_merge = MUST_ERHUA.contains(word_str) ||
                    (!NOT_ERHUA.contains(word_str) && !matches!(pos.as_str(), "a" | "j" | "nr"));

                if should_merge {
                    // Get the tone from the previous syllable
                    if let Some(prev_final) = word_finals.get(word_len - 2) {
                        if let Some(prev_tone) = prev_final.chars().last().filter(|c| c.is_ascii_digit()) {
                            // Update the last position (儿) to inherit the tone
                            if let Some(last_final) = word_finals.get_mut(word_len - 1) {
                                // Replace tone in the final
                                if last_final.chars().last().map(|c| c.is_ascii_digit()).unwrap_or(false) {
                                    last_final.pop();
                                }
                                last_final.push(prev_tone);
                            }
                        }
                    }
                }
            }

            // 6. Update pinyins with modified tones
            for (i, final_part) in word_finals.iter().enumerate() {
                let pinyin_idx = start_idx + i;
                if pinyin_idx < pinyins.len() {
                    if let Some(ref mut pinyin) = pinyins[pinyin_idx] {
                        // Update the tone in the pinyin
                        if let Some(new_tone) = final_part.chars().last() {
                            if new_tone.is_ascii_digit() {
                                // Replace the tone digit
                                if pinyin.chars().last().map(|c| c.is_ascii_digit()).unwrap_or(false) {
                                    pinyin.pop();
                                }
                                pinyin.push(new_tone);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Extract the final (vowel part + tone) from a pinyin string
/// e.g., "ni3" -> "i3", "hao3" -> "ao3", "zhuang1" -> "uang1"
fn extract_final_from_pinyin(pinyin: &str) -> String {
    let vowels = ['a', 'e', 'i', 'o', 'u', 'ü', 'v'];
    let chars: Vec<char> = pinyin.chars().collect();

    // Find the first vowel
    if let Some(vowel_pos) = chars.iter().position(|c| vowels.contains(&c.to_ascii_lowercase())) {
        chars[vowel_pos..].iter().collect()
    } else {
        // No vowel found, return whole string (might be a tone number only)
        pinyin.to_string()
    }
}

/// Apply tone sandhi rules to a list of pinyins and characters (character-level fallback)
/// Main rules:
/// 1. Polyphonic character corrections (e.g., 回应 → huí yìng4)
/// 2. 一 (yi) tone sandhi: yi2 before tone 4, yi4 before tone 1/2/3 (e.g., 一百 → yi4 bai3)
/// 3. Certain two-character words have neutral tone on last char (e.g., 部分 → bù fen5)
/// 4. Two consecutive tone 3 → first becomes tone 2 (e.g., 总统 zǒng tǒng → zóng tǒng)
fn apply_tone_sandhi(chars: &[char], pinyins: &mut [Option<String>]) {
    let neutral_words = must_neutral_tone_words();

    // Apply polyphonic corrections based on context
    for i in 0..chars.len() {
        if is_chinese_char(chars[i]) {
            let prev_char = if i > 0 && is_chinese_char(chars[i - 1]) {
                Some(chars[i - 1])
            } else {
                None
            };
            if let Some(correct_pinyin) = get_polyphonic_correction(prev_char, chars[i]) {
                pinyins[i] = Some(correct_pinyin.to_string());
            }
        }
    }

    // Apply 一 (yi) tone sandhi - copied from Python's _yi_sandhi
    // Chinese numeric characters (Python's isnumeric() returns True for these)
    let numeric_chars = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
                         '百', '千', '万', '亿', '零'];
    let punct_chars = [',', '.', '!', '?', '，', '。', '！', '？'];

    for i in 0..chars.len() {
        if chars[i] == '一' && pinyins[i].is_some() {
            // Check for reduplication pattern: X一X (e.g., 看一看)
            if i > 0 && i + 1 < chars.len() && chars[i - 1] == chars[i + 1] {
                if let Some(ref mut pinyin) = pinyins[i] {
                    if pinyin.chars().last().map(|c| c.is_ascii_digit()).unwrap_or(false) {
                        pinyin.pop();
                        pinyin.push('5');
                    }
                }
                continue;
            }

            // Check for ordinal: 第一
            if i > 0 && chars[i - 1] == '第' {
                // Keep yi1 for ordinals
                continue;
            }

            // Check if in pure numeric sequence (like 一八四五, 一零零, etc.)
            // Python's rule: if all other chars in the word are numeric (isnumeric() = True), skip sandhi
            // Since Rust doesn't have jieba segmentation, we check surrounding chars:
            // - Find the extent of consecutive numeric chars around 一
            // - If 一 is at start and followed by numeric, OR
            // - If 一 is preceded and followed by numeric chars, skip sandhi
            let prev_is_numeric = i > 0 && numeric_chars.contains(&chars[i - 1]);
            let next_is_numeric = i + 1 < chars.len() && numeric_chars.contains(&chars[i + 1]);

            // Pure numeric context: either at start of sequence or surrounded by numeric
            // This handles: 一八四五 (一 at start, 八 is numeric)
            //              二一九零 (一 in middle, both neighbors numeric)
            if next_is_numeric && (i == 0 || prev_is_numeric) {
                // In a pure digit-by-digit number reading, keep yi1
                continue;
            }

            // Also check for pattern like 一十X where prev char is numeric
            // This handles: 二一十 or 百一十
            if i + 1 < chars.len() && chars[i + 1] == '十' && prev_is_numeric {
                continue;
            }

            // Standard sandhi rules
            let next_char = if i + 1 < chars.len() { Some(chars[i + 1]) } else { None };
            let next_tone = if i + 1 < chars.len() {
                pinyins[i + 1].as_ref().and_then(|p| p.chars().last())
            } else {
                None
            };

            if let Some(ref mut pinyin) = pinyins[i] {
                // Skip if already modified by word-level sandhi (not yi1 anymore)
                // This prevents character-level sandhi from overriding word-level results
                if *pinyin != "yi1" {
                    continue;
                }

                // Skip if next char is punctuation
                if let Some(nc) = next_char {
                    if punct_chars.contains(&nc) {
                        continue;
                    }
                }

                match next_tone {
                    Some('4') => {
                        // Before tone 4: yi1 → yi2
                        *pinyin = "yi2".to_string();
                    }
                    Some('1') | Some('2') | Some('3') | Some('5') => {
                        // Before tone 1/2/3/5: yi1 → yi4
                        *pinyin = "yi4".to_string();
                    }
                    _ => {
                        // Alone or at end: keep yi1
                    }
                }
            }
        }
    }

    // Apply 个 (ge) neutral tone when used as measure word after numbers
    // Python rule: 个 after digit or 几有两半多各整每做是 → ge5
    let ge_prev_chars = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '零',
                         '两', '几', '有', '半', '多', '各', '整', '每', '做', '是'];
    for i in 1..chars.len() {
        if chars[i] == '个' && ge_prev_chars.contains(&chars[i - 1]) {
            if let Some(ref mut pinyin) = pinyins[i] {
                if pinyin.chars().last().map(|c| c.is_ascii_digit()).unwrap_or(false) {
                    pinyin.pop();
                    pinyin.push('5');
                }
            }
        }
    }

    // Apply neutral tone for specific words
    let mut i = 0;
    while i < chars.len() {
        if i + 1 < chars.len() && is_chinese_char(chars[i]) && is_chinese_char(chars[i + 1]) {
            // Check for two-character words that need neutral tone
            let word: String = chars[i..i+2].iter().collect();
            if neutral_words.contains(word.as_str()) {
                // Apply neutral tone to the second character
                if let Some(ref mut p) = pinyins[i + 1] {
                    if p.chars().last().map(|c| c.is_ascii_digit()).unwrap_or(false) {
                        p.pop();
                        p.push('5');
                    }
                }
            }
        }
        i += 1;
    }

    // Find consecutive tone 3 sequences and change all but the last to tone 2
    let mut i = 0;
    while i < pinyins.len() {
        if let Some(ref pinyin) = pinyins[i] {
            if pinyin.ends_with('3') {
                // Found a tone 3, check if next is also tone 3
                let mut j = i + 1;
                while j < pinyins.len() {
                    match &pinyins[j] {
                        Some(p) if p.ends_with('3') => j += 1,
                        _ => break,
                    }
                }
                // If we found consecutive tone 3s, change all but the last to tone 2
                if j > i + 1 {
                    for k in i..j-1 {
                        if let Some(ref mut p) = pinyins[k] {
                            if p.ends_with('3') {
                                p.pop();
                                p.push('2');
                            }
                        }
                    }
                }
                i = j;
            } else {
                i += 1;
            }
        } else {
            i += 1;
        }
    }
}

/// Convert Chinese text to phonemes with tone sandhi
/// Note: text should already be normalized (numbers converted to Chinese) before calling this
pub fn chinese_g2p(text: &str) -> (Vec<String>, Vec<i32>) {
    // First pass: collect all characters and their pinyins
    let chars: Vec<char> = text.chars().collect();
    let mut char_pinyins: Vec<Option<String>> = Vec::with_capacity(chars.len());

    for &c in &chars {
        if is_chinese_char(c) {
            char_pinyins.push(get_pinyin_for_char(c));
        } else {
            char_pinyins.push(None); // Non-Chinese chars don't participate in sandhi
        }
    }

    // Apply G2PW for polyphonic character disambiguation
    // G2PW uses BERT-based ML model for context-aware pronunciation
    let g2pw_pinyins = get_pinyin_with_g2pw(text);
    for (i, g2pw_pinyin) in g2pw_pinyins.into_iter().enumerate() {
        if let Some(pinyin) = g2pw_pinyin {
            // G2PW provides better pronunciation for polyphonic characters
            char_pinyins[i] = Some(pinyin);
        }
    }

    // Apply hard corrections for characters where G2PW/pinyin crate is wrong
    // These corrections take priority over G2PW
    let corrections = pinyin_corrections();
    for (i, &c) in chars.iter().enumerate() {
        if let Some(&corrected) = corrections.get(&c) {
            char_pinyins[i] = Some(corrected.to_string());
        }
    }

    // Apply polyphone corrections (word-context based) - fallback for chars G2PW doesn't cover
    apply_polyphone_corrections(&chars, &mut char_pinyins);

    // Apply tone sandhi using word-level processing (Python-compatible)
    // This uses jieba segmentation and the full tone_sandhi module
    apply_word_level_tone_sandhi(text, &mut char_pinyins);

    // Also apply character-level sandhi as fallback for edge cases
    apply_tone_sandhi(&chars, &mut char_pinyins);

    // Second pass: convert to phonemes
    let mut phonemes = Vec::new();
    let mut word2ph = Vec::new();

    for (i, c) in chars.iter().enumerate() {
        if c.is_whitespace() {
            // Skip whitespace
            continue;
        } else if *c == ',' || *c == '.' || *c == '!' || *c == '?' || *c == '-' || *c == '…'
            || *c == '，' || *c == '。' || *c == '！' || *c == '？' || *c == '、' || *c == '；'
            || *c == '：' || *c == '"' || *c == '"' || *c == '（' || *c == '）'
            || *c == '《' || *c == '》' || *c == '【' || *c == '】'
            || *c == '「' || *c == '」' || *c == '『' || *c == '』' || *c == '〈' || *c == '〉' {
            // Map punctuation to phonemes - keep valid symbols, skip brackets entirely
            // GPT-SoVITS symbol table has: "!" (0), "," (1), "-" (2), "." (3), "?" (4)
            let punct_phoneme = match *c {
                '.' | '。' => Some("."),
                ',' | '，' | '、' => Some(","),
                '!' | '！' => Some("!"),
                '?' | '？' => Some("?"),
                '-' | '—' | '–' => Some("-"),
                // Skip brackets and other punctuation entirely - don't produce SP
                '（' | '）' | '《' | '》' | '【' | '】' | '「' | '」' | '『' | '』' | '〈' | '〉'
                | '\u{201C}' | '\u{201D}' | '：' | '；' | '…' => None,
                _ => None,  // Skip unknown punctuation
            };
            if let Some(ph) = punct_phoneme {
                phonemes.push(ph.to_string());
                word2ph.push(1);
            } else {
                // Skipped punctuation still needs word2ph entry of 0 for BERT alignment
                word2ph.push(0);
            }
        } else if is_chinese_char(*c) {
            // Use the (possibly modified) pinyin from tone sandhi
            let char_phonemes = if let Some(ref pinyin) = char_pinyins[i] {
                let (initial, final_part) = get_initial_final(pinyin);
                let mut ph = Vec::new();
                if let Some(init) = initial {
                    if has_symbol(init) {
                        ph.push(init.to_string());
                    }
                }
                if has_symbol(&final_part) {
                    ph.push(final_part);
                }
                if ph.is_empty() {
                    ph.push(symbols::UNK.to_string());
                }
                ph
            } else {
                vec![symbols::UNK.to_string()]
            };
            let count = char_phonemes.len() as i32;
            phonemes.extend(char_phonemes);
            word2ph.push(count);
        } else if c.is_ascii_alphabetic() {
            phonemes.push(c.to_ascii_uppercase().to_string());
            word2ph.push(1);
        } else if c.is_ascii_digit() {
            let chinese_num = match c {
                '0' => '零', '1' => '一', '2' => '二', '3' => '三', '4' => '四',
                '5' => '五', '6' => '六', '7' => '七', '8' => '八', '9' => '九',
                _ => unreachable!(),
            };
            let char_phonemes = char_to_phonemes(chinese_num);
            let count = char_phonemes.len() as i32;
            phonemes.extend(char_phonemes);
            word2ph.push(count);
        }
    }

    (phonemes, word2ph)
}

/// Convert English text to phonemes using CMU dictionary and neural G2P for OOV words
pub fn english_g2p(text: &str) -> (Vec<String>, Vec<i32>) {
    use super::g2p_en;

    let mut phonemes = Vec::new();
    let mut word2ph = Vec::new();

    // Split text into words and numbers, preserving punctuation
    let mut current_word = String::new();
    let mut current_number = String::new();
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        if c.is_ascii_alphabetic() {
            // Flush any pending number
            if !current_number.is_empty() {
                let num_phonemes = number_to_english_phonemes(&current_number);
                for (ph, count) in num_phonemes {
                    phonemes.extend(ph);
                    word2ph.push(count);
                }
                current_number.clear();
            }
            current_word.push(c);
        } else if c == '\'' {
            // Apostrophe: Python tokenizes it separately, then replace_phs maps ' -> '-'
            // Flush current word first
            if !current_word.is_empty() {
                let word_phonemes = g2p_en::word_to_phonemes(&current_word);
                let count = word_phonemes.len() as i32;
                phonemes.extend(word_phonemes);
                word2ph.push(count);
                current_word.clear();
            }
            // Emit '-' phoneme for apostrophe
            phonemes.push("-".to_string());
            word2ph.push(1);
        } else if c.is_ascii_digit() {
            // Flush any pending word
            if !current_word.is_empty() {
                let word_phonemes = g2p_en::word_to_phonemes(&current_word);
                let count = word_phonemes.len() as i32;
                phonemes.extend(word_phonemes);
                word2ph.push(count);
                current_word.clear();
            }
            current_number.push(c);
        } else {
            // Process accumulated word
            if !current_word.is_empty() {
                let word_phonemes = g2p_en::word_to_phonemes(&current_word);
                let count = word_phonemes.len() as i32;
                phonemes.extend(word_phonemes);
                word2ph.push(count);
                current_word.clear();
            }
            // Process accumulated number
            if !current_number.is_empty() {
                let num_phonemes = number_to_english_phonemes(&current_number);
                for (ph, count) in num_phonemes {
                    phonemes.extend(ph);
                    word2ph.push(count);
                }
                current_number.clear();
            }

            // Handle punctuation and spaces
            if c.is_whitespace() {
                // Skip multiple spaces
                while chars.peek().map(|c| c.is_whitespace()).unwrap_or(false) {
                    chars.next();
                }
            } else if has_symbol(&c.to_string()) {
                // Map punctuation to phonemes - keep valid symbols, convert others to SP
                // GPT-SoVITS symbol table has: "!" (0), "," (1), "-" (2), "." (3), "?" (4)
                let punct_phoneme = match c {
                    '.' => ".",
                    ',' => ",",
                    '!' => "!",
                    '?' => "?",
                    '-' => "-",
                    _ => "SP",  // Other punctuation becomes short pause
                };
                phonemes.push(punct_phoneme.to_string());
                word2ph.push(1);
            }
        }
    }

    // Process final word if any
    if !current_word.is_empty() {
        let word_phonemes = g2p_en::word_to_phonemes(&current_word);
        let count = word_phonemes.len() as i32;
        phonemes.extend(word_phonemes);
        word2ph.push(count);
    }
    // Process final number if any
    if !current_number.is_empty() {
        let num_phonemes = number_to_english_phonemes(&current_number);
        for (ph, count) in num_phonemes {
            phonemes.extend(ph);
            word2ph.push(count);
        }
    }

    (phonemes, word2ph)
}

/// Convert a number string to English phonemes using num2en (like Python's inflect)
/// Returns a vector of (phonemes, word2ph_count) for each word
fn number_to_english_phonemes(num_str: &str) -> Vec<(Vec<String>, i32)> {
    use super::g2p_en;

    // Use num2en to convert number to English words (like Python's inflect)
    // e.g., 2001 → "two thousand one", 123 → "one hundred twenty-three"
    if let Ok(num) = num_str.parse::<u64>() {
        let words = num2en::u64_to_words(num);
        // Split into individual words and convert each to phonemes
        let mut result = Vec::new();
        for word in words.split(|c: char| c == ' ' || c == '-') {
            if !word.is_empty() {
                let ph = g2p_en::word_to_phonemes(word);
                if !ph.is_empty() {
                    result.push((ph.clone(), ph.len() as i32));
                }
            }
        }
        if !result.is_empty() {
            return result;
        }
    }

    // Fallback: read digits individually
    let mut result = Vec::new();
    for c in num_str.chars() {
        if let Some(digit) = c.to_digit(10) {
            let word = match digit {
                0 => "zero", 1 => "one", 2 => "two", 3 => "three", 4 => "four",
                5 => "five", 6 => "six", 7 => "seven", 8 => "eight", 9 => "nine",
                _ => unreachable!(),
            };
            let ph = g2p_en::word_to_phonemes(word);
            result.push((ph.clone(), ph.len() as i32));
        }
    }
    result
}

/// Language segment for mixed text processing
#[derive(Debug, Clone)]
pub struct LangSegment {
    pub text: String,
    pub is_english: bool,
}

/// Segment text into Chinese and English chunks
/// Digits are context-dependent:
/// - In English context (after letters): treated as English (e.g., "Room 404")
/// - Followed by Chinese units: treated as Chinese (e.g., "126.4亿斤")
pub fn segment_by_language(text: &str) -> Vec<LangSegment> {
    let mut segments = Vec::new();
    let mut current_text = String::new();
    let mut current_is_english: Option<bool> = None;

    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();

    for i in 0..len {
        let c = chars[i];
        let is_letter = c.is_ascii_alphabetic();
        let is_digit = c.is_ascii_digit() || c == '.';  // Include decimal point with digits
        let is_zh = is_chinese_char(c);
        let is_punct = is_punctuation(c) || c.is_whitespace();

        if is_letter {
            // English letter - always English
            if current_is_english == Some(false) && !current_text.is_empty() {
                segments.push(LangSegment { text: current_text.clone(), is_english: false });
                current_text.clear();
            }
            current_text.push(c);
            current_is_english = Some(true);
        } else if is_digit {
            // Digit - check context by looking ahead
            // Skip all consecutive digits/dots to find what follows
            let mut j = i + 1;
            while j < len && (chars[j].is_ascii_digit() || chars[j] == '.') {
                j += 1;
            }
            // Check what comes after the number
            let followed_by_chinese = j < len && is_chinese_char(chars[j]);
            let followed_by_english = j < len && chars[j].is_ascii_alphabetic();

            if followed_by_chinese && !followed_by_english {
                // Digits followed by Chinese (e.g., "126.4亿斤") - treat as Chinese
                if current_is_english == Some(true) && !current_text.is_empty() {
                    segments.push(LangSegment { text: current_text.clone(), is_english: true });
                    current_text.clear();
                }
                current_text.push(c);
                current_is_english = Some(false);
            } else {
                // Digits in English context or standalone
                if current_is_english == Some(false) && !current_text.is_empty() {
                    segments.push(LangSegment { text: current_text.clone(), is_english: false });
                    current_text.clear();
                }
                current_text.push(c);
                current_is_english = Some(true);
            }
        } else if is_zh {
            // Chinese character
            if current_is_english == Some(true) && !current_text.is_empty() {
                segments.push(LangSegment { text: current_text.clone(), is_english: true });
                current_text.clear();
            }
            current_text.push(c);
            current_is_english = Some(false);
        } else if is_punct {
            // Punctuation belongs to current segment
            current_text.push(c);
        }
        // Skip other characters
    }

    // Add final segment
    if !current_text.is_empty() {
        segments.push(LangSegment {
            text: current_text,
            is_english: current_is_english.unwrap_or(false)
        });
    }

    segments
}

/// Convert mixed Chinese/English text to phonemes
pub fn mixed_g2p(text: &str) -> (Vec<String>, Vec<i32>) {
    let segments = segment_by_language(text);
    let mut all_phonemes = Vec::new();
    let mut all_word2ph = Vec::new();

    for segment in segments {
        let (phonemes, word2ph) = if segment.is_english {
            // Word-level English G2P (matches Python's en_G2p which uses wordsegment
            // to split concatenated words back into real words for CMU dict lookup)
            english_g2p(&segment.text)
        } else {
            chinese_g2p(&segment.text)
        };
        all_phonemes.extend(phonemes);
        all_word2ph.extend(word2ph);
    }

    (all_phonemes, all_word2ph)
}

/// Spell English text letter-by-letter, matching Python's LangSegment behavior
/// where uppercase concatenated English is split into individual letters.
/// Each letter gets its CMU dict pronunciation (the letter name, not the sound).
/// Punctuation is kept as-is.
#[allow(dead_code)]
fn english_letter_spell(text: &str) -> (Vec<String>, Vec<i32>) {
    use super::g2p_en;

    let mut phonemes = Vec::new();
    let mut word2ph = Vec::new();

    for c in text.chars() {
        if c.is_ascii_alphabetic() {
            // Each letter gets its CMU pronunciation (letter name)
            let letter = c.to_ascii_uppercase().to_string();
            let letter_phones = g2p_en::word_to_phonemes(&letter);
            let count = letter_phones.len() as i32;
            phonemes.extend(letter_phones);
            word2ph.push(count);
        } else if c == ',' || c == '.' || c == '!' || c == '?' {
            // Punctuation
            phonemes.push(c.to_string());
            word2ph.push(1);
        }
        // Skip other characters (spaces etc.)
    }

    (phonemes, word2ph)
}

/// Text preprocessor configuration
#[derive(Debug, Clone)]
pub struct PreprocessorConfig {
    /// Default language if not detected
    pub default_language: Language,
    /// Whether to add BOS token
    pub add_bos: bool,
    /// Whether to add EOS token
    pub add_eos: bool,
}

impl Default for PreprocessorConfig {
    fn default() -> Self {
        Self {
            default_language: Language::Chinese,
            add_bos: true,
            add_eos: true,
        }
    }
}

/// Text preprocessor
pub struct TextPreprocessor {
    config: PreprocessorConfig,
}

impl TextPreprocessor {
    /// Create new preprocessor with config
    pub fn new(config: PreprocessorConfig) -> Self {
        Self { config }
    }

    /// Preprocess text to phonemes
    ///
    /// # Arguments
    /// * `text` - Input text
    /// * `language` - Optional language override (None for auto-detect)
    pub fn preprocess(&self, text: &str, language: Option<Language>) -> PreprocessorOutput {
        if text.trim().is_empty() {
            return PreprocessorOutput {
                phoneme_ids: if self.config.add_bos {
                    vec![bos_id(), eos_id()]
                } else {
                    vec![eos_id()]
                },
                phonemes: if self.config.add_bos {
                    vec![symbols::BOS.to_string(), symbols::EOS.to_string()]
                } else {
                    vec![symbols::EOS.to_string()]
                },
                word2ph: if self.config.add_bos { vec![1, 1] } else { vec![1] },
                text_normalized: String::new(),
                language: language.unwrap_or(self.config.default_language),
            };
        }

        // Detect language if not specified
        let language = language.unwrap_or_else(|| detect_language(text));

        // Normalize text
        // For mixed text: use mix_text_normalize to preserve English,
        // matching Python's all_zh path which calls mix_text_normalize
        // when English letters are detected.
        let (text_normalized, language) = match language {
            Language::Chinese => {
                // Check if text contains English - if so, treat as Mixed
                // This matches Python's all_zh path: if re.search(r'[A-Za-z]', text)
                if text.chars().any(|c| c.is_ascii_alphabetic()) {
                    (mix_text_normalize(text), Language::Mixed)
                } else {
                    (normalize_chinese(text), Language::Chinese)
                }
            }
            Language::English => (normalize_english(text), Language::English),
            Language::Mixed => (mix_text_normalize(text), Language::Mixed),
        };

        // Convert to phonemes
        let (mut phonemes, mut word2ph) = match language {
            Language::Chinese => chinese_g2p(&text_normalized),
            Language::English => english_g2p(&text_normalized),
            Language::Mixed => {
                // For mixed, segment by language and process each segment
                mixed_g2p(&text_normalized)
            }
        };

        // Add BOS/EOS tokens
        if self.config.add_bos {
            phonemes.insert(0, symbols::BOS.to_string());
            word2ph.insert(0, 1);
        }
        if self.config.add_eos {
            phonemes.push(symbols::EOS.to_string());
            word2ph.push(1);
        }

        // Convert to IDs
        let phoneme_ids: Vec<i32> = phonemes
            .iter()
            .map(|s| symbol_to_id(s))
            .collect();

        PreprocessorOutput {
            phoneme_ids,
            phonemes,
            word2ph,
            text_normalized,
            language,
        }
    }
}

impl Default for TextPreprocessor {
    fn default() -> Self {
        Self::new(PreprocessorConfig::default())
    }
}

/// Convenience function to preprocess text
pub fn preprocess_text(text: &str, language: Option<Language>) -> PreprocessorOutput {
    TextPreprocessor::default().preprocess(text, language)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_chinese_char() {
        assert!(is_chinese_char('你'));
        assert!(is_chinese_char('好'));
        assert!(is_chinese_char('世'));
        assert!(!is_chinese_char('a'));
        assert!(!is_chinese_char('1'));
        assert!(!is_chinese_char(' '));
    }

    #[test]
    fn test_detect_language() {
        assert_eq!(detect_language("你好世界"), Language::Chinese);
        assert_eq!(detect_language("hello world"), Language::English);
        // "你好 world" has both Chinese and English -> Mixed
        assert_eq!(detect_language("你好 world"), Language::Mixed);
        // Any mix of Chinese and English is Mixed
        assert_eq!(detect_language("你好wo"), Language::Mixed);
        assert_eq!(detect_language("Hello世界"), Language::Mixed);
    }

    #[test]
    fn test_normalize_chinese() {
        assert_eq!(normalize_chinese("你好，世界！"), "你好,世界!");
        // Parentheses are removed as they are non-phonetic characters
        assert_eq!(normalize_chinese("（测试）"), "测试");
    }

    #[test]
    fn test_get_initial_final() {
        let (init, final_) = get_initial_final("ni3");
        assert_eq!(init, Some("n"));
        assert_eq!(final_, "i3");

        let (init, final_) = get_initial_final("hao3");
        assert_eq!(init, Some("h"));
        assert_eq!(final_, "ao3");

        let (init, final_) = get_initial_final("shi4");
        assert_eq!(init, Some("sh"));
        assert_eq!(final_, "ir4");  // Retroflex i for sh/zh/ch/r

        let (init, final_) = get_initial_final("zhi1");
        assert_eq!(init, Some("zh"));
        assert_eq!(final_, "ir1");  // Retroflex i for sh/zh/ch/r
    }

    #[test]
    fn test_chinese_g2p() {
        let (phonemes, word2ph) = chinese_g2p("你好");
        // "你" -> "n" + "i3" (2 phonemes)
        // "好" -> "h" + "ao3" (2 phonemes)
        assert!(!phonemes.is_empty());
        assert_eq!(phonemes.len(), word2ph.iter().sum::<i32>() as usize);
    }

    #[test]
    fn test_english_g2p() {
        let (phonemes, word2ph) = english_g2p("hello world");
        assert!(!phonemes.is_empty());
        // CMU dictionary uses ARPABET notation
        // "hello" -> ["HH", "AH0", "L", "OW1"] or similar
        // "world" -> ["W", "ER1", "L", "D"]
        // Check that we have some phonemes and word2ph is reasonable
        assert!(phonemes.len() >= 4); // At least a few phonemes
        assert_eq!(word2ph.len(), 2); // Two words
    }

    #[test]
    fn test_preprocessor() {
        let preprocessor = TextPreprocessor::default();

        let output = preprocessor.preprocess("你好", Some(Language::Chinese));
        assert!(!output.phoneme_ids.is_empty());
        // BOS/EOS are mapped to SP in GPT-SoVITS symbol table
        assert!(output.phonemes.contains(&symbols::BOS.to_string()));
        assert!(output.phonemes.contains(&symbols::EOS.to_string()));
    }

    #[test]
    fn test_empty_text() {
        let preprocessor = TextPreprocessor::default();
        let output = preprocessor.preprocess("", None);
        assert_eq!(output.phonemes, vec![symbols::BOS, symbols::EOS]);
    }

    #[test]
    fn test_yi_tone_sandhi() {
        // 一 before tone 3 (百 = bai3) → yi4
        let (phonemes, _) = chinese_g2p("一百");
        // Find the yi phoneme
        let yi_phoneme = phonemes.iter().find(|p| p.starts_with("i"));
        assert_eq!(yi_phoneme, Some(&"i4".to_string()), "一 before 百(bai3) should become yi4");

        // 一 before tone 4 (样 = yang4) → yi2
        let (phonemes, _) = chinese_g2p("一样");
        let yi_phoneme = phonemes.iter().find(|p| p.starts_with("i"));
        assert_eq!(yi_phoneme, Some(&"i2".to_string()), "一 before 样(yang4) should become yi2");
    }

    #[test]
    fn test_polyphonic_wei() {
        // 为 in 改为 should be wei2 (not wei4)
        let (phonemes, _) = chinese_g2p("改为");
        // 改 = g ai3, 为 = w ei2
        // Phonemes should be: ["g", "ai3", "w", "ei2"]
        println!("Rust 改为: {:?}", phonemes);
        assert_eq!(phonemes, vec!["g", "ai3", "w", "ei2"],
            "改为 phonemes should match Python exactly");

        // Also verify 成为
        let (phonemes, _) = chinese_g2p("成为");
        println!("Rust 成为: {:?}", phonemes);
        assert!(phonemes.contains(&"ei2".to_string()),
            "为 in 成为 should be wei2, got phonemes: {:?}", phonemes);
    }

    #[test]
    fn test_gaiwei_in_mixed_content() {
        // Test the exact text that previously had the issue
        // "刊名改为经济学人" should have 改为 = g ai3 w ei2
        let text = "刊名改为经济学人";
        let (phonemes, _) = chinese_g2p(text);
        println!("Rust {}: {:?}", text, phonemes);

        // Find position of 为 phoneme (should be ei2 not ei4)
        // 刊 名 改 为 经 济 学 人
        // 0  1  2  3  4  5  6  7
        // Each character produces 2 phonemes (initial + final)
        // 改 = g ai3 (positions 4, 5)
        // 为 = w ei2 (positions 6, 7)
        assert!(phonemes.len() >= 8, "Should have at least 8 phonemes");

        // Check that we have ei2 for 为
        assert!(phonemes.contains(&"ei2".to_string()),
            "为 in 改为 should be wei2, got: {:?}", phonemes);

        // Verify no ei4 which would be wrong
        assert!(!phonemes.contains(&"ei4".to_string()),
            "Should NOT have ei4 for 为, got: {:?}", phonemes);
    }

    #[test]
    fn test_atlantic_mixed_content() {
        // Test complex mixed Chinese/English content
        let text = "2011年12月，TheAtlantic.com上开设了一个新的健康频道，内容涵盖食品以及与思想、身体、性爱、家庭和公共卫生有关的主题。";

        let norm = normalize_chinese(text);
        println!("Normalized ({} chars): {}", norm.chars().count(), norm);

        let (phones, word2ph) = chinese_g2p(&norm);
        println!("Phonemes ({}): {:?}", phones.len(), phones);
        println!("Word2Ph ({}): {:?}", word2ph.len(), word2ph);

        // Verify key properties
        assert!(!phones.is_empty(), "Should have phonemes");
        assert!(!word2ph.is_empty(), "Should have word2ph");
        assert_eq!(phones.len(), word2ph.iter().sum::<i32>() as usize,
            "Phoneme count should match sum of word2ph");

        // Verify 作为 has wei2 (not wei4)
        // The text contains "作为聚合器" so 为 should be wei2
    }

    #[test]
    fn test_atlantic_complex() {
        // Complex Atlantic text with names, dates, parentheses, etc.
        let text = "2017年7月28日，《大西洋杂志》宣布，亿万富翁投资者和慈善家劳伦·鲍威尔·乔布斯（前苹果公司总裁兼CEO史提夫·乔布斯的遗孀），她透过自己的爱默生集团获得了大部分股权，而爱默生集团的工作人员彼得·拉特曼（Peter Lattman）即时被任命为《大西洋杂志》的副董事长。大卫·G·布拉德利及大西洋媒体在该次交易中保留了少数股份[35]。";

        let norm = normalize_chinese(text);
        println!("Normalized ({} chars): {}", norm.chars().count(), norm);

        let (phones, word2ph) = chinese_g2p(&norm);
        println!("Phonemes ({}): {:?}", phones.len(), phones);
        println!("Word2Ph ({}): {:?}", word2ph.len(), word2ph);

        // Verify basic properties
        assert!(!phones.is_empty(), "Should have phonemes");
        assert!(!word2ph.is_empty(), "Should have word2ph");
        assert_eq!(phones.len(), word2ph.iter().sum::<i32>() as usize,
            "Phoneme count should match sum of word2ph");
    }

    #[test]
    fn test_yi_in_year_number() {
        // Test "二零一一年" - first 一 should stay yi1 (in number sequence)
        // second 一 should become yi4 (before 年 which is tone 2)
        // Python: yi1 yi4
        let (phones, _) = chinese_g2p("二零一一年");
        println!("Phonemes for 二零一一年: {:?}", phones);

        // 二 = EE er4 (0-1)
        // 零 = l ing2 (2-3)
        // 一 = y i? (4-5)
        // 一 = y i? (6-7)
        // 年 = n ian2 (8-9)

        // Check first 一 is yi1 (position 5)
        assert_eq!(phones[5], "i1",
            "First 一 in 二零一一 should be yi1, got: {}", phones[5]);

        // Check second 一 is yi4 (position 7)
        assert_eq!(phones[7], "i4",
            "Second 一 before 年 should be yi4, got: {}", phones[7]);
    }

    #[test]
    fn test_yige_sandhi() {
        // 一 before 个 (ge4) should become yi2
        let (phones, _) = chinese_g2p("一个");

        // 一 = y i2 (0-1) - yi sandhi: 一 before tone 4 → yi2
        // 个 = g e5 (2-3) - neutral tone
        assert_eq!(phones[1], "i2",
            "一 before 个 should be yi2, got: {}", phones[1]);
    }

    #[test]
    fn test_preprocess_text_convenience() {
        let output = preprocess_text("你好", Some(Language::Chinese));
        assert!(!output.phoneme_ids.is_empty());
        assert_eq!(output.language, Language::Chinese);
    }

    #[test]
    fn test_decimal_normalization() {
        assert_eq!(number_to_chinese_with_decimal("163.6"), "一百六十三点六");
        assert_eq!(number_to_chinese_with_decimal("0.5"), "零点五");
        assert_eq!(number_to_chinese_with_decimal("3.14"), "三点一四");
        assert_eq!(number_to_chinese_with_decimal("114.7"), "一百一十四点七");
        assert_eq!(number_to_chinese_with_decimal("126.4"), "一百二十六点四");
    }

    #[test]
    fn test_percentage_normalization() {
        assert_eq!(replace_percentage("70%"), "百分之七十");
        assert_eq!(replace_percentage("75%"), "百分之七十五");
        assert_eq!(replace_percentage("163.6%"), "百分之一百六十三点六");
    }

    #[test]
    fn test_full_number_normalization() {
        let result = normalize_numbers_to_chinese("增产163.6亿斤");
        assert_eq!(result, "增产一百六十三点六亿斤");

        let result = normalize_numbers_to_chinese("接近70%");
        assert_eq!(result, "接近百分之七十");

        let result = normalize_numbers_to_chinese("增量的75%");
        assert_eq!(result, "增量的百分之七十五");
    }

    #[test]
    fn test_negative_number() {
        let result = normalize_numbers_to_chinese("温度是-10度");
        assert!(result.contains("负十") || result.contains("零下"));

        let result = normalize_numbers_to_chinese("-25.5%");
        assert_eq!(result, "负百分之二十五点五");
    }

    #[test]
    fn test_fraction_normalization() {
        assert_eq!(replace_fraction("1/2"), "二分之一");
        assert_eq!(replace_fraction("3/4"), "四分之三");
        assert_eq!(replace_fraction("-1/2"), "负二分之一");
    }

    #[test]
    fn test_date_normalization() {
        let result = replace_date("2024-01-15");
        assert_eq!(result, "二零二四年一月十五日");

        let result = replace_date("2024/12/31");
        assert_eq!(result, "二零二四年十二月三十一日");
    }

    #[test]
    fn test_time_normalization() {
        let result = replace_time("14:30");
        assert_eq!(result, "十四点半");

        let result = replace_time("9:45");
        assert_eq!(result, "九点四十五分");

        let result = replace_time("18:05:30");
        assert_eq!(result, "十八点五分三十秒");
    }

    #[test]
    fn test_range_normalization() {
        let result = replace_range("1-10");
        assert_eq!(result, "一到十");

        let result = replace_range("0.5~1.5");
        assert_eq!(result, "零点五到一点五");
    }

    #[test]
    fn test_temperature_normalization() {
        let result = replace_temperature("-3°C");
        assert_eq!(result, "零下三摄氏度");

        let result = replace_temperature("25℃");
        assert_eq!(result, "二十五摄氏度");

        let result = replace_temperature("37度");
        assert_eq!(result, "三十七度");
    }

    #[test]
    fn test_consecutive_punctuation() {
        // Single punctuation should be preserved as-is
        assert_eq!(replace_consecutive_punctuation("你好,世界"), "你好,世界");
        assert_eq!(replace_consecutive_punctuation("你好.世界"), "你好.世界");
        assert_eq!(replace_consecutive_punctuation("你好!"), "你好!");
        assert_eq!(replace_consecutive_punctuation("你好?"), "你好?");

        // Consecutive punctuation should be deduplicated (keep first)
        assert_eq!(replace_consecutive_punctuation("你好..世界"), "你好.世界");
        assert_eq!(replace_consecutive_punctuation("你好...世界"), "你好.世界");
        assert_eq!(replace_consecutive_punctuation("你好!!世界"), "你好!世界");
        assert_eq!(replace_consecutive_punctuation("你好,,世界"), "你好,世界");
        assert_eq!(replace_consecutive_punctuation("你好!?世界"), "你好!世界");
    }

    /// Comparison test to output phonemes for Python comparison
    /// Run with: cargo test --features jieba compare_with_python -- --nocapture
    #[test]
    fn compare_with_python() {
        let preprocessor = TextPreprocessor::new(PreprocessorConfig {
            add_bos: false,
            add_eos: false,
            ..Default::default()
        });

        let test_cases = vec![
            "你好",
            "你好世界",
            "一百",
            "一样",
            "看一看",
            "第一",
            "一二三",
            "一个",
            "不对",
            "不好",
            "看不懂",
            "老虎",
            "展览馆",
            "朋友",
            "妈妈",
            "东西",
            "杂志",
            "改为",
            "成为",
        ];

        println!("\n{}", "=".repeat(60));
        println!("Rust Preprocessing Test Results");
        println!("{}", "=".repeat(60));

        for text in test_cases {
            let result = preprocessor.preprocess(text, Some(Language::Chinese));
            println!("\nInput: {}", text);
            println!("[Rust] Normalized: {}", result.text_normalized);
            println!("[Rust] Phonemes ({}): {:?}", result.phonemes.len(), result.phonemes);
            println!("[Rust] Word2Ph ({}): {:?}", result.word2ph.len(), result.word2ph);
        }

        println!("\n{}", "=".repeat(60));
        println!("COMPARISON SUMMARY");
        println!("{}", "=".repeat(60));

        // Expected Python outputs for comparison
        // These values are from real dora-primespeech Python (moyoyo_tts/text/chinese2.py)
        // using opencpop-strict.txt phoneme mapping with ir/i0/v notation
        let python_expected = vec![
            // Basic greetings
            ("你好", vec!["n", "i2", "h", "ao3"]),
            ("你好世界", vec!["n", "i2", "h", "ao3", "sh", "ir4", "j", "ie4"]),

            // Yi sandhi
            ("一百", vec!["y", "i4", "b", "ai3"]),  // yi before tone 3 -> yi4
            ("一样", vec!["y", "i2", "y", "ang4"]), // yi before tone 4 -> yi2
            ("看一看", vec!["k", "an4", "y", "i5", "k", "an4"]), // yi in reduplication -> yi5
            ("第一", vec!["d", "i4", "y", "i1"]),   // ordinal yi -> yi1
            ("一二三", vec!["y", "i1", "EE", "er4", "s", "an1"]), // yi in number sequence -> yi1
            ("一个", vec!["y", "i2", "g", "e5"]),   // yi ge -> yi2 (before tone 4)

            // Bu sandhi
            ("不对", vec!["b", "u2", "d", "ui4"]),  // bu before tone 4 -> bu2
            ("不好", vec!["b", "u4", "h", "ao3"]),  // bu before tone 3 -> bu4
            ("看不懂", vec!["k", "an4", "b", "u5", "d", "ong3"]), // bu in V不V -> bu5

            // Three-tone sandhi
            ("老虎", vec!["l", "ao2", "h", "u3"]),  // two tone-3 -> first becomes tone 2
            ("展览馆", vec!["zh", "an2", "l", "an2", "g", "uan3"]), // three tone-3

            // Neutral tone
            ("朋友", vec!["p", "eng2", "y", "ou5"]), // 友 -> neutral tone
            ("妈妈", vec!["m", "a1", "m", "a5"]),    // reduplication -> second neutral
            ("东西", vec!["d", "ong1", "x", "i5"]),  // 西 -> neutral tone

            // Polyphonic characters (verified from real dora-primespeech)
            ("杂志", vec!["z", "a2", "zh", "ir4"]),  // 杂志 uses ir4 for retroflex
            ("改为", vec!["g", "ai3", "w", "ei2"]),  // 为 as "change to" -> wei2
            ("成为", vec!["ch", "eng2", "w", "ei2"]), // 为 as "become" -> wei2

            // Complex phrases (verified from real dora-primespeech)
            ("大西洋杂志宣布", vec!["d", "a4", "x", "i1", "y", "ang2", "z", "a2", "zh", "ir4", "x", "van1", "b", "u4"]),
            ("亿万富翁投资者", vec!["y", "i4", "w", "an4", "f", "u4", "w", "eng1", "t", "ou2", "z", "i01", "zh", "e3"]),
            ("前苹果公司总裁", vec!["q", "ian2", "p", "ing2", "g", "uo3", "g", "ong1", "s", "i01", "z", "ong3", "c", "ai2"]),
            ("爱默生集团", vec!["AA", "ai4", "m", "o4", "sh", "eng1", "j", "i2", "t", "uan2"]), // 爱 uses AA initial
        ];

        let mut matches = 0;
        let mut mismatches = 0;

        for (text, expected_phones) in &python_expected {
            let result = preprocessor.preprocess(text, Some(Language::Chinese));
            let rust_phones: Vec<&str> = result.phonemes.iter().map(|s| s.as_str()).collect();

            if rust_phones == *expected_phones {
                println!("\n[MATCH] {}", text);
                matches += 1;
            } else {
                println!("\n[MISMATCH] {}", text);
                println!("  Python: {:?}", expected_phones);
                println!("  Rust:   {:?}", rust_phones);
                mismatches += 1;
            }
        }

        println!("\n{}", "=".repeat(60));
        println!("Results: {} matches, {} mismatches", matches, mismatches);
        println!("{}", "=".repeat(60));
    }

    /// Test mixed Chinese/English/numbers text (user's test case)
    /// Run with: cargo test --features jieba test_mixed_content -- --nocapture
    #[test]
    fn test_mixed_content() {
        let preprocessor = TextPreprocessor::new(PreprocessorConfig {
            add_bos: false,
            add_eos: false,
            ..Default::default()
        });

        let text = r#"1845年，在英国"铁路狂热"时期，该报一度与《银行公报》（Bankers' Gazette）及《铁路观察》（Railway Monitor）合并，并将刊名改为《经济学人：商业周报、银行公报及铁路观察——政治化和文学化的大众报纸》（The Economist, Weekly Commercial Times, Bankers' Gazette, and Railway Monitor. A Political, Literary and General Newspaper）。"#;

        let result = preprocessor.preprocess(text, Some(Language::Chinese));

        println!("\n{}", "=".repeat(60));
        println!("Mixed Content Test");
        println!("{}", "=".repeat(60));
        println!("\nInput: {}...", &text[..80.min(text.len())]);
        println!("\n[Rust] Normalized ({} chars):", result.text_normalized.len());
        println!("  {}", result.text_normalized);
        println!("\n[Rust] Phonemes ({}):", result.phonemes.len());
        println!("  {:?}", result.phonemes);
        println!("\n[Rust] Word2Ph ({}):", result.word2ph.len());
        println!("  {:?}", result.word2ph);

        // Basic sanity checks
        assert!(!result.phonemes.is_empty(), "Should produce some phonemes");
        assert!(!result.word2ph.is_empty(), "Should produce word2ph mappings");
        assert_eq!(
            result.phonemes.len(),
            result.word2ph.iter().map(|&x| x as usize).sum::<usize>(),
            "Phoneme count should match sum of word2ph"
        );
    }
}
