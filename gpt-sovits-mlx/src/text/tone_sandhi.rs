//! Complete Tone Sandhi implementation
//!
//! Port of Python's tone_sandhi.py with all rules:
//! - Bu (不) sandhi
//! - Yi (一) sandhi
//! - Neural/Neutral tone sandhi
//! - Three-tone sandhi
//! - Word pre-merging for correct sandhi application

use std::collections::HashSet;

/// Tone Sandhi processor for Mandarin Chinese
///
/// Implements all standard Mandarin tone sandhi rules:
/// 1. 不 (bù) tone changes
/// 2. 一 (yī) tone changes
/// 3. Neutral tone assignment
/// 4. Third tone sandhi (consecutive tone 3)
pub struct ToneSandhi {
    /// Words that MUST have neutral tone on last syllable
    must_neural_tone_words: HashSet<&'static str>,
    /// Words that must NOT have neutral tone
    must_not_neural_tone_words: HashSet<&'static str>,
    /// Punctuation characters
    punc: &'static str,
}

impl Default for ToneSandhi {
    fn default() -> Self {
        Self::new()
    }
}

impl ToneSandhi {
    pub fn new() -> Self {
        Self {
            must_neural_tone_words: Self::init_must_neural_tone_words(),
            must_not_neural_tone_words: Self::init_must_not_neural_tone_words(),
            punc: "：，；。？！\u{201C}\u{201D}\u{2018}\u{2019}':,;.?!",
        }
    }

    fn init_must_neural_tone_words() -> HashSet<&'static str> {
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
            "蜡烛", "姥爷", "照顾", "喉咙", "吉他", "弄堂", "蚂蚱", "凤凰", "拖沓", "寒碜",
            "糟蹋", "倒腾", "报复", "逻辑", "盘缠", "喽啰", "牢骚", "咖喱", "扫把", "惦记",
        ].into_iter().collect()
    }

    fn init_must_not_neural_tone_words() -> HashSet<&'static str> {
        [
            "男子", "女子", "分子", "原子", "量子", "莲子", "石子", "瓜子", "电子", "人人",
            "虎虎", "幺幺", "干嘛", "学子", "哈哈", "数数", "袅袅", "局地", "以下", "娃哈哈",
            "花花草草", "留得", "耕地", "想想", "熙熙", "攘攘", "卵子", "死死", "冉冉", "恳恳",
            "佼佼", "吵吵", "打打", "考考", "整整", "莘莘", "落地", "算子", "家家户户", "青青",
        ].into_iter().collect()
    }

    /// Apply all tone sandhi rules to a word
    ///
    /// # Arguments
    /// * `word` - The Chinese word
    /// * `pos` - Part-of-speech tag from jieba
    /// * `finals` - Mutable slice of finals with tone numbers (e.g., "a1", "ang3")
    pub fn modified_tone(&self, word: &str, pos: &str, finals: &mut [String]) {
        self.bu_sandhi(word, finals);
        self.yi_sandhi(word, finals);
        self.neural_sandhi(word, pos, finals);
        self.three_sandhi(word, finals);
    }

    /// 不 (bù) tone sandhi
    /// - 不 before tone 4 → bu2 (e.g., 不怕 bú pà)
    /// - 不 in X不X pattern → bu5 (e.g., 看不懂 kàn bu dǒng)
    fn bu_sandhi(&self, word: &str, finals: &mut [String]) {
        let chars: Vec<char> = word.chars().collect();

        // Pattern: X不X (e.g., 看不懂)
        if chars.len() == 3 && chars[1] == '不' {
            if let Some(f) = finals.get_mut(1) {
                Self::set_tone(f, '5');
            }
        } else {
            // 不 before tone 4 → bu2
            for (i, &c) in chars.iter().enumerate() {
                if c == '不' && i + 1 < chars.len() {
                    if let Some(next_final) = finals.get(i + 1) {
                        if next_final.ends_with('4') {
                            if let Some(f) = finals.get_mut(i) {
                                Self::set_tone(f, '2');
                            }
                        }
                    }
                }
            }
        }
    }

    /// 一 (yī) tone sandhi
    /// - 一 in number sequence (一零零) → yi1 (no change)
    /// - 一 in reduplication X一X (看一看) → yi5
    /// - 一 in ordinal 第一 → yi1 (no change)
    /// - 一 before tone 4 → yi2
    /// - 一 before non-tone-4 → yi4
    fn yi_sandhi(&self, word: &str, finals: &mut [String]) {
        let chars: Vec<char> = word.chars().collect();

        // Check if all chars (except 一) are numeric
        let yi_idx = word.find('一');
        if yi_idx.is_none() {
            return;
        }

        // Skip sandhi for pure digit sequences (e.g., 一零零, 二一零)
        // But apply sandhi for numbers with units like 一百, 一千
        let all_digits = chars.iter().all(|&c| Self::is_chinese_digit(c));
        let has_unit = chars.iter().any(|&c| Self::is_chinese_unit(c));

        if all_digits && !has_unit {
            return; // Keep yi1 for pure digit sequences
        }

        for (i, &c) in chars.iter().enumerate() {
            if c != '一' {
                continue;
            }

            // X一X reduplication pattern (e.g., 看一看)
            if i > 0 && i + 1 < chars.len() && chars[i - 1] == chars[i + 1] {
                if let Some(f) = finals.get_mut(i) {
                    Self::set_tone(f, '5');
                }
                continue;
            }

            // Ordinal: 第一 - keep yi1
            if i > 0 && chars[i - 1] == '第' {
                continue;
            }

            // Standard sandhi rules
            if i + 1 < chars.len() {
                let next_char = chars[i + 1];

                // Skip if next is punctuation
                if self.punc.contains(next_char) {
                    continue;
                }

                // Check next final's tone and determine new tone
                let new_tone = if let Some(next_final) = finals.get(i + 1) {
                    if next_final.ends_with('4') {
                        Some('2') // Before tone 4 → yi2
                    } else if next_final.ends_with('1')
                           || next_final.ends_with('2')
                           || next_final.ends_with('3')
                           || next_final.ends_with('5') {
                        Some('4') // Before non-tone-4 → yi4
                    } else {
                        None
                    }
                } else {
                    None
                };

                if let Some(tone) = new_tone {
                    if let Some(f) = finals.get_mut(i) {
                        Self::set_tone(f, tone);
                    }
                }
            }
        }
    }

    /// Neural (neutral) tone sandhi
    /// Applies neutral tone (tone 5) to various grammatical patterns
    fn neural_sandhi(&self, word: &str, pos: &str, finals: &mut [String]) {
        let chars: Vec<char> = word.chars().collect();
        let word_len = chars.len();

        if word_len == 0 || finals.is_empty() {
            return;
        }

        // 1. Reduplication words (奶奶, 试试, 旺旺) - n, v, a POS
        for j in 1..word_len {
            if chars[j] == chars[j - 1]
                && (pos.starts_with('n') || pos.starts_with('v') || pos.starts_with('a'))
                && !self.must_not_neural_tone_words.contains(word) {
                if let Some(f) = finals.get_mut(j) {
                    Self::set_tone(f, '5');
                }
            }
        }

        // 2. Sentence-final particles: 吧呢哈啊呐噻嘛吖嗨呐哦哒额滴哩哟喽啰耶喔诶
        let particles = "吧呢哈啊呐噻嘛吖嗨呐哦哒额滴哩哟喽啰耶喔诶";
        if word_len >= 1 && particles.contains(chars[word_len - 1]) {
            if let Some(f) = finals.last_mut() {
                Self::set_tone(f, '5');
            }
            return;
        }

        // 3. 的地得 → neutral
        if word_len >= 1 && "的地得".contains(chars[word_len - 1]) {
            if let Some(f) = finals.last_mut() {
                Self::set_tone(f, '5');
            }
            return;
        }

        // 4. Aspect markers: 了着过 with specific POS tags
        if word_len == 1 && "了着过".contains(chars[0]) {
            if pos == "ul" || pos == "uz" || pos == "ug" {
                if let Some(f) = finals.first_mut() {
                    Self::set_tone(f, '5');
                }
                return;
            }
        }

        // 5. 们子 suffix with r, n POS
        if word_len > 1 && "们子".contains(chars[word_len - 1]) {
            if (pos == "r" || pos == "n") && !self.must_not_neural_tone_words.contains(word) {
                if let Some(f) = finals.last_mut() {
                    Self::set_tone(f, '5');
                }
                return;
            }
        }

        // 6. Location suffixes: 上下里 with s, l, f POS
        if word_len > 1 && "上下里".contains(chars[word_len - 1]) {
            if pos == "s" || pos == "l" || pos == "f" {
                if let Some(f) = finals.last_mut() {
                    Self::set_tone(f, '5');
                }
                return;
            }
        }

        // 7. Directional complements: 上来/下去/进出/回过/起开 + 来去
        if word_len > 1 && "来去".contains(chars[word_len - 1]) {
            if "上下进出回过起开".contains(chars[word_len - 2]) {
                if let Some(f) = finals.last_mut() {
                    Self::set_tone(f, '5');
                }
                return;
            }
        }

        // 8. 个 as measure word after numbers
        if let Some(ge_idx) = chars.iter().position(|&c| c == '个') {
            let should_neutralize = if ge_idx >= 1 {
                let prev_char = chars[ge_idx - 1];
                prev_char.is_ascii_digit()
                    || Self::is_chinese_digit(prev_char)
                    || Self::is_chinese_unit(prev_char)
                    || "几有两半多各整每做是".contains(prev_char)
            } else {
                word == "个"
            };

            if should_neutralize {
                if let Some(f) = finals.get_mut(ge_idx) {
                    Self::set_tone(f, '5');
                }
                return;
            }
        }

        // 9. Must-have neutral tone words (dictionary lookup)
        if self.must_neural_tone_words.contains(word) {
            if let Some(f) = finals.last_mut() {
                Self::set_tone(f, '5');
            }
            return;
        }

        // Also check last 2 characters
        if word_len >= 2 {
            let last_two: String = chars[word_len - 2..].iter().collect();
            if self.must_neural_tone_words.contains(last_two.as_str()) {
                if let Some(f) = finals.last_mut() {
                    Self::set_tone(f, '5');
                }
            }
        }

        // 10. Sub-word analysis - split word and apply rules to each part
        // This is critical for compound words
        if word_len >= 2 {
            let (first_len, _) = self.split_word(word);
            if first_len > 0 && first_len < word_len {
                let first_word: String = chars[..first_len].iter().collect();
                let second_word: String = chars[first_len..].iter().collect();

                // Check if subwords need neutral tone
                // First subword
                if self.must_neural_tone_words.contains(first_word.as_str())
                    || (first_word.len() >= 2 && {
                        let last_two: String = first_word.chars().rev().take(2).collect::<String>().chars().rev().collect();
                        self.must_neural_tone_words.contains(last_two.as_str())
                    })
                {
                    if first_len > 0 {
                        if let Some(f) = finals.get_mut(first_len - 1) {
                            Self::set_tone(f, '5');
                        }
                    }
                }

                // Second subword
                if self.must_neural_tone_words.contains(second_word.as_str())
                    || (second_word.len() >= 2 && {
                        let last_two: String = second_word.chars().rev().take(2).collect::<String>().chars().rev().collect();
                        self.must_neural_tone_words.contains(last_two.as_str())
                    })
                {
                    if let Some(f) = finals.last_mut() {
                        Self::set_tone(f, '5');
                    }
                }
            }
        }
    }

    /// Third tone sandhi
    /// When multiple tone-3 syllables occur consecutively, all but the last change to tone 2
    fn three_sandhi(&self, word: &str, finals: &mut [String]) {
        let chars: Vec<char> = word.chars().collect();
        let word_len = chars.len();

        if word_len == 2 && self.all_tone_three(finals) {
            // Simple case: two tone-3 → first becomes tone-2
            if let Some(f) = finals.first_mut() {
                Self::set_tone(f, '2');
            }
        } else if word_len == 3 {
            // Three syllables - split into subwords
            let (first_len, _) = self.split_word(word);

            if self.all_tone_three(finals) {
                if first_len == 2 {
                    // disyllabic + monosyllabic (e.g., 蒙古/包)
                    if let Some(f) = finals.get_mut(0) {
                        Self::set_tone(f, '2');
                    }
                    if let Some(f) = finals.get_mut(1) {
                        Self::set_tone(f, '2');
                    }
                } else if first_len == 1 {
                    // monosyllabic + disyllabic (e.g., 纸/老虎)
                    if let Some(f) = finals.get_mut(1) {
                        Self::set_tone(f, '2');
                    }
                }
            } else {
                // Not all tone 3 - check subgroups
                // Compute conditions first to avoid borrow conflicts
                let should_modify_first = {
                    let finals_first = &finals[..first_len];
                    finals_first.len() == 2 && self.all_tone_three(finals_first)
                };

                let should_cross_modify = {
                    let finals_first = &finals[..first_len];
                    let finals_second = &finals[first_len..];
                    if !finals_first.is_empty() && !finals_second.is_empty() {
                        let last_first = finals_first.last().unwrap();
                        let first_second = finals_second.first().unwrap();
                        last_first.ends_with('3') && first_second.ends_with('3')
                    } else {
                        false
                    }
                };

                // Now do the mutations
                if should_modify_first {
                    if let Some(f) = finals.get_mut(0) {
                        Self::set_tone(f, '2');
                    }
                }

                if should_cross_modify {
                    if let Some(f) = finals.get_mut(first_len - 1) {
                        Self::set_tone(f, '2');
                    }
                }
            }
        } else if word_len == 4 {
            // Four syllables - split into 2+2 (idiom pattern)
            // Compute conditions first
            let modify_first_half = self.all_tone_three(&finals[..2]);
            let modify_second_half = self.all_tone_three(&finals[2..]);

            if modify_first_half {
                if let Some(f) = finals.get_mut(0) {
                    Self::set_tone(f, '2');
                }
            }
            if modify_second_half {
                if let Some(f) = finals.get_mut(2) {
                    Self::set_tone(f, '2');
                }
            }
        }
    }

    /// Check if all finals are tone 3
    fn all_tone_three(&self, finals: &[String]) -> bool {
        finals.iter().all(|f| f.ends_with('3'))
    }

    /// Split word into two subwords
    /// Uses jieba segmentation when available, falls back to heuristics
    /// Returns (first_subword_length, second_subword_length)
    #[cfg(feature = "jieba")]
    fn split_word(&self, word: &str) -> (usize, usize) {
        use super::jieba_seg::GLOBAL_SEGMENTER;

        let chars: Vec<char> = word.chars().collect();
        let len = chars.len();

        if len <= 1 {
            return (len, 0);
        }

        // Use jieba cut_for_search to find subwords
        let subwords: Vec<String> = GLOBAL_SEGMENTER.cut_for_search(word);

        if subwords.is_empty() {
            return (len / 2, len - len / 2);
        }

        // Sort by length (shortest first) like Python does
        let mut sorted_subwords = subwords.clone();
        sorted_subwords.sort_by_key(|s| s.chars().count());

        let first_subword = &sorted_subwords[0];
        let first_len = first_subword.chars().count();

        // Find position of first subword in original word
        if let Some(pos) = word.find(first_subword.as_str()) {
            if pos == 0 {
                // First subword is at the beginning
                let second_len = len - first_len;
                (first_len, second_len)
            } else {
                // First subword is at the end
                let second_len = first_len;
                let first_len = len - second_len;
                (first_len, second_len)
            }
        } else {
            // Fallback to simple split
            (len / 2, len - len / 2)
        }
    }

    /// Split word into two subwords (fallback without jieba)
    /// Returns (first_subword_length, second_subword_length)
    #[cfg(not(feature = "jieba"))]
    fn split_word(&self, word: &str) -> (usize, usize) {
        let chars: Vec<char> = word.chars().collect();
        let len = chars.len();

        // Simple heuristic: prefer 2+1 or 1+2 splits
        if len == 3 {
            // Default to 2+1 split (most common in Chinese)
            (2, 1)
        } else if len == 4 {
            (2, 2)
        } else {
            (len / 2, len - len / 2)
        }
    }

    /// Check if character is a Chinese numeric character
    /// Check if character is a basic Chinese digit (not units like 百/千/万)
    /// Only returns true for single digits that form sequences like 一二三
    fn is_chinese_digit(c: char) -> bool {
        "零一二三四五六七八九十两".contains(c)
    }

    /// Check if character is a Chinese numeric unit (百/千/万/亿)
    fn is_chinese_unit(c: char) -> bool {
        "百千万亿".contains(c)
    }

    /// Set the tone of a final
    fn set_tone(final_str: &mut String, tone: char) {
        if final_str.chars().last().map(|c| c.is_ascii_digit()).unwrap_or(false) {
            final_str.pop();
        }
        final_str.push(tone);
    }
}

// ============================================================================
// Word Pre-Merging for Tone Sandhi
// ============================================================================

/// Get pinyin finals with tone for a word
/// Uses the pinyin crate to get finals like ["a1", "o3", "e4"]
fn get_word_finals(word: &str) -> Vec<String> {
    use pinyin::ToPinyin;

    word.chars()
        .filter_map(|c| {
            c.to_pinyin().map(|p| {
                // Get the pinyin with tone number
                let py = p.with_tone_num_end();
                // Extract the final (vowel part + tone)
                // For "ni3" -> "i3", "hao3" -> "ao3"
                let chars: Vec<char> = py.chars().collect();
                if chars.is_empty() {
                    return "5".to_string(); // neutral tone fallback
                }

                // Find where the final starts (first vowel)
                let vowels = ['a', 'e', 'i', 'o', 'u', 'ü', 'v'];
                if let Some(vowel_pos) = chars.iter().position(|c| vowels.contains(&c.to_ascii_lowercase())) {
                    chars[vowel_pos..].iter().collect()
                } else {
                    // No vowel found, return whole pinyin
                    py.to_string()
                }
            })
        })
        .collect()
}

/// Check if all finals in a list are tone 3
fn all_finals_tone_three(finals: &[String]) -> bool {
    !finals.is_empty() && finals.iter().all(|f| f.ends_with('3'))
}

/// Check if word is a reduplication (AA pattern like 妈妈)
fn is_reduplication(word: &str) -> bool {
    let chars: Vec<char> = word.chars().collect();
    chars.len() == 2 && chars[0] == chars[1]
}

/// Segment with word and POS tag
#[derive(Debug, Clone)]
pub struct WordSegment {
    pub word: String,
    pub pos: String,
}

impl WordSegment {
    pub fn new(word: &str, pos: &str) -> Self {
        Self {
            word: word.to_string(),
            pos: pos.to_string(),
        }
    }
}

/// Pre-merge segments for correct tone sandhi application
/// Port of Python's pre_merge_for_modify()
pub fn pre_merge_for_modify(segments: Vec<WordSegment>) -> Vec<WordSegment> {
    let segments = merge_bu(segments);
    let segments = merge_yi(segments);
    let segments = merge_reduplication(segments);
    // Add continuous three-tone merges (critical for correct sandhi)
    let segments = merge_continuous_three_tones(segments);
    let segments = merge_continuous_three_tones_2(segments);
    let segments = merge_er(segments);
    segments
}

/// Merge consecutive words when both are all tone-3
/// E.g., "老" + "虎" (both tone 3) → "老虎"
/// Only merges if combined length ≤ 3 and neither is reduplication
fn merge_continuous_three_tones(segments: Vec<WordSegment>) -> Vec<WordSegment> {
    if segments.is_empty() {
        return segments;
    }

    // Get finals for all segments
    let finals_list: Vec<Vec<String>> = segments
        .iter()
        .map(|seg| get_word_finals(&seg.word))
        .collect();

    let mut result: Vec<WordSegment> = Vec::new();
    let mut merge_last = vec![false; segments.len()];

    for (i, seg) in segments.iter().enumerate() {
        if i >= 1
            && all_finals_tone_three(&finals_list[i - 1])
            && all_finals_tone_three(&finals_list[i])
            && !merge_last[i - 1]
        {
            // Check if we should merge
            let prev_word = &segments[i - 1].word;
            let should_merge = !is_reduplication(prev_word)
                && prev_word.chars().count() + seg.word.chars().count() <= 3;

            if should_merge {
                // Merge with previous
                if let Some(last) = result.last_mut() {
                    last.word.push_str(&seg.word);
                }
                merge_last[i] = true;
            } else {
                result.push(seg.clone());
            }
        } else {
            result.push(seg.clone());
        }
    }

    result
}

/// Merge words when last char of first word and first char of second word are both tone-3
/// E.g., "纸" (zhi3) + "老虎" (lao3hu3) → "纸老虎" when boundary chars are both tone 3
fn merge_continuous_three_tones_2(segments: Vec<WordSegment>) -> Vec<WordSegment> {
    if segments.is_empty() {
        return segments;
    }

    // Get finals for all segments
    let finals_list: Vec<Vec<String>> = segments
        .iter()
        .map(|seg| get_word_finals(&seg.word))
        .collect();

    let mut result: Vec<WordSegment> = Vec::new();
    let mut merge_last = vec![false; segments.len()];

    for (i, seg) in segments.iter().enumerate() {
        if i >= 1
            && !finals_list[i - 1].is_empty()
            && !finals_list[i].is_empty()
            && !merge_last[i - 1]
        {
            // Check if last char of prev word and first char of current word are tone 3
            let prev_last_tone3 = finals_list[i - 1]
                .last()
                .map(|f| f.ends_with('3'))
                .unwrap_or(false);
            let curr_first_tone3 = finals_list[i]
                .first()
                .map(|f| f.ends_with('3'))
                .unwrap_or(false);

            if prev_last_tone3 && curr_first_tone3 {
                let prev_word = &segments[i - 1].word;
                let should_merge = !is_reduplication(prev_word)
                    && prev_word.chars().count() + seg.word.chars().count() <= 3;

                if should_merge {
                    // Merge with previous
                    if let Some(last) = result.last_mut() {
                        last.word.push_str(&seg.word);
                    }
                    merge_last[i] = true;
                } else {
                    result.push(seg.clone());
                }
            } else {
                result.push(seg.clone());
            }
        } else {
            result.push(seg.clone());
        }
    }

    result
}

/// Merge 不 with the following word
fn merge_bu(segments: Vec<WordSegment>) -> Vec<WordSegment> {
    let mut result = Vec::new();
    let mut last_word = String::new();

    for seg in segments {
        if last_word == "不" {
            // Merge 不 with current word
            result.push(WordSegment::new(&format!("不{}", seg.word), &seg.pos));
            last_word.clear();
        } else if seg.word == "不" {
            last_word = "不".to_string();
        } else {
            result.push(seg);
            last_word.clear();
        }
    }

    // Handle trailing 不
    if last_word == "不" {
        result.push(WordSegment::new("不", "d"));
    }

    result
}

/// Merge 一 patterns:
/// 1. X一X reduplication (听一听)
/// 2. 一 + following word
fn merge_yi(segments: Vec<WordSegment>) -> Vec<WordSegment> {
    let mut result: Vec<WordSegment> = Vec::new();
    let seg_vec: Vec<_> = segments.into_iter().collect();
    let len = seg_vec.len();
    let mut skip_next = false;

    for i in 0..len {
        if skip_next {
            skip_next = false;
            continue;
        }

        let seg = &seg_vec[i];

        // Check X一X pattern
        if i >= 1 && i + 1 < len
            && seg.word == "一"
            && seg_vec[i - 1].word == seg_vec[i + 1].word
            && seg_vec[i - 1].pos == "v"
            && seg_vec[i + 1].pos == "v"
        {
            // Merge into previous: X一X
            if let Some(last) = result.last_mut() {
                last.word = format!("{}一{}", last.word, seg_vec[i - 1].word);
            }
            skip_next = true;
            continue;
        }

        // Check if this is X and previous was 一 in X一X (already handled above)
        if i >= 2
            && seg_vec[i - 1].word == "一"
            && seg_vec[i - 2].word == seg.word
            && seg.pos == "v"
            && seg_vec[i - 2].pos == "v"
        {
            continue; // Skip - already merged
        }

        result.push(seg.clone());
    }

    // Second pass: merge standalone 一 with following word
    // But DON'T merge if 一 is part of a pure number sequence (like 二零一一年)
    let mut final_result = Vec::new();
    let mut i = 0;
    while i < result.len() {
        if result[i].word == "一" && i + 1 < result.len() {
            // Check if this 一 is in a pure numeric context
            // If previous word is numeric digit (零一二三四五六七八九) AND
            // next word starts with numeric, don't merge
            let prev_is_numeric = if i > 0 {
                let prev = &result[i - 1].word;
                prev.chars().all(|c| "零一二三四五六七八九十两".contains(c))
            } else {
                false
            };
            let next_starts_numeric = result[i + 1].word.chars().next()
                .map(|c| "零一二三四五六七八九十".contains(c))
                .unwrap_or(false);

            if prev_is_numeric && next_starts_numeric {
                // In a pure digit sequence like 二零一一 - don't merge, keep 一 separate
                final_result.push(result[i].clone());
                i += 1;
            } else {
                // Merge 一 with next word (standard yi sandhi case)
                final_result.push(WordSegment::new(
                    &format!("一{}", result[i + 1].word),
                    &result[i + 1].pos,
                ));
                i += 2;
            }
        } else {
            final_result.push(result[i].clone());
            i += 1;
        }
    }

    final_result
}

/// Merge reduplication words (AA pattern)
fn merge_reduplication(segments: Vec<WordSegment>) -> Vec<WordSegment> {
    let mut result: Vec<WordSegment> = Vec::new();

    for seg in segments {
        if !result.is_empty() {
            let last = result.last().unwrap();
            if seg.word == last.word {
                // Merge: AA
                if let Some(last_mut) = result.last_mut() {
                    last_mut.word = format!("{}{}", last_mut.word, seg.word);
                }
                continue;
            }
        }
        result.push(seg);
    }

    result
}

/// Merge 儿 with previous word
fn merge_er(segments: Vec<WordSegment>) -> Vec<WordSegment> {
    let mut result: Vec<WordSegment> = Vec::new();

    for seg in segments {
        if seg.word == "儿" && !result.is_empty() {
            let last = result.last().unwrap();
            if last.word != "#" {
                if let Some(last_mut) = result.last_mut() {
                    last_mut.word.push('儿');
                }
                continue;
            }
        }
        result.push(seg);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bu_sandhi_before_tone4() {
        let sandhi = ToneSandhi::new();
        let mut finals = vec!["u4".to_string(), "a4".to_string()];
        sandhi.bu_sandhi("不怕", &mut finals);
        assert_eq!(finals[0], "u2"); // 不 before tone 4 → bu2
    }

    #[test]
    fn test_bu_sandhi_pattern() {
        let sandhi = ToneSandhi::new();
        let mut finals = vec!["an4".to_string(), "u4".to_string(), "ong3".to_string()];
        sandhi.bu_sandhi("看不懂", &mut finals);
        assert_eq!(finals[1], "u5"); // 不 in X不X → bu5
    }

    #[test]
    fn test_yi_sandhi_before_tone4() {
        let sandhi = ToneSandhi::new();
        let mut finals = vec!["i1".to_string(), "ang4".to_string()];
        sandhi.yi_sandhi("一样", &mut finals);
        assert_eq!(finals[0], "i2"); // 一 before tone 4 → yi2
    }

    #[test]
    fn test_yi_sandhi_before_tone3() {
        let sandhi = ToneSandhi::new();
        let mut finals = vec!["i1".to_string(), "ai3".to_string()];
        sandhi.yi_sandhi("一百", &mut finals);
        assert_eq!(finals[0], "i4"); // 一 before tone 3 → yi4
    }

    #[test]
    fn test_yi_sandhi_reduplication() {
        let sandhi = ToneSandhi::new();
        let mut finals = vec!["an4".to_string(), "i1".to_string(), "an4".to_string()];
        sandhi.yi_sandhi("看一看", &mut finals);
        assert_eq!(finals[1], "i5"); // 一 in X一X → yi5
    }

    #[test]
    fn test_neural_sandhi_reduplication() {
        let sandhi = ToneSandhi::new();
        let mut finals = vec!["ai3".to_string(), "ai3".to_string()];
        sandhi.neural_sandhi("奶奶", "n", &mut finals);
        assert_eq!(finals[1], "ai5"); // Second syllable → neutral
    }

    #[test]
    fn test_neural_sandhi_particle() {
        let sandhi = ToneSandhi::new();
        let mut finals = vec!["ao3".to_string(), "a5".to_string()];
        sandhi.neural_sandhi("好吧", "y", &mut finals);
        assert_eq!(finals[1], "a5"); // Particle already neutral
    }

    #[test]
    fn test_neural_sandhi_de() {
        let sandhi = ToneSandhi::new();
        let mut finals = vec!["o3".to_string(), "e4".to_string()];
        sandhi.neural_sandhi("我的", "u", &mut finals);
        assert_eq!(finals[1], "e5"); // 的 → neutral
    }

    #[test]
    fn test_three_sandhi_two_syllables() {
        let sandhi = ToneSandhi::new();
        let mut finals = vec!["i3".to_string(), "ao3".to_string()];
        sandhi.three_sandhi("你好", &mut finals);
        assert_eq!(finals[0], "i2"); // First tone 3 → tone 2
        assert_eq!(finals[1], "ao3"); // Last keeps tone 3
    }

    #[test]
    fn test_must_neural_word() {
        let sandhi = ToneSandhi::new();
        let mut finals = vec!["eng2".to_string(), "you3".to_string()];
        sandhi.neural_sandhi("朋友", "n", &mut finals);
        assert_eq!(finals[1], "you5"); // Dictionary word → neutral on last
    }

    #[test]
    fn test_get_word_finals() {
        let finals = get_word_finals("你好");
        assert_eq!(finals.len(), 2);
        // 你 = ni3 → i3, 好 = hao3 → ao3
        assert!(finals[0].ends_with('3'));
        assert!(finals[1].ends_with('3'));
    }

    #[test]
    fn test_is_reduplication() {
        assert!(is_reduplication("妈妈"));
        assert!(is_reduplication("爸爸"));
        assert!(!is_reduplication("你好"));
        assert!(!is_reduplication("我"));
    }

    #[test]
    fn test_all_finals_tone_three() {
        let finals = vec!["i3".to_string(), "ao3".to_string()];
        assert!(all_finals_tone_three(&finals));

        let mixed = vec!["i3".to_string(), "ao4".to_string()];
        assert!(!all_finals_tone_three(&mixed));
    }

    #[test]
    fn test_merge_continuous_three_tones() {
        // Two tone-3 words that should merge
        let segments = vec![
            WordSegment::new("老", "a"),  // lao3
            WordSegment::new("虎", "n"),  // hu3
        ];
        let merged = merge_continuous_three_tones(segments);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].word, "老虎");
    }

    #[test]
    fn test_merge_continuous_three_tones_no_merge_reduplication() {
        // Reduplication should not merge
        let segments = vec![
            WordSegment::new("奶", "n"),
            WordSegment::new("奶", "n"),  // This becomes 奶奶 via merge_reduplication
        ];
        // After reduplication merge, we have 奶奶 which is reduplication
        let merged_redup = merge_reduplication(segments);
        assert_eq!(merged_redup.len(), 1);
        assert_eq!(merged_redup[0].word, "奶奶");

        // Now if we have 奶奶 + another tone-3 word, it shouldn't merge
        // because 奶奶 is reduplication
        let segments2 = vec![
            WordSegment::new("奶奶", "n"),
            WordSegment::new("好", "a"),
        ];
        let merged2 = merge_continuous_three_tones(segments2);
        // Should not merge because prev word is reduplication
        assert_eq!(merged2.len(), 2);
    }

    #[test]
    fn test_merge_continuous_three_tones_2() {
        // Words where boundary chars are tone-3
        let segments = vec![
            WordSegment::new("纸", "n"),   // zhi3
            WordSegment::new("老", "a"),   // lao3
        ];
        let merged = merge_continuous_three_tones_2(segments);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].word, "纸老");
    }

    #[test]
    fn test_pre_merge_includes_three_tone_merges() {
        // Test that pre_merge_for_modify calls the three-tone merge functions
        let segments = vec![
            WordSegment::new("老", "a"),
            WordSegment::new("虎", "n"),
        ];
        let merged = pre_merge_for_modify(segments);
        // Should merge because both are tone-3
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].word, "老虎");
    }
}
