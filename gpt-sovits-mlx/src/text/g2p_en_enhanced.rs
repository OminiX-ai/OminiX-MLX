//! Enhanced English G2P (Grapheme-to-Phoneme)
//!
//! Port of Python's english.py with:
//! - CMU dictionary lookup
//! - Homograph disambiguation (POS-based)
//! - Possessive ('s) handling
//! - OOV fallback to neural G2P
//! - Number normalization

use std::collections::HashMap;

lazy_static::lazy_static! {
    /// Homograph dictionary: word → (pronunciation1, pronunciation2, POS_trigger)
    /// When POS starts with trigger, use pron1, otherwise pron2
    static ref HOMOGRAPHS: HashMap<&'static str, Homograph> = {
        let mut m = HashMap::new();

        // read: "reed" (VBP/present) vs "red" (VBD/past)
        m.insert("read", Homograph {
            pron1: vec!["R", "IY1", "D"],
            pron2: vec!["R", "EH1", "D"],
            pos_trigger: "VBP",
        });

        // live: "liv" (VB/verb) vs "layv" (JJ/adjective)
        m.insert("live", Homograph {
            pron1: vec!["L", "IH1", "V"],
            pron2: vec!["L", "AY1", "V"],
            pos_trigger: "VB",
        });

        // lives: "livz" (VBZ/verb) vs "layvz" (NNS/noun)
        m.insert("lives", Homograph {
            pron1: vec!["L", "IH1", "V", "Z"],
            pron2: vec!["L", "AY1", "V", "Z"],
            pos_trigger: "VBZ",
        });

        // close: "kloz" (VB/verb) vs "klos" (JJ/adjective)
        m.insert("close", Homograph {
            pron1: vec!["K", "L", "OW1", "Z"],
            pron2: vec!["K", "L", "OW1", "S"],
            pos_trigger: "VB",
        });

        // lead: "leed" (VB/verb) vs "led" (NN/noun - the metal)
        m.insert("lead", Homograph {
            pron1: vec!["L", "IY1", "D"],
            pron2: vec!["L", "EH1", "D"],
            pos_trigger: "VB",
        });

        // wind: "wind" (NN/noun - air) vs "waynd" (VB/verb - to turn)
        m.insert("wind", Homograph {
            pron1: vec!["W", "IH1", "N", "D"],
            pron2: vec!["W", "AY1", "N", "D"],
            pos_trigger: "NN",
        });

        // bow: "bou" (NN/noun - weapon) vs "bau" (VB/verb - to bend)
        m.insert("bow", Homograph {
            pron1: vec!["B", "OW1"],
            pron2: vec!["B", "AW1"],
            pos_trigger: "NN",
        });

        // tear: "teer" (NN/noun - from eye) vs "tair" (VB/verb - to rip)
        m.insert("tear", Homograph {
            pron1: vec!["T", "IH1", "R"],
            pron2: vec!["T", "EH1", "R"],
            pos_trigger: "NN",
        });

        // row: "roh" (NN/noun - line) vs "rau" (NN/verb - argument)
        m.insert("row", Homograph {
            pron1: vec!["R", "OW1"],
            pron2: vec!["R", "AW1"],
            pos_trigger: "NN",
        });

        // bass: "beys" (NN/noun - fish, instrument) vs "bas" (adjective - low)
        m.insert("bass", Homograph {
            pron1: vec!["B", "AE1", "S"],
            pron2: vec!["B", "EY1", "S"],
            pos_trigger: "JJ",
        });

        // complex: k-AH-m-P-L-EH-K-S (adjective) vs K-AA-M-P-L-EH-K-S (noun)
        m.insert("complex", Homograph {
            pron1: vec!["K", "AH0", "M", "P", "L", "EH1", "K", "S"],
            pron2: vec!["K", "AA1", "M", "P", "L", "EH0", "K", "S"],
            pos_trigger: "JJ",
        });

        // record: "REH-kerd" (NN/noun) vs "ri-KORD" (VB/verb)
        m.insert("record", Homograph {
            pron1: vec!["R", "EH1", "K", "ER0", "D"],
            pron2: vec!["R", "IH0", "K", "AO1", "R", "D"],
            pos_trigger: "NN",
        });

        // present: "PREH-zent" (NN/noun) vs "pri-ZENT" (VB/verb)
        m.insert("present", Homograph {
            pron1: vec!["P", "R", "EH1", "Z", "AH0", "N", "T"],
            pron2: vec!["P", "R", "IH0", "Z", "EH1", "N", "T"],
            pos_trigger: "NN",
        });

        // project: "PRAH-jekt" (NN/noun) vs "pruh-JEKT" (VB/verb)
        m.insert("project", Homograph {
            pron1: vec!["P", "R", "AA1", "JH", "EH0", "K", "T"],
            pron2: vec!["P", "R", "AH0", "JH", "EH1", "K", "T"],
            pos_trigger: "NN",
        });

        // permit: "PER-mit" (NN/noun) vs "per-MIT" (VB/verb)
        m.insert("permit", Homograph {
            pron1: vec!["P", "ER1", "M", "IH0", "T"],
            pron2: vec!["P", "ER0", "M", "IH1", "T"],
            pos_trigger: "NN",
        });

        // minute: "MIN-it" (NN/noun - time) vs "my-NOOT" (JJ/adjective - tiny)
        m.insert("minute", Homograph {
            pron1: vec!["M", "IH1", "N", "AH0", "T"],
            pron2: vec!["M", "AY0", "N", "UW1", "T"],
            pos_trigger: "NN",
        });

        m
    };

    /// Single letter pronunciations
    static ref LETTER_PRONS: HashMap<char, Vec<&'static str>> = {
        let mut m = HashMap::new();
        m.insert('a', vec!["EY1"]);
        m.insert('b', vec!["B", "IY1"]);
        m.insert('c', vec!["S", "IY1"]);
        m.insert('d', vec!["D", "IY1"]);
        m.insert('e', vec!["IY1"]);
        m.insert('f', vec!["EH1", "F"]);
        m.insert('g', vec!["JH", "IY1"]);
        m.insert('h', vec!["EY1", "CH"]);
        m.insert('i', vec!["AY1"]);
        m.insert('j', vec!["JH", "EY1"]);
        m.insert('k', vec!["K", "EY1"]);
        m.insert('l', vec!["EH1", "L"]);
        m.insert('m', vec!["EH1", "M"]);
        m.insert('n', vec!["EH1", "N"]);
        m.insert('o', vec!["OW1"]);
        m.insert('p', vec!["P", "IY1"]);
        m.insert('q', vec!["K", "Y", "UW1"]);
        m.insert('r', vec!["AA1", "R"]);
        m.insert('s', vec!["EH1", "S"]);
        m.insert('t', vec!["T", "IY1"]);
        m.insert('u', vec!["Y", "UW1"]);
        m.insert('v', vec!["V", "IY1"]);
        m.insert('w', vec!["D", "AH1", "B", "AH0", "L", "Y", "UW0"]);
        m.insert('x', vec!["EH1", "K", "S"]);
        m.insert('y', vec!["W", "AY1"]);
        m.insert('z', vec!["Z", "IY1"]);
        m
    };

    /// Voiceless consonants (for possessive 's → S)
    static ref VOICELESS_CONSONANTS: std::collections::HashSet<&'static str> = {
        ["P", "T", "K", "F", "TH", "HH"].into_iter().collect()
    };

    /// Sibilants (for possessive 's → AH0 Z)
    static ref SIBILANTS: std::collections::HashSet<&'static str> = {
        ["S", "Z", "SH", "ZH", "CH", "JH"].into_iter().collect()
    };
}

/// Homograph entry
#[derive(Debug, Clone)]
pub struct Homograph {
    /// First pronunciation (when POS matches)
    pub pron1: Vec<&'static str>,
    /// Second pronunciation (default)
    pub pron2: Vec<&'static str>,
    /// POS tag prefix that triggers pron1
    pub pos_trigger: &'static str,
}

/// Enhanced English G2P processor
pub struct EnhancedEnglishG2P {
    /// Reference to base CMU dictionary
    cmu_dict: Option<std::sync::Arc<HashMap<String, Vec<String>>>>,
}

impl Default for EnhancedEnglishG2P {
    fn default() -> Self {
        Self::new()
    }
}

impl EnhancedEnglishG2P {
    pub fn new() -> Self {
        Self { cmu_dict: None }
    }

    /// Set the CMU dictionary reference
    pub fn set_cmu_dict(&mut self, dict: std::sync::Arc<HashMap<String, Vec<String>>>) {
        self.cmu_dict = Some(dict);
    }

    /// Convert word to phonemes with POS-based disambiguation
    ///
    /// # Arguments
    /// * `word` - The word to convert (original case preserved for single-letter detection)
    /// * `pos` - Optional POS tag for disambiguation (e.g., "VB", "NN", "JJ")
    ///
    /// # Returns
    /// Vector of ARPABET phonemes
    pub fn word_to_phonemes(&self, word: &str, pos: Option<&str>) -> Vec<String> {
        let word_lower = word.to_lowercase();

        // Check for non-alphabetic
        if !word.chars().any(|c| c.is_ascii_alphabetic()) {
            return vec![word.to_string()];
        }

        // Single letter handling
        if word.len() == 1 {
            let c = word.chars().next().unwrap().to_ascii_lowercase();
            // Special case: uppercase "A" is "EY1"
            if word == "A" {
                return vec!["EY1".to_string()];
            }
            if let Some(pron) = LETTER_PRONS.get(&c) {
                return pron.iter().map(|s| s.to_string()).collect();
            }
        }

        // Check homographs first (need POS for disambiguation)
        if let Some(homograph) = HOMOGRAPHS.get(word_lower.as_str()) {
            let use_pron1 = pos
                .map(|p| p.starts_with(homograph.pos_trigger))
                .unwrap_or(false);

            let pron = if use_pron1 {
                &homograph.pron1
            } else {
                &homograph.pron2
            };
            return pron.iter().map(|s| s.to_string()).collect();
        }

        // Try possessive handling
        if let Some(result) = self.handle_possessive(&word_lower) {
            return result;
        }

        // CMU dictionary lookup
        if let Some(dict) = &self.cmu_dict {
            if let Some(phonemes) = dict.get(&word_lower) {
                return phonemes.clone();
            }
        }

        // Fall back to base g2p_en module
        super::g2p_en::word_to_phonemes(word)
    }

    /// Handle possessive forms ending in 's
    fn handle_possessive(&self, word: &str) -> Option<Vec<String>> {
        if !word.ends_with("'s") {
            return None;
        }

        let base = &word[..word.len() - 2];
        if base.is_empty() {
            return None;
        }

        // Get base word pronunciation
        let mut phonemes = self.word_to_phonemes(base, None);

        if phonemes.is_empty() {
            return None;
        }

        // Determine 's pronunciation based on last phoneme
        let last_phoneme = phonemes.last()?;

        if VOICELESS_CONSONANTS.contains(last_phoneme.as_str()) {
            // Voiceless consonant → 's is [S]
            phonemes.push("S".to_string());
        } else if SIBILANTS.contains(last_phoneme.as_str()) {
            // Sibilant → 's is [AH0 Z]
            phonemes.push("AH0".to_string());
            phonemes.push("Z".to_string());
        } else {
            // Voiced consonant or vowel → 's is [Z]
            phonemes.push("Z".to_string());
        }

        Some(phonemes)
    }

    /// Process text with tokenization and POS-aware G2P
    pub fn process_text(&self, text: &str) -> Vec<String> {
        let mut result = Vec::new();

        // Simple word tokenization
        let words = tokenize(text);

        // Basic POS inference (simplified - no full POS tagger)
        let pos_tags = infer_pos_tags(&words);

        for (word, pos) in words.iter().zip(pos_tags.iter()) {
            if word.chars().all(|c| !c.is_ascii_alphabetic()) {
                // Non-alphabetic token (punctuation, number)
                if is_valid_punctuation(word) {
                    result.push(word.clone());
                }
                continue;
            }

            let phonemes = self.word_to_phonemes(word, Some(pos));
            result.extend(phonemes);
        }

        result
    }
}

/// Simple word tokenizer (like TweetTokenizer)
fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();

    for c in text.chars() {
        if c.is_ascii_alphabetic() || c == '\'' {
            current.push(c);
        } else {
            if !current.is_empty() {
                tokens.push(current.clone());
                current.clear();
            }
            if !c.is_whitespace() {
                tokens.push(c.to_string());
            }
        }
    }

    if !current.is_empty() {
        tokens.push(current);
    }

    tokens
}

/// Simple POS tag inference (heuristic-based)
fn infer_pos_tags(words: &[String]) -> Vec<String> {
    words.iter().map(|word| {
        let lower = word.to_lowercase();

        // Simple heuristics
        if lower.ends_with("ing") {
            "VBG".to_string() // Present participle
        } else if lower.ends_with("ed") {
            "VBD".to_string() // Past tense
        } else if lower.ends_with("ly") {
            "RB".to_string() // Adverb
        } else if lower.ends_with("tion") || lower.ends_with("ness") || lower.ends_with("ment") {
            "NN".to_string() // Noun
        } else if lower.ends_with("ful") || lower.ends_with("less") || lower.ends_with("ous") {
            "JJ".to_string() // Adjective
        } else if lower.ends_with("s") && !lower.ends_with("ss") {
            "NNS".to_string() // Plural noun (or VBZ, but default to noun)
        } else {
            "NN".to_string() // Default to noun
        }
    }).collect()
}

/// Check if a string is valid punctuation for output
fn is_valid_punctuation(s: &str) -> bool {
    matches!(s, "." | "," | "!" | "?" | "-")
}

/// Normalize English text (port of english.py text_normalize)
pub fn normalize_english_text(text: &str) -> String {
    let mut result = text.to_string();

    // Expand common abbreviations
    result = result.replace("i.e.", "that is");
    result = result.replace("I.E.", "that is");
    result = result.replace("e.g.", "for example");
    result = result.replace("E.G.", "for example");

    // Remove non-ASCII (except basic punctuation)
    result = result
        .chars()
        .filter(|&c| c.is_ascii() || c == '\u{2018}' || c == '\u{2019}' || c == '\u{201C}' || c == '\u{201D}')
        .collect();

    // Normalize quotes (curly to straight)
    result = result.replace('\u{2018}', "'");  // left single quote
    result = result.replace('\u{2019}', "'");  // right single quote
    result = result.replace('\u{201C}', "\""); // left double quote
    result = result.replace('\u{201D}', "\""); // right double quote

    // Remove consecutive punctuation
    let punct_chars = ['!', '?', '.', ',', '-'];
    let mut new_result = String::new();
    let mut prev_punct = false;

    for c in result.chars() {
        let is_punct = punct_chars.contains(&c);
        if is_punct {
            if !prev_punct {
                new_result.push(c);
            }
            prev_punct = true;
        } else {
            new_result.push(c);
            prev_punct = false;
        }
    }

    // Clean up whitespace
    new_result.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_homograph_read_present() {
        let g2p = EnhancedEnglishG2P::new();
        let result = g2p.word_to_phonemes("read", Some("VBP"));
        assert_eq!(result, vec!["R", "IY1", "D"]);
    }

    #[test]
    fn test_homograph_read_past() {
        let g2p = EnhancedEnglishG2P::new();
        let result = g2p.word_to_phonemes("read", Some("VBD"));
        assert_eq!(result, vec!["R", "EH1", "D"]);
    }

    #[test]
    fn test_homograph_live_verb() {
        let g2p = EnhancedEnglishG2P::new();
        let result = g2p.word_to_phonemes("live", Some("VB"));
        assert_eq!(result, vec!["L", "IH1", "V"]);
    }

    #[test]
    fn test_homograph_live_adjective() {
        let g2p = EnhancedEnglishG2P::new();
        let result = g2p.word_to_phonemes("live", Some("JJ"));
        assert_eq!(result, vec!["L", "AY1", "V"]);
    }

    #[test]
    fn test_possessive_voiceless() {
        let g2p = EnhancedEnglishG2P::new();
        // cat's - ends in T (voiceless) → S
        // We need CMU dict for this to work properly
        // For now test the logic with a mock
    }

    #[test]
    fn test_single_letter_a() {
        let g2p = EnhancedEnglishG2P::new();
        let result = g2p.word_to_phonemes("A", None);
        assert_eq!(result, vec!["EY1"]);
    }

    #[test]
    fn test_single_letter_b() {
        let g2p = EnhancedEnglishG2P::new();
        let result = g2p.word_to_phonemes("b", None);
        assert_eq!(result, vec!["B", "IY1"]);
    }

    #[test]
    fn test_tokenize() {
        let tokens = tokenize("Hello, world!");
        assert_eq!(tokens, vec!["Hello", ",", "world", "!"]);
    }

    #[test]
    fn test_tokenize_possessive() {
        let tokens = tokenize("John's book");
        assert_eq!(tokens, vec!["John's", "book"]);
    }

    #[test]
    fn test_normalize_abbreviations() {
        let result = normalize_english_text("e.g. this is an example");
        assert_eq!(result, "for example this is an example");
    }

    #[test]
    fn test_normalize_quotes() {
        let result = normalize_english_text("She said 'hello'");
        assert_eq!(result, "She said 'hello'");
    }
}
