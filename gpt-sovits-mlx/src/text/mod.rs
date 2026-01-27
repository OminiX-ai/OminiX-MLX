//! Text processing for GPT-SoVITS
//!
//! This module provides text-to-phoneme conversion for TTS:
//! - Phoneme vocabulary and symbol mappings
//! - Text normalization (Chinese/English)
//! - Grapheme-to-phoneme conversion
//! - Language detection
//! - BERT feature extraction for TTS
//! - G2PW polyphonic character disambiguation
//!
//! ## New Modules (Python Parity)
//!
//! - `tone_sandhi`: Complete Mandarin tone sandhi rules (不/一/轻声/三声变调)
//! - `erhua`: 儿化 (r-coloring) handling
//! - `cn2an`: Chinese number conversion (Arabic → Chinese)
//! - `lang_segment`: Language detection and segmentation (zh/en/ja/ko)
//! - `g2p_en_enhanced`: Enhanced English G2P with homographs and possessives
//! - `text_normalizer`: Full text normalization (numbers, dates, units)
//! - `jieba_seg`: Word segmentation with POS tagging (via jieba-rs)

pub mod bert_features;
pub mod cmudict;
pub mod g2p_en;
pub mod g2pw;
pub mod preprocessor;
pub mod symbols;

// New modules for Python parity
pub mod tone_sandhi;
pub mod erhua;
pub mod cn2an;
pub mod lang_segment;
pub mod g2p_en_enhanced;
pub mod text_normalizer;
pub mod jieba_seg;

pub use bert_features::{BertFeatureExtractor, extract_bert_features};

pub use preprocessor::{
    Language, PreprocessorConfig, PreprocessorOutput, TextPreprocessor,
    detect_language, is_chinese_char, normalize_chinese, normalize_english,
    preprocess_text,
};

pub use symbols::{
    bos_id, eos_id, pad_id, sp_id, unk_id,
    has_symbol, id_to_symbol, ids_to_symbols, symbol_to_id, symbols_to_ids,
    vocab_size, all_symbols,
    PAD, UNK, BOS, EOS, SP,
};

// Re-export new modules
pub use tone_sandhi::{ToneSandhi, WordSegment, pre_merge_for_modify};
pub use erhua::{merge_erhua, apply_erhua_to_pinyin, has_potential_erhua};
pub use cn2an::{an2cn, digits_to_chinese, year_to_chinese, fraction_to_chinese, percentage_to_chinese};
pub use lang_segment::{Lang, LangText, LangSegment, is_japanese_char, is_korean_char};
pub use g2p_en_enhanced::{EnhancedEnglishG2P, Homograph, normalize_english_text};
