#![deny(unsafe_code)]

use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::io::{self, BufRead, Read, Write};

pub const SCHEMA: &str = "nsrl.corpus_trace.v1";
pub const TOKEN_SCHEMA: &str = "nsrl.token_trace.v1";
pub const LEXEME_TOKEN_SCHEMA: &str = "nsrl.lexeme_token_trace.v1";
pub const AUTHORITY: &str = "deterministic_corpus_preparation";
pub const TOKEN_AUTHORITY: &str = "deterministic_byte_tokenization";
pub const LEXEME_TOKEN_AUTHORITY: &str = "deterministic_lexeme_tokenization";
pub const TOKENIZER_ID: &str = "byte_identity_u8_v1";
pub const ASCII_LOWER_TOKENIZER_ID: &str = "byte_ascii_lower_text_u8_v1";
pub const LEXEME_TOKENIZER_ID: &str = "lexeme_ascii_lower_u16_v1";
pub const SHAKESPEARE_SOURCE_ID: &str = "gutenberg_ebook_100";
pub const SHAKESPEARE_SOURCE_URL: &str = "https://www.gutenberg.org/cache/epub/100/pg100.txt";
pub const SIMPLEWIKI_SOURCE_ID: &str = "simplewiki_latest_pages_articles";
pub const SIMPLEWIKI_SOURCE_URL: &str =
    "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2";

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

const KNOWN_NON_CLAIMS: [&str; 5] = [
    "not_tokenized_yet",
    "not_trained_yet",
    "simplewiki_xml_must_be_decompressed_before_input",
    "wiki_markup_cleaning_is_deliberately_minimal",
    "not_a_semantic_deduplication_pipeline",
];

const TOKEN_KNOWN_NON_CLAIMS: [&str; 4] = [
    "not_bpe_or_wordpiece",
    "not_trained_yet",
    "does_not_shuffle_windows",
    "does_not_materialize_window_rows",
];

const LEXEME_KNOWN_NON_CLAIMS: [&str; 5] = [
    "not_dynamic_vocabulary",
    "not_semantic_embeddings_yet",
    "not_bpe_or_wordpiece",
    "byte_fallback_reserved_for_unknown_lexemes",
    "does_not_materialize_window_rows",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CorpusConfig {
    pub max_simplewiki_pages: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenizeConfig {
    pub seq_len: usize,
    pub stride: usize,
    pub max_windows: Option<usize>,
    pub preview_tokens: usize,
    pub text_profile: TokenTextProfile,
}

impl Default for TokenizeConfig {
    fn default() -> Self {
        Self {
            seq_len: 128,
            stride: 1,
            max_windows: None,
            preview_tokens: 16,
            text_profile: TokenTextProfile::Identity,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenTextProfile {
    Identity,
    AsciiLower,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LexemeTokenizeConfig {
    pub seq_len: usize,
    pub stride: usize,
    pub max_windows: Option<usize>,
    pub preview_tokens: usize,
    pub max_vocab: usize,
    pub input_profile: LexemeInputProfile,
    pub vocab_profile: LexemeVocabProfile,
    pub vocab_frequency_cap: u32,
}

impl Default for LexemeTokenizeConfig {
    fn default() -> Self {
        Self {
            seq_len: 32,
            stride: 1,
            max_windows: None,
            preview_tokens: 16,
            max_vocab: 2048,
            input_profile: LexemeInputProfile::Plain,
            vocab_profile: LexemeVocabProfile::Frequency,
            vocab_frequency_cap: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LexemeInputProfile {
    Plain,
    ShakespeareGutenberg,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LexemeVocabProfile {
    Frequency,
    Balanced,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorpusTrace {
    pub shakespeare_input_bytes: usize,
    pub shakespeare_output_bytes: usize,
    pub simplewiki_input_bytes: usize,
    pub simplewiki_pages_seen: usize,
    pub simplewiki_pages_accepted: usize,
    pub simplewiki_pages_skipped_redirect: usize,
    pub simplewiki_pages_skipped_namespace: usize,
    pub simplewiki_pages_skipped_empty: usize,
    pub output_bytes: usize,
    pub output_lines: usize,
    pub output_hash: u64,
    pub max_simplewiki_pages: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenTrace {
    pub input_bytes: usize,
    pub output_bytes: usize,
    pub token_count: usize,
    pub vocab_size: usize,
    pub seq_len: usize,
    pub stride: usize,
    pub max_windows: Option<usize>,
    pub text_profile: TokenTextProfile,
    pub windows: usize,
    pub uncovered_tail_tokens: usize,
    pub token_hash: u64,
    pub window_hash: u64,
    pub first_input_preview: Vec<u8>,
    pub first_target_preview: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LexemeTokenTrace {
    pub input_bytes: usize,
    pub normalized_bytes: usize,
    pub lexeme_count: usize,
    pub output_bytes: usize,
    pub token_count: usize,
    pub vocab_size: usize,
    pub reserved_byte_tokens: usize,
    pub vocab_entries: usize,
    pub fallback_token_count: usize,
    pub known_lexeme_token_count: usize,
    pub seq_len: usize,
    pub stride: usize,
    pub max_windows: Option<usize>,
    pub max_vocab: usize,
    pub input_profile: LexemeInputProfile,
    pub vocab_profile: LexemeVocabProfile,
    pub vocab_frequency_cap: u32,
    pub windows: usize,
    pub uncovered_tail_tokens: usize,
    pub token_hash: u64,
    pub vocab_hash: u64,
    pub window_hash: u64,
    pub first_input_preview: Vec<u16>,
    pub first_target_preview: Vec<u16>,
    pub first_lexeme_preview: Vec<String>,
}

#[derive(Debug)]
pub enum CorpusError {
    Io(io::Error),
    InvalidInput(&'static str),
}

impl core::fmt::Display for CorpusError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "{error}"),
            Self::InvalidInput(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for CorpusError {}

impl From<io::Error> for CorpusError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

pub fn prepare_corpus<R: Read, B: BufRead, W: Write>(
    shakespeare: &mut R,
    simplewiki_xml: B,
    output: &mut W,
    config: CorpusConfig,
) -> Result<CorpusTrace, CorpusError> {
    let mut shakespeare_text = String::new();
    shakespeare.read_to_string(&mut shakespeare_text)?;
    let shakespeare_input_bytes = shakespeare_text.len();
    let shakespeare_clean = clean_shakespeare_text(&shakespeare_text);
    let shakespeare_output_bytes = shakespeare_clean.len();

    let mut corpus = String::new();
    corpus.push_str("<|source:shakespeare|>\n");
    corpus.push_str(&shakespeare_clean);
    if !corpus.ends_with('\n') {
        corpus.push('\n');
    }
    corpus.push_str("\n<|source:simplewiki|>\n");

    let wiki = extract_simplewiki_text(simplewiki_xml, config.max_simplewiki_pages)?;
    corpus.push_str(&wiki.text);
    if !corpus.ends_with('\n') {
        corpus.push('\n');
    }

    output.write_all(corpus.as_bytes())?;

    Ok(CorpusTrace {
        shakespeare_input_bytes,
        shakespeare_output_bytes,
        simplewiki_input_bytes: wiki.input_bytes,
        simplewiki_pages_seen: wiki.pages_seen,
        simplewiki_pages_accepted: wiki.pages_accepted,
        simplewiki_pages_skipped_redirect: wiki.pages_skipped_redirect,
        simplewiki_pages_skipped_namespace: wiki.pages_skipped_namespace,
        simplewiki_pages_skipped_empty: wiki.pages_skipped_empty,
        output_bytes: corpus.len(),
        output_lines: corpus.lines().count(),
        output_hash: hash_bytes(corpus.as_bytes()),
        max_simplewiki_pages: config.max_simplewiki_pages,
    })
}

pub fn tokenize_corpus<R: Read, W: Write>(
    corpus: &mut R,
    tokens_out: &mut W,
    config: TokenizeConfig,
) -> Result<TokenTrace, CorpusError> {
    if config.seq_len == 0 || config.stride == 0 {
        return Err(CorpusError::InvalidInput(
            "tokenization requires positive seq_len and stride",
        ));
    }

    let mut bytes = Vec::new();
    corpus.read_to_end(&mut bytes)?;
    let tokens = tokenize_bytes_for_profile(&bytes, config.text_profile);
    tokens_out.write_all(tokens.as_ref())?;

    let token_slice = tokens.as_ref();
    let window_stats = compute_window_stats(token_slice, config);
    let preview_len = config
        .preview_tokens
        .min(config.seq_len)
        .min(token_slice.len());
    let first_input_preview = token_slice
        .iter()
        .copied()
        .take(preview_len)
        .collect::<Vec<_>>();
    let first_target_preview = if token_slice.len() > 1 {
        tokens
            .iter()
            .copied()
            .skip(1)
            .take(preview_len)
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };

    Ok(TokenTrace {
        input_bytes: bytes.len(),
        output_bytes: token_slice.len(),
        token_count: token_slice.len(),
        vocab_size: 256,
        seq_len: config.seq_len,
        stride: config.stride,
        max_windows: config.max_windows,
        text_profile: config.text_profile,
        windows: window_stats.windows,
        uncovered_tail_tokens: window_stats.uncovered_tail_tokens,
        token_hash: hash_bytes(token_slice),
        window_hash: window_stats.hash,
        first_input_preview,
        first_target_preview,
    })
}

pub fn tokenize_lexeme_corpus<R: Read, TW: Write, VW: Write>(
    corpus: &mut R,
    tokens_out: &mut TW,
    vocab_out: &mut VW,
    config: LexemeTokenizeConfig,
) -> Result<LexemeTokenTrace, CorpusError> {
    if config.seq_len == 0 || config.stride == 0 {
        return Err(CorpusError::InvalidInput(
            "lexeme tokenization requires positive seq_len and stride",
        ));
    }
    if !(257..=65_536).contains(&config.max_vocab) {
        return Err(CorpusError::InvalidInput(
            "lexeme tokenization requires max_vocab in 257..=65536",
        ));
    }

    let mut bytes = Vec::new();
    corpus.read_to_end(&mut bytes)?;
    let semantic_bytes = lexeme_input_bytes_for_profile(&bytes, config.input_profile);
    let normalized = ascii_lower_lexeme_tokens(&semantic_bytes);
    let scan = scan_ascii_lower_lexeme_counts(&normalized, config.preview_tokens);
    let vocab = build_lexeme_vocab(
        scan.counts,
        config.max_vocab,
        config.vocab_profile,
        config.vocab_frequency_cap,
    );
    let vocab_bytes = lexeme_vocab_tsv_bytes(&vocab);
    let encoded = encode_lexeme_tokens(&normalized, &vocab);

    write_u16_tokens_le(&encoded.tokens, tokens_out)?;
    vocab_out.write_all(&vocab_bytes)?;

    let window_stats = compute_window_stats_u16(&encoded.tokens, config);
    let preview_len = config
        .preview_tokens
        .min(config.seq_len)
        .min(encoded.tokens.len());
    let first_input_preview = encoded
        .tokens
        .iter()
        .copied()
        .take(preview_len)
        .collect::<Vec<_>>();
    let first_target_preview = if encoded.tokens.len() > 1 {
        encoded
            .tokens
            .iter()
            .copied()
            .skip(1)
            .take(preview_len)
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    Ok(LexemeTokenTrace {
        input_bytes: bytes.len(),
        normalized_bytes: normalized.len(),
        lexeme_count: scan.lexeme_count,
        output_bytes: encoded.tokens.len() * 2,
        token_count: encoded.tokens.len(),
        vocab_size: 256 + vocab.len(),
        reserved_byte_tokens: 256,
        vocab_entries: vocab.len(),
        fallback_token_count: encoded.fallback_token_count,
        known_lexeme_token_count: encoded.known_lexeme_token_count,
        seq_len: config.seq_len,
        stride: config.stride,
        max_windows: config.max_windows,
        max_vocab: config.max_vocab,
        input_profile: config.input_profile,
        vocab_profile: config.vocab_profile,
        vocab_frequency_cap: config.vocab_frequency_cap,
        windows: window_stats.windows,
        uncovered_tail_tokens: window_stats.uncovered_tail_tokens,
        token_hash: hash_u16_tokens(&encoded.tokens),
        vocab_hash: hash_bytes(&vocab_bytes),
        window_hash: window_stats.hash,
        first_input_preview,
        first_target_preview,
        first_lexeme_preview: scan.first_lexeme_preview,
    })
}

pub fn clean_shakespeare_text(input: &str) -> String {
    let cleaned = clean_gutenberg_text(input);
    strip_shakespeare_collection_front_matter(&cleaned).to_string()
}

pub fn clean_gutenberg_text(input: &str) -> String {
    let body = strip_gutenberg_boilerplate(input);
    normalize_text(body)
}

pub fn clean_wiki_text(input: &str) -> String {
    let decoded = decode_xml_entities(input);
    let without_refs = strip_tag_blocks(&decoded, "ref");
    let without_tables = strip_balanced(&without_refs, "{|", "|}");
    let without_templates = strip_balanced(&without_tables, "{{", "}}");
    let links = rewrite_wiki_links(&without_templates);
    let tags = strip_angle_tags(&links);
    let headings = tags.replace('=', " ");
    normalize_text(&headings)
}

impl CorpusTrace {
    pub fn to_json_line(&self) -> String {
        let mut out = String::new();
        out.push('{');
        push_string_field(&mut out, "schema", SCHEMA);
        comma(&mut out);
        push_string_field(&mut out, "authority", AUTHORITY);
        comma(&mut out);
        out.push_str("\"sources\":[");
        push_source_object(
            &mut out,
            SHAKESPEARE_SOURCE_ID,
            SHAKESPEARE_SOURCE_URL,
            "plain_text",
        );
        comma(&mut out);
        push_source_object(
            &mut out,
            SIMPLEWIKI_SOURCE_ID,
            SIMPLEWIKI_SOURCE_URL,
            "decompressed_mediawiki_xml",
        );
        out.push(']');
        comma(&mut out);
        out.push_str("\"config\":{");
        push_optional_usize_field(&mut out, "max_simplewiki_pages", self.max_simplewiki_pages);
        out.push('}');
        comma(&mut out);
        out.push_str("\"shakespeare\":{");
        push_usize_field(&mut out, "input_bytes", self.shakespeare_input_bytes);
        comma(&mut out);
        push_usize_field(&mut out, "output_bytes", self.shakespeare_output_bytes);
        out.push('}');
        comma(&mut out);
        out.push_str("\"simplewiki\":{");
        push_usize_field(&mut out, "input_bytes", self.simplewiki_input_bytes);
        comma(&mut out);
        push_usize_field(&mut out, "pages_seen", self.simplewiki_pages_seen);
        comma(&mut out);
        push_usize_field(&mut out, "pages_accepted", self.simplewiki_pages_accepted);
        comma(&mut out);
        push_usize_field(
            &mut out,
            "pages_skipped_redirect",
            self.simplewiki_pages_skipped_redirect,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "pages_skipped_namespace",
            self.simplewiki_pages_skipped_namespace,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "pages_skipped_empty",
            self.simplewiki_pages_skipped_empty,
        );
        out.push('}');
        comma(&mut out);
        out.push_str("\"output\":{");
        push_usize_field(&mut out, "bytes", self.output_bytes);
        comma(&mut out);
        push_usize_field(&mut out, "lines", self.output_lines);
        comma(&mut out);
        push_hash_field(&mut out, "hash", self.output_hash);
        out.push('}');
        comma(&mut out);
        push_string_array_field(&mut out, "known_non_claims", &KNOWN_NON_CLAIMS);
        out.push('}');
        out.push('\n');
        out
    }
}

impl TokenTrace {
    pub fn to_json_line(&self) -> String {
        let mut out = String::new();
        out.push('{');
        push_string_field(&mut out, "schema", TOKEN_SCHEMA);
        comma(&mut out);
        push_string_field(&mut out, "authority", TOKEN_AUTHORITY);
        comma(&mut out);
        push_string_field(&mut out, "tokenizer", self.tokenizer_id());
        comma(&mut out);
        out.push_str("\"config\":{");
        push_usize_field(&mut out, "seq_len", self.seq_len);
        comma(&mut out);
        push_usize_field(&mut out, "stride", self.stride);
        comma(&mut out);
        push_optional_usize_field(&mut out, "max_windows", self.max_windows);
        comma(&mut out);
        push_string_field(&mut out, "text_profile", self.text_profile_name());
        out.push('}');
        comma(&mut out);
        out.push_str("\"input\":{");
        push_usize_field(&mut out, "bytes", self.input_bytes);
        out.push('}');
        comma(&mut out);
        out.push_str("\"tokens\":{");
        push_usize_field(&mut out, "count", self.token_count);
        comma(&mut out);
        push_usize_field(&mut out, "vocab_size", self.vocab_size);
        comma(&mut out);
        push_hash_field(&mut out, "hash", self.token_hash);
        out.push('}');
        comma(&mut out);
        out.push_str("\"windows\":{");
        push_usize_field(&mut out, "count", self.windows);
        comma(&mut out);
        push_usize_field(
            &mut out,
            "uncovered_tail_tokens",
            self.uncovered_tail_tokens,
        );
        comma(&mut out);
        push_hash_field(&mut out, "hash", self.window_hash);
        out.push('}');
        comma(&mut out);
        out.push_str("\"preview\":{");
        push_u8_array_field(&mut out, "first_input_tokens", &self.first_input_preview);
        comma(&mut out);
        push_u8_array_field(&mut out, "first_target_tokens", &self.first_target_preview);
        out.push('}');
        comma(&mut out);
        out.push_str("\"output\":{");
        push_usize_field(&mut out, "bytes", self.output_bytes);
        out.push('}');
        comma(&mut out);
        push_string_array_field(&mut out, "known_non_claims", &TOKEN_KNOWN_NON_CLAIMS);
        out.push('}');
        out.push('\n');
        out
    }
}

impl LexemeTokenTrace {
    pub fn to_json_line(&self) -> String {
        let mut out = String::new();
        out.push('{');
        push_string_field(&mut out, "schema", LEXEME_TOKEN_SCHEMA);
        comma(&mut out);
        push_string_field(&mut out, "authority", LEXEME_TOKEN_AUTHORITY);
        comma(&mut out);
        push_string_field(&mut out, "tokenizer", LEXEME_TOKENIZER_ID);
        comma(&mut out);
        out.push_str("\"config\":{");
        push_usize_field(&mut out, "seq_len", self.seq_len);
        comma(&mut out);
        push_usize_field(&mut out, "stride", self.stride);
        comma(&mut out);
        push_optional_usize_field(&mut out, "max_windows", self.max_windows);
        comma(&mut out);
        push_usize_field(&mut out, "max_vocab", self.max_vocab);
        comma(&mut out);
        push_string_field(&mut out, "input_profile", self.input_profile.profile_name());
        comma(&mut out);
        push_string_field(&mut out, "vocab_profile", self.vocab_profile.profile_name());
        comma(&mut out);
        push_usize_field(
            &mut out,
            "vocab_frequency_cap",
            self.vocab_frequency_cap as usize,
        );
        comma(&mut out);
        push_string_field(&mut out, "text_profile", "ascii-lower");
        comma(&mut out);
        push_string_field(&mut out, "token_width", "u16_le");
        out.push('}');
        comma(&mut out);
        out.push_str("\"input\":{");
        push_usize_field(&mut out, "bytes", self.input_bytes);
        comma(&mut out);
        push_usize_field(&mut out, "normalized_bytes", self.normalized_bytes);
        out.push('}');
        comma(&mut out);
        out.push_str("\"lexemes\":{");
        push_usize_field(&mut out, "count", self.lexeme_count);
        comma(&mut out);
        push_usize_field(
            &mut out,
            "known_lexeme_tokens",
            self.known_lexeme_token_count,
        );
        comma(&mut out);
        push_usize_field(&mut out, "fallback_byte_tokens", self.fallback_token_count);
        out.push('}');
        comma(&mut out);
        out.push_str("\"vocab\":{");
        push_usize_field(&mut out, "size", self.vocab_size);
        comma(&mut out);
        push_usize_field(&mut out, "reserved_byte_tokens", self.reserved_byte_tokens);
        comma(&mut out);
        push_usize_field(&mut out, "entries", self.vocab_entries);
        comma(&mut out);
        push_hash_field(&mut out, "hash", self.vocab_hash);
        out.push('}');
        comma(&mut out);
        out.push_str("\"tokens\":{");
        push_usize_field(&mut out, "count", self.token_count);
        comma(&mut out);
        push_usize_field(&mut out, "bytes", self.output_bytes);
        comma(&mut out);
        push_hash_field(&mut out, "hash", self.token_hash);
        out.push('}');
        comma(&mut out);
        out.push_str("\"windows\":{");
        push_usize_field(&mut out, "count", self.windows);
        comma(&mut out);
        push_usize_field(
            &mut out,
            "uncovered_tail_tokens",
            self.uncovered_tail_tokens,
        );
        comma(&mut out);
        push_hash_field(&mut out, "hash", self.window_hash);
        out.push('}');
        comma(&mut out);
        out.push_str("\"preview\":{");
        push_u16_array_field(&mut out, "first_input_tokens", &self.first_input_preview);
        comma(&mut out);
        push_u16_array_field(&mut out, "first_target_tokens", &self.first_target_preview);
        comma(&mut out);
        push_string_vec_field(&mut out, "first_lexemes", &self.first_lexeme_preview);
        out.push('}');
        comma(&mut out);
        push_string_array_field(&mut out, "known_non_claims", &LEXEME_KNOWN_NON_CLAIMS);
        out.push('}');
        out.push('\n');
        out
    }
}

impl TokenTrace {
    fn tokenizer_id(&self) -> &'static str {
        tokenizer_id_for_profile(self.text_profile)
    }

    fn text_profile_name(&self) -> &'static str {
        text_profile_name(self.text_profile)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WindowStats {
    windows: usize,
    uncovered_tail_tokens: usize,
    hash: u64,
}

fn compute_window_stats(tokens: &[u8], config: TokenizeConfig) -> WindowStats {
    let mut windows = 0_usize;
    let mut start = 0_usize;
    let mut last_end = 0_usize;
    let mut hash = FNV_OFFSET;
    hash = fnv_update_usize(hash, tokens.len());
    hash = fnv_update_usize(hash, config.seq_len);
    hash = fnv_update_usize(hash, config.stride);
    hash = fnv_update_usize(hash, config.max_windows.unwrap_or(usize::MAX));

    while start
        .checked_add(config.seq_len)
        .is_some_and(|target_index| target_index < tokens.len())
    {
        if config.max_windows.is_some_and(|limit| windows >= limit) {
            break;
        }

        let end = start + config.seq_len + 1;
        hash = fnv_update_usize(hash, start);
        hash = fnv_update_bytes(hash, &tokens[start..end]);
        windows += 1;
        last_end = end;
        start = start.saturating_add(config.stride);
    }

    WindowStats {
        windows,
        uncovered_tail_tokens: tokens.len().saturating_sub(last_end),
        hash,
    }
}

fn compute_window_stats_u16(tokens: &[u16], config: LexemeTokenizeConfig) -> WindowStats {
    let mut windows = 0_usize;
    let mut start = 0_usize;
    let mut last_end = 0_usize;
    let mut hash = FNV_OFFSET;
    hash = fnv_update_usize(hash, tokens.len());
    hash = fnv_update_usize(hash, config.seq_len);
    hash = fnv_update_usize(hash, config.stride);
    hash = fnv_update_usize(hash, config.max_windows.unwrap_or(usize::MAX));
    hash = fnv_update_usize(hash, config.max_vocab);
    hash = fnv_update_usize(hash, config.input_profile.profile_id());
    hash = fnv_update_usize(hash, config.vocab_profile.profile_id());
    hash = fnv_update_usize(hash, config.vocab_frequency_cap as usize);

    while start
        .checked_add(config.seq_len)
        .is_some_and(|target_index| target_index < tokens.len())
    {
        if config.max_windows.is_some_and(|limit| windows >= limit) {
            break;
        }

        let end = start + config.seq_len + 1;
        hash = fnv_update_usize(hash, start);
        hash = fnv_update_u16_tokens(hash, &tokens[start..end]);
        windows += 1;
        last_end = end;
        start = start.saturating_add(config.stride);
    }

    WindowStats {
        windows,
        uncovered_tail_tokens: tokens.len().saturating_sub(last_end),
        hash,
    }
}

fn tokenize_bytes_for_profile(input: &[u8], text_profile: TokenTextProfile) -> Cow<'_, [u8]> {
    match text_profile {
        TokenTextProfile::Identity => Cow::Borrowed(input),
        TokenTextProfile::AsciiLower => Cow::Owned(ascii_lower_text_tokens(input)),
    }
}

fn strip_corpus_marker_lines(input: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len());
    for line in input.split_inclusive(|&byte| byte == b'\n') {
        let trimmed = trim_ascii_line_start(line);
        if trimmed.starts_with(b"<|source:") || trimmed.starts_with(b"<|page:") {
            continue;
        }
        out.extend_from_slice(line);
    }
    out
}

fn lexeme_input_bytes_for_profile(input: &[u8], profile: LexemeInputProfile) -> Vec<u8> {
    match profile {
        LexemeInputProfile::Plain => strip_corpus_marker_lines(input),
        LexemeInputProfile::ShakespeareGutenberg => {
            let text = String::from_utf8_lossy(input);
            clean_shakespeare_text(&text).into_bytes()
        }
    }
}

fn strip_shakespeare_collection_front_matter(input: &str) -> &str {
    let marker = "THE SONNETS";
    let Some(first) = input.find(marker) else {
        return input;
    };
    let after_first = first + marker.len();
    let Some(second_relative) = input[after_first..].find(marker) else {
        return input;
    };
    &input[after_first + second_relative..]
}

fn trim_ascii_line_start(mut input: &[u8]) -> &[u8] {
    while let Some((&byte, rest)) = input.split_first() {
        if byte == b' ' || byte == b'\t' || byte == b'\r' || byte == b'\n' {
            input = rest;
        } else {
            break;
        }
    }
    input
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LexemeVocabEntry {
    id: u16,
    lexeme: Vec<u8>,
    count: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct EncodedLexemeTokens {
    tokens: Vec<u16>,
    fallback_token_count: usize,
    known_lexeme_token_count: usize,
}

struct LexemeScan {
    counts: HashMap<Vec<u8>, u32>,
    lexeme_count: usize,
    first_lexeme_preview: Vec<String>,
}

fn scan_ascii_lower_lexeme_counts(input: &[u8], preview_tokens: usize) -> LexemeScan {
    let mut counts = HashMap::<Vec<u8>, u32>::new();
    let mut lexeme_count = 0_usize;
    let mut first_lexeme_preview = Vec::new();
    for_each_ascii_lower_lexeme(input, |lexeme| {
        lexeme_count = lexeme_count.saturating_add(1);
        if first_lexeme_preview.len() < preview_tokens {
            first_lexeme_preview.push(String::from_utf8_lossy(lexeme).to_string());
        }
        let entry = counts.entry(lexeme.to_vec()).or_insert(0);
        *entry = entry.saturating_add(1);
    });

    LexemeScan {
        counts,
        lexeme_count,
        first_lexeme_preview,
    }
}

fn build_lexeme_vocab(
    counts: HashMap<Vec<u8>, u32>,
    max_vocab: usize,
    profile: LexemeVocabProfile,
    frequency_cap: u32,
) -> Vec<LexemeVocabEntry> {
    let mut ranked = counts
        .into_iter()
        .map(|(lexeme, count)| RankedLexeme {
            score: lexeme_vocab_rank_score(count, profile, frequency_cap),
            lexeme,
            count,
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .score
            .cmp(&left.score)
            .then_with(|| right.count.cmp(&left.count))
            .then_with(|| left.lexeme.cmp(&right.lexeme))
    });

    ranked
        .into_iter()
        .take(max_vocab.saturating_sub(256))
        .enumerate()
        .map(|(index, ranked)| LexemeVocabEntry {
            id: 256_u16 + index as u16,
            lexeme: ranked.lexeme,
            count: ranked.count,
        })
        .collect()
}

struct RankedLexeme {
    lexeme: Vec<u8>,
    count: u32,
    score: u32,
}

fn lexeme_vocab_rank_score(count: u32, profile: LexemeVocabProfile, frequency_cap: u32) -> u32 {
    match profile {
        LexemeVocabProfile::Frequency => count,
        LexemeVocabProfile::Balanced => capped_sqrt_count_score(count, frequency_cap),
    }
}

fn capped_sqrt_count_score(count: u32, frequency_cap: u32) -> u32 {
    if count == 0 {
        return 0;
    }
    if frequency_cap == 0 {
        return integer_sqrt_u32(count);
    }
    if count <= frequency_cap {
        return count;
    }
    frequency_cap.saturating_add(integer_sqrt_u32(count - frequency_cap))
}

fn integer_sqrt_u32(value: u32) -> u32 {
    if value < 2 {
        return value;
    }
    let mut low = 1_u32;
    let mut high = value.min(65_536);
    let mut answer = 1_u32;
    while low <= high {
        let mid = low + ((high - low) >> 1);
        let square = u64::from(mid) * u64::from(mid);
        if square <= u64::from(value) {
            answer = mid;
            low = mid.saturating_add(1);
        } else {
            high = mid.saturating_sub(1);
        }
    }
    answer
}

fn encode_lexeme_tokens(input: &[u8], vocab: &[LexemeVocabEntry]) -> EncodedLexemeTokens {
    let mut lookup = HashMap::<&[u8], u16>::with_capacity(vocab.len());
    for entry in vocab {
        lookup.insert(entry.lexeme.as_slice(), entry.id);
    }

    let mut tokens = Vec::new();
    let mut fallback_token_count = 0_usize;
    let mut known_lexeme_token_count = 0_usize;
    for_each_ascii_lower_lexeme(input, |lexeme| {
        if let Some(&id) = lookup.get(lexeme) {
            tokens.push(id);
            known_lexeme_token_count = known_lexeme_token_count.saturating_add(1);
        } else {
            for &byte in lexeme {
                tokens.push(u16::from(byte));
                fallback_token_count = fallback_token_count.saturating_add(1);
            }
        }
    });

    EncodedLexemeTokens {
        tokens,
        fallback_token_count,
        known_lexeme_token_count,
    }
}

fn for_each_ascii_lower_lexeme<F>(input: &[u8], mut visit: F)
where
    F: FnMut(&[u8]),
{
    let mut index = 0_usize;

    while index < input.len() {
        let byte = input[index];
        if byte == b' ' {
            index += 1;
            continue;
        }
        if is_lexeme_word_byte(byte) {
            let start = index;
            index += 1;
            while index < input.len() {
                let current = input[index];
                let joins_word = (current == b'\'' || current == b'-')
                    && index + 1 < input.len()
                    && is_lexeme_word_byte(input[index - 1])
                    && is_lexeme_word_byte(input[index + 1]);
                if is_lexeme_word_byte(current) || joins_word {
                    index += 1;
                } else {
                    break;
                }
            }
            visit(&input[start..index]);
            continue;
        }
        if is_lexeme_punctuation(byte) {
            visit(&input[index..index + 1]);
        }
        index += 1;
    }
}

fn lexeme_vocab_tsv_bytes(vocab: &[LexemeVocabEntry]) -> Vec<u8> {
    let mut out = String::new();
    out.push_str("token_id\tlexeme\tcount\n");
    for entry in vocab {
        let lexeme = String::from_utf8_lossy(&entry.lexeme);
        let _ = writeln!(&mut out, "{}\t{}\t{}", entry.id, lexeme, entry.count);
    }
    out.into_bytes()
}

fn write_u16_tokens_le<W: Write>(tokens: &[u16], output: &mut W) -> Result<(), CorpusError> {
    const BUFFER_TOKENS: usize = 8192;
    let mut buffer = Vec::with_capacity(BUFFER_TOKENS * 2);
    for chunk in tokens.chunks(BUFFER_TOKENS) {
        buffer.clear();
        for &token in chunk {
            buffer.extend_from_slice(&token.to_le_bytes());
        }
        output.write_all(&buffer)?;
    }
    Ok(())
}

fn is_lexeme_word_byte(byte: u8) -> bool {
    byte.is_ascii_alphabetic()
}

fn is_lexeme_punctuation(byte: u8) -> bool {
    matches!(byte, b'.' | b',' | b';' | b':' | b'?' | b'!' | b'\'' | b'-')
}

fn ascii_lower_text_tokens(input: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len());
    let mut pending_space = false;

    for &byte in input {
        match ascii_lower_text_byte(byte) {
            Some(token) => {
                if pending_space && !out.is_empty() {
                    out.push(b' ');
                }
                pending_space = false;
                out.push(token);
            }
            None => {
                pending_space = true;
            }
        }
    }

    out
}

fn ascii_lower_lexeme_tokens(input: &[u8]) -> Vec<u8> {
    if input.is_ascii() {
        return ascii_lower_lexeme_ascii_tokens(input);
    }

    if let Ok(text) = std::str::from_utf8(input) {
        return ascii_lower_lexeme_str_tokens(text);
    }

    let text = String::from_utf8_lossy(input);
    ascii_lower_lexeme_str_tokens(&text)
}

fn ascii_lower_lexeme_ascii_tokens(input: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len());
    let mut pending_space = false;

    for &byte in input {
        match ascii_lower_lexeme_ascii_byte(byte) {
            Some(token) => {
                if pending_space && !out.is_empty() {
                    out.push(b' ');
                }
                pending_space = false;
                out.push(token);
            }
            None => {
                pending_space = true;
            }
        }
    }

    out
}

fn ascii_lower_lexeme_str_tokens(input: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len());
    let mut pending_space = false;

    for ch in input.chars() {
        match ascii_lower_lexeme_char(ch) {
            Some(token) => {
                if pending_space && !out.is_empty() {
                    out.push(b' ');
                }
                pending_space = false;
                out.push(token);
            }
            None => {
                pending_space = true;
            }
        }
    }

    out
}

fn ascii_lower_lexeme_ascii_byte(byte: u8) -> Option<u8> {
    match byte {
        b'A'..=b'Z' => Some(byte.to_ascii_lowercase()),
        b'a'..=b'z' | b'.' | b',' | b';' | b':' | b'?' | b'!' | b'\'' | b'-' => Some(byte),
        b' ' | b'\n' | b'\r' | b'\t' => None,
        _ => None,
    }
}

fn ascii_lower_lexeme_char(ch: char) -> Option<u8> {
    match ch {
        'A'..='Z' => Some(ch.to_ascii_lowercase() as u8),
        'a'..='z' | '.' | ',' | ';' | ':' | '?' | '!' => Some(ch as u8),
        '\'' | '\u{2018}' | '\u{2019}' | '\u{02bc}' => Some(b'\''),
        '-' | '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2013}' | '\u{2014}' => Some(b'-'),
        _ if ch.is_whitespace() => None,
        _ => None,
    }
}

fn ascii_lower_text_byte(byte: u8) -> Option<u8> {
    match byte {
        b'A'..=b'Z' => Some(byte.to_ascii_lowercase()),
        b'a'..=b'z' | b'0'..=b'9' | b'.' | b',' | b';' | b':' | b'?' | b'!' | b'\'' | b'-' => {
            Some(byte)
        }
        b' ' | b'\n' | b'\r' | b'\t' => None,
        _ => None,
    }
}

fn tokenizer_id_for_profile(text_profile: TokenTextProfile) -> &'static str {
    match text_profile {
        TokenTextProfile::Identity => TOKENIZER_ID,
        TokenTextProfile::AsciiLower => ASCII_LOWER_TOKENIZER_ID,
    }
}

fn text_profile_name(text_profile: TokenTextProfile) -> &'static str {
    match text_profile {
        TokenTextProfile::Identity => "identity",
        TokenTextProfile::AsciiLower => "ascii-lower",
    }
}

impl LexemeInputProfile {
    fn profile_name(self) -> &'static str {
        match self {
            Self::Plain => "plain",
            Self::ShakespeareGutenberg => "shakespeare-gutenberg",
        }
    }

    fn profile_id(self) -> usize {
        match self {
            Self::Plain => 0,
            Self::ShakespeareGutenberg => 1,
        }
    }
}

impl LexemeVocabProfile {
    fn profile_name(self) -> &'static str {
        match self {
            Self::Frequency => "frequency",
            Self::Balanced => "balanced",
        }
    }

    fn profile_id(self) -> usize {
        match self {
            Self::Frequency => 0,
            Self::Balanced => 1,
        }
    }
}

#[derive(Default)]
struct WikiExtraction {
    text: String,
    input_bytes: usize,
    pages_seen: usize,
    pages_accepted: usize,
    pages_skipped_redirect: usize,
    pages_skipped_namespace: usize,
    pages_skipped_empty: usize,
}

#[derive(Default)]
struct PageState {
    title: String,
    namespace: String,
    text: String,
    redirect: bool,
    in_page: bool,
    in_text: bool,
    saw_title: bool,
    saw_namespace: bool,
}

fn extract_simplewiki_text<B: BufRead>(
    reader: B,
    max_pages: Option<usize>,
) -> Result<WikiExtraction, CorpusError> {
    let mut out = WikiExtraction::default();
    let mut page = PageState::default();

    for line in reader.lines() {
        let line = line?;
        out.input_bytes = out.input_bytes.saturating_add(line.len()).saturating_add(1);
        let trimmed = line.trim();

        if trimmed.starts_with("<page>") {
            page = PageState::default();
            page.in_page = true;
            out.pages_seen = out.pages_seen.saturating_add(1);
        }

        if !page.in_page {
            continue;
        }

        if !page.in_text {
            if !page.saw_title
                && let Some(title) = extract_tag_text(trimmed, "title")
            {
                page.title = decode_xml_entities(title);
                page.saw_title = true;
            }
            if !page.saw_namespace
                && let Some(namespace) = extract_tag_text(trimmed, "ns")
            {
                page.namespace = namespace.to_string();
                page.saw_namespace = true;
            }
            if trimmed.starts_with("<redirect") {
                page.redirect = true;
            }
        }

        consume_text_line(trimmed, &mut page);

        if trimmed.starts_with("</page>") {
            finish_page(&mut out, &page, max_pages);
            if max_pages.is_some_and(|limit| out.pages_accepted >= limit) {
                break;
            }
            page = PageState::default();
        }
    }

    Ok(out)
}

fn finish_page(out: &mut WikiExtraction, page: &PageState, max_pages: Option<usize>) {
    if max_pages.is_some_and(|limit| out.pages_accepted >= limit) {
        return;
    }

    if page.redirect {
        out.pages_skipped_redirect = out.pages_skipped_redirect.saturating_add(1);
        return;
    }

    if page.namespace != "0" {
        out.pages_skipped_namespace = out.pages_skipped_namespace.saturating_add(1);
        return;
    }

    let cleaned = clean_wiki_text(&page.text);
    if cleaned.is_empty() {
        out.pages_skipped_empty = out.pages_skipped_empty.saturating_add(1);
        return;
    }

    out.pages_accepted = out.pages_accepted.saturating_add(1);
    out.text.push_str("<|page:");
    out.text.push_str(&normalize_marker(&page.title));
    out.text.push_str("|>\n");
    out.text.push_str(&cleaned);
    out.text.push_str("\n\n");
}

fn consume_text_line(line: &str, page: &mut PageState) {
    if page.in_text {
        if let Some(end) = line.find("</text>") {
            if end > 0 {
                page.text.push_str(&line[..end]);
            }
            page.in_text = false;
        } else {
            page.text.push_str(line);
            page.text.push('\n');
        }
        return;
    }

    let Some(start) = line.find("<text") else {
        return;
    };
    let Some(close) = line[start..].find('>') else {
        return;
    };
    let text_start = start + close + 1;
    let remainder = &line[text_start..];
    if let Some(end) = remainder.find("</text>") {
        page.text.push_str(&remainder[..end]);
    } else {
        page.text.push_str(remainder);
        page.text.push('\n');
        page.in_text = true;
    }
}

fn strip_gutenberg_boilerplate(input: &str) -> &str {
    let after_start = if let Some(index) = input.find("*** START OF") {
        input[index..]
            .find('\n')
            .map(|line_end| &input[index + line_end + 1..])
            .unwrap_or(input)
    } else {
        input
    };

    let end_marker = ["*** END OF", "End of Project Gutenberg"]
        .iter()
        .filter_map(|marker| after_start.find(marker))
        .min();

    if let Some(index) = end_marker {
        &after_start[..index]
    } else {
        after_start
    }
}

fn extract_tag_text<'a>(line: &'a str, tag: &str) -> Option<&'a str> {
    let mut open = String::new();
    let mut close = String::new();
    write!(&mut open, "<{tag}>").ok()?;
    write!(&mut close, "</{tag}>").ok()?;
    let start = line.find(&open)? + open.len();
    let end = line[start..].find(&close)? + start;
    Some(&line[start..end])
}

fn decode_xml_entities(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '&' {
            out.push(ch);
            continue;
        }

        let mut entity = String::new();
        while let Some(&next) = chars.peek() {
            chars.next();
            if next == ';' {
                break;
            }
            entity.push(next);
            if entity.len() > 16 {
                break;
            }
        }

        match entity.as_str() {
            "amp" => out.push('&'),
            "lt" => out.push('<'),
            "gt" => out.push('>'),
            "quot" => out.push('"'),
            "apos" => out.push('\''),
            _ => {
                out.push('&');
                out.push_str(&entity);
                out.push(';');
            }
        }
    }
    out
}

fn strip_tag_blocks(input: &str, tag: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut index = 0_usize;
    let open_prefix = format!("<{tag}");
    let close = format!("</{tag}>");

    while let Some(relative_start) = input[index..].find(&open_prefix) {
        let start = index + relative_start;
        out.push_str(&input[index..start]);
        let Some(relative_end) = input[start..].find(&close) else {
            index = input.len();
            break;
        };
        index = start + relative_end + close.len();
    }
    out.push_str(&input[index..]);
    out
}

fn strip_balanced(input: &str, open: &str, close: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut index = 0_usize;

    while index < input.len() {
        if input[index..].starts_with(open) {
            let mut depth = 1_usize;
            index += open.len();
            while index < input.len() && depth > 0 {
                if input[index..].starts_with(open) {
                    depth += 1;
                    index += open.len();
                } else if input[index..].starts_with(close) {
                    depth -= 1;
                    index += close.len();
                } else {
                    index += input[index..]
                        .chars()
                        .next()
                        .map(char::len_utf8)
                        .unwrap_or(1);
                }
            }
        } else {
            let ch = input[index..].chars().next().unwrap_or_default();
            out.push(ch);
            index += ch.len_utf8();
        }
    }

    out
}

fn rewrite_wiki_links(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut index = 0_usize;
    while let Some(relative_start) = input[index..].find("[[") {
        let start = index + relative_start;
        out.push_str(&input[index..start]);
        let content_start = start + 2;
        let Some(relative_end) = input[content_start..].find("]]") else {
            out.push_str(&input[start..]);
            return out;
        };
        let end = content_start + relative_end;
        let content = &input[content_start..end];
        if !content.starts_with("File:")
            && !content.starts_with("Image:")
            && !content.starts_with("Category:")
        {
            let display = content.rsplit_once('|').map_or(content, |(_, text)| text);
            out.push_str(display);
        }
        index = end + 2;
    }
    out.push_str(&input[index..]);
    out
}

fn strip_angle_tags(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut in_tag = false;
    for ch in input.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => {
                if in_tag {
                    in_tag = false;
                } else {
                    out.push(ch);
                }
            }
            _ if !in_tag => out.push(ch),
            _ => {}
        }
    }
    out
}

fn normalize_text(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut blank_lines = 0_usize;

    for raw_line in input.lines() {
        let mut line = String::new();
        let mut pending_space = false;
        for ch in raw_line.chars() {
            if ch.is_control() && ch != '\t' {
                continue;
            }
            if ch.is_whitespace() {
                pending_space = true;
            } else {
                if pending_space && !line.is_empty() {
                    line.push(' ');
                }
                pending_space = false;
                line.push(ch);
            }
        }

        let line = line.trim();
        if line.is_empty() {
            blank_lines = blank_lines.saturating_add(1);
            if blank_lines == 1 && !out.is_empty() && !out.ends_with("\n\n") {
                out.push('\n');
            }
        } else {
            blank_lines = 0;
            out.push_str(line);
            out.push('\n');
        }
    }

    out.trim().to_string()
}

fn normalize_marker(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut pending_dash = false;
    for ch in input.chars() {
        if ch.is_ascii_alphanumeric() {
            if pending_dash && !out.is_empty() {
                out.push('-');
            }
            pending_dash = false;
            out.push(ch.to_ascii_lowercase());
        } else {
            pending_dash = true;
        }
    }
    out
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    fnv_update_bytes(FNV_OFFSET, bytes)
}

fn hash_u16_tokens(tokens: &[u16]) -> u64 {
    fnv_update_u16_tokens(FNV_OFFSET, tokens)
}

fn fnv_update_bytes(mut value: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        value ^= u64::from(byte);
        value = value.wrapping_mul(FNV_PRIME);
    }
    value
}

fn fnv_update_usize(value: u64, number: usize) -> u64 {
    fnv_update_bytes(value, &(number as u64).to_le_bytes())
}

fn fnv_update_u16_tokens(mut value: u64, tokens: &[u16]) -> u64 {
    for &token in tokens {
        value = fnv_update_bytes(value, &token.to_le_bytes());
    }
    value
}

fn comma(out: &mut String) {
    out.push(',');
}

fn push_source_object(out: &mut String, id: &str, url: &str, format: &str) {
    out.push('{');
    push_string_field(out, "id", id);
    comma(out);
    push_string_field(out, "url", url);
    comma(out);
    push_string_field(out, "format", format);
    out.push('}');
}

fn push_string_field(out: &mut String, name: &str, value: &str) {
    push_quoted(out, name);
    out.push(':');
    push_quoted(out, value);
}

fn push_string_array_field(out: &mut String, name: &str, values: &[&str]) {
    push_quoted(out, name);
    out.push_str(":[");
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            comma(out);
        }
        push_quoted(out, value);
    }
    out.push(']');
}

fn push_string_vec_field(out: &mut String, name: &str, values: &[String]) {
    push_quoted(out, name);
    out.push_str(":[");
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            comma(out);
        }
        push_quoted(out, value);
    }
    out.push(']');
}

fn push_optional_usize_field(out: &mut String, name: &str, value: Option<usize>) {
    push_quoted(out, name);
    out.push(':');
    if let Some(value) = value {
        out.push_str(&value.to_string());
    } else {
        out.push_str("null");
    }
}

fn push_usize_field(out: &mut String, name: &str, value: usize) {
    push_quoted(out, name);
    out.push(':');
    out.push_str(&value.to_string());
}

fn push_u8_array_field(out: &mut String, name: &str, values: &[u8]) {
    push_quoted(out, name);
    out.push_str(":[");
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            comma(out);
        }
        out.push_str(&value.to_string());
    }
    out.push(']');
}

fn push_u16_array_field(out: &mut String, name: &str, values: &[u16]) {
    push_quoted(out, name);
    out.push_str(":[");
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            comma(out);
        }
        out.push_str(&value.to_string());
    }
    out.push(']');
}

fn push_hash_field(out: &mut String, name: &str, value: u64) {
    push_quoted(out, name);
    out.push(':');
    push_quoted(out, &format!("0x{value:016x}"));
}

fn push_quoted(out: &mut String, value: &str) {
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch.is_control() => out.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => out.push(ch),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shakespeare_boilerplate_is_removed() {
        let input = "header\n*** START OF THE PROJECT GUTENBERG EBOOK 100 ***\nA  line\n\nB\n*** END OF THE PROJECT GUTENBERG EBOOK 100 ***\nlicense";

        assert_eq!(clean_shakespeare_text(input), "A line\n\nB");
    }

    #[test]
    fn wiki_markup_is_cleaned_deterministically() {
        let input = "== Lead ==\n[[Earth|The Earth]] is {{age|old}} &amp; round.<ref>note</ref>\n[[Category:Planets]]";

        assert_eq!(clean_wiki_text(input), "Lead\nThe Earth is & round.");
    }

    #[test]
    fn simplewiki_extraction_filters_namespace_redirects_and_empty_pages() {
        let xml = r#"
<mediawiki>
<page>
<title>Apple</title>
<ns>0</ns>
<revision><text xml:space="preserve">[[Apple]] is a fruit.</text></revision>
</page>
<page>
<title>Talk:Apple</title>
<ns>1</ns>
<revision><text xml:space="preserve">skip me</text></revision>
</page>
<page>
<title>Macintosh</title>
<ns>0</ns>
<redirect title="Apple" />
<revision><text xml:space="preserve">skip redirect</text></revision>
</page>
<page>
<title>Blank</title>
<ns>0</ns>
<revision><text xml:space="preserve">{{stub}}</text></revision>
</page>
</mediawiki>
"#;

        let wiki = extract_simplewiki_text(xml.as_bytes(), None).expect("extract");

        assert_eq!(wiki.pages_seen, 4);
        assert_eq!(wiki.pages_accepted, 1);
        assert_eq!(wiki.pages_skipped_namespace, 1);
        assert_eq!(wiki.pages_skipped_redirect, 1);
        assert_eq!(wiki.pages_skipped_empty, 1);
        assert_eq!(wiki.text, "<|page:apple|>\nApple is a fruit.\n\n");
    }

    #[test]
    fn corpus_trace_is_byte_stable() {
        let shakespeare = b"*** START OF TEST ***\nTo be.\n*** END OF TEST ***";
        let wiki = br#"
<mediawiki>
<page>
<title>Bee</title>
<ns>0</ns>
<revision><text xml:space="preserve">A [[bee]] flies.</text></revision>
</page>
</mediawiki>
"#;
        let mut output_a = Vec::new();
        let mut output_b = Vec::new();
        let trace_a = prepare_corpus(
            &mut &shakespeare[..],
            &wiki[..],
            &mut output_a,
            CorpusConfig {
                max_simplewiki_pages: None,
            },
        )
        .expect("left");
        let trace_b = prepare_corpus(
            &mut &shakespeare[..],
            &wiki[..],
            &mut output_b,
            CorpusConfig {
                max_simplewiki_pages: None,
            },
        )
        .expect("right");

        assert_eq!(output_a, output_b);
        assert_eq!(trace_a, trace_b);
        assert_eq!(trace_a.to_json_line(), trace_b.to_json_line());
        assert!(
            trace_a
                .to_json_line()
                .contains("\"schema\":\"nsrl.corpus_trace.v1\"")
        );
        assert!(
            String::from_utf8(output_a)
                .unwrap()
                .contains("<|source:simplewiki|>")
        );
    }

    #[test]
    fn byte_tokenizer_writes_identity_tokens_and_counts_windows() {
        let corpus = b"abcdef";
        let mut tokens = Vec::new();
        let trace = tokenize_corpus(
            &mut &corpus[..],
            &mut tokens,
            TokenizeConfig {
                seq_len: 3,
                stride: 2,
                max_windows: None,
                preview_tokens: 4,
                text_profile: TokenTextProfile::Identity,
            },
        )
        .expect("tokenize");

        assert_eq!(tokens, corpus);
        assert_eq!(trace.input_bytes, 6);
        assert_eq!(trace.output_bytes, 6);
        assert_eq!(trace.token_count, 6);
        assert_eq!(trace.vocab_size, 256);
        assert_eq!(trace.windows, 2);
        assert_eq!(trace.uncovered_tail_tokens, 0);
        assert_eq!(trace.first_input_preview, b"abc".to_vec());
        assert_eq!(trace.first_target_preview, b"bcd".to_vec());
    }

    #[test]
    fn byte_tokenizer_trace_is_byte_stable() {
        let corpus = b"to be or not to be";
        let config = TokenizeConfig {
            seq_len: 4,
            stride: 1,
            max_windows: Some(3),
            preview_tokens: 8,
            text_profile: TokenTextProfile::Identity,
        };
        let mut left_tokens = Vec::new();
        let mut right_tokens = Vec::new();
        let left = tokenize_corpus(&mut &corpus[..], &mut left_tokens, config)
            .expect("left")
            .to_json_line();
        let right = tokenize_corpus(&mut &corpus[..], &mut right_tokens, config)
            .expect("right")
            .to_json_line();

        assert_eq!(left_tokens, right_tokens);
        assert_eq!(left, right);
        assert!(left.contains("\"schema\":\"nsrl.token_trace.v1\""));
        assert!(left.contains("\"tokenizer\":\"byte_identity_u8_v1\""));
        assert!(left.contains("\"text_profile\":\"identity\""));
        assert!(left.contains("\"count\":3"));
    }

    #[test]
    fn ascii_lower_tokenizer_writes_text_curriculum_tokens() {
        let corpus = b"To  BE!\n<|Page:Apple|>  Caf\xc3\xa9\t& raw";
        let mut tokens = Vec::new();
        let trace = tokenize_corpus(
            &mut &corpus[..],
            &mut tokens,
            TokenizeConfig {
                seq_len: 4,
                stride: 1,
                max_windows: None,
                preview_tokens: 8,
                text_profile: TokenTextProfile::AsciiLower,
            },
        )
        .expect("tokenize");

        assert_eq!(tokens, b"to be! page:apple caf raw");
        assert_eq!(trace.input_bytes, corpus.len());
        assert_eq!(trace.output_bytes, tokens.len());
        assert_eq!(trace.text_profile, TokenTextProfile::AsciiLower);
        assert_eq!(trace.first_input_preview, b"to b".to_vec());
        let json = trace.to_json_line();
        assert!(json.contains("\"tokenizer\":\"byte_ascii_lower_text_u8_v1\""));
        assert!(json.contains("\"text_profile\":\"ascii-lower\""));
    }

    #[test]
    fn lexeme_tokenizer_promotes_common_chunks_and_falls_back_to_bytes() {
        let corpus = b"To be, or not to be. Zebra";
        let mut token_bytes = Vec::new();
        let mut vocab_bytes = Vec::new();
        let trace = tokenize_lexeme_corpus(
            &mut &corpus[..],
            &mut token_bytes,
            &mut vocab_bytes,
            LexemeTokenizeConfig {
                seq_len: 4,
                stride: 1,
                max_windows: None,
                preview_tokens: 8,
                max_vocab: 260,
                input_profile: LexemeInputProfile::Plain,
                vocab_profile: LexemeVocabProfile::Frequency,
                vocab_frequency_cap: 0,
            },
        )
        .expect("lexeme tokenize");

        let tokens = decode_u16_le_tokens(&token_bytes);
        assert_eq!(
            String::from_utf8(vocab_bytes.clone()).unwrap(),
            "token_id\tlexeme\tcount\n256\tbe\t2\n257\tto\t2\n258\t,\t1\n259\t.\t1\n"
        );
        assert_eq!(
            tokens,
            vec![
                257,
                256,
                258,
                u16::from(b'o'),
                u16::from(b'r'),
                u16::from(b'n'),
                u16::from(b'o'),
                u16::from(b't'),
                257,
                256,
                259,
                u16::from(b'z'),
                u16::from(b'e'),
                u16::from(b'b'),
                u16::from(b'r'),
                u16::from(b'a'),
            ]
        );
        assert_eq!(trace.vocab_size, 260);
        assert_eq!(trace.vocab_entries, 4);
        assert_eq!(trace.known_lexeme_token_count, 6);
        assert_eq!(trace.fallback_token_count, 10);
        assert_eq!(trace.output_bytes, tokens.len() * 2);
        assert_eq!(
            trace.first_input_preview,
            vec![257, 256, 258, u16::from(b'o')]
        );
        let json = trace.to_json_line();
        assert!(json.contains("\"schema\":\"nsrl.lexeme_token_trace.v1\""));
        assert!(json.contains("\"tokenizer\":\"lexeme_ascii_lower_u16_v1\""));
        assert!(json.contains("\"vocab_profile\":\"frequency\""));
        assert!(json.contains("\"vocab_frequency_cap\":0"));
        assert!(json.contains("\"token_width\":\"u16_le\""));
        assert!(json.contains("\"first_lexemes\":[\"to\",\"be\",\",\",\"or\""));
    }

    #[test]
    fn lexeme_tokenizer_balanced_vocab_reduces_repetition_power() {
        let corpus = b"the the the the the the the the the the the the the the the the the the the the king king king queen queen crown";
        let mut token_bytes = Vec::new();
        let mut vocab_bytes = Vec::new();
        let trace = tokenize_lexeme_corpus(
            &mut &corpus[..],
            &mut token_bytes,
            &mut vocab_bytes,
            LexemeTokenizeConfig {
                seq_len: 4,
                stride: 1,
                max_windows: None,
                preview_tokens: 8,
                max_vocab: 259,
                input_profile: LexemeInputProfile::Plain,
                vocab_profile: LexemeVocabProfile::Balanced,
                vocab_frequency_cap: 4,
            },
        )
        .expect("lexeme tokenize");

        let vocab = String::from_utf8(vocab_bytes).unwrap();
        assert_eq!(
            vocab,
            "token_id\tlexeme\tcount\n256\tthe\t20\n257\tking\t3\n258\tqueen\t2\n"
        );
        assert_eq!(trace.vocab_entries, 3);
        let json = trace.to_json_line();
        assert!(json.contains("\"vocab_profile\":\"balanced\""));
        assert!(json.contains("\"vocab_frequency_cap\":4"));
    }

    #[test]
    fn lexeme_tokenizer_trace_is_stable() {
        let corpus = b"Summer's day, summer's lease";
        let config = LexemeTokenizeConfig {
            seq_len: 3,
            stride: 2,
            max_windows: Some(4),
            preview_tokens: 6,
            max_vocab: 512,
            input_profile: LexemeInputProfile::Plain,
            vocab_profile: LexemeVocabProfile::Frequency,
            vocab_frequency_cap: 0,
        };
        let mut left_tokens = Vec::new();
        let mut left_vocab = Vec::new();
        let mut right_tokens = Vec::new();
        let mut right_vocab = Vec::new();
        let left =
            tokenize_lexeme_corpus(&mut &corpus[..], &mut left_tokens, &mut left_vocab, config)
                .expect("left")
                .to_json_line();
        let right = tokenize_lexeme_corpus(
            &mut &corpus[..],
            &mut right_tokens,
            &mut right_vocab,
            config,
        )
        .expect("right")
        .to_json_line();

        assert_eq!(left_tokens, right_tokens);
        assert_eq!(left_vocab, right_vocab);
        assert_eq!(left, right);
        assert!(left.contains("\"max_vocab\":512"));
    }

    #[test]
    fn lexeme_tokenizer_skips_nsrl_corpus_marker_lines() {
        let corpus = b"<|source:shakespeare|>\nTo be\n  <|page:apple|>\nApple grows";
        let mut token_bytes = Vec::new();
        let mut vocab_bytes = Vec::new();
        let trace = tokenize_lexeme_corpus(
            &mut &corpus[..],
            &mut token_bytes,
            &mut vocab_bytes,
            LexemeTokenizeConfig {
                seq_len: 2,
                stride: 1,
                max_windows: None,
                preview_tokens: 6,
                max_vocab: 512,
                input_profile: LexemeInputProfile::Plain,
                vocab_profile: LexemeVocabProfile::Frequency,
                vocab_frequency_cap: 0,
            },
        )
        .expect("lexeme tokenize");

        assert_eq!(
            trace.first_lexeme_preview,
            vec!["to", "be", "apple", "grows"]
        );
        let vocab = String::from_utf8(vocab_bytes).unwrap();
        assert!(!vocab.contains("source"));
        assert!(!vocab.contains("page"));
    }

    #[test]
    fn lexeme_tokenizer_can_clean_raw_gutenberg_shakespeare() {
        let corpus = "Release date: January 1, 1994\n*** START OF TEST ***\nTitle\nContents\nTHE SONNETS\nALL'S WELL\nTHE SONNETS\nKing\u{2019}s 2042 keeper.\n*** END OF TEST ***\nProject Gutenberg license";
        let mut input = corpus.as_bytes();
        let mut token_bytes = Vec::new();
        let mut vocab_bytes = Vec::new();
        let trace = tokenize_lexeme_corpus(
            &mut input,
            &mut token_bytes,
            &mut vocab_bytes,
            LexemeTokenizeConfig {
                seq_len: 2,
                stride: 1,
                max_windows: None,
                preview_tokens: 4,
                max_vocab: 512,
                input_profile: LexemeInputProfile::ShakespeareGutenberg,
                vocab_profile: LexemeVocabProfile::Frequency,
                vocab_frequency_cap: 0,
            },
        )
        .expect("lexeme tokenize");

        assert_eq!(
            trace.first_lexeme_preview,
            vec!["the", "sonnets", "king's", "keeper"]
        );
        assert_eq!(
            trace.input_profile,
            LexemeInputProfile::ShakespeareGutenberg
        );
        let vocab = String::from_utf8(vocab_bytes).unwrap();
        assert!(vocab.contains("king's"));
        assert!(vocab.contains("keeper"));
        assert!(!vocab.contains("release"));
        assert!(!vocab.contains("gutenberg"));
        assert!(!vocab.contains("contents"));
        assert!(!vocab.contains("2042"));
        let json = trace.to_json_line();
        assert!(json.contains("\"input_profile\":\"shakespeare-gutenberg\""));
    }

    fn decode_u16_le_tokens(bytes: &[u8]) -> Vec<u16> {
        bytes
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
            .collect()
    }
}
