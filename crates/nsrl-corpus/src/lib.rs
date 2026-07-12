#![deny(unsafe_code)]

use std::borrow::Cow;
use std::fmt::Write as _;
use std::io::{self, BufRead, Read, Write};

pub const SCHEMA: &str = "nsrl.corpus_trace.v1";
pub const SIMPLEWIKI_EXTRACT_SCHEMA: &str = "nsrl.simplewiki_extract_trace.v1";
pub const TOKEN_SCHEMA: &str = "nsrl.token_trace.v1";
pub const AUTHORITY: &str = "deterministic_corpus_preparation";
pub const TOKEN_AUTHORITY: &str = "deterministic_byte_tokenization";
pub const TOKENIZER_ID: &str = "byte_identity_u8_v1";
pub const ASCII_LOWER_TOKENIZER_ID: &str = "byte_ascii_lower_text_u8_v1";
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
pub struct SimpleWikiExtractTrace {
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

pub fn extract_simplewiki_corpus<B: BufRead, W: Write>(
    simplewiki_xml: B,
    output: &mut W,
    config: CorpusConfig,
) -> Result<SimpleWikiExtractTrace, CorpusError> {
    let wiki = extract_simplewiki_text(simplewiki_xml, config.max_simplewiki_pages)?;
    let mut corpus = String::new();
    corpus.push_str("<|source:simplewiki|>\n");
    corpus.push_str(&wiki.text);
    if !corpus.ends_with('\n') {
        corpus.push('\n');
    }

    output.write_all(corpus.as_bytes())?;

    Ok(SimpleWikiExtractTrace {
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
    let without_media = drop_wiki_media_lines(&without_templates);
    let links = rewrite_wiki_links(&without_media);
    let tags = strip_angle_tags(&links);
    let markup = tags
        .replace("'''", "")
        .replace("''", "")
        .replace("[[", "")
        .replace("]]", "");
    let headings = markup.replace('=', " ");
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

impl SimpleWikiExtractTrace {
    pub fn to_json_line(&self) -> String {
        let mut out = String::new();
        out.push('{');
        push_string_field(&mut out, "schema", SIMPLEWIKI_EXTRACT_SCHEMA);
        comma(&mut out);
        push_string_field(&mut out, "authority", AUTHORITY);
        comma(&mut out);
        out.push_str("\"sources\":[");
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

fn tokenize_bytes_for_profile(input: &[u8], text_profile: TokenTextProfile) -> Cow<'_, [u8]> {
    match text_profile {
        TokenTextProfile::Identity => Cow::Borrowed(input),
        TokenTextProfile::AsciiLower => Cow::Owned(ascii_lower_text_tokens(input)),
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
        let lower_content = content.to_ascii_lowercase();
        if !lower_content.starts_with("file:")
            && !lower_content.starts_with("image:")
            && !lower_content.starts_with("category:")
        {
            let display = content.rsplit_once('|').map_or(content, |(_, text)| text);
            out.push_str(display);
        }
        index = end + 2;
    }
    out.push_str(&input[index..]);
    out
}

fn drop_wiki_media_lines(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for line in input.lines() {
        let lower = line.to_ascii_lowercase();
        if lower.contains("[[file:") || lower.contains("[[image:") {
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
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
        let input = "== '''Lead''' ==\n[[Earth|The Earth]] is {{age|old}} &amp; round.<ref>note</ref>\n[[File:Earth.jpg|thumb|The planet [[Earth]]]]\n[[Category:Planets]]";

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
    fn simplewiki_corpus_extraction_writes_source_trace() {
        let xml = br#"
<mediawiki>
<page>
<title>Bee</title>
<ns>0</ns>
<revision><text xml:space="preserve">A [[bee]] flies.</text></revision>
</page>
</mediawiki>
"#;
        let mut output = Vec::new();
        let trace = extract_simplewiki_corpus(
            &xml[..],
            &mut output,
            CorpusConfig {
                max_simplewiki_pages: None,
            },
        )
        .expect("extract simplewiki corpus");

        assert_eq!(
            String::from_utf8(output).expect("utf8"),
            "<|source:simplewiki|>\n<|page:bee|>\nA bee flies.\n\n"
        );
        assert_eq!(trace.simplewiki_pages_accepted, 1);
        assert_eq!(trace.output_lines, 4);
        assert!(
            trace
                .to_json_line()
                .contains("\"schema\":\"nsrl.simplewiki_extract_trace.v1\"")
        );
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
}
