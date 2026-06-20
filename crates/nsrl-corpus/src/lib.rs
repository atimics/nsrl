use std::fmt::Write as _;
use std::io::{self, BufRead, Read, Write};

pub const SCHEMA: &str = "nsrl.corpus_trace.v1";
pub const AUTHORITY: &str = "deterministic_corpus_preparation";
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CorpusConfig {
    pub max_simplewiki_pages: Option<usize>,
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

pub fn clean_shakespeare_text(input: &str) -> String {
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
            if !page.saw_title {
                if let Some(title) = extract_tag_text(trimmed, "title") {
                    page.title = decode_xml_entities(title);
                    page.saw_title = true;
                }
            }
            if !page.saw_namespace {
                if let Some(namespace) = extract_tag_text(trimmed, "ns") {
                    page.namespace = namespace.to_string();
                    page.saw_namespace = true;
                }
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

    if let Some(index) = after_start.find("*** END OF") {
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
    let mut value = FNV_OFFSET;
    for &byte in bytes {
        value ^= u64::from(byte);
        value = value.wrapping_mul(FNV_PRIME);
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
}
