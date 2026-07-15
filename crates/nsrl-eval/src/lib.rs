#![deny(unsafe_code)]

pub mod contract;
pub mod open_generation;
pub mod successor;

use std::collections::HashSet;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EvalPartition {
    Train,
    Eval,
    Gold,
}

impl EvalPartition {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Eval => "eval",
            Self::Gold => "gold",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayCheck {
    pub actual_hash: String,
    pub expected_hash: Option<String>,
    pub passed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SourceGroundingMetrics {
    pub exact_span: bool,
    pub longest_run: usize,
    pub bigram_per_mille: usize,
    pub trigram_per_mille: usize,
}

pub fn partition_by_gold_and_bucket(
    record_hash: &str,
    partition_bucket: usize,
    gold_hashes: &HashSet<String>,
    eval_permille: usize,
) -> EvalPartition {
    if gold_hashes.contains(&record_hash.to_ascii_lowercase()) {
        EvalPartition::Gold
    } else if partition_bucket < eval_permille {
        EvalPartition::Eval
    } else {
        EvalPartition::Train
    }
}

pub fn read_gold_hashes(path: &Path) -> Result<HashSet<String>, Box<dyn std::error::Error>> {
    if !path.exists() {
        return Ok(HashSet::new());
    }
    let text = fs::read_to_string(path)?;
    let mut hashes = HashSet::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let first = trimmed.split('\t').next().unwrap_or("").trim();
        if first == "prompt_hash" {
            continue;
        }
        if first.len() != 8 || !first.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(format!("{} has invalid gold hash {}", path.display(), first).into());
        }
        hashes.insert(first.to_ascii_lowercase());
    }
    Ok(hashes)
}

pub fn append_ledger(path: &Path, row: &str) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(file, "{row}")?;
    Ok(())
}

pub fn read_u16_le_tokens(path: &Path) -> Result<Vec<u16>, Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    if bytes.len() % 2 != 0 {
        return Err(format!("{} is not a little-endian u16 token stream", path.display()).into());
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect())
}

pub fn source_grounding_metrics(source: &[u16], generated: &[u16]) -> SourceGroundingMetrics {
    if generated.is_empty() {
        return SourceGroundingMetrics::default();
    }
    let exact_span = source_contains(source, generated, 0, generated.len());
    let mut longest_run = 0_usize;
    for len in (1..=generated.len()).rev() {
        let mut found = false;
        for start in 0..=generated.len() - len {
            if source_contains(source, generated, start, len) {
                found = true;
                break;
            }
        }
        if found {
            longest_run = len;
            break;
        }
    }
    SourceGroundingMetrics {
        exact_span,
        longest_run,
        bigram_per_mille: ngram_coverage_per_mille(source, generated, 2),
        trigram_per_mille: ngram_coverage_per_mille(source, generated, 3),
    }
}

fn source_contains(source: &[u16], needle: &[u16], start: usize, len: usize) -> bool {
    if len == 0 || len > source.len() || start.saturating_add(len) > needle.len() {
        return false;
    }
    'source: for index in 0..=source.len() - len {
        for offset in 0..len {
            if source[index + offset] != needle[start + offset] {
                continue 'source;
            }
        }
        return true;
    }
    false
}

fn ngram_coverage_per_mille(source: &[u16], generated: &[u16], order: usize) -> usize {
    if order == 0 || generated.len() < order {
        return 0;
    }
    let total = generated.len() - order + 1;
    let mut seen = HashSet::<Vec<u16>>::new();
    for window in source.windows(order) {
        seen.insert(window.to_vec());
    }
    let hits = generated
        .windows(order)
        .filter(|window| seen.contains(*window))
        .count();
    hits.saturating_mul(1000).saturating_add(total / 2) / total
}

pub fn unix_timestamp() -> Result<u64, Box<dyn std::error::Error>> {
    Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs())
}

pub fn replay_check(row: &str, expected_hash: Option<&str>) -> ReplayCheck {
    let actual_hash = trace_row_hash(row);
    let expected_hash = expected_hash.map(|value| value.to_ascii_lowercase());
    let passed = expected_hash
        .as_deref()
        .map(|expected| expected == actual_hash)
        .unwrap_or(true);
    ReplayCheck {
        actual_hash,
        expected_hash,
        passed,
    }
}

pub fn trace_row_hash(row: &str) -> String {
    stable_hex_u32(stable_hash_bytes(row.trim_end().as_bytes()))
}

pub fn stable_hash_bytes(bytes: &[u8]) -> u32 {
    let mut hash = 2_166_136_261_u32;
    for &byte in bytes {
        hash ^= u32::from(byte);
        hash = hash.wrapping_mul(16_777_619);
    }
    hash | 1
}

pub fn stable_hex_u32(value: u32) -> String {
    format!("{value:08x}")
}

pub fn json_escape(value: &str) -> String {
    let mut out = String::new();
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch.is_control() => out.push(' '),
            _ => out.push(ch),
        }
    }
    out
}

pub fn escape_tsv(value: &str) -> String {
    value
        .replace('\t', " ")
        .replace(['\r', '\n'], " ")
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partition_prefers_gold_over_eval_bucket() {
        let mut gold = HashSet::new();
        gold.insert("abcdef01".to_string());

        assert_eq!(
            partition_by_gold_and_bucket("abcdef01", 1, &gold, 100),
            EvalPartition::Gold
        );
        assert_eq!(
            partition_by_gold_and_bucket("deadbeef", 1, &gold, 100),
            EvalPartition::Eval
        );
        assert_eq!(
            partition_by_gold_and_bucket("deadbeef", 100, &gold, 100),
            EvalPartition::Train
        );
    }

    #[test]
    fn replay_check_hashes_trimmed_rows() {
        let row = "{\"schema\":\"demo\"}\n";
        let hash = trace_row_hash(row);
        assert_eq!(hash, trace_row_hash(row.trim_end()));
        assert!(replay_check(row, Some(&hash)).passed);
        assert!(!replay_check(row, Some("00000000")).passed);
    }

    #[test]
    fn source_grounding_detects_exact_span_and_ngrams() {
        let source = [1, 2, 3, 4, 5, 8, 9, 10];
        let generated = [2, 3, 4];
        let metrics = source_grounding_metrics(&source, &generated);

        assert!(metrics.exact_span);
        assert_eq!(metrics.longest_run, 3);
        assert_eq!(metrics.bigram_per_mille, 1000);
        assert_eq!(metrics.trigram_per_mille, 1000);
    }

    #[test]
    fn source_grounding_scores_partial_coverage() {
        let source = [1, 2, 3, 7, 8, 9];
        let generated = [2, 3, 4, 8, 9];
        let metrics = source_grounding_metrics(&source, &generated);

        assert!(!metrics.exact_span);
        assert_eq!(metrics.longest_run, 2);
        assert_eq!(metrics.bigram_per_mille, 500);
        assert_eq!(metrics.trigram_per_mille, 0);
    }
}
