use std::collections::HashSet;
use std::fs;
use std::path::{Component, Path, PathBuf};

use crate::json_escape;

pub const OPEN_GENERATION_CONTRACT_SCHEMA: &str = "nsrl.open_generation_contract.v1";
pub const OPEN_GENERATION_MANIFEST_SCHEMA: &str = "nsrl.open_generation_manifest.v1";
pub const OPEN_GENERATION_PROMPT_SCHEMA: &str = "nsrl.open_generation_prompt.v1";
pub const OPEN_GENERATION_CONTRACT_ID: &str = "open-generation-v1";
pub const OPEN_GENERATION_MANIFEST_HEADER: &str = "schema\tcontract\ttokenizer\ttokenizer_hash\tdevelopment_panel\tdevelopment_panel_hash\thidden_test_sha256\tprompt_count\tmax_prompt_bytes\tgeneration_tokens\tsampling_seeds\tretained_improvement_per_mille\tmax_repeat_4gram_share_per_mille\tmin_unique_4gram_share_per_mille\tmin_entropy_q10\tmin_utf8_valid_per_mille\tmin_context_use_per_mille\tmin_distractor_resistance_per_mille\tmin_human_preference_delta_per_mille";
pub const OPEN_GENERATION_PANEL_HEADER: &str =
    "schema\tcontract\tpartition\tid\tcategory\tmax_new_tokens\trequired_phrase_hex\tprompt_hex";

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
const REQUIRED_CATEGORIES: [&str; 6] = [
    "continuation",
    "constrained-style",
    "explanation",
    "dialogue",
    "long-context-reference",
    "adversarial-repetition",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenGenerationManifest {
    pub manifest_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub tokenizer_hash: String,
    pub development_panel_path: PathBuf,
    pub development_panel_hash: String,
    pub hidden_test_sha256: String,
    pub prompt_count: usize,
    pub max_prompt_bytes: usize,
    pub generation_tokens: usize,
    pub sampling_seeds: Vec<u64>,
    pub retained_improvement_per_mille: usize,
    pub max_repeat_4gram_share_per_mille: usize,
    pub min_unique_4gram_share_per_mille: usize,
    pub min_entropy_q10: usize,
    pub min_utf8_valid_per_mille: usize,
    pub min_context_use_per_mille: usize,
    pub min_distractor_resistance_per_mille: usize,
    pub min_human_preference_delta_per_mille: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenGenerationPrompt {
    pub id: String,
    pub category: String,
    pub max_new_tokens: usize,
    pub required_phrase: Vec<u8>,
    pub prompt: Vec<u8>,
}

impl OpenGenerationManifest {
    pub fn to_json_line(&self) -> String {
        let seeds = self
            .sampling_seeds
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(",");
        format!(
            concat!(
                "{{\"schema\":\"{}\",\"contract\":\"{}\",\"valid\":true,",
                "\"manifest\":\"{}\",\"tokenizer\":{{\"path\":\"{}\",\"hash\":\"{}\"}},",
                "\"development_panel\":{{\"path\":\"{}\",\"hash\":\"{}\",\"prompts\":{}}},",
                "\"hidden_test_sha256\":\"{}\",\"max_prompt_bytes\":{},\"generation_tokens\":{},",
                "\"sampling_seeds\":[{}],\"thresholds\":{{\"retained_improvement_per_mille\":{},",
                "\"max_repeat_4gram_share_per_mille\":{},\"min_unique_4gram_share_per_mille\":{},",
                "\"min_entropy_q10\":{},\"min_utf8_valid_per_mille\":{},\"min_context_use_per_mille\":{},",
                "\"min_distractor_resistance_per_mille\":{},\"min_human_preference_delta_per_mille\":{}}}}}\n"
            ),
            OPEN_GENERATION_MANIFEST_SCHEMA,
            OPEN_GENERATION_CONTRACT_ID,
            json_escape(&self.manifest_path.to_string_lossy()),
            json_escape(&self.tokenizer_path.to_string_lossy()),
            self.tokenizer_hash,
            json_escape(&self.development_panel_path.to_string_lossy()),
            self.development_panel_hash,
            self.prompt_count,
            self.hidden_test_sha256,
            self.max_prompt_bytes,
            self.generation_tokens,
            seeds,
            self.retained_improvement_per_mille,
            self.max_repeat_4gram_share_per_mille,
            self.min_unique_4gram_share_per_mille,
            self.min_entropy_q10,
            self.min_utf8_valid_per_mille,
            self.min_context_use_per_mille,
            self.min_distractor_resistance_per_mille,
            self.min_human_preference_delta_per_mille,
        )
    }
}

pub fn open_generation_contract_json_line() -> String {
    format!(
        concat!(
            "{{\"schema\":\"{}\",\"contract\":\"{}\",\"status\":\"development_frozen\",",
            "\"generation_mode\":\"native_unassisted\",\"primary_modeling_metric\":\"bits_per_original_utf8_byte\",",
            "\"required_baselines\":[\"byte-ngram\",\"retrieval\",\"best-smaller-nsrl\",\"same-shape-float-twin\"],",
            "\"required_prompt_categories\":[\"continuation\",\"constrained-style\",\"explanation\",\"dialogue\",\"long-context-reference\",\"adversarial-repetition\"],",
            "\"forbidden_assistance\":[\"retrieval\",\"corpus-prior\",\"memory-injection\",\"target-lookup\",\"routing-oracle\"],",
            "\"evidence_layers\":[\"modeling\",\"generation-health\",\"blinded-human-product-quality\"],",
            "\"promotion_requires_integer_transformer_proof_v1\":true}}\n"
        ),
        OPEN_GENERATION_CONTRACT_SCHEMA, OPEN_GENERATION_CONTRACT_ID,
    )
}

pub fn load_open_generation_manifest(path: &Path) -> Result<OpenGenerationManifest, String> {
    let input = fs::read_to_string(path).map_err(|error| {
        format!(
            "cannot read open-generation manifest {}: {error}",
            path.display()
        )
    })?;
    let mut lines = input.lines();
    if lines.next() != Some(OPEN_GENERATION_MANIFEST_HEADER) {
        return Err("open-generation manifest header is invalid".to_string());
    }
    let row = lines
        .next()
        .ok_or("open-generation manifest row is missing")?;
    if lines.any(|line| !line.trim().is_empty()) {
        return Err("open-generation manifest must contain one row".to_string());
    }
    let fields = row.split('\t').collect::<Vec<_>>();
    if fields.len() != 19
        || fields[0] != OPEN_GENERATION_MANIFEST_SCHEMA
        || fields[1] != OPEN_GENERATION_CONTRACT_ID
    {
        return Err("open-generation manifest schema or contract is invalid".to_string());
    }
    let directory = path.parent().unwrap_or_else(|| Path::new("."));
    let tokenizer_path = resolve_path(directory, fields[2], "tokenizer")?;
    let development_panel_path = resolve_path(directory, fields[4], "development_panel")?;
    validate_fnv_hash(fields[3], "tokenizer_hash")?;
    validate_fnv_hash(fields[5], "development_panel_hash")?;
    validate_sha256(fields[6], "hidden_test_sha256")?;
    let tokenizer = fs::read(&tokenizer_path).map_err(|error| {
        format!(
            "cannot read tokenizer {}: {error}",
            tokenizer_path.display()
        )
    })?;
    if !tokenizer.starts_with(b"NSRLBPE1") {
        return Err("open-generation tokenizer is not NSRLBPE1".to_string());
    }
    if fnv_hash(&tokenizer) != fields[3] {
        return Err("open-generation tokenizer hash mismatch".to_string());
    }
    let panel = fs::read(&development_panel_path).map_err(|error| {
        format!(
            "cannot read development panel {}: {error}",
            development_panel_path.display()
        )
    })?;
    if fnv_hash(&panel) != fields[5] {
        return Err("open-generation development panel hash mismatch".to_string());
    }
    let prompt_count = parse_positive(fields[7], "prompt_count")?;
    let max_prompt_bytes = parse_positive(fields[8], "max_prompt_bytes")?;
    let generation_tokens = parse_positive(fields[9], "generation_tokens")?;
    let sampling_seeds = parse_seeds(fields[10])?;
    let retained_improvement_per_mille = parse_per_mille(fields[11], "retained improvement")?;
    let max_repeat_4gram_share_per_mille = parse_per_mille(fields[12], "repeat 4gram share")?;
    let min_unique_4gram_share_per_mille = parse_per_mille(fields[13], "unique 4gram share")?;
    let min_entropy_q10 = parse_positive(fields[14], "min_entropy_q10")?;
    let min_utf8_valid_per_mille = parse_per_mille(fields[15], "UTF-8 validity")?;
    let min_context_use_per_mille = parse_per_mille(fields[16], "context use")?;
    let min_distractor_resistance_per_mille = parse_per_mille(fields[17], "distractor resistance")?;
    let min_human_preference_delta_per_mille = fields[18]
        .parse::<i32>()
        .map_err(|_| "human preference delta must be an integer".to_string())?;
    if !(-1000..=1000).contains(&min_human_preference_delta_per_mille) {
        return Err("human preference delta must be in -1000..1000".to_string());
    }
    validate_development_panel(&panel, prompt_count, max_prompt_bytes, generation_tokens)?;
    Ok(OpenGenerationManifest {
        manifest_path: path.to_path_buf(),
        tokenizer_path,
        tokenizer_hash: fields[3].to_string(),
        development_panel_path,
        development_panel_hash: fields[5].to_string(),
        hidden_test_sha256: fields[6].to_string(),
        prompt_count,
        max_prompt_bytes,
        generation_tokens,
        sampling_seeds,
        retained_improvement_per_mille,
        max_repeat_4gram_share_per_mille,
        min_unique_4gram_share_per_mille,
        min_entropy_q10,
        min_utf8_valid_per_mille,
        min_context_use_per_mille,
        min_distractor_resistance_per_mille,
        min_human_preference_delta_per_mille,
    })
}

pub fn load_open_generation_development_panel(
    manifest: &OpenGenerationManifest,
) -> Result<Vec<OpenGenerationPrompt>, String> {
    let panel = fs::read(&manifest.development_panel_path).map_err(|error| {
        format!(
            "cannot read development panel {}: {error}",
            manifest.development_panel_path.display()
        )
    })?;
    if fnv_hash(&panel) != manifest.development_panel_hash {
        return Err("open-generation development panel hash mismatch".to_string());
    }
    parse_development_panel(
        &panel,
        manifest.prompt_count,
        manifest.max_prompt_bytes,
        manifest.generation_tokens,
    )
}

fn validate_development_panel(
    bytes: &[u8],
    expected_prompts: usize,
    max_prompt_bytes: usize,
    generation_tokens: usize,
) -> Result<(), String> {
    parse_development_panel(bytes, expected_prompts, max_prompt_bytes, generation_tokens)
        .map(|_| ())
}

fn parse_development_panel(
    bytes: &[u8],
    expected_prompts: usize,
    max_prompt_bytes: usize,
    generation_tokens: usize,
) -> Result<Vec<OpenGenerationPrompt>, String> {
    let input = core::str::from_utf8(bytes)
        .map_err(|_| "development panel must be valid UTF-8 TSV".to_string())?;
    let mut lines = input.lines();
    if lines.next() != Some(OPEN_GENERATION_PANEL_HEADER) {
        return Err("development panel header is invalid".to_string());
    }
    let mut ids = HashSet::new();
    let mut categories = HashSet::new();
    let mut prompts = Vec::with_capacity(expected_prompts);
    for (index, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != 8
            || fields[0] != OPEN_GENERATION_PROMPT_SCHEMA
            || fields[1] != OPEN_GENERATION_CONTRACT_ID
            || fields[2] != "development"
        {
            return Err(format!("development prompt row {} is invalid", index + 2));
        }
        if fields[3].is_empty()
            || !fields[3]
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
            || !ids.insert(fields[3])
        {
            return Err(format!(
                "development prompt id {} is invalid or duplicate",
                fields[3]
            ));
        }
        if !REQUIRED_CATEGORIES.contains(&fields[4]) {
            return Err(format!("unknown development prompt category {}", fields[4]));
        }
        categories.insert(fields[4]);
        let max_new_tokens = parse_positive(fields[5], "max_new_tokens")?;
        if max_new_tokens < generation_tokens {
            return Err(
                "every development prompt must permit the frozen generation length".to_string(),
            );
        }
        let required_phrase = if fields[6] == "-" {
            Vec::new()
        } else {
            let required = decode_hex(fields[6], "required phrase")?;
            if required.is_empty() || core::str::from_utf8(&required).is_err() {
                return Err("required phrase must be non-empty UTF-8".to_string());
            }
            required
        };
        let prompt = decode_hex(fields[7], "prompt")?;
        if prompt.is_empty()
            || prompt.len() > max_prompt_bytes
            || core::str::from_utf8(&prompt).is_err()
        {
            return Err("prompt must be non-empty UTF-8 within max_prompt_bytes".to_string());
        }
        prompts.push(OpenGenerationPrompt {
            id: fields[3].to_string(),
            category: fields[4].to_string(),
            max_new_tokens,
            required_phrase,
            prompt,
        });
    }
    if prompts.len() != expected_prompts {
        return Err(format!(
            "development panel has {} prompts, expected {expected_prompts}",
            prompts.len()
        ));
    }
    if REQUIRED_CATEGORIES
        .iter()
        .any(|category| !categories.contains(category))
    {
        return Err("development panel is missing a required category".to_string());
    }
    Ok(prompts)
}

fn resolve_path(directory: &Path, value: &str, field: &str) -> Result<PathBuf, String> {
    let relative = Path::new(value);
    if relative.is_absolute()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!(
            "open-generation {field} must be a simple relative path"
        ));
    }
    Ok(directory.join(relative))
}

fn decode_hex(value: &str, field: &str) -> Result<Vec<u8>, String> {
    if !value.len().is_multiple_of(2) || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("{field} must be lowercase hexadecimal"));
    }
    (0..value.len())
        .step_by(2)
        .map(|index| {
            u8::from_str_radix(&value[index..index + 2], 16).map_err(|_| format!("bad {field}"))
        })
        .collect()
}

fn parse_seeds(value: &str) -> Result<Vec<u64>, String> {
    let seeds = value
        .split(',')
        .map(|seed| {
            seed.parse::<u64>()
                .map_err(|_| "sampling seeds must be integers".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    if seeds.len() < 2 || seeds.iter().copied().collect::<HashSet<_>>().len() != seeds.len() {
        return Err("sampling seeds must contain at least two distinct values".to_string());
    }
    Ok(seeds)
}

fn parse_positive(value: &str, field: &str) -> Result<usize, String> {
    let value = value
        .parse::<usize>()
        .map_err(|_| format!("{field} must be positive"))?;
    (value > 0)
        .then_some(value)
        .ok_or_else(|| format!("{field} must be positive"))
}

fn parse_per_mille(value: &str, field: &str) -> Result<usize, String> {
    let value = value
        .parse::<usize>()
        .map_err(|_| format!("{field} must be an integer"))?;
    (value <= 1000)
        .then_some(value)
        .ok_or_else(|| format!("{field} must be <= 1000"))
}

fn validate_fnv_hash(value: &str, field: &str) -> Result<(), String> {
    if value.len() != 18
        || !value.starts_with("0x")
        || !value[2..].bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(format!("{field} must be a 64-bit hexadecimal hash"));
    }
    Ok(())
}

fn validate_sha256(value: &str, field: &str) -> Result<(), String> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("{field} must be a SHA-256 hexadecimal digest"));
    }
    Ok(())
}

fn fnv_hash(bytes: &[u8]) -> String {
    let hash = bytes.iter().fold(FNV_OFFSET, |mut hash, &byte| {
        hash ^= u64::from(byte);
        hash.wrapping_mul(FNV_PRIME)
    });
    format!("0x{hash:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(id: &str, category: &str) -> String {
        format!(
            "{OPEN_GENERATION_PROMPT_SCHEMA}\t{OPEN_GENERATION_CONTRACT_ID}\tdevelopment\t{id}\t{category}\t512\t-\t74657374"
        )
    }

    fn valid_panel() -> Vec<u8> {
        let mut rows = vec![OPEN_GENERATION_PANEL_HEADER.to_string()];
        for (index, category) in REQUIRED_CATEGORIES.iter().enumerate() {
            rows.push(row(&format!("prompt-{index}"), category));
        }
        format!("{}\n", rows.join("\n")).into_bytes()
    }

    #[test]
    fn contract_locks_unassisted_generation() {
        let contract = open_generation_contract_json_line();
        assert!(contract.contains("native_unassisted"));
        assert!(contract.contains("same-shape-float-twin"));
        assert!(contract.contains("memory-injection"));
    }

    #[test]
    fn panel_requires_all_categories_and_unique_ids() {
        let panel = valid_panel();
        assert!(validate_development_panel(&panel, 6, 128, 512).is_ok());
        let missing = panel
            .split(|&byte| byte == b'\n')
            .take(6)
            .collect::<Vec<_>>()
            .join(&b'\n');
        assert!(validate_development_panel(&missing, 5, 128, 512).is_err());
    }

    #[test]
    fn panel_rejects_short_generation_and_bad_hex() {
        let short = valid_panel();
        let short = String::from_utf8(short)
            .expect("UTF-8 panel")
            .replace("\t512\t", "\t511\t");
        assert!(validate_development_panel(short.as_bytes(), 6, 128, 512).is_err());
        assert!(decode_hex("xyz", "fixture").is_err());
    }
}
