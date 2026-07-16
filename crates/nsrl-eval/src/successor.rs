use std::collections::HashSet;
use std::fs;
use std::path::{Component, Path, PathBuf};

use crate::contract::{proof_dataset_hash, proof_target_count};
use crate::json_escape;

pub const SUCCESSOR_CONTRACT_SCHEMA: &str = "nsrl.integer_transformer_successor_contract.v2";
pub const SUCCESSOR_MANIFEST_SCHEMA: &str = "nsrl.integer_transformer_successor_manifest.v2";
pub const SUCCESSOR_RESULT_SCHEMA: &str = "nsrl.integer_transformer_successor_result.v2";
pub const SUCCESSOR_CONTRACT_ID: &str = "integer-transformer-successor-v2";
pub const SUCCESSOR_FROZEN_DATASET_HASH: &str = "0x8fe7b86378f81951";
pub const SUCCESSOR_FROZEN_TARGETS: usize = 5_896;
pub const SUCCESSOR_TOKENIZER: &str = "byte_identity_u8_v1";
pub const SUCCESSOR_RESULTS_HEADER: &str = "schema\tcontract\tsuite\tpartition\tdataset_hash\ttokenizer_hash\tcandidate_model_hash\tevaluator_hash\trunner_hash\tsystem\ttargets\tmistakes\ttotal_nll_millibits\tzero_probability_windows\tsuffix_memory\tretrieval_assistance\trouting_oracle\treplay_hash";
pub const SUCCESSOR_MANIFEST_HEADER: &str = "schema\tcontract\ttrain\teval\tcandidate\tcontext\tstride\ttargets\tdataset_hash\ttokenizer\ttokenizer_hash\tcandidate_model_hash\tcandidate_artifact_hash\tevaluator_hash\trunner_hash\tmatrix_hash\tevidence_hash\ttransformer_replay_hash\tuniform_replay_hash\tretrieval_replay_hash\tbyte_ngram_replay_hash\tfloat_transformer_replay_hash";

const FNV64_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV64_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SuccessorSystem {
    TransformerOnly,
    Uniform,
    Retrieval,
    ByteNgram,
    FloatTransformer,
}

impl SuccessorSystem {
    pub const REQUIRED: [Self; 5] = [
        Self::TransformerOnly,
        Self::Uniform,
        Self::Retrieval,
        Self::ByteNgram,
        Self::FloatTransformer,
    ];
    pub const BASELINES: [Self; 4] = [
        Self::Uniform,
        Self::Retrieval,
        Self::ByteNgram,
        Self::FloatTransformer,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TransformerOnly => "transformer-only",
            Self::Uniform => "uniform",
            Self::Retrieval => "retrieval",
            Self::ByteNgram => "byte-ngram",
            Self::FloatTransformer => "float-transformer",
        }
    }

    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "transformer-only" => Ok(Self::TransformerOnly),
            "uniform" => Ok(Self::Uniform),
            "retrieval" => Ok(Self::Retrieval),
            "byte-ngram" => Ok(Self::ByteNgram),
            "float-transformer" => Ok(Self::FloatTransformer),
            _ => Err(format!("unknown successor system {value}")),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SuccessorManifest {
    pub manifest_path: PathBuf,
    pub train_path: PathBuf,
    pub eval_path: PathBuf,
    pub candidate_path: PathBuf,
    pub context: usize,
    pub stride: usize,
    pub targets: usize,
    pub dataset_hash: String,
    pub tokenizer: String,
    pub tokenizer_hash: String,
    pub candidate_model_hash: String,
    pub candidate_artifact_hash: String,
    pub evaluator_hash: String,
    pub runner_hash: String,
    pub matrix_hash: String,
    pub evidence_hash: String,
    pub replay_hashes: [String; 5],
}

impl SuccessorManifest {
    pub fn replay_hash(&self, system: SuccessorSystem) -> &str {
        &self.replay_hashes[system_index(system)]
    }

    pub fn to_json_line(&self) -> String {
        format!(
            concat!(
                "{{\"schema\":\"{}\",\"contract\":\"{}\",",
                "\"manifest\":\"{}\",\"train\":\"{}\",\"eval\":\"{}\",",
                "\"candidate\":\"{}\",\"context\":{},\"stride\":{},\"targets\":{},",
                "\"dataset_hash\":\"{}\",\"tokenizer\":\"{}\",\"tokenizer_hash\":\"{}\",",
                "\"candidate_model_hash\":\"{}\",\"candidate_artifact_hash\":\"{}\",",
                "\"evaluator_hash\":\"{}\",\"runner_hash\":\"{}\",",
                "\"matrix_hash\":\"{}\",\"evidence_hash\":\"{}\",\"valid\":true}}\n"
            ),
            SUCCESSOR_MANIFEST_SCHEMA,
            SUCCESSOR_CONTRACT_ID,
            json_escape(&self.manifest_path.to_string_lossy()),
            json_escape(&self.train_path.to_string_lossy()),
            json_escape(&self.eval_path.to_string_lossy()),
            json_escape(&self.candidate_path.to_string_lossy()),
            self.context,
            self.stride,
            self.targets,
            self.dataset_hash,
            self.tokenizer,
            self.tokenizer_hash,
            self.candidate_model_hash,
            self.candidate_artifact_hash,
            self.evaluator_hash,
            self.runner_hash,
            self.matrix_hash,
            self.evidence_hash,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SuccessorResult {
    pub system: SuccessorSystem,
    pub targets: usize,
    pub mistakes: usize,
    pub total_nll_millibits: u64,
    pub zero_probability_windows: usize,
    pub replay_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SuccessorCheck {
    pub dataset_hash: String,
    pub targets: usize,
    pub candidate: SuccessorResult,
    pub baselines: Vec<SuccessorResult>,
    pub matrix_hash: String,
    pub evidence_hash: String,
    pub passed: bool,
}

impl SuccessorCheck {
    pub fn to_json_line(&self) -> String {
        let baselines = self
            .baselines
            .iter()
            .map(result_json)
            .collect::<Vec<_>>()
            .join(",");
        format!(
            concat!(
                "{{\"schema\":\"{}\",\"contract\":\"{}\",",
                "\"dataset_hash\":\"{}\",\"targets\":{},",
                "\"objective\":\"integer_base2_softmax_nll_millibits\",",
                "\"matrix_hash\":\"{}\",\"evidence_hash\":\"{}\",",
                "\"candidate\":{},\"baselines\":[{}],\"passed\":{}}}\n"
            ),
            SUCCESSOR_RESULT_SCHEMA,
            SUCCESSOR_CONTRACT_ID,
            json_escape(&self.dataset_hash),
            self.targets,
            self.matrix_hash,
            self.evidence_hash,
            result_json(&self.candidate),
            baselines,
            self.passed,
        )
    }
}

pub fn successor_contract_json_line() -> String {
    format!(
        concat!(
            "{{\"schema\":\"{}\",\"contract\":\"{}\",",
            "\"frozen_dataset_hash\":\"{}\",\"frozen_targets\":{},",
            "\"tokenizer\":\"{}\",\"candidate\":\"transformer-only\",",
            "\"required_baselines\":[\"uniform\",\"retrieval\",\"byte-ngram\",\"float-transformer\"],",
            "\"primary_metric\":\"total_nll_millibits\",",
            "\"objective\":\"integer_base2_softmax_nll_millibits\",",
            "\"objective_properties\":{{\"base_score\":\"negative_log_likelihood\",",
            "\"normalization_independent\":true,\"logit_shift_invariant\":true,",
            "\"zero_probability_floor_millibits\":32000}},",
            "\"secondary_metric\":\"mistakes\",\"secondary_is_gate\":false,",
            "\"pass_rule\":\"transformer_only_nll_strictly_lower_than_every_baseline\",",
            "\"forbidden_candidate_assistance\":[\"suffix-memory\",\"retrieval\",\"routing-oracle\"],",
            "\"integrity\":[\"candidate_artifact_hash\",\"evaluator_hash\",\"runner_hash\",",
            "\"exact_matrix_hash\",\"evidence_hash\",\"per_system_replay_hashes\"]}}\n"
        ),
        SUCCESSOR_CONTRACT_SCHEMA,
        SUCCESSOR_CONTRACT_ID,
        SUCCESSOR_FROZEN_DATASET_HASH,
        SUCCESSOR_FROZEN_TARGETS,
        SUCCESSOR_TOKENIZER,
    )
}

pub fn load_successor_manifest(path: &Path) -> Result<SuccessorManifest, String> {
    let input = fs::read_to_string(path)
        .map_err(|error| format!("cannot read successor manifest {}: {error}", path.display()))?;
    let mut lines = input.lines();
    if lines.next() != Some(SUCCESSOR_MANIFEST_HEADER) {
        return Err(format!(
            "successor manifest header must be {SUCCESSOR_MANIFEST_HEADER}"
        ));
    }
    let row = lines.next().ok_or("successor manifest row is missing")?;
    if lines.any(|line| !line.trim().is_empty()) {
        return Err("successor manifest must contain exactly one row".to_string());
    }
    let fields = row.split('\t').collect::<Vec<_>>();
    if fields.len() != 22
        || fields[0] != SUCCESSOR_MANIFEST_SCHEMA
        || fields[1] != SUCCESSOR_CONTRACT_ID
    {
        return Err("successor manifest schema or contract is invalid".to_string());
    }
    let directory = path.parent().unwrap_or_else(|| Path::new("."));
    let train_path = resolve_manifest_path(directory, fields[2], "train")?;
    let eval_path = resolve_manifest_path(directory, fields[3], "eval")?;
    let candidate_path = resolve_manifest_path(directory, fields[4], "candidate")?;
    let context = parse_positive(fields[5], "context")?;
    let stride = parse_positive(fields[6], "stride")?;
    let targets = parse_positive(fields[7], "targets")?;
    if targets != SUCCESSOR_FROZEN_TARGETS {
        return Err(format!(
            "successor target count {targets} does not equal frozen count {SUCCESSOR_FROZEN_TARGETS}"
        ));
    }
    for (field, name) in [
        (fields[8], "dataset_hash"),
        (fields[10], "tokenizer_hash"),
        (fields[11], "candidate_model_hash"),
        (fields[12], "candidate_artifact_hash"),
        (fields[13], "evaluator_hash"),
        (fields[14], "runner_hash"),
        (fields[15], "matrix_hash"),
        (fields[16], "evidence_hash"),
        (fields[17], "transformer_replay_hash"),
        (fields[18], "uniform_replay_hash"),
        (fields[19], "retrieval_replay_hash"),
        (fields[20], "byte_ngram_replay_hash"),
        (fields[21], "float_transformer_replay_hash"),
    ] {
        validate_hash(field, name)?;
    }
    if fields[8] != SUCCESSOR_FROZEN_DATASET_HASH {
        return Err(format!(
            "successor dataset hash {} does not equal frozen hash {SUCCESSOR_FROZEN_DATASET_HASH}",
            fields[8]
        ));
    }
    if fields[9] != SUCCESSOR_TOKENIZER {
        return Err(format!(
            "successor tokenizer {} does not equal frozen tokenizer {SUCCESSOR_TOKENIZER}",
            fields[9]
        ));
    }
    let expected_tokenizer_hash = fnv64_hex(SUCCESSOR_TOKENIZER.as_bytes());
    if fields[10] != expected_tokenizer_hash {
        return Err(format!(
            "successor tokenizer_hash {} does not match {expected_tokenizer_hash}",
            fields[10]
        ));
    }
    let train = fs::read(&train_path).map_err(|error| {
        format!(
            "cannot read successor train {}: {error}",
            train_path.display()
        )
    })?;
    let eval = fs::read(&eval_path).map_err(|error| {
        format!(
            "cannot read successor eval {}: {error}",
            eval_path.display()
        )
    })?;
    let dataset_hash = proof_dataset_hash(&train, &eval);
    if dataset_hash != fields[8] {
        return Err(format!(
            "successor dataset_hash {} does not match corpus hash {dataset_hash}",
            fields[8]
        ));
    }
    let actual_targets = proof_target_count(eval.len(), context, stride);
    if actual_targets != targets {
        return Err(format!(
            "successor target count {targets} does not match evaluation surface {actual_targets}"
        ));
    }
    let candidate = fs::read(&candidate_path).map_err(|error| {
        format!(
            "cannot read successor candidate {}: {error}",
            candidate_path.display()
        )
    })?;
    let candidate_artifact_hash = fnv64_hex(&candidate);
    if candidate_artifact_hash != fields[12] {
        return Err(format!(
            "successor candidate_artifact_hash {} does not match {candidate_artifact_hash}",
            fields[12]
        ));
    }
    Ok(SuccessorManifest {
        manifest_path: path.to_path_buf(),
        train_path,
        eval_path,
        candidate_path,
        context,
        stride,
        targets,
        dataset_hash,
        tokenizer: fields[9].to_string(),
        tokenizer_hash: fields[10].to_string(),
        candidate_model_hash: fields[11].to_string(),
        candidate_artifact_hash,
        evaluator_hash: fields[13].to_string(),
        runner_hash: fields[14].to_string(),
        matrix_hash: fields[15].to_string(),
        evidence_hash: fields[16].to_string(),
        replay_hashes: fields[17..22]
            .iter()
            .map(|value| (*value).to_string())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap(),
    })
}

pub fn check_successor_results(
    input: &str,
    evidence: &[u8],
    manifest: &SuccessorManifest,
) -> Result<SuccessorCheck, String> {
    let actual_matrix_hash = fnv64_hex(input.as_bytes());
    if actual_matrix_hash != manifest.matrix_hash {
        return Err(format!(
            "successor matrix hash {actual_matrix_hash} does not match frozen hash {}",
            manifest.matrix_hash
        ));
    }
    let actual_evidence_hash = fnv64_hex(evidence);
    if actual_evidence_hash != manifest.evidence_hash {
        return Err(format!(
            "successor evidence hash {actual_evidence_hash} does not match frozen hash {}",
            manifest.evidence_hash
        ));
    }
    let mut lines = input.lines();
    if lines.next() != Some(SUCCESSOR_RESULTS_HEADER) {
        return Err("successor results header mismatch".to_string());
    }
    let mut systems = HashSet::new();
    let mut results = Vec::new();
    for (line_index, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != 18 {
            return Err(format!(
                "successor result line {} has {} fields, expected 18",
                line_index + 2,
                fields.len()
            ));
        }
        if fields[0] != SUCCESSOR_RESULT_SCHEMA
            || fields[1] != SUCCESSOR_CONTRACT_ID
            || fields[2] != "substrate"
            || fields[3] != "eval"
        {
            return Err(format!(
                "successor result line {} has invalid contract metadata",
                line_index + 2
            ));
        }
        for (actual, expected, name) in [
            (fields[4], manifest.dataset_hash.as_str(), "dataset_hash"),
            (
                fields[5],
                manifest.tokenizer_hash.as_str(),
                "tokenizer_hash",
            ),
            (
                fields[6],
                manifest.candidate_model_hash.as_str(),
                "candidate_model_hash",
            ),
            (
                fields[7],
                manifest.evaluator_hash.as_str(),
                "evaluator_hash",
            ),
            (fields[8], manifest.runner_hash.as_str(), "runner_hash"),
        ] {
            if actual != expected {
                return Err(format!(
                    "successor result line {} {name} {actual} does not match frozen {expected}",
                    line_index + 2
                ));
            }
        }
        let system = SuccessorSystem::parse(fields[9])?;
        if !systems.insert(system) {
            return Err(format!("duplicate successor system {}", system.as_str()));
        }
        let targets = parse_positive(fields[10], "targets")?;
        if targets != manifest.targets {
            return Err(format!(
                "successor result line {} targets {targets} do not match frozen {}",
                line_index + 2,
                manifest.targets
            ));
        }
        let mistakes = parse_nonnegative(fields[11], "mistakes")?;
        let zero_probability_windows = parse_nonnegative(fields[13], "zero_probability_windows")?;
        if mistakes > targets || zero_probability_windows > targets {
            return Err(format!(
                "successor result line {} has a count larger than targets",
                line_index + 2
            ));
        }
        if fields[14] != "false" || fields[15] != "false" || fields[16] != "false" {
            return Err(format!(
                "successor result line {} declares forbidden candidate assistance",
                line_index + 2
            ));
        }
        let replay_hash = fields[17].to_ascii_lowercase();
        if replay_hash != manifest.replay_hash(system) {
            return Err(format!(
                "successor result line {} replay hash {} does not match frozen {}",
                line_index + 2,
                replay_hash,
                manifest.replay_hash(system)
            ));
        }
        results.push(SuccessorResult {
            system,
            targets,
            mistakes,
            total_nll_millibits: fields[12]
                .parse()
                .map_err(|_| "total_nll_millibits must be a non-negative integer".to_string())?,
            zero_probability_windows,
            replay_hash,
        });
    }
    for required in SuccessorSystem::REQUIRED {
        if !systems.contains(&required) {
            return Err(format!(
                "missing required successor system {}",
                required.as_str()
            ));
        }
    }
    if results.len() != SuccessorSystem::REQUIRED.len() {
        return Err("successor results must contain exactly five systems".to_string());
    }
    let candidate = results
        .iter()
        .find(|result| result.system == SuccessorSystem::TransformerOnly)
        .cloned()
        .ok_or("missing transformer-only candidate")?;
    let baselines = SuccessorSystem::BASELINES
        .iter()
        .map(|system| {
            results
                .iter()
                .find(|result| result.system == *system)
                .cloned()
                .ok_or_else(|| format!("missing baseline {}", system.as_str()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let passed = baselines
        .iter()
        .all(|baseline| candidate.total_nll_millibits < baseline.total_nll_millibits);
    Ok(SuccessorCheck {
        dataset_hash: manifest.dataset_hash.clone(),
        targets: manifest.targets,
        candidate,
        baselines,
        matrix_hash: actual_matrix_hash,
        evidence_hash: actual_evidence_hash,
        passed,
    })
}

pub fn fnv64_hex(bytes: &[u8]) -> String {
    let mut hash = FNV64_OFFSET;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV64_PRIME);
    }
    format!("0x{hash:016x}")
}

fn system_index(system: SuccessorSystem) -> usize {
    match system {
        SuccessorSystem::TransformerOnly => 0,
        SuccessorSystem::Uniform => 1,
        SuccessorSystem::Retrieval => 2,
        SuccessorSystem::ByteNgram => 3,
        SuccessorSystem::FloatTransformer => 4,
    }
}

fn parse_positive(value: &str, field: &str) -> Result<usize, String> {
    let parsed = parse_nonnegative(value, field)?;
    if parsed == 0 {
        return Err(format!("{field} must be positive"));
    }
    Ok(parsed)
}

fn parse_nonnegative(value: &str, field: &str) -> Result<usize, String> {
    value
        .parse()
        .map_err(|_| format!("{field} must be a non-negative integer"))
}

fn validate_hash(value: &str, field: &str) -> Result<(), String> {
    if value.len() != 18
        || !value.starts_with("0x")
        || !value[2..].bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(format!(
            "{field} must be 0x followed by 16 hexadecimal digits"
        ));
    }
    Ok(())
}

fn resolve_manifest_path(directory: &Path, value: &str, field: &str) -> Result<PathBuf, String> {
    let relative = Path::new(value);
    if relative.is_absolute()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!(
            "successor manifest {field} path must be a simple relative path"
        ));
    }
    Ok(directory.join(relative))
}

fn result_json(result: &SuccessorResult) -> String {
    format!(
        "{{\"system\":\"{}\",\"targets\":{},\"mistakes\":{},\"total_nll_millibits\":{},\"zero_probability_windows\":{},\"replay_hash\":\"{}\"}}",
        result.system.as_str(),
        result.targets,
        result.mistakes,
        result.total_nll_millibits,
        result.zero_probability_windows,
        json_escape(&result.replay_hash),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result_row(system: &str, nll: u64) -> String {
        format!(
            "{SUCCESSOR_RESULT_SCHEMA}\t{SUCCESSOR_CONTRACT_ID}\tsubstrate\teval\t0x8fe7b86378f81951\t0x1111111111111111\t0x2222222222222222\t0x3333333333333333\t0x4444444444444444\t{system}\t5896\t40\t{nll}\t0\tfalse\tfalse\tfalse\t0x5555555555555555"
        )
    }

    fn input_with(candidate: String) -> String {
        [
            SUCCESSOR_RESULTS_HEADER.to_string(),
            candidate,
            result_row("uniform", 100_000),
            result_row("retrieval", 101_000),
            result_row("byte-ngram", 102_000),
            result_row("float-transformer", 103_000),
        ]
        .join("\n")
            + "\n"
    }

    fn manifest_for(input: &str, evidence: &[u8]) -> SuccessorManifest {
        SuccessorManifest {
            manifest_path: PathBuf::new(),
            train_path: PathBuf::new(),
            eval_path: PathBuf::new(),
            candidate_path: PathBuf::new(),
            context: 64,
            stride: 1,
            targets: SUCCESSOR_FROZEN_TARGETS,
            dataset_hash: SUCCESSOR_FROZEN_DATASET_HASH.to_string(),
            tokenizer: SUCCESSOR_TOKENIZER.to_string(),
            tokenizer_hash: "0x1111111111111111".to_string(),
            candidate_model_hash: "0x2222222222222222".to_string(),
            candidate_artifact_hash: "0xaaaaaaaaaaaaaaaa".to_string(),
            evaluator_hash: "0x3333333333333333".to_string(),
            runner_hash: "0x4444444444444444".to_string(),
            matrix_hash: fnv64_hex(input.as_bytes()),
            evidence_hash: fnv64_hex(evidence),
            replay_hashes: std::array::from_fn(|_| "0x5555555555555555".to_string()),
        }
    }

    #[test]
    fn successor_contract_promotes_canonical_nll_and_transformer_only() {
        let contract = successor_contract_json_line();
        assert!(contract.contains("integer_base2_softmax_nll_millibits"));
        assert!(contract.contains("\"candidate\":\"transformer-only\""));
        assert!(contract.contains("\"float-transformer\""));
        assert!(contract.contains(SUCCESSOR_FROZEN_DATASET_HASH));
        assert!(contract.contains("5896"));
        assert!(!contract.contains("probability_error_q15"));
    }

    #[test]
    fn successor_check_uses_nll_as_the_only_promotion_gate() {
        let evidence = b"evidence";
        let input = input_with(result_row("transformer-only", 99_000));
        let manifest = manifest_for(&input, evidence);
        let check =
            check_successor_results(&input, evidence, &manifest).expect("valid successor results");
        assert!(check.passed);
        assert_eq!(check.candidate.system, SuccessorSystem::TransformerOnly);
        assert_eq!(check.baselines.len(), 4);
    }

    #[test]
    fn successor_check_rejects_fabricated_metrics_even_with_valid_rows() {
        let evidence = b"evidence";
        let frozen = input_with(result_row("transformer-only", 99_000));
        let manifest = manifest_for(&frozen, evidence);
        let fabricated = frozen.replacen("\t99000\t", "\t1\t", 1);
        assert!(check_successor_results(&fabricated, evidence, &manifest).is_err());
    }

    #[test]
    fn successor_check_rejects_wrong_dataset() {
        let evidence = b"evidence";
        let input = input_with(result_row("transformer-only", 99_000));
        let fabricated = input.replacen(SUCCESSOR_FROZEN_DATASET_HASH, "0x0123456789abcdef", 1);
        let manifest = manifest_for(&fabricated, evidence);
        let mut frozen = manifest.clone();
        frozen.dataset_hash = SUCCESSOR_FROZEN_DATASET_HASH.to_string();
        assert!(check_successor_results(&fabricated, evidence, &frozen).is_err());
    }

    #[test]
    fn successor_check_rejects_wrong_target_count() {
        let evidence = b"evidence";
        let input = input_with(result_row("transformer-only", 99_000));
        let fabricated = input.replacen("\t5896\t", "\t5895\t", 1);
        let manifest = manifest_for(&fabricated, evidence);
        assert!(check_successor_results(&fabricated, evidence, &manifest).is_err());
    }

    #[test]
    fn successor_check_rejects_wrong_candidate_model_hash() {
        let evidence = b"evidence";
        let input = input_with(result_row("transformer-only", 99_000));
        let fabricated = input.replacen("0x2222222222222222", "0x9999999999999999", 1);
        let manifest = manifest_for(&fabricated, evidence);
        assert!(check_successor_results(&fabricated, evidence, &manifest).is_err());
    }

    #[test]
    fn successor_check_rejects_assisted_candidate() {
        let evidence = b"evidence";
        let input = input_with(result_row("transformer-only", 99_000));
        for field in ["suffix", "retrieval", "routing"] {
            let fabricated = match field {
                "suffix" => input.replacen("\tfalse\tfalse\tfalse\t", "\ttrue\tfalse\tfalse\t", 1),
                "retrieval" => {
                    input.replacen("\tfalse\tfalse\tfalse\t", "\tfalse\ttrue\tfalse\t", 1)
                }
                _ => input.replacen("\tfalse\tfalse\tfalse\t", "\tfalse\tfalse\ttrue\t", 1),
            };
            let manifest = manifest_for(&fabricated, evidence);
            assert!(check_successor_results(&fabricated, evidence, &manifest).is_err());
        }
    }

    #[test]
    fn successor_check_requires_a_real_float_transformer_baseline() {
        let evidence = b"evidence";
        let input = input_with(result_row("transformer-only", 99_000))
            .replace("\tfloat-transformer\t", "\tfloat-reference\t");
        let manifest = manifest_for(&input, evidence);
        assert!(check_successor_results(&input, evidence, &manifest).is_err());
    }
}
