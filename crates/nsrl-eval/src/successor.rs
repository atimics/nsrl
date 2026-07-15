use std::collections::HashSet;
use std::fs;
use std::path::{Component, Path, PathBuf};

use crate::json_escape;

pub const SUCCESSOR_CONTRACT_SCHEMA: &str = "nsrl.integer_transformer_successor_contract.v2";
pub const SUCCESSOR_MANIFEST_SCHEMA: &str = "nsrl.integer_transformer_successor_manifest.v2";
pub const SUCCESSOR_RESULT_SCHEMA: &str = "nsrl.integer_transformer_successor_result.v2";
pub const SUCCESSOR_CONTRACT_ID: &str = "integer-transformer-successor-v2";
pub const SUCCESSOR_DATASET_HASH: &str = "0x8fe7b86378f81951";
pub const SUCCESSOR_TARGETS: usize = 5_896;
pub const SUCCESSOR_ASSISTANCE: &str = "suffix-memory=off,retrieval=off,routing-oracle=off";
pub const SUCCESSOR_MANIFEST_HEADER: &str = "schema\tcontract\ttrain\teval\tcontext\tstride\ttargets\tdataset_hash\tcandidate\tcandidate_artifact_hash\tcandidate_hash\tmodel_hash\trunner\trunner_hash\tassistance\tfloat_model\tfloat_model_hash\tfloat_runner\tfloat_runner_hash";
pub const SUCCESSOR_RESULTS_HEADER: &str = "schema\tcontract\tsuite\tpartition\tdataset_hash\tcandidate_hash\tmodel_hash\trunner_hash\tassistance_hash\tsystem\ttargets\tmistakes\ttotal_nll_millibits\tzero_probability_windows\treplay_hash";

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
    pub context: usize,
    pub stride: usize,
    pub targets: usize,
    pub dataset_hash: String,
    pub candidate_path: PathBuf,
    pub candidate_artifact_hash: String,
    pub candidate_hash: String,
    pub model_hash: String,
    pub runner_path: PathBuf,
    pub runner_hash: String,
    pub assistance: String,
    pub assistance_hash: String,
    pub float_model_path: PathBuf,
    pub float_model_hash: String,
    pub float_runner_path: PathBuf,
    pub float_runner_hash: String,
}

impl SuccessorManifest {
    pub fn to_json_line(&self) -> String {
        format!(
            concat!(
                "{{\"schema\":\"{}\",\"contract\":\"{}\",",
                "\"manifest\":\"{}\",\"train\":\"{}\",\"eval\":\"{}\",",
                "\"context\":{},\"stride\":{},\"targets\":{},\"dataset_hash\":\"{}\",",
                "\"candidate\":{{\"artifact\":\"{}\",\"artifact_hash\":\"{}\",",
                "\"candidate_hash\":\"{}\",\"model_hash\":\"{}\"}},",
                "\"runner\":{{\"path\":\"{}\",\"hash\":\"{}\"}},",
                "\"assistance\":{{\"profile\":\"{}\",\"hash\":\"{}\"}},",
                "\"float_transformer\":{{\"model\":\"{}\",\"model_hash\":\"{}\",",
                "\"runner\":\"{}\",\"runner_hash\":\"{}\"}},\"valid\":true}}\n"
            ),
            SUCCESSOR_MANIFEST_SCHEMA,
            SUCCESSOR_CONTRACT_ID,
            json_escape(&self.manifest_path.to_string_lossy()),
            json_escape(&self.train_path.to_string_lossy()),
            json_escape(&self.eval_path.to_string_lossy()),
            self.context,
            self.stride,
            self.targets,
            self.dataset_hash,
            json_escape(&self.candidate_path.to_string_lossy()),
            self.candidate_artifact_hash,
            self.candidate_hash,
            self.model_hash,
            json_escape(&self.runner_path.to_string_lossy()),
            self.runner_hash,
            json_escape(&self.assistance),
            self.assistance_hash,
            json_escape(&self.float_model_path.to_string_lossy()),
            self.float_model_hash,
            json_escape(&self.float_runner_path.to_string_lossy()),
            self.float_runner_hash,
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
    pub candidate_hash: String,
    pub model_hash: String,
    pub runner_hash: String,
    pub assistance_hash: String,
    pub float_model_hash: String,
    pub float_runner_hash: String,
    pub candidate: SuccessorResult,
    pub baselines: Vec<SuccessorResult>,
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
                "{{\"schema\":\"{}\",\"contract\":\"{}\",\"dataset_hash\":\"{}\",",
                "\"targets\":{},\"candidate_hash\":\"{}\",\"model_hash\":\"{}\",",
                "\"runner_hash\":\"{}\",\"assistance_hash\":\"{}\",",
                "\"float_model_hash\":\"{}\",\"float_runner_hash\":\"{}\",",
                "\"objective\":\"integer_base2_softmax_nll_millibits\",",
                "\"candidate\":{},\"baselines\":[{}],\"passed\":{}}}\n"
            ),
            SUCCESSOR_RESULT_SCHEMA,
            SUCCESSOR_CONTRACT_ID,
            json_escape(&self.dataset_hash),
            self.targets,
            self.candidate_hash,
            self.model_hash,
            self.runner_hash,
            self.assistance_hash,
            self.float_model_hash,
            self.float_runner_hash,
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
            "\"dataset_hash\":\"{}\",\"targets\":{},\"manifest_required\":true,",
            "\"candidate\":\"transformer-only\",",
            "\"required_baselines\":[\"uniform\",\"retrieval\",\"byte-ngram\",\"float-transformer\"],",
            "\"primary_metric\":\"total_nll_millibits\",",
            "\"objective\":\"integer_base2_softmax_nll_millibits\",",
            "\"objective_properties\":{{\"base_score\":\"negative_log_likelihood\",",
            "\"normalization_independent\":true,\"logit_shift_invariant\":true,",
            "\"zero_probability_floor_millibits\":32000}},",
            "\"secondary_metric\":\"mistakes\",\"secondary_is_gate\":false,",
            "\"pass_rule\":\"transformer_only_nll_strictly_lower_than_every_baseline\",",
            "\"required_bindings\":[\"candidate_hash\",\"model_hash\",\"runner_hash\"],",
            "\"required_assistance\":\"{}\"}}\n"
        ),
        SUCCESSOR_CONTRACT_SCHEMA,
        SUCCESSOR_CONTRACT_ID,
        SUCCESSOR_DATASET_HASH,
        SUCCESSOR_TARGETS,
        SUCCESSOR_ASSISTANCE,
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
    if fields.len() != 19
        || fields[0] != SUCCESSOR_MANIFEST_SCHEMA
        || fields[1] != SUCCESSOR_CONTRACT_ID
    {
        return Err("successor manifest schema or contract is invalid".to_string());
    }
    let directory = path.parent().unwrap_or_else(|| Path::new("."));
    let train_path = resolve_manifest_path(directory, fields[2], "train")?;
    let eval_path = resolve_manifest_path(directory, fields[3], "eval")?;
    let context = parse_positive(fields[4], "context")?;
    let stride = parse_positive(fields[5], "stride")?;
    let targets = parse_positive(fields[6], "targets")?;
    validate_hash(fields[7], "dataset_hash")?;
    let train = read_file(&train_path, "train corpus")?;
    let eval = read_file(&eval_path, "eval corpus")?;
    let dataset_hash = successor_dataset_hash(&train, &eval);
    if fields[7] != SUCCESSOR_DATASET_HASH || dataset_hash != SUCCESSOR_DATASET_HASH {
        return Err(format!(
            "successor dataset must be {SUCCESSOR_DATASET_HASH}, found manifest {} and corpus {dataset_hash}",
            fields[7]
        ));
    }
    let actual_targets = target_count(eval.len(), context, stride);
    if targets != SUCCESSOR_TARGETS || actual_targets != SUCCESSOR_TARGETS {
        return Err(format!(
            "successor evaluation must contain {SUCCESSOR_TARGETS} targets, found manifest {targets} and corpus {actual_targets}"
        ));
    }

    let candidate_path = resolve_manifest_path(directory, fields[8], "candidate")?;
    validate_hash(fields[9], "candidate_artifact_hash")?;
    validate_file_hash(&candidate_path, fields[9], "candidate")?;
    validate_hash(fields[10], "candidate_hash")?;
    validate_hash(fields[11], "model_hash")?;
    let runner_path = resolve_manifest_path(directory, fields[12], "runner")?;
    validate_hash(fields[13], "runner_hash")?;
    validate_file_hash(&runner_path, fields[13], "runner")?;
    if fields[14] != SUCCESSOR_ASSISTANCE {
        return Err(format!(
            "successor assistance must be {SUCCESSOR_ASSISTANCE}"
        ));
    }
    let float_model_path = resolve_manifest_path(directory, fields[15], "float_model")?;
    validate_hash(fields[16], "float_model_hash")?;
    validate_file_hash(&float_model_path, fields[16], "float model")?;
    let float_runner_path = resolve_manifest_path(directory, fields[17], "float_runner")?;
    validate_hash(fields[18], "float_runner_hash")?;
    validate_file_hash(&float_runner_path, fields[18], "float runner")?;

    Ok(SuccessorManifest {
        manifest_path: path.to_path_buf(),
        train_path,
        eval_path,
        context,
        stride,
        targets,
        dataset_hash,
        candidate_path,
        candidate_artifact_hash: fields[9].to_string(),
        candidate_hash: fields[10].to_string(),
        model_hash: fields[11].to_string(),
        runner_path,
        runner_hash: fields[13].to_string(),
        assistance: fields[14].to_string(),
        assistance_hash: stable_hex_hash(fields[14].as_bytes()),
        float_model_path,
        float_model_hash: fields[16].to_string(),
        float_runner_path,
        float_runner_hash: fields[18].to_string(),
    })
}

pub fn check_successor_results(
    input: &str,
    manifest: &SuccessorManifest,
) -> Result<SuccessorCheck, String> {
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
        if fields.len() != 15 {
            return Err(format!(
                "successor result line {} has {} fields, expected 15",
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
        let bindings = [
            (fields[4], manifest.dataset_hash.as_str(), "dataset_hash"),
            (
                fields[5],
                manifest.candidate_hash.as_str(),
                "candidate_hash",
            ),
            (fields[6], manifest.model_hash.as_str(), "model_hash"),
            (fields[7], manifest.runner_hash.as_str(), "runner_hash"),
            (
                fields[8],
                manifest.assistance_hash.as_str(),
                "assistance_hash",
            ),
        ];
        for (actual, expected, name) in bindings {
            if actual != expected {
                return Err(format!(
                    "successor result line {} {name} does not match the frozen manifest",
                    line_index + 2
                ));
            }
        }
        let system = SuccessorSystem::parse(fields[9])?;
        if !systems.insert(system) {
            return Err(format!("duplicate successor system {}", system.as_str()));
        }
        let targets = parse_usize(fields[10], "targets")?;
        if targets != manifest.targets {
            return Err(format!(
                "successor result line {} targets do not match the frozen manifest",
                line_index + 2
            ));
        }
        let mistakes = parse_usize(fields[11], "mistakes")?;
        let zero_probability_windows = parse_usize(fields[13], "zero_probability_windows")?;
        if mistakes > targets || zero_probability_windows > targets {
            return Err(format!(
                "successor result line {} has a count larger than targets",
                line_index + 2
            ));
        }
        validate_hash(fields[14], "replay_hash")?;
        results.push(SuccessorResult {
            system,
            targets,
            mistakes,
            total_nll_millibits: fields[12]
                .parse()
                .map_err(|_| "total_nll_millibits must be a non-negative integer".to_string())?,
            zero_probability_windows,
            replay_hash: fields[14].to_string(),
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
        candidate_hash: manifest.candidate_hash.clone(),
        model_hash: manifest.model_hash.clone(),
        runner_hash: manifest.runner_hash.clone(),
        assistance_hash: manifest.assistance_hash.clone(),
        float_model_hash: manifest.float_model_hash.clone(),
        float_runner_hash: manifest.float_runner_hash.clone(),
        candidate,
        baselines,
        passed,
    })
}

pub fn successor_dataset_hash(train: &[u8], eval: &[u8]) -> String {
    let mut hash = FNV64_OFFSET;
    for &byte in train.iter().chain([0xff].iter()).chain(eval.iter()) {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV64_PRIME);
    }
    format!("0x{hash:016x}")
}

pub fn stable_hex_hash(bytes: &[u8]) -> String {
    let mut hash = FNV64_OFFSET;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV64_PRIME);
    }
    format!("0x{hash:016x}")
}

fn validate_file_hash(path: &Path, expected: &str, name: &str) -> Result<(), String> {
    let bytes = read_file(path, name)?;
    let actual = stable_hex_hash(&bytes);
    if actual != expected {
        return Err(format!(
            "{name} hash {actual} does not match frozen {expected}"
        ));
    }
    Ok(())
}

fn read_file(path: &Path, name: &str) -> Result<Vec<u8>, String> {
    fs::read(path).map_err(|error| format!("cannot read {name} {}: {error}", path.display()))
}

fn target_count(bytes: usize, context: usize, stride: usize) -> usize {
    if bytes <= context {
        0
    } else {
        (bytes - context).div_ceil(stride)
    }
}

fn resolve_manifest_path(directory: &Path, value: &str, name: &str) -> Result<PathBuf, String> {
    let relative = Path::new(value);
    if relative.is_absolute()
        || relative
            .components()
            .any(|component| matches!(component, Component::RootDir | Component::Prefix(_)))
    {
        return Err(format!("successor manifest {name} path must be relative"));
    }
    Ok(directory.join(relative))
}

fn parse_positive(value: &str, field: &str) -> Result<usize, String> {
    let parsed = parse_usize(value, field)?;
    if parsed == 0 {
        return Err(format!("{field} must be positive"));
    }
    Ok(parsed)
}

fn parse_usize(value: &str, field: &str) -> Result<usize, String> {
    value
        .parse()
        .map_err(|_| format!("{field} must be a non-negative integer"))
}

fn validate_hash(value: &str, field: &str) -> Result<(), String> {
    if value.starts_with("0x")
        && value.len() == 18
        && value[2..].bytes().all(|byte| byte.is_ascii_hexdigit())
        && value == value.to_ascii_lowercase()
    {
        Ok(())
    } else {
        Err(format!(
            "{field} must be a lowercase 64-bit hexadecimal hash"
        ))
    }
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

    fn manifest() -> SuccessorManifest {
        SuccessorManifest {
            manifest_path: PathBuf::from("manifest.tsv"),
            train_path: PathBuf::from("train.txt"),
            eval_path: PathBuf::from("eval.txt"),
            context: 64,
            stride: 1,
            targets: SUCCESSOR_TARGETS,
            dataset_hash: SUCCESSOR_DATASET_HASH.to_string(),
            candidate_path: PathBuf::from("candidate.nsrlmt"),
            candidate_artifact_hash: "0xaaaaaaaaaaaaaaaa".to_string(),
            candidate_hash: "0xbbbbbbbbbbbbbbbb".to_string(),
            model_hash: "0xcccccccccccccccc".to_string(),
            runner_path: PathBuf::from("runner.mjs"),
            runner_hash: "0xdddddddddddddddd".to_string(),
            assistance: SUCCESSOR_ASSISTANCE.to_string(),
            assistance_hash: stable_hex_hash(SUCCESSOR_ASSISTANCE.as_bytes()),
            float_model_path: PathBuf::from("float.npy"),
            float_model_hash: "0xeeeeeeeeeeeeeeee".to_string(),
            float_runner_path: PathBuf::from("float.py"),
            float_runner_hash: "0xffffffffffffffff".to_string(),
        }
    }

    fn result_row(manifest: &SuccessorManifest, system: &str, nll: u64) -> String {
        format!(
            "{SUCCESSOR_RESULT_SCHEMA}\t{SUCCESSOR_CONTRACT_ID}\tsubstrate\teval\t{}\t{}\t{}\t{}\t{}\t{system}\t{}\t4000\t{nll}\t0\t0x0123456789abcdef",
            manifest.dataset_hash,
            manifest.candidate_hash,
            manifest.model_hash,
            manifest.runner_hash,
            manifest.assistance_hash,
            manifest.targets,
        )
    }

    fn passing_input(manifest: &SuccessorManifest) -> String {
        [
            SUCCESSOR_RESULTS_HEADER.to_string(),
            result_row(manifest, "transformer-only", 40_000_000),
            result_row(manifest, "uniform", 50_000_000),
            result_row(manifest, "retrieval", 51_000_000),
            result_row(manifest, "byte-ngram", 52_000_000),
            result_row(manifest, "float-transformer", 53_000_000),
        ]
        .join("\n")
    }

    #[test]
    fn successor_contract_freezes_dataset_identities_and_assistance() {
        let contract = successor_contract_json_line();
        assert!(contract.contains(SUCCESSOR_DATASET_HASH));
        assert!(contract.contains("\"targets\":5896"));
        assert!(contract.contains("\"manifest_required\":true"));
        assert!(contract.contains("candidate_hash"));
        assert!(contract.contains("model_hash"));
        assert!(contract.contains("runner_hash"));
        assert!(contract.contains(SUCCESSOR_ASSISTANCE));
        assert!(contract.contains("float-transformer"));
    }

    #[test]
    fn successor_check_uses_nll_as_the_only_promotion_gate() {
        let manifest = manifest();
        let check = check_successor_results(&passing_input(&manifest), &manifest)
            .expect("valid successor results");
        assert!(check.passed);
        assert_eq!(check.candidate.system, SuccessorSystem::TransformerOnly);
        assert_eq!(check.baselines.len(), 4);
    }

    #[test]
    fn successor_check_rejects_stale_candidate_model_and_runner_bindings() {
        let manifest = manifest();
        for frozen in [
            manifest.candidate_hash.as_str(),
            manifest.model_hash.as_str(),
            manifest.runner_hash.as_str(),
        ] {
            let input = passing_input(&manifest).replace(frozen, "0x9999999999999999");
            assert!(check_successor_results(&input, &manifest).is_err());
        }
    }

    #[test]
    fn successor_check_rejects_wrong_dataset_targets_and_assistance() {
        let manifest = manifest();
        let input = passing_input(&manifest).replace(SUCCESSOR_DATASET_HASH, "0x9999999999999999");
        assert!(check_successor_results(&input, &manifest).is_err());

        let input = passing_input(&manifest).replace("\t5896\t", "\t5895\t");
        assert!(check_successor_results(&input, &manifest).is_err());

        let input = passing_input(&manifest)
            .replace(manifest.assistance_hash.as_str(), "0x9999999999999999");
        assert!(check_successor_results(&input, &manifest).is_err());
    }

    #[test]
    fn successor_check_requires_a_real_float_transformer_baseline() {
        let manifest = manifest();
        let input = passing_input(&manifest).replace("float-transformer", "float-reference");
        assert!(check_successor_results(&input, &manifest).is_err());
    }

    #[test]
    fn successor_dataset_hash_matches_the_frozen_separator_contract() {
        assert_eq!(
            successor_dataset_hash(b"train", b"eval"),
            "0x062848796793ad9c"
        );
    }
}
