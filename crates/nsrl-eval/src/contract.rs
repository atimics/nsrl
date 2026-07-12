use std::collections::HashSet;
use std::fs;
use std::path::{Component, Path, PathBuf};

use crate::json_escape;

pub const PROOF_CONTRACT_SCHEMA: &str = "nsrl.integer_transformer_proof_contract.v1";
pub const PROOF_MANIFEST_SCHEMA: &str = "nsrl.integer_transformer_proof_manifest.v1";
pub const PROOF_RESULT_SCHEMA: &str = "nsrl.integer_transformer_proof_result.v1";
pub const PROOF_CONTRACT_ID: &str = "integer-transformer-proof-v1";
pub const PROOF_PARTITION: &str = "eval";
pub const PROOF_RESULTS_HEADER: &str = "schema\tcontract\tsuite\tpartition\tdataset_hash\tsystem\ttargets\tmistakes\tprobability_error_q15\treplay_hash";
pub const PROOF_MANIFEST_HEADER: &str =
    "schema\tcontract\ttrain\teval\tcontext\tstride\tmin_targets\tdataset_hash";

const FNV64_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV64_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExperimentSuite {
    Substrate,
    Literary,
    Solomon,
}

impl ExperimentSuite {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Substrate => "substrate",
            Self::Literary => "literary",
            Self::Solomon => "solomon",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProofSystem {
    Candidate,
    Retrieval,
    ByteNgram,
    FloatReference,
}

impl ProofSystem {
    pub const REQUIRED: [Self; 4] = [
        Self::Candidate,
        Self::Retrieval,
        Self::ByteNgram,
        Self::FloatReference,
    ];

    pub const BASELINES: [Self; 3] = [Self::Retrieval, Self::ByteNgram, Self::FloatReference];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Candidate => "candidate",
            Self::Retrieval => "retrieval",
            Self::ByteNgram => "byte-ngram",
            Self::FloatReference => "float-reference",
        }
    }

    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "candidate" => Ok(Self::Candidate),
            "retrieval" => Ok(Self::Retrieval),
            "byte-ngram" => Ok(Self::ByteNgram),
            "float-reference" => Ok(Self::FloatReference),
            _ => Err(format!("unknown proof system {value}")),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProofResult {
    pub system: ProofSystem,
    pub targets: usize,
    pub mistakes: usize,
    pub probability_error_q15: u64,
    pub replay_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProofCheck {
    pub dataset_hash: String,
    pub targets: usize,
    pub candidate: ProofResult,
    pub baselines: Vec<ProofResult>,
    pub passed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProofManifest {
    pub manifest_path: PathBuf,
    pub train_path: PathBuf,
    pub eval_path: PathBuf,
    pub context: usize,
    pub stride: usize,
    pub min_targets: usize,
    pub dataset_hash: String,
    pub targets: usize,
}

impl ProofManifest {
    pub fn to_json_line(&self) -> String {
        format!(
            "{{\"schema\":\"{}\",\"contract\":\"{}\",\"manifest\":\"{}\",\"train\":\"{}\",\"eval\":\"{}\",\"context\":{},\"stride\":{},\"min_targets\":{},\"targets\":{},\"dataset_hash\":\"{}\",\"valid\":true}}\n",
            PROOF_MANIFEST_SCHEMA,
            PROOF_CONTRACT_ID,
            json_escape(&self.manifest_path.to_string_lossy()),
            json_escape(&self.train_path.to_string_lossy()),
            json_escape(&self.eval_path.to_string_lossy()),
            self.context,
            self.stride,
            self.min_targets,
            self.targets,
            self.dataset_hash,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProofBaselineCheck {
    pub dataset_hash: String,
    pub targets: usize,
    pub baselines: Vec<ProofResult>,
}

impl ProofBaselineCheck {
    pub fn to_json_line(&self) -> String {
        let baselines = self
            .baselines
            .iter()
            .map(result_json)
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\"schema\":\"{}\",\"contract\":\"{}\",\"dataset_hash\":\"{}\",\"targets\":{},\"baselines\":[{}],\"valid\":true}}\n",
            PROOF_RESULT_SCHEMA, PROOF_CONTRACT_ID, self.dataset_hash, self.targets, baselines,
        )
    }
}

#[derive(Debug)]
struct ParsedProofResults {
    dataset_hash: String,
    targets: usize,
    systems: HashSet<ProofSystem>,
    results: Vec<ProofResult>,
}

impl ProofCheck {
    pub fn to_json_line(&self) -> String {
        let baselines = self
            .baselines
            .iter()
            .map(result_json)
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\"schema\":\"{}\",\"contract\":\"{}\",\"dataset_hash\":\"{}\",\"targets\":{},\"candidate\":{},\"baselines\":[{}],\"passed\":{}}}\n",
            PROOF_RESULT_SCHEMA,
            PROOF_CONTRACT_ID,
            json_escape(&self.dataset_hash),
            self.targets,
            result_json(&self.candidate),
            baselines,
            self.passed,
        )
    }
}

pub fn proof_contract_json_line() -> String {
    format!(
        "{{\"schema\":\"{}\",\"contract\":\"{}\",\"headline_suite\":\"substrate\",\"partition\":\"{}\",\"candidate\":\"candidate\",\"required_baselines\":[\"retrieval\",\"byte-ngram\",\"float-reference\"],\"primary_metric\":\"probability_error_q15\",\"secondary_metric\":\"mistakes\",\"pass_rule\":\"candidate_probability_error_strictly_lower_and_mistakes_not_higher_than_every_baseline\",\"experiment_suites\":[\"literary\",\"solomon\"]}}\n",
        PROOF_CONTRACT_SCHEMA, PROOF_CONTRACT_ID, PROOF_PARTITION,
    )
}

pub fn check_proof_results(input: &str) -> Result<ProofCheck, String> {
    let parsed = parse_proof_results(input)?;
    for required in ProofSystem::REQUIRED {
        if !parsed.systems.contains(&required) {
            return Err(format!(
                "missing required proof system {}",
                required.as_str()
            ));
        }
    }
    if parsed.results.len() != ProofSystem::REQUIRED.len() {
        return Err("full proof results must contain exactly four systems".to_string());
    }
    let candidate = parsed
        .results
        .iter()
        .find(|result| result.system == ProofSystem::Candidate)
        .cloned()
        .ok_or("missing candidate")?;
    let baselines = required_baselines(&parsed.results)?;
    let passed = baselines.iter().all(|baseline| {
        candidate.probability_error_q15 < baseline.probability_error_q15
            && candidate.mistakes <= baseline.mistakes
    });
    Ok(ProofCheck {
        dataset_hash: parsed.dataset_hash,
        targets: parsed.targets,
        candidate,
        baselines,
        passed,
    })
}

pub fn check_proof_baselines(
    input: &str,
    manifest: &ProofManifest,
) -> Result<ProofBaselineCheck, String> {
    let parsed = parse_proof_results(input)?;
    if parsed.systems.contains(&ProofSystem::Candidate) || parsed.results.len() != 3 {
        return Err(
            "baseline artifact must contain exactly the three required baselines".to_string(),
        );
    }
    let baselines = required_baselines(&parsed.results)?;
    if parsed.dataset_hash != manifest.dataset_hash || parsed.targets != manifest.targets {
        return Err("baseline artifact does not match the frozen manifest".to_string());
    }
    Ok(ProofBaselineCheck {
        dataset_hash: parsed.dataset_hash,
        targets: parsed.targets,
        baselines,
    })
}

pub fn load_proof_manifest(path: &Path) -> Result<ProofManifest, String> {
    let input = fs::read_to_string(path)
        .map_err(|error| format!("cannot read manifest {}: {error}", path.display()))?;
    let mut lines = input.lines();
    if lines.next() != Some(PROOF_MANIFEST_HEADER) {
        return Err(format!(
            "proof manifest header must be {PROOF_MANIFEST_HEADER}"
        ));
    }
    let row = lines.next().ok_or("proof manifest row is missing")?;
    if lines.any(|line| !line.trim().is_empty()) {
        return Err("proof manifest must contain exactly one row".to_string());
    }
    let fields = row.split('\t').collect::<Vec<_>>();
    if fields.len() != 8 || fields[0] != PROOF_MANIFEST_SCHEMA || fields[1] != PROOF_CONTRACT_ID {
        return Err("proof manifest schema or contract is invalid".to_string());
    }
    let directory = path.parent().unwrap_or_else(|| Path::new("."));
    let train_path = resolve_manifest_path(directory, fields[2], "train")?;
    let eval_path = resolve_manifest_path(directory, fields[3], "eval")?;
    let context = parse_positive(fields[4], "context")?;
    let stride = parse_positive(fields[5], "stride")?;
    let min_targets = parse_positive(fields[6], "min_targets")?;
    validate_hash(fields[7], "dataset_hash")?;
    let train = fs::read(&train_path)
        .map_err(|error| format!("cannot read train corpus {}: {error}", train_path.display()))?;
    let eval = fs::read(&eval_path)
        .map_err(|error| format!("cannot read eval corpus {}: {error}", eval_path.display()))?;
    let dataset_hash = proof_dataset_hash(&train, &eval);
    if dataset_hash != fields[7] {
        return Err(format!(
            "manifest dataset_hash {} does not match {dataset_hash}",
            fields[7]
        ));
    }
    let targets = proof_target_count(eval.len(), context, stride);
    if targets < min_targets {
        return Err(format!(
            "benchmark has {targets} targets, below frozen minimum {min_targets}"
        ));
    }
    Ok(ProofManifest {
        manifest_path: path.to_path_buf(),
        train_path,
        eval_path,
        context,
        stride,
        min_targets,
        dataset_hash,
        targets,
    })
}

pub fn proof_dataset_hash(train: &[u8], eval: &[u8]) -> String {
    let mut hash = FNV64_OFFSET;
    for byte in train
        .iter()
        .copied()
        .chain([u8::MAX])
        .chain(eval.iter().copied())
    {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV64_PRIME);
    }
    format!("0x{hash:016x}")
}

pub fn proof_target_count(eval_bytes: usize, context: usize, stride: usize) -> usize {
    if context == 0 || stride == 0 || eval_bytes <= context {
        return 0;
    }
    (eval_bytes - context).div_ceil(stride)
}

fn parse_proof_results(input: &str) -> Result<ParsedProofResults, String> {
    let mut lines = input.lines();
    let header = lines.next().ok_or("proof results are empty")?;
    if header != PROOF_RESULTS_HEADER {
        return Err(format!(
            "proof results header must be {PROOF_RESULTS_HEADER}"
        ));
    }

    let mut dataset_hash = None;
    let mut targets = None;
    let mut systems = HashSet::new();
    let mut results = Vec::new();
    for (index, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != 10 {
            return Err(format!(
                "proof result row {} must have 10 fields",
                index + 2
            ));
        }
        if fields[0] != PROOF_RESULT_SCHEMA
            || fields[1] != PROOF_CONTRACT_ID
            || fields[2] != ExperimentSuite::Substrate.as_str()
            || fields[3] != PROOF_PARTITION
        {
            return Err(format!(
                "proof result row {} violates the frozen contract",
                index + 2
            ));
        }
        validate_hash(fields[4], "dataset_hash")?;
        validate_hash(fields[9], "replay_hash")?;
        if dataset_hash.get_or_insert(fields[4].to_string()) != fields[4] {
            return Err("all proof rows must use the same dataset_hash".to_string());
        }
        let system = ProofSystem::parse(fields[5])?;
        if !systems.insert(system) {
            return Err(format!("duplicate proof system {}", system.as_str()));
        }
        let row_targets = parse_positive(fields[6], "targets")?;
        if *targets.get_or_insert(row_targets) != row_targets {
            return Err("all proof rows must use the same target count".to_string());
        }
        let mistakes = fields[7]
            .parse::<usize>()
            .map_err(|_| "mistakes must be a non-negative integer".to_string())?;
        if mistakes > row_targets {
            return Err("mistakes cannot exceed targets".to_string());
        }
        let probability_error_q15 = fields[8]
            .parse::<u64>()
            .map_err(|_| "probability_error_q15 must be a non-negative integer".to_string())?;
        results.push(ProofResult {
            system,
            targets: row_targets,
            mistakes,
            probability_error_q15,
            replay_hash: fields[9].to_string(),
        });
    }

    Ok(ParsedProofResults {
        dataset_hash: dataset_hash.ok_or("missing dataset_hash")?,
        targets: targets.ok_or("missing target count")?,
        systems,
        results,
    })
}

fn required_baselines(results: &[ProofResult]) -> Result<Vec<ProofResult>, String> {
    ProofSystem::BASELINES
        .iter()
        .map(|system| {
            results
                .iter()
                .find(|result| result.system == *system)
                .cloned()
                .ok_or_else(|| format!("missing baseline {}", system.as_str()))
        })
        .collect()
}

fn resolve_manifest_path(directory: &Path, value: &str, field: &str) -> Result<PathBuf, String> {
    let relative = Path::new(value);
    if relative.is_absolute()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!(
            "manifest {field} path must be a simple relative path"
        ));
    }
    Ok(directory.join(relative))
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

fn parse_positive(value: &str, field: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("{field} must be a positive integer"))?;
    if parsed == 0 {
        return Err(format!("{field} must be positive"));
    }
    Ok(parsed)
}

fn result_json(result: &ProofResult) -> String {
    format!(
        "{{\"system\":\"{}\",\"targets\":{},\"mistakes\":{},\"probability_error_q15\":{},\"replay_hash\":\"{}\"}}",
        result.system.as_str(),
        result.targets,
        result.mistakes,
        result.probability_error_q15,
        json_escape(&result.replay_hash),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEADER: &str = PROOF_RESULTS_HEADER;

    fn row(system: &str, mistakes: usize, error: u64, replay: &str) -> String {
        format!(
            "{PROOF_RESULT_SCHEMA}\t{PROOF_CONTRACT_ID}\tsubstrate\teval\t0x1111111111111111\t{system}\t100\t{mistakes}\t{error}\t{replay}"
        )
    }

    #[test]
    fn proof_requires_candidate_to_beat_every_baseline() {
        let input = [
            HEADER.to_string(),
            row("candidate", 10, 100, "0xaaaaaaaaaaaaaaaa"),
            row("retrieval", 10, 101, "0xbbbbbbbbbbbbbbbb"),
            row("byte-ngram", 11, 120, "0xcccccccccccccccc"),
            row("float-reference", 12, 130, "0xdddddddddddddddd"),
        ]
        .join("\n");
        let check = check_proof_results(&input).expect("valid proof");
        assert!(check.passed);
    }

    #[test]
    fn proof_rejects_missing_or_duplicate_systems() {
        let missing = [
            HEADER.to_string(),
            row("candidate", 10, 100, "0xaaaaaaaaaaaaaaaa"),
        ]
        .join("\n");
        assert!(check_proof_results(&missing).is_err());

        let duplicate = [
            HEADER.to_string(),
            row("candidate", 10, 100, "0xaaaaaaaaaaaaaaaa"),
            row("candidate", 10, 100, "0xeeeeeeeeeeeeeeee"),
        ]
        .join("\n");
        assert!(check_proof_results(&duplicate).is_err());
    }

    #[test]
    fn proof_fails_when_candidate_only_wins_primary_metric() {
        let input = [
            HEADER.to_string(),
            row("candidate", 20, 100, "0xaaaaaaaaaaaaaaaa"),
            row("retrieval", 10, 101, "0xbbbbbbbbbbbbbbbb"),
            row("byte-ngram", 21, 120, "0xcccccccccccccccc"),
            row("float-reference", 22, 130, "0xdddddddddddddddd"),
        ]
        .join("\n");
        assert!(!check_proof_results(&input).expect("valid proof").passed);
    }

    #[test]
    fn dataset_hash_binds_train_eval_boundary() {
        assert_eq!(
            proof_dataset_hash(b"abc", b"def"),
            proof_dataset_hash(b"abc", b"def")
        );
        assert_ne!(
            proof_dataset_hash(b"abc", b"def"),
            proof_dataset_hash(b"abcd", b"ef")
        );
    }

    #[test]
    fn target_count_matches_context_windows() {
        assert_eq!(proof_target_count(65, 64, 1), 1);
        assert_eq!(proof_target_count(70, 64, 2), 3);
        assert_eq!(proof_target_count(64, 64, 1), 0);
    }
}
