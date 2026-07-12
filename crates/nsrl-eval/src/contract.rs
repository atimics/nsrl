use std::collections::HashSet;

use crate::json_escape;

pub const PROOF_CONTRACT_SCHEMA: &str = "nsrl.integer_transformer_proof_contract.v1";
pub const PROOF_RESULT_SCHEMA: &str = "nsrl.integer_transformer_proof_result.v1";
pub const PROOF_CONTRACT_ID: &str = "integer-transformer-proof-v1";
pub const PROOF_PARTITION: &str = "eval";
pub const PROOF_RESULTS_HEADER: &str = "schema\tcontract\tsuite\tpartition\tdataset_hash\tsystem\ttargets\tmistakes\tprobability_error_q15\treplay_hash";

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

    for required in ProofSystem::REQUIRED {
        if !systems.contains(&required) {
            return Err(format!(
                "missing required proof system {}",
                required.as_str()
            ));
        }
    }
    let candidate = results
        .iter()
        .find(|result| result.system == ProofSystem::Candidate)
        .cloned()
        .ok_or("missing candidate")?;
    let baselines = ProofSystem::BASELINES
        .iter()
        .map(|system| {
            results
                .iter()
                .find(|result| result.system == *system)
                .cloned()
                .ok_or_else(|| format!("missing baseline {}", system.as_str()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let passed = baselines.iter().all(|baseline| {
        candidate.probability_error_q15 < baseline.probability_error_q15
            && candidate.mistakes <= baseline.mistakes
    });
    Ok(ProofCheck {
        dataset_hash: dataset_hash.ok_or("missing dataset_hash")?,
        targets: targets.ok_or("missing target count")?,
        candidate,
        baselines,
        passed,
    })
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
}
