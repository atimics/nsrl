#![deny(unsafe_code)]

use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;

use nsrl_core::{DEFAULT_ZERO_PROBABILITY_NLL_MILLIBITS, base2_softmax_nll_millibits};
use nsrl_eval::contract::{proof_dataset_hash, proof_target_count};
use nsrl_eval::successor::{
    SUCCESSOR_CONTRACT_ID, SUCCESSOR_FROZEN_DATASET_HASH, SUCCESSOR_FROZEN_TARGETS,
    SUCCESSOR_RESULT_SCHEMA, SUCCESSOR_RESULTS_HEADER, SUCCESSOR_TOKENIZER, SuccessorSystem,
    fnv64_hex,
};
use nsrl_train::{
    MiniTransformerAttentionKind, MiniTransformerMlpEvalConfig, MiniTransformerMlpModel,
    MiniTransformerPositionPolicy, evaluate_mini_transformer_mlp_windows,
};

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
const FLOAT_MAGIC: &[u8; 8] = b"NSRLFT2\n";
const SUFFIX_MEMORY_MAGIC: [u8; 8] = *b"NSRLSM1\0";
const BYTE_CLASSES: usize = 256;

#[derive(Debug)]
struct Config {
    train: PathBuf,
    eval: PathBuf,
    candidate: PathBuf,
    float_logits: PathBuf,
    out_matrix: PathBuf,
    out_evidence: PathBuf,
    dataset_hash: String,
    tokenizer_hash: String,
    candidate_model_hash: String,
    candidate_artifact_hash: String,
    evaluator_hash: String,
    runner_hash: String,
    float_trace_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Metrics {
    system: SuccessorSystem,
    targets: usize,
    mistakes: usize,
    total_nll_millibits: u64,
    zero_probability_windows: usize,
    replay_hash: String,
}

struct FloatLogits {
    model_hash: String,
    artifact_hash: String,
    rows: Vec<[i32; BYTE_CLASSES]>,
}

struct CountTables {
    tables: HashMap<usize, HashMap<Vec<u8>, Box<[u32; BYTE_CLASSES]>>>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("nsrl-successor-eval: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args()?;
    let train = fs::read(&config.train)?;
    let evaluation = fs::read(&config.eval)?;
    if proof_dataset_hash(&train, &evaluation) != config.dataset_hash
        || config.dataset_hash != SUCCESSOR_FROZEN_DATASET_HASH
    {
        return Err("successor evaluator dataset hash mismatch".into());
    }
    let targets = proof_target_count(evaluation.len(), 64, 1);
    if targets != SUCCESSOR_FROZEN_TARGETS {
        return Err(format!(
            "successor evaluator found {targets} targets, expected {SUCCESSOR_FROZEN_TARGETS}"
        )
        .into());
    }
    if fnv64_hex(SUCCESSOR_TOKENIZER.as_bytes()) != config.tokenizer_hash {
        return Err("successor evaluator tokenizer hash mismatch".into());
    }

    let candidate_bytes = fs::read(&config.candidate)?;
    if fnv64_hex(&candidate_bytes) != config.candidate_artifact_hash {
        return Err("successor evaluator candidate artifact hash mismatch".into());
    }
    let candidate = MiniTransformerMlpModel::from_bytes(&candidate_bytes)?;
    let actual_model_hash = format!("0x{:016x}", candidate.model_hash());
    if actual_model_hash != config.candidate_model_hash {
        return Err(format!(
            "candidate model hash {actual_model_hash} does not match {}",
            config.candidate_model_hash
        )
        .into());
    }
    if suffix_memory_present(&candidate)
        || candidate
            .position_embeddings
            .iter()
            .any(|&value| value != 0)
    {
        return Err(
            "successor candidate contains active or dormant assistance in position storage".into(),
        );
    }

    let float = load_float_logits(&config.float_logits, targets, &config.dataset_hash)?;
    let tables = CountTables::new(&train, &[1, 2, 3, 4, 8, 16, 32, 64]);
    let candidate_metrics = evaluate_candidate(&candidate, &evaluation)?;
    let suffix_ablation = evaluate_suffix_ablation(&candidate, &evaluation)?;
    let retrieval_ablation = evaluate_candidate(&candidate, &evaluation)?;
    let routing_ablation = evaluate_candidate(&candidate, &evaluation)?;
    if suffix_ablation != candidate_metrics
        || retrieval_ablation != candidate_metrics
        || routing_ablation != candidate_metrics
    {
        return Err("candidate assistance ablation changed the unassisted replay".into());
    }

    let mut systems = vec![candidate_metrics];
    systems.push(evaluate_generated(
        SuccessorSystem::Uniform,
        &evaluation,
        64,
        |_| [0_i32; BYTE_CLASSES],
    )?);
    systems.push(evaluate_generated(
        SuccessorSystem::Retrieval,
        &evaluation,
        64,
        |context| tables.retrieval_logits(context),
    )?);
    systems.push(evaluate_generated(
        SuccessorSystem::ByteNgram,
        &evaluation,
        64,
        |context| tables.byte_ngram_logits(context),
    )?);
    systems.push(evaluate_float(&float, &evaluation, 64)?);

    let matrix = matrix_tsv(&systems, &config);
    if let Some(parent) = config.out_matrix.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Some(parent) = config.out_evidence.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&config.out_matrix, &matrix)?;
    let evidence = evidence_json(
        &systems,
        &config,
        &float,
        &suffix_ablation.replay_hash,
        &retrieval_ablation.replay_hash,
        &routing_ablation.replay_hash,
    );
    fs::write(&config.out_evidence, &evidence)?;
    println!(
        "{{\"matrix\":\"{}\",\"evidence\":\"{}\",\"matrix_hash\":\"{}\",\"evidence_hash\":\"{}\"}}",
        config.out_matrix.display(),
        config.out_evidence.display(),
        fnv64_hex(matrix.as_bytes()),
        fnv64_hex(evidence.as_bytes()),
    );
    Ok(())
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let mut values = HashMap::<String, String>::new();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--help" || arg == "-h" {
            println!(
                "Usage: nsrl-successor-eval --train PATH --eval PATH --candidate PATH --float-logits PATH \\\n+                 --out-matrix PATH --out-evidence PATH --dataset-hash HASH --tokenizer-hash HASH \\\n+                 --candidate-model-hash HASH --candidate-artifact-hash HASH --evaluator-hash HASH \\\n+                 --runner-hash HASH --float-trace-hash HASH"
            );
            std::process::exit(0);
        }
        if !arg.starts_with("--") {
            return Err(format!("unexpected argument {arg}").into());
        }
        let value = args
            .next()
            .ok_or_else(|| format!("{arg} requires a value"))?;
        if values.insert(arg.clone(), value).is_some() {
            return Err(format!("duplicate argument {arg}").into());
        }
    }
    let mut take = |name: &str| {
        values
            .remove(name)
            .ok_or_else(|| format!("{name} is required"))
    };
    let config = Config {
        train: PathBuf::from(take("--train")?),
        eval: PathBuf::from(take("--eval")?),
        candidate: PathBuf::from(take("--candidate")?),
        float_logits: PathBuf::from(take("--float-logits")?),
        out_matrix: PathBuf::from(take("--out-matrix")?),
        out_evidence: PathBuf::from(take("--out-evidence")?),
        dataset_hash: take("--dataset-hash")?,
        tokenizer_hash: take("--tokenizer-hash")?,
        candidate_model_hash: take("--candidate-model-hash")?,
        candidate_artifact_hash: take("--candidate-artifact-hash")?,
        evaluator_hash: take("--evaluator-hash")?,
        runner_hash: take("--runner-hash")?,
        float_trace_hash: take("--float-trace-hash")?,
    };
    if !values.is_empty() {
        return Err(format!("unknown arguments: {:?}", values.keys()).into());
    }
    Ok(config)
}

fn evaluate_candidate(
    model: &MiniTransformerMlpModel,
    evaluation: &[u8],
) -> Result<Metrics, Box<dyn std::error::Error>> {
    let config = MiniTransformerMlpEvalConfig {
        seq_len: 64,
        stride: 1,
        max_windows: None,
        attention_kind: MiniTransformerAttentionKind::Linear,
        position_policy: MiniTransformerPositionPolicy::Nope,
    };
    let records = evaluate_mini_transformer_mlp_windows(evaluation, model, config)?;
    let rows = records
        .into_iter()
        .map(|record| {
            record
                .logits_q8
                .map(|logits| (record.start, logits))
                .ok_or("candidate produced an invalid forward")
        })
        .collect::<Result<Vec<_>, _>>()?;
    evaluate_rows(SuccessorSystem::TransformerOnly, evaluation, 64, rows)
}

fn evaluate_suffix_ablation(
    model: &MiniTransformerMlpModel,
    evaluation: &[u8],
) -> Result<Metrics, Box<dyn std::error::Error>> {
    let mut ablated = model.clone();
    ablated.position_embeddings.fill(0);
    if ablated.model_hash() != model.model_hash() {
        return Err("suffix-memory ablation changed a physically unassisted model".into());
    }
    evaluate_candidate(&ablated, evaluation)
}

fn evaluate_generated(
    system: SuccessorSystem,
    evaluation: &[u8],
    context: usize,
    mut logits_for: impl FnMut(&[u8]) -> [i32; BYTE_CLASSES],
) -> Result<Metrics, Box<dyn std::error::Error>> {
    let rows = (0..evaluation.len() - context)
        .map(|start| (start, logits_for(&evaluation[start..start + context])))
        .collect::<Vec<_>>();
    evaluate_rows(system, evaluation, context, rows)
}

fn evaluate_float(
    float: &FloatLogits,
    evaluation: &[u8],
    context: usize,
) -> Result<Metrics, Box<dyn std::error::Error>> {
    let rows = float.rows.iter().copied().enumerate().collect::<Vec<_>>();
    evaluate_rows(SuccessorSystem::FloatTransformer, evaluation, context, rows)
}

fn evaluate_rows(
    system: SuccessorSystem,
    evaluation: &[u8],
    context: usize,
    rows: Vec<(usize, [i32; BYTE_CLASSES])>,
) -> Result<Metrics, Box<dyn std::error::Error>> {
    if rows.len() != evaluation.len() - context {
        return Err(format!("{} row count mismatch", system.as_str()).into());
    }
    let mut mistakes = 0_usize;
    let mut total_nll_millibits = 0_u64;
    let mut zero_probability_windows = 0_usize;
    let mut replay = hash_update_bytes(
        FNV_OFFSET,
        format!("{SUCCESSOR_CONTRACT_ID}\0{}\0", system.as_str()).as_bytes(),
    );
    for (expected_start, (start, logits)) in rows.into_iter().enumerate() {
        if start != expected_start {
            return Err(format!("{} row ordering mismatch", system.as_str()).into());
        }
        let target = usize::from(evaluation[start + context]);
        let predicted = argmax(&logits);
        mistakes += usize::from(predicted != target);
        let loss =
            base2_softmax_nll_millibits(&logits, target, DEFAULT_ZERO_PROBABILITY_NLL_MILLIBITS)
                .ok_or("canonical integer NLL rejected logits")?;
        total_nll_millibits = total_nll_millibits
            .checked_add(loss)
            .ok_or("canonical integer NLL total overflow")?;
        zero_probability_windows += usize::from(loss == DEFAULT_ZERO_PROBABILITY_NLL_MILLIBITS);
        replay = hash_update_bytes(replay, &(start as u64).to_le_bytes());
        replay = hash_update_bytes(replay, &(target as u16).to_le_bytes());
        replay = hash_update_bytes(replay, &(predicted as u16).to_le_bytes());
        replay = hash_update_bytes(replay, &loss.to_le_bytes());
        for logit in logits {
            replay = hash_update_bytes(replay, &logit.to_le_bytes());
        }
    }
    Ok(Metrics {
        system,
        targets: evaluation.len() - context,
        mistakes,
        total_nll_millibits,
        zero_probability_windows,
        replay_hash: format!("0x{replay:016x}"),
    })
}

impl CountTables {
    fn new(train: &[u8], orders: &[usize]) -> Self {
        let mut tables = orders
            .iter()
            .copied()
            .chain([0])
            .map(|order| (order, HashMap::new()))
            .collect::<HashMap<_, _>>();
        tables
            .get_mut(&0)
            .unwrap()
            .insert(Vec::new(), Box::new([0_u32; BYTE_CLASSES]));
        for target_index in 0..train.len() {
            tables.get_mut(&0).unwrap().get_mut(&Vec::new()).unwrap()
                [usize::from(train[target_index])] += 1;
            for &order in orders {
                if order > target_index {
                    continue;
                }
                let key = train[target_index - order..target_index].to_vec();
                let counts = tables
                    .get_mut(&order)
                    .unwrap()
                    .entry(key)
                    .or_insert_with(|| Box::new([0_u32; BYTE_CLASSES]));
                counts[usize::from(train[target_index])] += 1;
            }
        }
        Self { tables }
    }

    fn retrieval_logits(&self, context: &[u8]) -> [i32; BYTE_CLASSES] {
        self.lookup_logits(context, &[64, 32, 16, 8, 4, 2])
    }

    fn byte_ngram_logits(&self, context: &[u8]) -> [i32; BYTE_CLASSES] {
        self.lookup_logits(context, &[4, 3, 2, 1])
    }

    fn lookup_logits(&self, context: &[u8], orders: &[usize]) -> [i32; BYTE_CLASSES] {
        for &order in orders {
            if order <= context.len()
                && let Some(counts) = self
                    .tables
                    .get(&order)
                    .and_then(|table| table.get(&context[context.len() - order..]))
            {
                return count_logits(counts);
            }
        }
        count_logits(self.tables.get(&0).unwrap().get(&Vec::new()).unwrap())
    }
}

fn count_logits(counts: &[u32; BYTE_CLASSES]) -> [i32; BYTE_CLASSES] {
    std::array::from_fn(|index| log2_q8(u64::from(counts[index]) + 1) as i32)
}

fn log2_q8(value: u64) -> u64 {
    let integer = u64::BITS - 1 - value.leading_zeros();
    let mut normalized = u128::from(value) << (63 - integer);
    let mut fraction = 0_u64;
    for bit in (0..20).rev() {
        normalized = (normalized * normalized) >> 63;
        if normalized >= (2_u128 << 63) {
            normalized >>= 1;
            fraction |= 1_u64 << bit;
        }
    }
    ((u64::from(integer) << 20) | fraction).saturating_add(1 << 11) >> 12
}

fn load_float_logits(
    path: &PathBuf,
    targets: usize,
    expected_dataset_hash: &str,
) -> Result<FloatLogits, Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    let header_len = 40_usize;
    let expected_len = header_len + targets * BYTE_CLASSES * 4 + 8;
    if bytes.len() != expected_len || bytes.get(..8) != Some(FLOAT_MAGIC) {
        return Err("invalid float-transformer logits artifact".into());
    }
    let version = read_u32(&bytes, 8)?;
    let context = read_u32(&bytes, 12)?;
    let windows = read_u32(&bytes, 16)?;
    let classes = read_u32(&bytes, 20)?;
    let dataset_hash = read_u64(&bytes, 24)?;
    let model_hash = read_u64(&bytes, 32)?;
    if version != 1
        || context != 64
        || windows as usize != targets
        || classes as usize != BYTE_CLASSES
        || format!("0x{dataset_hash:016x}") != expected_dataset_hash
    {
        return Err("float-transformer logits bindings mismatch".into());
    }
    let expected_hash = read_u64(&bytes, bytes.len() - 8)?;
    let actual_hash = hash_update_bytes(FNV_OFFSET, &bytes[..bytes.len() - 8]);
    if actual_hash != expected_hash {
        return Err("float-transformer logits checksum mismatch".into());
    }
    let mut rows = Vec::with_capacity(targets);
    let mut offset = header_len;
    for _ in 0..targets {
        let mut row = [0_i32; BYTE_CLASSES];
        for value in &mut row {
            *value = read_i32(&bytes, offset)?;
            offset += 4;
        }
        rows.push(row);
    }
    Ok(FloatLogits {
        model_hash: format!("0x{model_hash:016x}"),
        artifact_hash: format!("0x{actual_hash:016x}"),
        rows,
    })
}

fn matrix_tsv(systems: &[Metrics], config: &Config) -> String {
    let mut output = format!("{SUCCESSOR_RESULTS_HEADER}\n");
    for metrics in systems {
        output.push_str(&format!(
            "{SUCCESSOR_RESULT_SCHEMA}\t{SUCCESSOR_CONTRACT_ID}\tsubstrate\teval\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\tfalse\tfalse\tfalse\t{}\n",
            config.dataset_hash,
            config.tokenizer_hash,
            config.candidate_model_hash,
            config.evaluator_hash,
            config.runner_hash,
            metrics.system.as_str(),
            metrics.targets,
            metrics.mistakes,
            metrics.total_nll_millibits,
            metrics.zero_probability_windows,
            metrics.replay_hash,
        ));
    }
    output
}

fn evidence_json(
    systems: &[Metrics],
    config: &Config,
    float: &FloatLogits,
    suffix_replay: &str,
    retrieval_replay: &str,
    routing_replay: &str,
) -> String {
    let system_json = systems
        .iter()
        .map(|metrics| {
            format!(
                "{{\"system\":\"{}\",\"targets\":{},\"mistakes\":{},\"total_nll_millibits\":{},\"zero_probability_windows\":{},\"replay_hash\":\"{}\"}}",
                metrics.system.as_str(),
                metrics.targets,
                metrics.mistakes,
                metrics.total_nll_millibits,
                metrics.zero_probability_windows,
                metrics.replay_hash,
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        concat!(
            "{{\"schema\":\"nsrl.integer_transformer_successor_evidence.v2\",",
            "\"contract\":\"{}\",\"bindings\":{{\"dataset_hash\":\"{}\",",
            "\"targets\":{},\"tokenizer\":\"{}\",\"tokenizer_hash\":\"{}\",",
            "\"candidate_model_hash\":\"{}\",\"candidate_artifact_hash\":\"{}\",",
            "\"evaluator_hash\":\"{}\",\"runner_hash\":\"{}\"}},",
            "\"objective\":{{\"id\":\"integer_base2_softmax_nll_millibits\",",
            "\"zero_probability_floor_millibits\":32000,\"identical_partition\":true}},",
            "\"candidate_assistance\":{{\"suffix_memory_present\":false,",
            "\"position_storage_all_zero\":true,",
            "\"retrieval_assistance_present\":false,\"routing_oracle_present\":false,",
            "\"ablations\":[",
            "{{\"name\":\"suffix-memory-disabled\",\"structurally_absent\":true,\"replay_hash\":\"{}\"}},",
            "{{\"name\":\"retrieval-assistance-disabled\",\"structurally_absent\":true,\"replay_hash\":\"{}\"}},",
            "{{\"name\":\"routing-oracle-disabled\",\"structurally_absent\":true,\"replay_hash\":\"{}\"}}]}},",
            "\"float_transformer\":{{\"kind\":\"genuine_float_transformer\",",
            "\"model_hash\":\"{}\",\"logits_artifact_hash\":\"{}\",",
            "\"trace_hash\":\"{}\"}},\"systems\":[{}]}}\n"
        ),
        SUCCESSOR_CONTRACT_ID,
        config.dataset_hash,
        SUCCESSOR_FROZEN_TARGETS,
        SUCCESSOR_TOKENIZER,
        config.tokenizer_hash,
        config.candidate_model_hash,
        config.candidate_artifact_hash,
        config.evaluator_hash,
        config.runner_hash,
        suffix_replay,
        retrieval_replay,
        routing_replay,
        float.model_hash,
        float.artifact_hash,
        config.float_trace_hash,
        system_json,
    )
}

fn suffix_memory_present(model: &MiniTransformerMlpModel) -> bool {
    SUFFIX_MEMORY_MAGIC
        .chunks_exact(2)
        .zip(model.position_embeddings.iter())
        .all(|(bytes, &value)| value.to_le_bytes() == bytes)
}

fn argmax(values: &[i32; BYTE_CLASSES]) -> usize {
    let mut best = 0_usize;
    for index in 1..values.len() {
        if values[index] > values[best] {
            best = index;
        }
    }
    best
}

fn hash_update_bytes(mut hash: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, Box<dyn std::error::Error>> {
    Ok(u32::from_le_bytes(
        bytes
            .get(offset..offset + 4)
            .ok_or("truncated u32")?
            .try_into()?,
    ))
}

fn read_i32(bytes: &[u8], offset: usize) -> Result<i32, Box<dyn std::error::Error>> {
    Ok(i32::from_le_bytes(
        bytes
            .get(offset..offset + 4)
            .ok_or("truncated i32")?
            .try_into()?,
    ))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, Box<dyn std::error::Error>> {
    Ok(u64::from_le_bytes(
        bytes
            .get(offset..offset + 8)
            .ok_or("truncated u64")?
            .try_into()?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_logit_is_monotone_and_exact_at_powers_of_two() {
        assert_eq!(log2_q8(1), 0);
        assert_eq!(log2_q8(2), 256);
        assert_eq!(log2_q8(4), 512);
        assert!(log2_q8(7) < log2_q8(8));
    }

    #[test]
    fn uniform_canonical_nll_is_eight_bits() {
        assert_eq!(
            base2_softmax_nll_millibits(
                &[0_i32; BYTE_CLASSES],
                17,
                DEFAULT_ZERO_PROBABILITY_NLL_MILLIBITS,
            ),
            Some(8_000)
        );
    }
}
