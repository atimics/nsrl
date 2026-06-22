#![deny(unsafe_code)]

use std::collections::{BTreeMap, HashMap, HashSet};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use nsrl_train::solomon_latent::{
    DEFAULT_EVAL_PERMILLE, DEFAULT_PROMPT_SPLIT_SEED, LatentTextModel, PromptRecord,
    SIGNATURE_BINS, TextIndexRow, default_gold_path, dot_i16, json_escape, latent_abs_error,
    mean_q8, read_gold_hashes, read_latent_model, read_prompt_records, read_text_index_rows,
    signature_abs_error, stable_hash_bytes, stable_hex_u32, text_features,
};

const SCHEMA: &str = "nsrl.solomon_eval_ledger.v1";

#[derive(Debug, Clone)]
struct Config {
    prompts_path: PathBuf,
    text_index_path: PathBuf,
    model_path: PathBuf,
    ledger_path: PathBuf,
    partition_path: Option<PathBuf>,
    gold_path: Option<PathBuf>,
    eval_permille: usize,
    split_seed: String,
    prompt_set_version: String,
    append_ledger: bool,
}

impl Default for Config {
    fn default() -> Self {
        let prompt_dir = PathBuf::from("data/processed/key-solomon-goetia-latent-v1");
        Self {
            prompts_path: prompt_dir.join("prompts.jsonl"),
            text_index_path: PathBuf::from(
                "data/processed/key-solomon-goetia-text-index-pg72679/solomon-spirit-text-signatures.tsv",
            ),
            model_path: prompt_dir.join("model.nsrllat"),
            ledger_path: prompt_dir.join("eval-ledger.jsonl"),
            partition_path: Some(prompt_dir.join("partition.tsv")),
            gold_path: None,
            eval_permille: DEFAULT_EVAL_PERMILLE,
            split_seed: String::from(DEFAULT_PROMPT_SPLIT_SEED),
            prompt_set_version: String::from("key-solomon-goetia-latent-v1"),
            append_ledger: true,
        }
    }
}

#[derive(Debug, Clone)]
struct PartitionedPrompt {
    prompt: PromptRecord,
    partition: Partition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Partition {
    Train,
    Eval,
    Gold,
}

impl Partition {
    fn as_str(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Eval => "eval",
            Self::Gold => "gold",
        }
    }
}

#[derive(Debug, Clone, Default)]
struct Metrics {
    count: usize,
    top1: usize,
    top5: usize,
    rank_total: u64,
    text_signature_abs: u64,
    text_image_latent_abs: u64,
}

#[derive(Debug, Clone)]
struct FinalMetrics {
    count: usize,
    top1: usize,
    top5: usize,
    top1_per_mille: usize,
    top5_per_mille: usize,
    mean_rank_q8: u64,
    text_signature_mae_q8: u64,
    text_image_latent_mae_q8: u64,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("nsrl-solomon-eval: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args(env::args().skip(1))?;
    if config.eval_permille > 900 {
        return Err("--eval-permille must be <= 900".into());
    }
    let model_bytes = fs::read(&config.model_path)?;
    let model_hash = stable_hex_u32(stable_hash_bytes(&model_bytes));
    let model = read_latent_model(&config.model_path)?;
    let rows = read_text_index_rows(&config.text_index_path)?;
    let prompts = read_prompt_records(&config.prompts_path, &config.split_seed)?;
    let gold_path = config
        .gold_path
        .clone()
        .unwrap_or_else(|| default_gold_path(&config.prompts_path));
    let gold_hashes = read_gold_hashes(&gold_path)?;
    let partitioned = partition_prompts(prompts, &gold_hashes, config.eval_permille);
    if let Some(path) = &config.partition_path {
        write_partition(path, &partitioned)?;
    }

    let eval_prompts: Vec<&PartitionedPrompt> = partitioned
        .iter()
        .filter(|prompt| prompt.partition == Partition::Eval)
        .collect();
    let gold_prompts: Vec<&PartitionedPrompt> = partitioned
        .iter()
        .filter(|prompt| prompt.partition == Partition::Gold)
        .collect();
    let eval_metrics = evaluate_prompts(&eval_prompts, &rows, &model)?;
    let gold_metrics = evaluate_prompts(&gold_prompts, &rows, &model)?;
    let train_count = partitioned
        .iter()
        .filter(|prompt| prompt.partition == Partition::Train)
        .count();
    let timestamp = unix_timestamp()?;
    let ledger_row = ledger_row(
        &config,
        timestamp,
        &model_hash,
        train_count,
        &eval_metrics,
        &gold_metrics,
    );
    if config.append_ledger {
        append_ledger(&config.ledger_path, &ledger_row)?;
    }
    println!("{ledger_row}");
    Ok(())
}

fn usage() {
    println!(
        "Usage: nsrl-solomon-eval [--prompts PATH] [--text-index PATH] [--model PATH] [--ledger PATH] [--gold PATH] [--eval-permille N] [--split-seed TEXT] [--prompt-set-version TEXT] [--partition-out PATH|--no-partition] [--no-ledger]"
    );
}

fn parse_args<I>(mut args: I) -> Result<Config, Box<dyn std::error::Error>>
where
    I: Iterator<Item = String>,
{
    let mut config = Config::default();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                usage();
                std::process::exit(0);
            }
            "--prompts" => {
                config.prompts_path = PathBuf::from(args.next().ok_or("--prompts requires PATH")?);
                if config.gold_path.is_none() {
                    config.gold_path = Some(default_gold_path(&config.prompts_path));
                }
            }
            "--text-index" => {
                config.text_index_path =
                    PathBuf::from(args.next().ok_or("--text-index requires PATH")?);
            }
            "--model" => {
                config.model_path = PathBuf::from(args.next().ok_or("--model requires PATH")?);
            }
            "--ledger" => {
                config.ledger_path = PathBuf::from(args.next().ok_or("--ledger requires PATH")?);
            }
            "--gold" => {
                config.gold_path = Some(PathBuf::from(args.next().ok_or("--gold requires PATH")?));
            }
            "--eval-permille" => {
                config.eval_permille = args.next().ok_or("--eval-permille requires N")?.parse()?;
            }
            "--split-seed" => {
                config.split_seed = args.next().ok_or("--split-seed requires TEXT")?;
            }
            "--prompt-set-version" => {
                config.prompt_set_version =
                    args.next().ok_or("--prompt-set-version requires TEXT")?;
            }
            "--partition-out" => {
                config.partition_path = Some(PathBuf::from(
                    args.next().ok_or("--partition-out requires PATH")?,
                ));
            }
            "--no-partition" => {
                config.partition_path = None;
            }
            "--no-ledger" => {
                config.append_ledger = false;
            }
            _ => return Err(format!("unknown option: {arg}").into()),
        }
    }
    Ok(config)
}

fn partition_prompts(
    prompts: Vec<PromptRecord>,
    gold_hashes: &HashSet<String>,
    eval_permille: usize,
) -> Vec<PartitionedPrompt> {
    prompts
        .into_iter()
        .map(|prompt| {
            let partition = if gold_hashes.contains(&prompt.prompt_hash) {
                Partition::Gold
            } else if prompt.bucket < eval_permille {
                Partition::Eval
            } else {
                Partition::Train
            };
            PartitionedPrompt { prompt, partition }
        })
        .collect()
}

fn write_partition(
    path: &Path,
    prompts: &[PartitionedPrompt],
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut out = String::new();
    out.push_str("prompt_hash\tspirit_id\tbucket\tpartition\ttier\tsource\tcluster\ttext\n");
    for prompt in prompts {
        out.push_str(&prompt.prompt.prompt_hash);
        out.push('\t');
        out.push_str(&prompt.prompt.spirit_id.to_string());
        out.push('\t');
        out.push_str(&prompt.prompt.bucket.to_string());
        out.push('\t');
        out.push_str(prompt.partition.as_str());
        out.push('\t');
        out.push_str(&escape_tsv(&prompt.prompt.tier));
        out.push('\t');
        out.push_str(&escape_tsv(&prompt.prompt.source));
        out.push('\t');
        out.push_str(&escape_tsv(&prompt.prompt.cluster));
        out.push('\t');
        out.push_str(&escape_tsv(&prompt.prompt.text));
        out.push('\n');
    }
    fs::write(path, out)?;
    Ok(())
}

fn evaluate_prompts(
    prompts: &[&PartitionedPrompt],
    rows: &[TextIndexRow],
    model: &LatentTextModel,
) -> Result<BTreeMap<String, FinalMetrics>, Box<dyn std::error::Error>> {
    let mut seen_targets = HashSet::new();
    let mut candidate_indices = Vec::new();
    let mut target_lookup = HashMap::new();
    for (row_index, row) in rows.iter().enumerate() {
        if seen_targets.insert(row.number) {
            target_lookup.insert(row.number, candidate_indices.len());
            candidate_indices.push(row_index);
        }
    }
    let mut image_latents = Vec::with_capacity(candidate_indices.len());
    for &row_index in &candidate_indices {
        image_latents.push(model.encode_image(&rows[row_index].image_features)?);
    }
    let mut metrics_by_tier: BTreeMap<String, Metrics> = BTreeMap::new();
    let mut all = Metrics::default();
    for prompt in prompts {
        let target_index = *target_lookup
            .get(&prompt.prompt.spirit_id)
            .ok_or_else(|| format!("missing target spirit_id {}", prompt.prompt.spirit_id))?;
        let row = &rows[candidate_indices[target_index]];
        let features = text_features(&prompt.prompt.text, model.text_feature_count);
        let text_latent = model.encode_text(&features)?;
        let text_prediction = model.decode_signature(&text_latent);
        let positive_score = dot_i16(&text_latent, &image_latents[target_index]);
        let mut rank = 1_usize;
        for (candidate_index, image_latent) in image_latents.iter().enumerate() {
            if candidate_index == target_index {
                continue;
            }
            let candidate = &rows[candidate_indices[candidate_index]];
            let score = dot_i16(&text_latent, image_latent);
            if score > positive_score || (score == positive_score && candidate.number < row.number)
            {
                rank = rank.saturating_add(1);
            }
        }
        let text_abs = signature_abs_error(&text_prediction, &row.signature);
        let latent_abs = latent_abs_error(&text_latent, &image_latents[target_index]);
        update_metrics(&mut all, rank, text_abs, latent_abs)?;
        let tier = metrics_by_tier
            .entry(prompt.prompt.tier.clone())
            .or_default();
        update_metrics(tier, rank, text_abs, latent_abs)?;
    }
    let mut final_metrics = BTreeMap::new();
    final_metrics.insert("all".to_string(), finalize_metrics(&all, model.latent_dim)?);
    for (tier, metrics) in metrics_by_tier {
        final_metrics.insert(tier, finalize_metrics(&metrics, model.latent_dim)?);
    }
    Ok(final_metrics)
}

fn update_metrics(
    metrics: &mut Metrics,
    rank: usize,
    text_abs: u64,
    latent_abs: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    metrics.count = metrics.count.saturating_add(1);
    if rank == 1 {
        metrics.top1 = metrics.top1.saturating_add(1);
    }
    if rank <= 5 {
        metrics.top5 = metrics.top5.saturating_add(1);
    }
    metrics.rank_total = metrics.rank_total.saturating_add(u64::try_from(rank)?);
    metrics.text_signature_abs = metrics.text_signature_abs.saturating_add(text_abs);
    metrics.text_image_latent_abs = metrics.text_image_latent_abs.saturating_add(latent_abs);
    Ok(())
}

fn finalize_metrics(
    metrics: &Metrics,
    latent_dim: usize,
) -> Result<FinalMetrics, Box<dyn std::error::Error>> {
    let count = metrics.count;
    let signature_count = u64::try_from(count.saturating_mul(SIGNATURE_BINS))?;
    let latent_count = u64::try_from(count.saturating_mul(latent_dim))?;
    Ok(FinalMetrics {
        count,
        top1: metrics.top1,
        top5: metrics.top5,
        top1_per_mille: per_mille(metrics.top1, count),
        top5_per_mille: per_mille(metrics.top5, count),
        mean_rank_q8: mean_q8(metrics.rank_total, u64::try_from(count)?),
        text_signature_mae_q8: mean_q8(metrics.text_signature_abs, signature_count),
        text_image_latent_mae_q8: mean_q8(metrics.text_image_latent_abs, latent_count),
    })
}

fn ledger_row(
    config: &Config,
    timestamp: u64,
    model_hash: &str,
    train_count: usize,
    eval_metrics: &BTreeMap<String, FinalMetrics>,
    gold_metrics: &BTreeMap<String, FinalMetrics>,
) -> String {
    let mut out = String::new();
    out.push('{');
    json_pair(&mut out, "schema", SCHEMA, true);
    number_pair(&mut out, "timestamp_unix_s", timestamp, true);
    json_pair(
        &mut out,
        "model",
        &config.model_path.display().to_string(),
        true,
    );
    json_pair(&mut out, "model_hash", model_hash, true);
    json_pair(
        &mut out,
        "prompt_set_version",
        &config.prompt_set_version,
        true,
    );
    json_pair(
        &mut out,
        "prompts",
        &config.prompts_path.display().to_string(),
        true,
    );
    json_pair(&mut out, "split_seed", &config.split_seed, true);
    number_pair(
        &mut out,
        "eval_permille",
        u64::try_from(config.eval_permille).unwrap_or(0),
        true,
    );
    number_pair(
        &mut out,
        "n_train_prompts",
        u64::try_from(train_count).unwrap_or(0),
        true,
    );
    out.push_str("\"retrieval_eval\":");
    metrics_object(&mut out, eval_metrics);
    out.push(',');
    out.push_str("\"retrieval_gold\":");
    metrics_object(&mut out, gold_metrics);
    out.push('}');
    out
}

fn metrics_object(out: &mut String, metrics: &BTreeMap<String, FinalMetrics>) {
    out.push('{');
    for (index, (tier, metric)) in metrics.iter().enumerate() {
        if index != 0 {
            out.push(',');
        }
        out.push('"');
        out.push_str(&json_escape(tier));
        out.push_str("\":{");
        number_pair(out, "count", u64::try_from(metric.count).unwrap_or(0), true);
        number_pair(out, "top1", u64::try_from(metric.top1).unwrap_or(0), true);
        number_pair(out, "top5", u64::try_from(metric.top5).unwrap_or(0), true);
        number_pair(
            out,
            "top1_per_mille",
            u64::try_from(metric.top1_per_mille).unwrap_or(0),
            true,
        );
        number_pair(
            out,
            "top5_per_mille",
            u64::try_from(metric.top5_per_mille).unwrap_or(0),
            true,
        );
        number_pair(out, "mean_rank_q8", metric.mean_rank_q8, true);
        number_pair(out, "mae_q8", metric.text_signature_mae_q8, true);
        number_pair(
            out,
            "text_image_latent_mae_q8",
            metric.text_image_latent_mae_q8,
            false,
        );
        out.push('}');
    }
    out.push('}');
}

fn append_ledger(path: &Path, row: &str) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(file, "{row}")?;
    Ok(())
}

fn unix_timestamp() -> Result<u64, Box<dyn std::error::Error>> {
    Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs())
}

fn per_mille(count: usize, total: usize) -> usize {
    if total == 0 {
        return 0;
    }
    count.saturating_mul(1000) / total
}

fn json_pair(out: &mut String, key: &str, value: &str, comma: bool) {
    out.push('"');
    out.push_str(key);
    out.push_str("\":\"");
    out.push_str(&json_escape(value));
    out.push('"');
    if comma {
        out.push(',');
    }
}

fn number_pair(out: &mut String, key: &str, value: u64, comma: bool) {
    out.push('"');
    out.push_str(key);
    out.push_str("\":");
    out.push_str(&value.to_string());
    if comma {
        out.push(',');
    }
}

fn escape_tsv(value: &str) -> String {
    value
        .replace('\t', " ")
        .replace(['\r', '\n'], " ")
        .trim()
        .to_string()
}
