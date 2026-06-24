#![deny(unsafe_code)]

use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use nsrl_train::solomon_latent::{
    DEFAULT_EVAL_PERMILLE, DEFAULT_PROMPT_SPLIT_SEED, default_gold_path, prompt_partition_bucket,
    read_gold_hashes, read_prompt_records, read_text_index_rows,
};

const SCHEMA: &str = "nsrl.solomon_latent_trace.v1";
const MODEL_MAGIC: &[u8; 8] = b"NSRLLAT1";
// Match the shared solomon_latent module's 16x16 signature grid (256 bins).
const SIGNATURE_GRID: usize = 16;
const SIGNATURE_BINS: usize = SIGNATURE_GRID * SIGNATURE_GRID;

#[derive(Debug, Clone)]
struct Config {
    text_index_path: PathBuf,
    prompts_path: Option<PathBuf>,
    gold_path: Option<PathBuf>,
    out_dir: PathBuf,
    model_out: PathBuf,
    epochs: usize,
    latent_dim: usize,
    text_feature_count: usize,
    text_encoder_shift: u8,
    image_encoder_shift: u8,
    decoder_shift: u8,
    encoder_learning_shift: u8,
    decoder_learning_shift: u8,
    align_learning_shift: u8,
    contrastive_learning_shift: u8,
    contrastive_negatives: usize,
    contrastive_margin: i64,
    bias_learning_shift: u8,
    max_weight_delta: i16,
    max_bias_delta: i16,
    eval_permille: usize,
    split_seed: String,
    seed: String,
}

impl Default for Config {
    fn default() -> Self {
        let out_dir = PathBuf::from("data/processed/key-solomon-goetia-latent-v1");
        Self {
            text_index_path: PathBuf::from(
                "data/processed/key-solomon-goetia-text-index-pg72679/solomon-spirit-text-signatures.tsv",
            ),
            prompts_path: None,
            gold_path: None,
            model_out: out_dir.join("model.nsrllat"),
            out_dir,
            epochs: 120,
            latent_dim: 64,
            text_feature_count: 512,
            text_encoder_shift: 8,
            image_encoder_shift: 6,
            decoder_shift: 8,
            encoder_learning_shift: 20,
            decoder_learning_shift: 16,
            align_learning_shift: 12,
            contrastive_learning_shift: 10,
            contrastive_negatives: 4,
            contrastive_margin: 4096,
            bias_learning_shift: 5,
            max_weight_delta: 8,
            max_bias_delta: 16,
            eval_permille: DEFAULT_EVAL_PERMILLE,
            split_seed: String::from(DEFAULT_PROMPT_SPLIT_SEED),
            seed: String::from("solomon-latent-v1"),
        }
    }
}

#[derive(Debug, Clone)]
struct Row {
    number: usize,
    primary_name: String,
    aliases: String,
    slice_id: String,
    variant_id: String,
    source_lanes: String,
    text: String,
    signature: [u16; SIGNATURE_BINS],
    target_latent: Vec<i16>,
    text_features: Vec<i16>,
    image_features: [i16; SIGNATURE_BINS],
}

#[derive(Debug, Clone)]
struct LatentModel {
    latent_dim: usize,
    text_feature_count: usize,
    text_encoder_shift: u8,
    image_encoder_shift: u8,
    decoder_shift: u8,
    text_weights: Vec<i8>,
    text_biases: Vec<i16>,
    image_weights: Vec<i8>,
    image_biases: Vec<i16>,
    decoder_weights: Vec<i8>,
    decoder_biases: [i16; SIGNATURE_BINS],
}

#[derive(Debug, Clone)]
struct EpochTrace {
    epoch: usize,
    text_signature_mae_q8: u64,
    image_signature_mae_q8: u64,
    text_image_latent_mae_q8: u64,
}

#[derive(Debug, Clone)]
struct EvalTrace {
    retrieval_top1: usize,
    retrieval_top5: usize,
    retrieval_top1_per_mille: usize,
    retrieval_top5_per_mille: usize,
    mean_rank_q8: u64,
    text_signature_mae_q8: u64,
    image_signature_mae_q8: u64,
    text_image_latent_mae_q8: u64,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("nsrl-solomon-latent-train: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args(env::args().skip(1))?;
    if config.epochs == 0
        || config.latent_dim == 0
        || config.text_feature_count == 0
        || config.latent_dim > 512
        || config.text_feature_count > 4096
    {
        return Err(
            "epochs, latent dim, and text feature count must be positive bounded values".into(),
        );
    }
    if config.max_weight_delta < 0 || config.max_bias_delta < 0 {
        return Err("max deltas must be non-negative".into());
    }
    if config.eval_permille > 900 {
        return Err("--eval-permille must be <= 900".into());
    }
    fs::create_dir_all(&config.out_dir)?;

    let rows = read_rows(&config)?;
    let mut model = LatentModel::new(&config, &rows)?;
    let mut epochs = Vec::with_capacity(config.epochs);
    for epoch in 0..config.epochs {
        epochs.push(train_epoch(&config, &rows, &mut model, epoch)?);
    }
    write_model(&config.model_out, &model)?;
    let eval = evaluate(&rows, &model)?;
    write_trace(&config, &model, &rows, &epochs, &eval)?;

    println!(
        "{{\"schema\":\"{}\",\"model\":\"{}\",\"rows\":{},\"retrieval_top1\":{},\"text_signature_mae\":\"{}\",\"image_signature_mae\":\"{}\"}}",
        SCHEMA,
        json_escape(&config.model_out.display().to_string()),
        rows.len(),
        eval.retrieval_top1,
        format_q8(eval.text_signature_mae_q8),
        format_q8(eval.image_signature_mae_q8),
    );
    Ok(())
}

fn usage() {
    println!(
        "Usage: nsrl-solomon-latent-train [--text-index PATH] [--prompts PATH] [--gold PATH] [--eval-permille N] [--split-seed TEXT] [--out-dir PATH] [--model-out PATH] [--epochs N] [--latent-dim N] [--text-features N] [--seed TEXT]"
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
            "--text-index" => {
                config.text_index_path =
                    PathBuf::from(args.next().ok_or("--text-index requires PATH")?);
            }
            "--prompts" => {
                config.prompts_path =
                    Some(PathBuf::from(args.next().ok_or("--prompts requires PATH")?));
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
            "--out-dir" => {
                config.out_dir = PathBuf::from(args.next().ok_or("--out-dir requires PATH")?);
                config.model_out = config.out_dir.join("model.nsrllat");
            }
            "--model-out" => {
                config.model_out = PathBuf::from(args.next().ok_or("--model-out requires PATH")?);
            }
            "--epochs" => {
                config.epochs = args.next().ok_or("--epochs requires N")?.parse()?;
            }
            "--latent-dim" => {
                config.latent_dim = args.next().ok_or("--latent-dim requires N")?.parse()?;
            }
            "--text-features" => {
                config.text_feature_count =
                    args.next().ok_or("--text-features requires N")?.parse()?;
            }
            "--text-encoder-shift" => {
                config.text_encoder_shift = args
                    .next()
                    .ok_or("--text-encoder-shift requires N")?
                    .parse()?;
            }
            "--image-encoder-shift" => {
                config.image_encoder_shift = args
                    .next()
                    .ok_or("--image-encoder-shift requires N")?
                    .parse()?;
            }
            "--decoder-shift" => {
                config.decoder_shift = args.next().ok_or("--decoder-shift requires N")?.parse()?;
            }
            "--encoder-learning-shift" => {
                config.encoder_learning_shift = args
                    .next()
                    .ok_or("--encoder-learning-shift requires N")?
                    .parse()?;
            }
            "--decoder-learning-shift" => {
                config.decoder_learning_shift = args
                    .next()
                    .ok_or("--decoder-learning-shift requires N")?
                    .parse()?;
            }
            "--align-learning-shift" => {
                config.align_learning_shift = args
                    .next()
                    .ok_or("--align-learning-shift requires N")?
                    .parse()?;
            }
            "--contrastive-learning-shift" => {
                config.contrastive_learning_shift = args
                    .next()
                    .ok_or("--contrastive-learning-shift requires N")?
                    .parse()?;
            }
            "--contrastive-negatives" => {
                config.contrastive_negatives = args
                    .next()
                    .ok_or("--contrastive-negatives requires N")?
                    .parse()?;
            }
            "--contrastive-margin" => {
                config.contrastive_margin = args
                    .next()
                    .ok_or("--contrastive-margin requires N")?
                    .parse()?;
            }
            "--bias-learning-shift" => {
                config.bias_learning_shift = args
                    .next()
                    .ok_or("--bias-learning-shift requires N")?
                    .parse()?;
            }
            "--max-weight-delta" => {
                config.max_weight_delta = args
                    .next()
                    .ok_or("--max-weight-delta requires N")?
                    .parse()?;
            }
            "--max-bias-delta" => {
                config.max_bias_delta =
                    args.next().ok_or("--max-bias-delta requires N")?.parse()?;
            }
            "--seed" => {
                config.seed = args.next().ok_or("--seed requires TEXT")?;
            }
            _ => return Err(format!("unknown option: {arg}").into()),
        }
    }
    Ok(config)
}

fn read_rows(config: &Config) -> Result<Vec<Row>, Box<dyn std::error::Error>> {
    if let Some(prompts_path) = &config.prompts_path {
        return read_prompt_training_rows(config, prompts_path);
    }
    let text = fs::read_to_string(&config.text_index_path)?;
    let mut rows = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        if line_index == 0 || line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 9 {
            return Err(format!(
                "{} line {} has {} fields, expected 9",
                config.text_index_path.display(),
                line_index + 1,
                fields.len()
            )
            .into());
        }
        let signature = parse_signature(fields[7], &config.text_index_path, line_index + 1)?;
        let mut image_features = [0_i16; SIGNATURE_BINS];
        for (out, &value) in image_features.iter_mut().zip(signature.iter()) {
            *out = i16::try_from(value)?.saturating_sub(64).clamp(-511, 511);
        }
        let text = format!(
            "{} {} {}",
            fields[1],
            fields[2].replace('|', " "),
            fields[8]
        );
        let variant_id = fields
            .get(9)
            .filter(|value| !value.trim().is_empty())
            .map(|value| (*value).to_string())
            .unwrap_or_else(|| "canonical".to_string());
        let source_lanes = fields
            .get(10)
            .filter(|value| !value.trim().is_empty())
            .map(|value| (*value).to_string())
            .unwrap_or_else(|| "goetia".to_string());
        rows.push(Row {
            number: fields[0].parse()?,
            primary_name: fields[1].to_string(),
            aliases: fields[2].to_string(),
            slice_id: fields[3].to_string(),
            variant_id,
            source_lanes,
            text: fields[8].to_string(),
            signature,
            target_latent: signature_latent(
                &signature,
                config.latent_dim,
                &config.seed,
                config.image_encoder_shift,
            ),
            text_features: text_features(&text, config.text_feature_count),
            image_features,
        });
    }
    rows.sort_by(|left, right| {
        left.number
            .cmp(&right.number)
            .then_with(|| left.variant_id.cmp(&right.variant_id))
    });
    if rows.is_empty() {
        return Err(format!("{} has no rows", config.text_index_path.display()).into());
    }
    Ok(rows)
}

fn read_prompt_training_rows(
    config: &Config,
    prompts_path: &Path,
) -> Result<Vec<Row>, Box<dyn std::error::Error>> {
    let base_rows = read_text_index_rows(&config.text_index_path)?;
    let mut by_number = HashMap::new();
    for row in base_rows {
        by_number.entry(row.number).or_insert(row);
    }
    let prompts = read_prompt_records(prompts_path, &config.split_seed)?;
    let gold_path = config
        .gold_path
        .clone()
        .unwrap_or_else(|| default_gold_path(prompts_path));
    let gold_hashes = read_gold_hashes(&gold_path)?;
    let mut rows = Vec::new();
    for prompt in prompts {
        if gold_hashes.contains(&prompt.prompt_hash)
            || prompt_partition_bucket(&prompt, &config.split_seed) < config.eval_permille
        {
            continue;
        }
        let base = by_number
            .get(&prompt.spirit_id)
            .ok_or_else(|| format!("prompt references missing spirit_id {}", prompt.spirit_id))?;
        rows.push(Row {
            number: base.number,
            primary_name: base.primary_name.clone(),
            aliases: base.aliases.clone(),
            slice_id: base.slice_id.clone(),
            variant_id: prompt.prompt_hash,
            source_lanes: format!("{}:{}", prompt.source, prompt.tier),
            text: prompt.text.clone(),
            signature: base.signature,
            target_latent: signature_latent(
                &base.signature,
                config.latent_dim,
                &config.seed,
                config.image_encoder_shift,
            ),
            text_features: text_features(&prompt.text, config.text_feature_count),
            image_features: base.image_features,
        });
    }
    rows.sort_by(|left, right| {
        left.number
            .cmp(&right.number)
            .then_with(|| left.variant_id.cmp(&right.variant_id))
    });
    if rows.is_empty() {
        return Err(format!(
            "{} leaves no train prompts after gold/eval filtering",
            prompts_path.display()
        )
        .into());
    }
    Ok(rows)
}

fn parse_signature(
    text: &str,
    path: &Path,
    line_number: usize,
) -> Result<[u16; SIGNATURE_BINS], Box<dyn std::error::Error>> {
    let mut signature = [0_u16; SIGNATURE_BINS];
    let parts: Vec<&str> = text.split(',').collect();
    if parts.len() != SIGNATURE_BINS {
        return Err(format!(
            "{} line {} has {} signature bins, expected {}",
            path.display(),
            line_number,
            parts.len(),
            SIGNATURE_BINS
        )
        .into());
    }
    for (index, part) in parts.iter().enumerate() {
        signature[index] = part.parse()?;
    }
    Ok(signature)
}

impl LatentModel {
    fn new(config: &Config, rows: &[Row]) -> Result<Self, Box<dyn std::error::Error>> {
        let text_weight_count = config
            .latent_dim
            .checked_mul(config.text_feature_count)
            .ok_or("text weight count overflow")?;
        let image_weight_count = config
            .latent_dim
            .checked_mul(SIGNATURE_BINS)
            .ok_or("image weight count overflow")?;
        let decoder_weight_count = SIGNATURE_BINS
            .checked_mul(config.latent_dim)
            .ok_or("decoder weight count overflow")?;
        let mut text_weights = vec![0_i8; text_weight_count];
        let mut image_weights = vec![0_i8; image_weight_count];
        let mut decoder_weights = vec![0_i8; decoder_weight_count];
        for dim in 0..config.latent_dim {
            for feature in 0..config.text_feature_count {
                text_weights[dim * config.text_feature_count + feature] =
                    initial_weight(&config.seed, "text", dim, feature);
            }
            for feature in 0..SIGNATURE_BINS {
                image_weights[dim * SIGNATURE_BINS + feature] =
                    initial_projection_weight(&config.seed, dim, feature);
            }
        }
        for bin in 0..SIGNATURE_BINS {
            for dim in 0..config.latent_dim {
                decoder_weights[bin * config.latent_dim + dim] =
                    initial_weight(&config.seed, "decode", bin, dim);
            }
        }
        let mut decoder_biases = [0_i16; SIGNATURE_BINS];
        let target_indices = unique_target_indices(rows);
        for (bin, bias) in decoder_biases.iter_mut().enumerate().take(SIGNATURE_BINS) {
            let mut total = 0_u64;
            for &row_index in &target_indices {
                let row = &rows[row_index];
                total = total.saturating_add(u64::from(row.signature[bin]));
            }
            *bias = i16::try_from(
                (total + u64::try_from(target_indices.len())? / 2)
                    / u64::try_from(target_indices.len())?,
            )?;
        }
        Ok(Self {
            latent_dim: config.latent_dim,
            text_feature_count: config.text_feature_count,
            text_encoder_shift: config.text_encoder_shift,
            image_encoder_shift: config.image_encoder_shift,
            decoder_shift: config.decoder_shift,
            text_weights,
            text_biases: vec![0; config.latent_dim],
            image_weights,
            image_biases: vec![0; config.latent_dim],
            decoder_weights,
            decoder_biases,
        })
    }

    fn encode_text(&self, features: &[i16]) -> Vec<i16> {
        encode_latent(
            features,
            &self.text_weights,
            &self.text_biases,
            self.text_feature_count,
            self.latent_dim,
            self.text_encoder_shift,
        )
    }

    fn encode_image(&self, features: &[i16; SIGNATURE_BINS]) -> Vec<i16> {
        encode_latent(
            features,
            &self.image_weights,
            &self.image_biases,
            SIGNATURE_BINS,
            self.latent_dim,
            self.image_encoder_shift,
        )
    }

    fn decode_signature(&self, latent: &[i16]) -> [u16; SIGNATURE_BINS] {
        let mut out = [0_u16; SIGNATURE_BINS];
        for (bin, out_value) in out.iter_mut().enumerate() {
            let mut acc = i64::from(self.decoder_biases[bin]) << self.decoder_shift;
            for (dim, latent_value) in latent.iter().enumerate().take(self.latent_dim) {
                let weight = self.decoder_weights[bin * self.latent_dim + dim];
                acc = acc.saturating_add(i64::from(weight) * i64::from(*latent_value));
            }
            let value = signed_round_shift(acc, self.decoder_shift).clamp(0, 255);
            *out_value = u16::try_from(value).unwrap_or(0);
        }
        out
    }
}

fn encode_latent(
    features: &[i16],
    weights: &[i8],
    biases: &[i16],
    feature_count: usize,
    latent_dim: usize,
    shift: u8,
) -> Vec<i16> {
    let mut out = vec![0_i16; latent_dim];
    for dim in 0..latent_dim {
        let mut acc = 0_i64;
        for (feature, &value) in features.iter().enumerate().take(feature_count) {
            let weight = weights[dim * feature_count + feature];
            acc = acc.saturating_add(i64::from(weight) * i64::from(value));
        }
        let value = signed_round_shift(acc, shift).saturating_add(i64::from(biases[dim]));
        out[dim] = value.clamp(-511, 511) as i16;
    }
    out
}

fn unique_target_indices(rows: &[Row]) -> Vec<usize> {
    let mut indices = Vec::new();
    let mut last_number = None;
    for (index, row) in rows.iter().enumerate() {
        if last_number != Some(row.number) {
            indices.push(index);
            last_number = Some(row.number);
        }
    }
    indices
}

fn unique_target_count(rows: &[Row]) -> usize {
    unique_target_indices(rows).len()
}

fn is_first_target_row(rows: &[Row], row_index: usize) -> bool {
    row_index == 0 || rows[row_index - 1].number != rows[row_index].number
}

fn train_epoch(
    config: &Config,
    rows: &[Row],
    model: &mut LatentModel,
    epoch: usize,
) -> Result<EpochTrace, Box<dyn std::error::Error>> {
    let mut text_abs = 0_u64;
    let mut image_abs = 0_u64;
    let mut latent_abs = 0_u64;
    for row_index in 0..rows.len() {
        let row = &rows[row_index];
        if is_first_target_row(rows, row_index) {
            let image_latent = model.encode_image(&row.image_features);
            let image_prediction = model.decode_signature(&image_latent);
            image_abs =
                image_abs.saturating_add(signature_abs_error(&image_prediction, &row.signature));
            update_decoder(
                config,
                model,
                &row.target_latent,
                &row.signature,
                &image_prediction,
            );
            update_decoder(
                config,
                model,
                &image_latent,
                &row.signature,
                &image_prediction,
            );
        }

        let image_latent = model.encode_image(&row.image_features);
        let text_latent = model.encode_text(&row.text_features);
        let text_prediction = model.decode_signature(&text_latent);
        text_abs = text_abs.saturating_add(signature_abs_error(&text_prediction, &row.signature));
        update_decoder(
            config,
            model,
            &text_latent,
            &row.signature,
            &text_prediction,
        );
        update_text_encoder_reconstruction(
            config,
            model,
            &row.text_features,
            &row.signature,
            &text_prediction,
        );
        update_text_encoder_alignment(
            config,
            model,
            &row.text_features,
            &row.target_latent,
            &text_latent,
        );
        latent_abs = latent_abs.saturating_add(latent_abs_error(&image_latent, &text_latent));
        update_contrastive_rank(config, rows, model, row_index, epoch);
    }
    let target_count = unique_target_count(rows);
    let text_signature_count = u64::try_from(rows.len().saturating_mul(SIGNATURE_BINS))?;
    let image_signature_count = u64::try_from(target_count.saturating_mul(SIGNATURE_BINS))?;
    let latent_count = u64::try_from(rows.len().saturating_mul(model.latent_dim))?;
    Ok(EpochTrace {
        epoch: epoch + 1,
        text_signature_mae_q8: mean_q8(text_abs, text_signature_count),
        image_signature_mae_q8: mean_q8(image_abs, image_signature_count),
        text_image_latent_mae_q8: mean_q8(latent_abs, latent_count),
    })
}

fn update_decoder(
    config: &Config,
    model: &mut LatentModel,
    latent: &[i16],
    target: &[u16; SIGNATURE_BINS],
    prediction: &[u16; SIGNATURE_BINS],
) {
    for bin in 0..SIGNATURE_BINS {
        let error = i64::from(target[bin]) - i64::from(prediction[bin]);
        let bias_delta = signed_round_shift(error, config.bias_learning_shift).clamp(
            -i64::from(config.max_bias_delta),
            i64::from(config.max_bias_delta),
        );
        model.decoder_biases[bin] = saturating_i16_add(model.decoder_biases[bin], bias_delta);
        for (dim, &latent_value) in latent.iter().enumerate().take(model.latent_dim) {
            let product = error.saturating_mul(i64::from(latent_value));
            let delta = signed_round_shift(product, config.decoder_learning_shift).clamp(
                -i64::from(config.max_weight_delta),
                i64::from(config.max_weight_delta),
            );
            let index = bin * model.latent_dim + dim;
            model.decoder_weights[index] = saturating_i8_add(model.decoder_weights[index], delta);
        }
    }
}

fn update_text_encoder_reconstruction(
    config: &Config,
    model: &mut LatentModel,
    features: &[i16],
    target: &[u16; SIGNATURE_BINS],
    prediction: &[u16; SIGNATURE_BINS],
) {
    let gradients = latent_gradients(model, target, prediction);
    update_encoder_weights(
        &mut model.text_weights,
        &mut model.text_biases,
        model.text_feature_count,
        model.latent_dim,
        features,
        &gradients,
        config.encoder_learning_shift,
        config.bias_learning_shift,
        config.max_weight_delta,
        config.max_bias_delta,
    );
}

fn update_text_encoder_alignment(
    config: &Config,
    model: &mut LatentModel,
    features: &[i16],
    image_latent: &[i16],
    text_latent: &[i16],
) {
    let gradients: Vec<i64> = image_latent
        .iter()
        .zip(text_latent.iter())
        .map(|(&image, &text)| i64::from(image) - i64::from(text))
        .collect();
    update_encoder_weights(
        &mut model.text_weights,
        &mut model.text_biases,
        model.text_feature_count,
        model.latent_dim,
        features,
        &gradients,
        config.align_learning_shift,
        config.bias_learning_shift,
        config.max_weight_delta,
        config.max_bias_delta,
    );
}

fn update_contrastive_rank(
    config: &Config,
    rows: &[Row],
    model: &mut LatentModel,
    row_index: usize,
    epoch: usize,
) {
    if rows.len() < 2 || config.contrastive_negatives == 0 {
        return;
    }
    let row = &rows[row_index];
    for negative_offset in 0..config.contrastive_negatives {
        let mut negative_index = row_index.saturating_add(1).saturating_add(
            (epoch.saturating_add(1)).saturating_mul(negative_offset.saturating_mul(17) + 7),
        ) % rows.len();
        let mut attempts = 0_usize;
        while rows[negative_index].number == row.number && attempts < rows.len() {
            negative_index = (negative_index + 1) % rows.len();
            attempts = attempts.saturating_add(1);
        }
        if rows[negative_index].number == row.number {
            continue;
        }
        let negative = &rows[negative_index];
        let text_latent = model.encode_text(&row.text_features);
        let positive_latent = &row.target_latent;
        let negative_latent = &negative.target_latent;
        let positive_score = dot_i16(&text_latent, positive_latent);
        let negative_score = dot_i16(&text_latent, negative_latent);
        if positive_score > negative_score.saturating_add(config.contrastive_margin) {
            continue;
        }
        let text_gradients: Vec<i64> = positive_latent
            .iter()
            .zip(negative_latent.iter())
            .map(|(&positive, &negative)| i64::from(positive) - i64::from(negative))
            .collect();
        update_encoder_weights(
            &mut model.text_weights,
            &mut model.text_biases,
            model.text_feature_count,
            model.latent_dim,
            &row.text_features,
            &text_gradients,
            config.contrastive_learning_shift,
            config.bias_learning_shift,
            config.max_weight_delta,
            config.max_bias_delta,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn update_encoder_weights(
    weights: &mut [i8],
    biases: &mut [i16],
    feature_count: usize,
    latent_dim: usize,
    features: &[i16],
    gradients: &[i64],
    learning_shift: u8,
    bias_learning_shift: u8,
    max_weight_delta: i16,
    max_bias_delta: i16,
) {
    for dim in 0..latent_dim {
        let gradient = gradients[dim];
        let bias_delta = signed_round_shift(gradient, bias_learning_shift)
            .clamp(-i64::from(max_bias_delta), i64::from(max_bias_delta));
        biases[dim] = saturating_i16_add(biases[dim], bias_delta);
        for (feature, &feature_value) in features.iter().enumerate().take(feature_count) {
            if feature_value == 0 || gradient == 0 {
                continue;
            }
            let product = gradient.saturating_mul(i64::from(feature_value));
            let delta = signed_round_shift(product, learning_shift)
                .clamp(-i64::from(max_weight_delta), i64::from(max_weight_delta));
            let index = dim * feature_count + feature;
            weights[index] = saturating_i8_add(weights[index], delta);
        }
    }
}

fn latent_gradients(
    model: &LatentModel,
    target: &[u16; SIGNATURE_BINS],
    prediction: &[u16; SIGNATURE_BINS],
) -> Vec<i64> {
    let mut gradients = vec![0_i64; model.latent_dim];
    for bin in 0..SIGNATURE_BINS {
        let error = i64::from(target[bin]) - i64::from(prediction[bin]);
        for (dim, gradient) in gradients.iter_mut().enumerate() {
            let weight = model.decoder_weights[bin * model.latent_dim + dim];
            *gradient = gradient.saturating_add(error.saturating_mul(i64::from(weight)));
        }
    }
    for gradient in &mut gradients {
        *gradient = signed_round_shift(*gradient, 4);
    }
    gradients
}

fn evaluate(rows: &[Row], model: &LatentModel) -> Result<EvalTrace, Box<dyn std::error::Error>> {
    let target_indices = unique_target_indices(rows);
    let mut image_latents = Vec::with_capacity(target_indices.len());
    for &row_index in &target_indices {
        image_latents.push(model.encode_image(&rows[row_index].image_features));
    }
    let mut retrieval_top1 = 0_usize;
    let mut retrieval_top5 = 0_usize;
    let mut rank_total = 0_u64;
    let mut text_abs = 0_u64;
    let mut image_abs = 0_u64;
    let mut latent_abs = 0_u64;
    for (row_index, row) in rows.iter().enumerate() {
        let text_latent = model.encode_text(&row.text_features);
        let text_prediction = model.decode_signature(&text_latent);
        let target_latent_index = target_indices
            .iter()
            .position(|&target_index| rows[target_index].number == row.number)
            .ok_or("missing image target for row")?;
        let image_prediction = model.decode_signature(&image_latents[target_latent_index]);
        text_abs = text_abs.saturating_add(signature_abs_error(&text_prediction, &row.signature));
        if is_first_target_row(rows, row_index) {
            image_abs =
                image_abs.saturating_add(signature_abs_error(&image_prediction, &row.signature));
        }
        latent_abs = latent_abs.saturating_add(latent_abs_error(
            &text_latent,
            &image_latents[target_latent_index],
        ));
        let positive_score = dot_i16(&text_latent, &image_latents[target_latent_index]);
        let mut rank = 1_usize;
        for (candidate_index, image_latent) in image_latents.iter().enumerate() {
            let candidate_row = &rows[target_indices[candidate_index]];
            if candidate_row.number == row.number {
                continue;
            }
            let score = dot_i16(&text_latent, image_latent);
            if score > positive_score
                || (score == positive_score && candidate_row.number < row.number)
            {
                rank = rank.saturating_add(1);
            }
        }
        if rank == 1 {
            retrieval_top1 = retrieval_top1.saturating_add(1);
        }
        if rank <= 5 {
            retrieval_top5 = retrieval_top5.saturating_add(1);
        }
        rank_total = rank_total.saturating_add(u64::try_from(rank)?);
    }
    let row_count = rows.len();
    let target_count = target_indices.len();
    let text_signature_count = u64::try_from(row_count.saturating_mul(SIGNATURE_BINS))?;
    let image_signature_count = u64::try_from(target_count.saturating_mul(SIGNATURE_BINS))?;
    let latent_count = u64::try_from(row_count.saturating_mul(model.latent_dim))?;
    Ok(EvalTrace {
        retrieval_top1,
        retrieval_top5,
        retrieval_top1_per_mille: retrieval_top1.saturating_mul(1000) / row_count,
        retrieval_top5_per_mille: retrieval_top5.saturating_mul(1000) / row_count,
        mean_rank_q8: mean_q8(rank_total, u64::try_from(row_count)?),
        text_signature_mae_q8: mean_q8(text_abs, text_signature_count),
        image_signature_mae_q8: mean_q8(image_abs, image_signature_count),
        text_image_latent_mae_q8: mean_q8(latent_abs, latent_count),
    })
}

fn write_model(path: &Path, model: &LatentModel) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut bytes = Vec::new();
    bytes.extend_from_slice(MODEL_MAGIC);
    bytes.extend_from_slice(&checked_u32(model.latent_dim, "latent_dim")?.to_le_bytes());
    bytes.extend_from_slice(
        &checked_u32(model.text_feature_count, "text_feature_count")?.to_le_bytes(),
    );
    bytes.extend_from_slice(&checked_u32(SIGNATURE_BINS, "signature_bins")?.to_le_bytes());
    bytes.extend_from_slice(&u32::from(model.text_encoder_shift).to_le_bytes());
    bytes.extend_from_slice(&u32::from(model.image_encoder_shift).to_le_bytes());
    bytes.extend_from_slice(&u32::from(model.decoder_shift).to_le_bytes());
    bytes.extend_from_slice(&checked_u32(SIGNATURE_GRID, "signature_grid")?.to_le_bytes());
    bytes.extend_from_slice(
        &model
            .text_weights
            .iter()
            .map(|&value| value as u8)
            .collect::<Vec<_>>(),
    );
    for &bias in &model.text_biases {
        bytes.extend_from_slice(&bias.to_le_bytes());
    }
    bytes.extend_from_slice(
        &model
            .image_weights
            .iter()
            .map(|&value| value as u8)
            .collect::<Vec<_>>(),
    );
    for &bias in &model.image_biases {
        bytes.extend_from_slice(&bias.to_le_bytes());
    }
    bytes.extend_from_slice(
        &model
            .decoder_weights
            .iter()
            .map(|&value| value as u8)
            .collect::<Vec<_>>(),
    );
    for &bias in &model.decoder_biases {
        bytes.extend_from_slice(&bias.to_le_bytes());
    }
    fs::write(path, bytes)?;
    Ok(())
}

fn write_trace(
    config: &Config,
    model: &LatentModel,
    rows: &[Row],
    epochs: &[EpochTrace],
    eval: &EvalTrace,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut out = String::new();
    out.push_str("{\n");
    json_field(&mut out, "schema", SCHEMA, true);
    json_field(
        &mut out,
        "text_index",
        &config.text_index_path.display().to_string(),
        true,
    );
    json_field(
        &mut out,
        "model_out",
        &config.model_out.display().to_string(),
        true,
    );
    json_field(&mut out, "seed", &config.seed, true);
    json_field(&mut out, "split_seed", &config.split_seed, true);
    if let Some(prompts_path) = &config.prompts_path {
        json_field(
            &mut out,
            "prompts",
            &prompts_path.display().to_string(),
            true,
        );
        let gold_path = config
            .gold_path
            .clone()
            .unwrap_or_else(|| default_gold_path(prompts_path));
        json_field(&mut out, "gold", &gold_path.display().to_string(), true);
    }
    number_field(&mut out, "eval_permille", config.eval_permille, true);
    number_field(&mut out, "rows", rows.len(), true);
    number_field(&mut out, "image_targets", unique_target_count(rows), true);
    number_field(&mut out, "latent_dim", model.latent_dim, true);
    number_field(
        &mut out,
        "text_feature_count",
        model.text_feature_count,
        true,
    );
    number_field(&mut out, "signature_bins", SIGNATURE_BINS, true);
    number_field(&mut out, "epochs", config.epochs, true);
    number_field(&mut out, "retrieval_top1", eval.retrieval_top1, true);
    number_field(&mut out, "retrieval_top5", eval.retrieval_top5, true);
    number_field(
        &mut out,
        "retrieval_top1_per_mille",
        eval.retrieval_top1_per_mille,
        true,
    );
    number_field(
        &mut out,
        "retrieval_top5_per_mille",
        eval.retrieval_top5_per_mille,
        true,
    );
    number_field(
        &mut out,
        "mean_rank_q8",
        usize::try_from(eval.mean_rank_q8)?,
        true,
    );
    number_field(
        &mut out,
        "text_signature_mae_q8",
        usize::try_from(eval.text_signature_mae_q8)?,
        true,
    );
    number_field(
        &mut out,
        "image_signature_mae_q8",
        usize::try_from(eval.image_signature_mae_q8)?,
        true,
    );
    number_field(
        &mut out,
        "text_image_latent_mae_q8",
        usize::try_from(eval.text_image_latent_mae_q8)?,
        true,
    );
    out.push_str("  \"row_sample\":[");
    for (index, row) in rows.iter().take(8).enumerate() {
        if index != 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "{{\"number\":{},\"name\":\"{}\",\"slice_id\":\"{}\",\"variant_id\":\"{}\",\"source_lanes\":\"{}\",\"text_bytes\":{},\"alias_bytes\":{}}}",
            row.number,
            json_escape(&row.primary_name),
            json_escape(&row.slice_id),
            json_escape(&row.variant_id),
            json_escape(&row.source_lanes),
            row.text.len(),
            row.aliases.len(),
        ));
    }
    out.push_str("],\n");
    out.push_str("  \"epoch_trace\":[");
    for (index, epoch) in epochs.iter().enumerate() {
        if index != 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "{{\"epoch\":{},\"text_signature_mae_q8\":{},\"image_signature_mae_q8\":{},\"text_image_latent_mae_q8\":{}}}",
            epoch.epoch,
            epoch.text_signature_mae_q8,
            epoch.image_signature_mae_q8,
            epoch.text_image_latent_mae_q8,
        ));
    }
    out.push_str("]\n");
    out.push_str("}\n");
    fs::write(config.out_dir.join("trace.json"), out)?;
    Ok(())
}

fn text_features(text: &str, feature_count: usize) -> Vec<i16> {
    let mut features = vec![0_i16; feature_count];
    for (position, token) in tokenize_text(text).into_iter().enumerate() {
        if token.len() < 2 {
            continue;
        }
        let hash = hash_text(&token);
        let bin = usize::try_from(hash).unwrap_or(0) % feature_count;
        let length = i16::try_from(token.len().min(12)).unwrap_or(12);
        let position_bonus = i16::try_from(position % 5).unwrap_or(0).saturating_mul(4);
        let value = 64_i16
            .saturating_add(length.saturating_mul(12))
            .saturating_add(position_bonus);
        let signed = if hash & 0x8000_0000 == 0 {
            value
        } else {
            -value
        };
        features[bin] = features[bin].saturating_add(signed).clamp(-511, 511);
    }
    features
}

fn signature_latent(
    signature: &[u16; SIGNATURE_BINS],
    latent_dim: usize,
    seed: &str,
    shift: u8,
) -> Vec<i16> {
    let mut latent = vec![0_i16; latent_dim];
    for (dim, out) in latent.iter_mut().enumerate() {
        let mut acc = 0_i64;
        for (bin, &value) in signature.iter().enumerate() {
            let centered = i64::from(value).saturating_sub(64);
            let weight = i64::from(initial_projection_weight(seed, dim, bin));
            acc = acc.saturating_add(centered.saturating_mul(weight));
        }
        *out = signed_round_shift(acc, shift).clamp(-511, 511) as i16;
    }
    latent
}

fn initial_projection_weight(seed: &str, dim: usize, bin: usize) -> i8 {
    let hash = hash_parts(&[seed, "latent-signature", &dim.to_string(), &bin.to_string()]);
    let value = i8::try_from(hash % 9).unwrap_or(0) - 4;
    if value == 0 { 1 } else { value }
}

fn tokenize_text(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for byte in text.bytes() {
        if byte.is_ascii_alphanumeric() {
            current.push(char::from(byte.to_ascii_lowercase()));
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

fn initial_weight(seed: &str, lane: &str, left: usize, right: usize) -> i8 {
    let hash = hash_parts(&[seed, lane, &left.to_string(), &right.to_string()]);
    match hash % 5 {
        0 => -2,
        1 => -1,
        2 => 0,
        3 => 1,
        _ => 2,
    }
}

fn signature_abs_error(left: &[u16; SIGNATURE_BINS], right: &[u16; SIGNATURE_BINS]) -> u64 {
    left.iter()
        .zip(right.iter())
        .map(|(&left, &right)| abs_diff_u16(left, right))
        .sum()
}

fn latent_abs_error(left: &[i16], right: &[i16]) -> u64 {
    left.iter()
        .zip(right.iter())
        .map(|(&left, &right)| abs_diff_i16(left, right))
        .sum()
}

fn dot_i16(left: &[i16], right: &[i16]) -> i64 {
    left.iter()
        .zip(right.iter())
        .fold(0_i64, |acc, (&left, &right)| {
            acc.saturating_add(i64::from(left) * i64::from(right))
        })
}

fn abs_diff_u16(left: u16, right: u16) -> u64 {
    if left >= right {
        u64::from(left - right)
    } else {
        u64::from(right - left)
    }
}

fn abs_diff_i16(left: i16, right: i16) -> u64 {
    if left >= right {
        u64::from((left - right) as u16)
    } else {
        u64::from((right - left) as u16)
    }
}

fn signed_round_shift(value: i64, shift: u8) -> i64 {
    if shift == 0 {
        return value;
    }
    let rounding = 1_i64 << (shift - 1);
    if value >= 0 {
        value.saturating_add(rounding) >> shift
    } else {
        -((value.saturating_neg().saturating_add(rounding)) >> shift)
    }
}

fn saturating_i8_add(value: i8, delta: i64) -> i8 {
    i64::from(value)
        .saturating_add(delta)
        .clamp(i64::from(i8::MIN), i64::from(i8::MAX)) as i8
}

fn saturating_i16_add(value: i16, delta: i64) -> i16 {
    i64::from(value)
        .saturating_add(delta)
        .clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i16
}

fn mean_q8(total: u64, count: u64) -> u64 {
    if count == 0 {
        return 0;
    }
    ((u128::from(total) * 256_u128) / u128::from(count)) as u64
}

fn format_q8(value: u64) -> String {
    let mut whole = value / 256;
    let mut fraction = ((value % 256) * 100 + 128) / 256;
    if fraction == 100 {
        whole = whole.saturating_add(1);
        fraction = 0;
    }
    format!("{whole}.{fraction:02}")
}

fn checked_u32(value: usize, label: &str) -> Result<u32, Box<dyn std::error::Error>> {
    u32::try_from(value).map_err(|_| format!("{label} exceeds u32").into())
}

fn hash_text(text: &str) -> u32 {
    hash_parts(&[text])
}

fn hash_parts(parts: &[&str]) -> u32 {
    let mut hash = 2_166_136_261_u32;
    for part in parts {
        for byte in part.bytes() {
            hash ^= u32::from(byte);
            hash = hash.wrapping_mul(16_777_619);
        }
        hash ^= 255;
        hash = hash.wrapping_mul(16_777_619);
    }
    hash | 1
}

fn json_field(out: &mut String, key: &str, value: &str, comma: bool) {
    out.push_str("  \"");
    out.push_str(key);
    out.push_str("\":\"");
    out.push_str(&json_escape(value));
    out.push('"');
    if comma {
        out.push(',');
    }
    out.push('\n');
}

fn number_field(out: &mut String, key: &str, value: usize, comma: bool) {
    out.push_str("  \"");
    out.push_str(key);
    out.push_str("\":");
    out.push_str(&value.to_string());
    if comma {
        out.push(',');
    }
    out.push('\n');
}

fn json_escape(value: &str) -> String {
    let mut out = String::new();
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(ch),
        }
    }
    out
}
