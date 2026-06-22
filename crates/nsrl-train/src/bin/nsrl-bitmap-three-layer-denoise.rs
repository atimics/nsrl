#![deny(unsafe_code)]

use std::env;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

const SCHEMA: &str = "nsrl.bitmap_denoise_three_layer_conv_trace.v1";
const MODEL_MAGIC: &[u8; 8] = b"NSRLCV3\n";
const KERNEL: usize = 9;
const CORRUPTION_KINDS: [&str; 8] = [
    "pixel-dropout",
    "salt-pepper",
    "block-mask",
    "stroke-thin",
    "stroke-thicken",
    "line-drop",
    "mixed-noise",
    "coarse-erase",
];

#[derive(Debug, Clone)]
struct Config {
    dataset_root: PathBuf,
    out_dir: PathBuf,
    model_out: PathBuf,
    image_size: usize,
    timesteps: usize,
    layers: usize,
    epochs: usize,
    output_shift: u8,
    learning_shift: u8,
    bias_learning_shift: u8,
    max_weight_delta: i16,
    max_bias_delta: i16,
    preview_pairs: usize,
}

impl Default for Config {
    fn default() -> Self {
        let dataset_root = PathBuf::from("data/processed/key-solomon-goetia-denoise-v1");
        let out_dir = dataset_root.join("baseline-three-layer-conv");
        let model_out = out_dir.join("model.nsrlcv3");
        Self {
            dataset_root,
            out_dir,
            model_out,
            image_size: 128,
            timesteps: 8,
            layers: 3,
            epochs: 8,
            output_shift: 12,
            learning_shift: 31,
            bias_learning_shift: 30,
            max_weight_delta: 4,
            max_bias_delta: 12,
            preview_pairs: 32,
        }
    }
}

#[derive(Debug, Clone)]
struct PairRow {
    corruption: String,
    timestep: usize,
}

#[derive(Debug)]
struct SplitData {
    input: Vec<u8>,
    target: Vec<u8>,
    rows: Vec<PairRow>,
}

#[derive(Debug)]
struct ConvLayer {
    weights: [i8; KERNEL],
    condition_biases: Vec<i16>,
    condition_blend_shifts: Vec<u8>,
}

#[derive(Debug)]
struct LayeredConvModel {
    image_size: usize,
    timesteps: usize,
    output_shift: u8,
    layers: Vec<ConvLayer>,
}

#[derive(Debug, Clone)]
struct EpochTrace {
    layer: usize,
    epoch: usize,
    train_raw_mean_abs_q8: u64,
    weights: [i8; KERNEL],
    bias_min: i16,
    bias_max: i16,
}

#[derive(Debug)]
struct Metrics {
    pair_count: usize,
    pixel_count: u64,
    input_abs_error: u64,
    layer_abs_errors: Vec<u64>,
    predicted_abs_error: u64,
    input_mean_abs_q8: u64,
    layer_mean_abs_q8: Vec<u64>,
    predicted_mean_abs_q8: u64,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("nsrl-bitmap-three-layer-denoise: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args(env::args().skip(1))?;
    if config.image_size == 0 || config.timesteps == 0 || config.epochs == 0 || config.layers == 0 {
        return Err("image size, timesteps, layers, and epochs must be positive".into());
    }
    fs::create_dir_all(&config.out_dir)?;

    let train = read_split(&config, "train")?;
    let eval = read_split(&config, "eval")?;
    let (model, epochs) = train_model(&config, &train)?;
    write_model(&config.model_out, &model)?;

    let train_metrics = evaluate_split(&config, &model, "train", &train, config.preview_pairs)?;
    let eval_metrics = evaluate_split(&config, &model, "eval", &eval, config.preview_pairs)?;
    write_trace(&config, &model, &epochs, &train_metrics, &eval_metrics)?;

    println!(
        "{{\"schema\":\"{}\",\"model\":\"{}\",\"train_predicted_mae\":\"{}\",\"eval_predicted_mae\":\"{}\",\"eval_input_copy_mae\":\"{}\"}}",
        SCHEMA,
        json_escape(&config.model_out.display().to_string()),
        format_q8(train_metrics.predicted_mean_abs_q8),
        format_q8(eval_metrics.predicted_mean_abs_q8),
        format_q8(eval_metrics.input_mean_abs_q8),
    );
    Ok(())
}

fn usage() {
    println!(
        "Usage: nsrl-bitmap-three-layer-denoise [--dataset PATH] [--out-dir PATH] [--model-out PATH] [--image-size N] [--timesteps N] [--layers N] [--epochs N] [--output-shift N] [--learning-shift N] [--bias-learning-shift N] [--max-weight-delta N] [--max-bias-delta N] [--preview-pairs N]"
    );
}

fn parse_args<I>(mut args: I) -> Result<Config, Box<dyn std::error::Error>>
where
    I: Iterator<Item = String>,
{
    let default = Config::default();
    let mut config = default.clone();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                usage();
                std::process::exit(0);
            }
            "--dataset" | "--dataset-root" => {
                config.dataset_root = PathBuf::from(args.next().ok_or("--dataset requires PATH")?);
                if config.out_dir == default.out_dir {
                    config.out_dir = config.dataset_root.join("baseline-three-layer-conv");
                    config.model_out = config.out_dir.join("model.nsrlcv3");
                }
            }
            "--out-dir" => {
                config.out_dir = PathBuf::from(args.next().ok_or("--out-dir requires PATH")?);
                if config.model_out == default.model_out {
                    config.model_out = config.out_dir.join("model.nsrlcv3");
                }
            }
            "--model-out" => {
                config.model_out = PathBuf::from(args.next().ok_or("--model-out requires PATH")?);
            }
            "--image-size" => {
                config.image_size = args.next().ok_or("--image-size requires N")?.parse()?
            }
            "--timesteps" => {
                config.timesteps = args.next().ok_or("--timesteps requires N")?.parse()?
            }
            "--layers" => config.layers = args.next().ok_or("--layers requires N")?.parse()?,
            "--epochs" => config.epochs = args.next().ok_or("--epochs requires N")?.parse()?,
            "--output-shift" => {
                config.output_shift = args.next().ok_or("--output-shift requires N")?.parse()?
            }
            "--learning-shift" => {
                config.learning_shift = args.next().ok_or("--learning-shift requires N")?.parse()?
            }
            "--bias-learning-shift" => {
                config.bias_learning_shift = args
                    .next()
                    .ok_or("--bias-learning-shift requires N")?
                    .parse()?
            }
            "--max-weight-delta" => {
                config.max_weight_delta = args
                    .next()
                    .ok_or("--max-weight-delta requires N")?
                    .parse()?
            }
            "--max-bias-delta" => {
                config.max_bias_delta = args.next().ok_or("--max-bias-delta requires N")?.parse()?
            }
            "--preview-pairs" => {
                config.preview_pairs = args.next().ok_or("--preview-pairs requires N")?.parse()?
            }
            _ => return Err(format!("unknown option: {arg}").into()),
        }
    }
    Ok(config)
}

fn read_split(config: &Config, split: &str) -> Result<SplitData, Box<dyn std::error::Error>> {
    let image_bytes = checked_image_bytes(config.image_size)?;
    let input_path = config
        .dataset_root
        .join("pairs")
        .join(format!("{split}.input.ink{}.u8", config.image_size));
    let target_path = config
        .dataset_root
        .join("pairs")
        .join(format!("{split}.target.ink{}.u8", config.image_size));
    let rows_path = config
        .dataset_root
        .join("rows")
        .join(format!("{split}.pairs.jsonl"));
    let input = fs::read(&input_path)?;
    let target = fs::read(&target_path)?;
    if input.len() != target.len() {
        return Err(format!(
            "{split} input and target byte counts differ: {} vs {}",
            input.len(),
            target.len()
        )
        .into());
    }
    if input.len() % image_bytes != 0 {
        return Err(format!(
            "{split} byte count {} is not a multiple of image bytes {image_bytes}",
            input.len()
        )
        .into());
    }
    let rows = read_pair_rows(&rows_path)?;
    let pair_count = input.len() / image_bytes;
    if rows.len() != pair_count {
        return Err(format!(
            "{split} row count {} does not match pair count {pair_count}",
            rows.len()
        )
        .into());
    }
    Ok(SplitData {
        input,
        target,
        rows,
    })
}

fn read_pair_rows(path: &Path) -> Result<Vec<PairRow>, Box<dyn std::error::Error>> {
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut rows = Vec::new();
    for (index, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let corruption = json_string_field(&line, "corruption")
            .ok_or_else(|| format!("{}:{} missing corruption", path.display(), index + 1))?;
        let timestep = json_usize_field(&line, "timestep")
            .ok_or_else(|| format!("{}:{} missing timestep", path.display(), index + 1))?;
        rows.push(PairRow {
            corruption,
            timestep,
        });
    }
    Ok(rows)
}

fn train_model(
    config: &Config,
    train: &SplitData,
) -> Result<(LayeredConvModel, Vec<EpochTrace>), Box<dyn std::error::Error>> {
    let mut model = LayeredConvModel {
        image_size: config.image_size,
        timesteps: config.timesteps,
        output_shift: config.output_shift,
        layers: Vec::new(),
    };
    let mut traces = Vec::new();
    let mut layer_inputs = train.input.clone();
    for layer_index in 0..config.layers {
        let layer_number = layer_index + 1;
        let (mut layer, mut layer_traces) =
            train_layer(config, train, &layer_inputs, layer_number)?;
        tune_condition_blends(config, &mut layer, train, &layer_inputs)?;
        layer_inputs = apply_layer_to_split(config, &layer, train, &layer_inputs)?;
        model.layers.push(layer);
        traces.append(&mut layer_traces);
    }
    Ok((model, traces))
}

fn train_layer(
    config: &Config,
    train: &SplitData,
    layer_inputs: &[u8],
    layer_number: usize,
) -> Result<(ConvLayer, Vec<EpochTrace>), Box<dyn std::error::Error>> {
    let condition_count = condition_count(config)?;
    if layer_inputs.len() != train.target.len() {
        return Err("layer input and target byte counts differ".into());
    }
    let mut layer = ConvLayer {
        weights: [0; KERNEL],
        condition_biases: vec![0; condition_count],
        condition_blend_shifts: vec![u8::MAX; condition_count],
    };
    let mut traces = Vec::new();
    let image_bytes = checked_image_bytes(config.image_size)?;
    for epoch in 0..config.epochs {
        let mut weight_grads = [0_i64; KERNEL];
        let mut bias_grads = vec![0_i64; condition_count];
        let mut raw_error = 0_u64;
        for pair_index in 0..train.rows.len() {
            let row = &train.rows[pair_index];
            let condition = condition_index(config, row)?;
            let input = &layer_inputs[pair_index * image_bytes..(pair_index + 1) * image_bytes];
            let target = &train.target[pair_index * image_bytes..(pair_index + 1) * image_bytes];
            for y in 0..config.image_size {
                for x in 0..config.image_size {
                    let index = pixel_index(config.image_size, x, y);
                    let mut features = [0_i16; KERNEL];
                    local_features(input, config.image_size, x, y, &mut features);
                    let predicted = predict_raw_pixel(
                        &layer,
                        config.output_shift,
                        condition,
                        input[index],
                        &features,
                    );
                    let error = i16::from(target[index]) - i16::from(predicted);
                    raw_error = raw_error.saturating_add(abs_i16(error));
                    for kernel_index in 0..KERNEL {
                        weight_grads[kernel_index] = weight_grads[kernel_index]
                            .saturating_add(i64::from(error) * i64::from(features[kernel_index]));
                    }
                    bias_grads[condition] = bias_grads[condition].saturating_add(i64::from(error));
                }
            }
        }
        for (weight, grad) in layer.weights.iter_mut().zip(weight_grads.iter()) {
            let delta = signed_round_shift(*grad, config.learning_shift).clamp(
                -i64::from(config.max_weight_delta),
                i64::from(config.max_weight_delta),
            );
            let next = i16::from(*weight)
                .saturating_add(delta as i16)
                .clamp(-127, 127);
            *weight = next as i8;
        }
        for (bias, grad) in layer.condition_biases.iter_mut().zip(bias_grads.iter()) {
            let delta = signed_round_shift(*grad, config.bias_learning_shift).clamp(
                -i64::from(config.max_bias_delta),
                i64::from(config.max_bias_delta),
            );
            *bias = bias.saturating_add(delta as i16);
        }
        traces.push(EpochTrace {
            layer: layer_number,
            epoch: epoch + 1,
            train_raw_mean_abs_q8: mean_q8(raw_error, train_pixel_count(config, train)?),
            weights: layer.weights,
            bias_min: *layer.condition_biases.iter().min().unwrap_or(&0),
            bias_max: *layer.condition_biases.iter().max().unwrap_or(&0),
        });
    }
    Ok((layer, traces))
}

fn tune_condition_blends(
    config: &Config,
    layer: &mut ConvLayer,
    train: &SplitData,
    layer_inputs: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    let condition_count = condition_count(config)?;
    let candidates = [u8::MAX, 0, 1, 2, 3, 4, 5, 6, 7, 8];
    let mut errors = vec![0_u64; condition_count * candidates.len()];
    let image_bytes = checked_image_bytes(config.image_size)?;
    for pair_index in 0..train.rows.len() {
        let row = &train.rows[pair_index];
        let condition = condition_index(config, row)?;
        let input = &layer_inputs[pair_index * image_bytes..(pair_index + 1) * image_bytes];
        let target = &train.target[pair_index * image_bytes..(pair_index + 1) * image_bytes];
        for y in 0..config.image_size {
            for x in 0..config.image_size {
                let index = pixel_index(config.image_size, x, y);
                let mut features = [0_i16; KERNEL];
                local_features(input, config.image_size, x, y, &mut features);
                let raw = predict_raw_pixel(
                    layer,
                    config.output_shift,
                    condition,
                    input[index],
                    &features,
                );
                for (candidate_index, &candidate) in candidates.iter().enumerate() {
                    let predicted = blend_pixel(input[index], raw, candidate);
                    let offset = condition * candidates.len() + candidate_index;
                    errors[offset] =
                        errors[offset].saturating_add(abs_diff_u8(predicted, target[index]));
                }
            }
        }
    }
    layer
        .condition_blend_shifts
        .resize(condition_count, u8::MAX);
    for condition in 0..condition_count {
        let mut best_index = 0_usize;
        let mut best_error = u64::MAX;
        for (candidate_index, _) in candidates.iter().enumerate() {
            let error = errors[condition * candidates.len() + candidate_index];
            if error < best_error {
                best_error = error;
                best_index = candidate_index;
            }
        }
        layer.condition_blend_shifts[condition] = candidates[best_index];
    }
    Ok(())
}

fn apply_layer_to_split(
    config: &Config,
    layer: &ConvLayer,
    data: &SplitData,
    layer_inputs: &[u8],
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let image_bytes = checked_image_bytes(config.image_size)?;
    let mut out = vec![0_u8; layer_inputs.len()];
    for pair_index in 0..data.rows.len() {
        let input = &layer_inputs[pair_index * image_bytes..(pair_index + 1) * image_bytes];
        let prediction = predict_layer_image(config, layer, &data.rows[pair_index], input)?;
        out[pair_index * image_bytes..(pair_index + 1) * image_bytes].copy_from_slice(&prediction);
    }
    Ok(out)
}

fn evaluate_split(
    config: &Config,
    model: &LayeredConvModel,
    split: &str,
    data: &SplitData,
    preview_pairs: usize,
) -> Result<Metrics, Box<dyn std::error::Error>> {
    let image_bytes = checked_image_bytes(config.image_size)?;
    let mut input_abs_error = 0_u64;
    let mut layer_abs_errors = vec![0_u64; model.layers.len()];
    let mut predicted_abs_error = 0_u64;
    let mut preview = Vec::new();
    for pair_index in 0..data.rows.len() {
        let input = &data.input[pair_index * image_bytes..(pair_index + 1) * image_bytes];
        let target = &data.target[pair_index * image_bytes..(pair_index + 1) * image_bytes];
        let mut prediction = input.to_vec();
        for (layer_index, layer) in model.layers.iter().enumerate() {
            prediction = predict_layer_image(config, layer, &data.rows[pair_index], &prediction)?;
            for index in 0..image_bytes {
                layer_abs_errors[layer_index] = layer_abs_errors[layer_index]
                    .saturating_add(abs_diff_u8(prediction[index], target[index]));
            }
        }
        for index in 0..image_bytes {
            input_abs_error =
                input_abs_error.saturating_add(abs_diff_u8(input[index], target[index]));
            predicted_abs_error =
                predicted_abs_error.saturating_add(abs_diff_u8(prediction[index], target[index]));
        }
        if preview.len() < preview_pairs {
            preview.push((input.to_vec(), prediction, target.to_vec()));
        }
    }
    write_preview(
        &config
            .out_dir
            .join(format!("{split}.input-pred-target.pgm")),
        config.image_size,
        &preview,
    )?;
    let pixel_count = train_pixel_count(config, data)?;
    let layer_mean_abs_q8 = layer_abs_errors
        .iter()
        .map(|&error| mean_q8(error, pixel_count))
        .collect();
    Ok(Metrics {
        pair_count: data.rows.len(),
        pixel_count,
        input_abs_error,
        layer_abs_errors,
        predicted_abs_error,
        input_mean_abs_q8: mean_q8(input_abs_error, pixel_count),
        layer_mean_abs_q8,
        predicted_mean_abs_q8: mean_q8(predicted_abs_error, pixel_count),
    })
}

fn predict_layer_image(
    config: &Config,
    layer: &ConvLayer,
    row: &PairRow,
    input: &[u8],
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let condition = condition_index(config, row)?;
    let blend_shift = layer
        .condition_blend_shifts
        .get(condition)
        .copied()
        .unwrap_or(u8::MAX);
    let mut prediction = vec![0_u8; input.len()];
    for y in 0..config.image_size {
        for x in 0..config.image_size {
            let index = pixel_index(config.image_size, x, y);
            let mut features = [0_i16; KERNEL];
            local_features(input, config.image_size, x, y, &mut features);
            let raw = predict_raw_pixel(
                layer,
                config.output_shift,
                condition,
                input[index],
                &features,
            );
            prediction[index] = blend_pixel(input[index], raw, blend_shift);
        }
    }
    Ok(prediction)
}

fn local_features(input: &[u8], image_size: usize, x: usize, y: usize, out: &mut [i16; KERNEL]) {
    let center = i16::from(input[pixel_index(image_size, x, y)]);
    let mut index = 0_usize;
    for dy in 0..3 {
        for dx in 0..3 {
            let nx = x.checked_add(dx).and_then(|value| value.checked_sub(1));
            let ny = y.checked_add(dy).and_then(|value| value.checked_sub(1));
            let neighbor = if let (Some(nx), Some(ny)) = (nx, ny) {
                if nx < image_size && ny < image_size {
                    i16::from(input[pixel_index(image_size, nx, ny)])
                } else {
                    center
                }
            } else {
                center
            };
            out[index] = neighbor - center;
            index += 1;
        }
    }
}

fn predict_raw_pixel(
    layer: &ConvLayer,
    output_shift: u8,
    condition: usize,
    input_center: u8,
    features: &[i16; KERNEL],
) -> u8 {
    let mut acc = i64::from(*layer.condition_biases.get(condition).unwrap_or(&0));
    for (weight, feature) in layer.weights.iter().zip(features.iter()) {
        acc = acc.saturating_add(i64::from(*weight) * i64::from(*feature));
    }
    let residual = signed_round_shift(acc, output_shift);
    let predicted = i64::from(input_center).saturating_add(residual);
    predicted.clamp(0, i64::from(u8::MAX)) as u8
}

fn blend_pixel(input: u8, raw: u8, blend_shift: u8) -> u8 {
    if blend_shift == u8::MAX {
        return input;
    }
    if blend_shift == 0 {
        return raw;
    }
    let delta = i16::from(raw) - i16::from(input);
    let rounding = 1_i16 << (blend_shift - 1);
    let adjustment = if delta >= 0 {
        (delta + rounding) >> blend_shift
    } else {
        -(((-delta) + rounding) >> blend_shift)
    };
    let out = i16::from(input) + adjustment;
    out.clamp(0, i16::from(u8::MAX)) as u8
}

fn condition_count(config: &Config) -> Result<usize, Box<dyn std::error::Error>> {
    CORRUPTION_KINDS
        .len()
        .checked_mul(config.timesteps)
        .ok_or_else(|| "condition count overflow".into())
}

fn condition_index(config: &Config, row: &PairRow) -> Result<usize, Box<dyn std::error::Error>> {
    if row.timestep == 0 || row.timestep > config.timesteps {
        return Err(format!(
            "timestep {} is outside 1..={}",
            row.timestep, config.timesteps
        )
        .into());
    }
    let corruption = CORRUPTION_KINDS
        .iter()
        .position(|&kind| kind == row.corruption)
        .ok_or_else(|| format!("unknown corruption kind: {}", row.corruption))?;
    Ok(corruption * config.timesteps + (row.timestep - 1))
}

fn pixel_index(image_size: usize, x: usize, y: usize) -> usize {
    y * image_size + x
}

fn checked_image_bytes(image_size: usize) -> Result<usize, Box<dyn std::error::Error>> {
    image_size
        .checked_mul(image_size)
        .ok_or_else(|| "image byte count overflow".into())
}

fn train_pixel_count(config: &Config, data: &SplitData) -> Result<u64, Box<dyn std::error::Error>> {
    Ok(u64::try_from(data.rows.len())?
        .saturating_mul(u64::try_from(checked_image_bytes(config.image_size)?)?))
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

fn abs_i16(value: i16) -> u64 {
    if value >= 0 {
        u64::from(value as u16)
    } else {
        u64::from(value.unsigned_abs())
    }
}

fn abs_diff_u8(left: u8, right: u8) -> u64 {
    if left >= right {
        u64::from(left - right)
    } else {
        u64::from(right - left)
    }
}

fn mean_q8(total: u64, count: u64) -> u64 {
    if count == 0 {
        return 0;
    }
    ((u128::from(total) * 256_u128) / u128::from(count)) as u64
}

fn format_q8(value: u64) -> String {
    let whole = value / 256;
    let fraction = ((value % 256) * 100 + 128) / 256;
    format!("{whole}.{fraction:02}")
}

fn write_model(path: &Path, model: &LayeredConvModel) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut bytes = Vec::new();
    bytes.extend_from_slice(MODEL_MAGIC);
    bytes.extend_from_slice(&checked_u32(model.image_size, "image_size")?.to_le_bytes());
    bytes.extend_from_slice(&checked_u32(model.timesteps, "timesteps")?.to_le_bytes());
    bytes.extend_from_slice(&u32::from(model.output_shift).to_le_bytes());
    bytes
        .extend_from_slice(&checked_u32(CORRUPTION_KINDS.len(), "corruption count")?.to_le_bytes());
    bytes.extend_from_slice(&checked_u32(model.layers.len(), "layer count")?.to_le_bytes());
    for layer in &model.layers {
        bytes.extend_from_slice(&layer.weights.map(|value| value as u8));
        for bias in &layer.condition_biases {
            bytes.extend_from_slice(&bias.to_le_bytes());
        }
        bytes.extend_from_slice(&layer.condition_blend_shifts);
    }
    fs::write(path, bytes)?;
    Ok(())
}

fn checked_u32(value: usize, label: &str) -> Result<u32, Box<dyn std::error::Error>> {
    u32::try_from(value).map_err(|_| format!("{label} exceeds u32").into())
}

fn write_trace(
    config: &Config,
    model: &LayeredConvModel,
    epochs: &[EpochTrace],
    train: &Metrics,
    eval: &Metrics,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut out = String::new();
    out.push_str("{\n");
    json_field(&mut out, "schema", SCHEMA, true);
    json_field(
        &mut out,
        "dataset_root",
        &config.dataset_root.display().to_string(),
        true,
    );
    json_field(
        &mut out,
        "model_out",
        &config.model_out.display().to_string(),
        true,
    );
    number_field(&mut out, "image_size", config.image_size, true);
    number_field(&mut out, "timesteps", config.timesteps, true);
    number_field(&mut out, "layers", model.layers.len(), true);
    number_field(&mut out, "epochs", config.epochs, true);
    number_field(
        &mut out,
        "output_shift",
        usize::from(config.output_shift),
        true,
    );
    number_field(
        &mut out,
        "learning_shift",
        usize::from(config.learning_shift),
        true,
    );
    out.push_str("  \"layer_models\":[");
    for (layer_index, layer) in model.layers.iter().enumerate() {
        if layer_index != 0 {
            out.push(',');
        }
        out.push_str(&format!("{{\"layer\":{},\"weights\":[", layer_index + 1));
        for (index, weight) in layer.weights.iter().enumerate() {
            if index != 0 {
                out.push(',');
            }
            out.push_str(&weight.to_string());
        }
        out.push_str("],\"condition_blend_shifts\":[");
        for (index, shift) in layer.condition_blend_shifts.iter().enumerate() {
            if index != 0 {
                out.push(',');
            }
            if *shift == u8::MAX {
                out.push_str("\"copy\"");
            } else {
                out.push_str(&shift.to_string());
            }
        }
        out.push_str("]}");
    }
    out.push_str("],\n");
    out.push_str("  \"epochs_trace\":[");
    for (index, epoch) in epochs.iter().enumerate() {
        if index != 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "{{\"layer\":{},\"epoch\":{},\"train_raw_mean_abs\":\"{}\",\"weights\":[{},{},{},{},{},{},{},{},{}],\"bias_min\":{},\"bias_max\":{}}}",
            epoch.layer,
            epoch.epoch,
            format_q8(epoch.train_raw_mean_abs_q8),
            epoch.weights[0],
            epoch.weights[1],
            epoch.weights[2],
            epoch.weights[3],
            epoch.weights[4],
            epoch.weights[5],
            epoch.weights[6],
            epoch.weights[7],
            epoch.weights[8],
            epoch.bias_min,
            epoch.bias_max,
        ));
    }
    out.push_str("],\n");
    metrics_field(&mut out, "train", train, true);
    metrics_field(&mut out, "eval", eval, false);
    out.push_str("}\n");
    fs::write(config.out_dir.join("trace.json"), out)?;
    Ok(())
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

fn metrics_field(out: &mut String, key: &str, metrics: &Metrics, comma: bool) {
    out.push_str("  \"");
    out.push_str(key);
    out.push_str("\":{");
    out.push_str(&format!(
        "\"pair_count\":{},\"pixel_count\":{},\"input_abs_error\":{},",
        metrics.pair_count, metrics.pixel_count, metrics.input_abs_error,
    ));
    out.push_str("\"layer_abs_errors\":[");
    for (index, error) in metrics.layer_abs_errors.iter().enumerate() {
        if index != 0 {
            out.push(',');
        }
        out.push_str(&error.to_string());
    }
    out.push_str("],");
    out.push_str(&format!(
        "\"predicted_abs_error\":{},\"input_mean_abs_q8\":{},",
        metrics.predicted_abs_error, metrics.input_mean_abs_q8,
    ));
    out.push_str("\"layer_mean_abs_q8\":[");
    for (index, value) in metrics.layer_mean_abs_q8.iter().enumerate() {
        if index != 0 {
            out.push(',');
        }
        out.push_str(&value.to_string());
    }
    out.push_str("],\"layer_mean_abs\":[");
    for (index, value) in metrics.layer_mean_abs_q8.iter().enumerate() {
        if index != 0 {
            out.push(',');
        }
        out.push('"');
        out.push_str(&format_q8(*value));
        out.push('"');
    }
    out.push_str(&format!(
        "],\"predicted_mean_abs_q8\":{},\"input_mean_abs\":\"{}\",\"predicted_mean_abs\":\"{}\"",
        metrics.predicted_mean_abs_q8,
        format_q8(metrics.input_mean_abs_q8),
        format_q8(metrics.predicted_mean_abs_q8),
    ));
    out.push('}');
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

fn json_string_field(line: &str, field: &str) -> Option<String> {
    let needle = format!("\"{field}\":\"");
    let start = line.find(&needle)? + needle.len();
    let mut out = String::new();
    let mut escaped = false;
    for ch in line[start..].chars() {
        if escaped {
            out.push(ch);
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == '"' {
            return Some(out);
        } else {
            out.push(ch);
        }
    }
    None
}

fn json_usize_field(line: &str, field: &str) -> Option<usize> {
    let needle = format!("\"{field}\":");
    let start = line.find(&needle)? + needle.len();
    let mut end = start;
    let bytes = line.as_bytes();
    while end < bytes.len() && bytes[end].is_ascii_digit() {
        end += 1;
    }
    line[start..end].parse().ok()
}

fn write_preview(
    path: &Path,
    image_size: usize,
    triples: &[(Vec<u8>, Vec<u8>, Vec<u8>)],
) -> Result<(), Box<dyn std::error::Error>> {
    if triples.is_empty() {
        return Ok(());
    }
    let columns = 2_usize;
    let gap = 4_usize;
    let triple_width = image_size * 3 + gap * 2;
    let rows = triples.len().div_ceil(columns);
    let width = columns * triple_width + (columns + 1) * gap;
    let height = rows * image_size + (rows + 1) * gap;
    let mut sheet = vec![255_u8; width * height];
    for (triple_index, (input, prediction, target)) in triples.iter().enumerate() {
        let col = triple_index % columns;
        let row = triple_index / columns;
        let x0 = gap + col * (triple_width + gap);
        let y0 = gap + row * (image_size + gap);
        for (side, image) in [input, prediction, target].iter().enumerate() {
            let side_x = x0 + side * (image_size + gap);
            for y in 0..image_size {
                for x in 0..image_size {
                    let value = 255_u8.saturating_sub(image[pixel_index(image_size, x, y)]);
                    sheet[(y0 + y) * width + side_x + x] = value;
                }
            }
        }
    }
    let mut file = fs::File::create(path)?;
    write!(file, "P5\n{width} {height}\n255\n")?;
    file.write_all(&sheet)?;
    Ok(())
}
