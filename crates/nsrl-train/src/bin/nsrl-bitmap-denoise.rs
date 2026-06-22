#![deny(unsafe_code)]

use std::env;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

const SCHEMA: &str = "nsrl.bitmap_denoise_local_table_trace.v1";
const MODEL_MAGIC: &[u8; 8] = b"NSRLBM1\n";
const MASK_COUNT: usize = 512;
const CORRUPTION_KINDS: [&str; 9] = [
    "pixel-dropout",
    "salt-pepper",
    "block-mask",
    "stroke-thin",
    "stroke-thicken",
    "line-drop",
    "mixed-noise",
    "coarse-erase",
    "box-blur",
];

#[derive(Debug, Clone)]
struct Config {
    dataset_root: PathBuf,
    out_dir: PathBuf,
    model_out: PathBuf,
    image_size: usize,
    threshold: u8,
    timesteps: usize,
    preview_pairs: usize,
}

impl Default for Config {
    fn default() -> Self {
        let dataset_root = PathBuf::from("data/processed/key-solomon-goetia-denoise-v1");
        let out_dir = dataset_root.join("baseline-local-table");
        let model_out = out_dir.join("model.nsrlbmd");
        Self {
            dataset_root,
            out_dir,
            model_out,
            image_size: 128,
            threshold: 64,
            timesteps: 8,
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
struct LocalTableAccumulator {
    conditional_sums: Vec<u64>,
    conditional_counts: Vec<u32>,
    global_sums: [u64; MASK_COUNT],
    global_counts: [u32; MASK_COUNT],
    total_sum: u64,
    total_count: u64,
}

#[derive(Debug)]
struct LocalTableModel {
    image_size: usize,
    threshold: u8,
    timesteps: usize,
    global_mean: u8,
    global_values: [u8; MASK_COUNT],
    conditional_values: Vec<u8>,
    condition_blend_shifts: Vec<u8>,
}

#[derive(Debug)]
struct Metrics {
    pair_count: usize,
    pixel_count: u64,
    input_abs_error: u64,
    predicted_abs_error: u64,
    input_mean_abs_q8: u64,
    predicted_mean_abs_q8: u64,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("nsrl-bitmap-denoise: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args(env::args().skip(1))?;
    if config.image_size == 0 || config.timesteps == 0 {
        return Err("image size and timesteps must be positive".into());
    }
    fs::create_dir_all(&config.out_dir)?;

    let train = read_split(&config, "train")?;
    let eval = read_split(&config, "eval")?;
    let mut model = train_model(&config, &train)?;
    tune_condition_blends(&config, &mut model, &train)?;
    write_model(&config.model_out, &model)?;

    let train_metrics = evaluate_split(&config, &model, "train", &train, config.preview_pairs)?;
    let eval_metrics = evaluate_split(&config, &model, "eval", &eval, config.preview_pairs)?;
    write_trace(&config, &model, &train_metrics, &eval_metrics)?;

    println!(
        "{{\"schema\":\"{}\",\"model\":\"{}\",\"train_predicted_mae\":\"{}\",\"eval_predicted_mae\":\"{}\"}}",
        SCHEMA,
        json_escape(&config.model_out.display().to_string()),
        format_q8(train_metrics.predicted_mean_abs_q8),
        format_q8(eval_metrics.predicted_mean_abs_q8),
    );
    Ok(())
}

fn usage() {
    println!(
        "Usage: nsrl-bitmap-denoise [--dataset PATH] [--out-dir PATH] [--model-out PATH] [--image-size N] [--threshold N] [--timesteps N] [--preview-pairs N]"
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
            "--dataset" | "--dataset-root" => {
                config.dataset_root = PathBuf::from(args.next().ok_or("--dataset requires PATH")?);
                if config.out_dir == Config::default().out_dir {
                    config.out_dir = config.dataset_root.join("baseline-local-table");
                    config.model_out = config.out_dir.join("model.nsrlbmd");
                }
            }
            "--out-dir" => {
                config.out_dir = PathBuf::from(args.next().ok_or("--out-dir requires PATH")?);
                if config.model_out == Config::default().model_out {
                    config.model_out = config.out_dir.join("model.nsrlbmd");
                }
            }
            "--model-out" => {
                config.model_out = PathBuf::from(args.next().ok_or("--model-out requires PATH")?);
            }
            "--image-size" => {
                config.image_size = args.next().ok_or("--image-size requires N")?.parse()?;
            }
            "--threshold" => {
                config.threshold = args.next().ok_or("--threshold requires N")?.parse()?;
            }
            "--timesteps" => {
                config.timesteps = args.next().ok_or("--timesteps requires N")?.parse()?;
            }
            "--preview-pairs" => {
                config.preview_pairs = args.next().ok_or("--preview-pairs requires N")?.parse()?;
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

fn train_model(
    config: &Config,
    train: &SplitData,
) -> Result<LocalTableModel, Box<dyn std::error::Error>> {
    let condition_count = CORRUPTION_KINDS
        .len()
        .checked_mul(config.timesteps)
        .ok_or("condition count overflow")?;
    let mut acc = LocalTableAccumulator {
        conditional_sums: vec![0; condition_count * MASK_COUNT],
        conditional_counts: vec![0; condition_count * MASK_COUNT],
        global_sums: [0; MASK_COUNT],
        global_counts: [0; MASK_COUNT],
        total_sum: 0,
        total_count: 0,
    };
    let image_bytes = checked_image_bytes(config.image_size)?;
    for pair_index in 0..train.rows.len() {
        let row = &train.rows[pair_index];
        let condition = condition_index(config, row)?;
        let input = &train.input[pair_index * image_bytes..(pair_index + 1) * image_bytes];
        let target = &train.target[pair_index * image_bytes..(pair_index + 1) * image_bytes];
        for y in 0..config.image_size {
            for x in 0..config.image_size {
                let mask = neighborhood_mask(input, config.image_size, config.threshold, x, y);
                let target_value = u64::from(target[pixel_index(config.image_size, x, y)]);
                let offset = condition * MASK_COUNT + mask;
                acc.conditional_sums[offset] =
                    acc.conditional_sums[offset].saturating_add(target_value);
                acc.conditional_counts[offset] = acc.conditional_counts[offset].saturating_add(1);
                acc.global_sums[mask] = acc.global_sums[mask].saturating_add(target_value);
                acc.global_counts[mask] = acc.global_counts[mask].saturating_add(1);
                acc.total_sum = acc.total_sum.saturating_add(target_value);
                acc.total_count = acc.total_count.saturating_add(1);
            }
        }
    }
    let global_mean = averaged_byte(acc.total_sum, acc.total_count);
    let mut global_values = [0_u8; MASK_COUNT];
    for (index, value) in global_values.iter_mut().enumerate() {
        *value = if acc.global_counts[index] == 0 {
            global_mean
        } else {
            averaged_byte(acc.global_sums[index], u64::from(acc.global_counts[index]))
        };
    }
    let mut conditional_values = vec![0_u8; condition_count * MASK_COUNT];
    for condition in 0..condition_count {
        for mask in 0..MASK_COUNT {
            let offset = condition * MASK_COUNT + mask;
            conditional_values[offset] = if acc.conditional_counts[offset] == 0 {
                global_values[mask]
            } else {
                averaged_byte(
                    acc.conditional_sums[offset],
                    u64::from(acc.conditional_counts[offset]),
                )
            };
        }
    }
    Ok(LocalTableModel {
        image_size: config.image_size,
        threshold: config.threshold,
        timesteps: config.timesteps,
        global_mean,
        global_values,
        conditional_values,
        condition_blend_shifts: vec![u8::MAX; condition_count],
    })
}

fn tune_condition_blends(
    config: &Config,
    model: &mut LocalTableModel,
    train: &SplitData,
) -> Result<(), Box<dyn std::error::Error>> {
    let condition_count = CORRUPTION_KINDS
        .len()
        .checked_mul(config.timesteps)
        .ok_or("condition count overflow")?;
    let candidates = [u8::MAX, 0, 1, 2, 3, 4, 5, 6, 7, 8];
    let mut errors = vec![0_u64; condition_count * candidates.len()];
    let image_bytes = checked_image_bytes(config.image_size)?;
    for pair_index in 0..train.rows.len() {
        let row = &train.rows[pair_index];
        let condition = condition_index(config, row)?;
        let input = &train.input[pair_index * image_bytes..(pair_index + 1) * image_bytes];
        let target = &train.target[pair_index * image_bytes..(pair_index + 1) * image_bytes];
        for y in 0..config.image_size {
            for x in 0..config.image_size {
                let index = pixel_index(config.image_size, x, y);
                let mask = neighborhood_mask(input, config.image_size, config.threshold, x, y);
                let table_value = model.conditional_values[condition * MASK_COUNT + mask];
                for (candidate_index, &candidate) in candidates.iter().enumerate() {
                    let predicted = blend_pixel(input[index], table_value, candidate);
                    let offset = condition * candidates.len() + candidate_index;
                    errors[offset] =
                        errors[offset].saturating_add(abs_diff_u8(predicted, target[index]));
                }
            }
        }
    }
    model
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
        model.condition_blend_shifts[condition] = candidates[best_index];
    }
    Ok(())
}

fn evaluate_split(
    config: &Config,
    model: &LocalTableModel,
    split: &str,
    data: &SplitData,
    preview_pairs: usize,
) -> Result<Metrics, Box<dyn std::error::Error>> {
    let image_bytes = checked_image_bytes(config.image_size)?;
    let mut input_abs_error = 0_u64;
    let mut predicted_abs_error = 0_u64;
    let mut preview = Vec::new();

    for pair_index in 0..data.rows.len() {
        let input = &data.input[pair_index * image_bytes..(pair_index + 1) * image_bytes];
        let target = &data.target[pair_index * image_bytes..(pair_index + 1) * image_bytes];
        let prediction = predict_image(config, model, &data.rows[pair_index], input)?;
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
    let pixel_count = u64::try_from(data.rows.len())?.saturating_mul(u64::try_from(image_bytes)?);
    Ok(Metrics {
        pair_count: data.rows.len(),
        pixel_count,
        input_abs_error,
        predicted_abs_error,
        input_mean_abs_q8: mean_q8(input_abs_error, pixel_count),
        predicted_mean_abs_q8: mean_q8(predicted_abs_error, pixel_count),
    })
}

fn predict_image(
    config: &Config,
    model: &LocalTableModel,
    row: &PairRow,
    input: &[u8],
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let condition = condition_index(config, row)?;
    let blend_shift = model
        .condition_blend_shifts
        .get(condition)
        .copied()
        .unwrap_or(u8::MAX);
    let mut prediction = vec![0_u8; input.len()];
    for y in 0..config.image_size {
        for x in 0..config.image_size {
            let mask = neighborhood_mask(input, config.image_size, config.threshold, x, y);
            let index = pixel_index(config.image_size, x, y);
            let table_value = model.conditional_values[condition * MASK_COUNT + mask];
            prediction[index] = blend_pixel(input[index], table_value, blend_shift);
        }
    }
    Ok(prediction)
}

fn blend_pixel(input: u8, table_value: u8, blend_shift: u8) -> u8 {
    if blend_shift == u8::MAX {
        return input;
    }
    if blend_shift == 0 {
        return table_value;
    }
    let delta = i16::from(table_value) - i16::from(input);
    let rounding = 1_i16 << (blend_shift - 1);
    let adjustment = if delta >= 0 {
        (delta + rounding) >> blend_shift
    } else {
        -(((-delta) + rounding) >> blend_shift)
    };
    let out = i16::from(input) + adjustment;
    out.clamp(0, i16::from(u8::MAX)) as u8
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

fn neighborhood_mask(input: &[u8], image_size: usize, threshold: u8, x: usize, y: usize) -> usize {
    let mut mask = 0_usize;
    let mut bit = 0_usize;
    for dy in 0..3 {
        for dx in 0..3 {
            let nx = x.checked_add(dx).and_then(|value| value.checked_sub(1));
            let ny = y.checked_add(dy).and_then(|value| value.checked_sub(1));
            if let (Some(nx), Some(ny)) = (nx, ny) {
                if nx < image_size
                    && ny < image_size
                    && input[pixel_index(image_size, nx, ny)] > threshold
                {
                    mask |= 1_usize << bit;
                }
            }
            bit += 1;
        }
    }
    mask
}

fn pixel_index(image_size: usize, x: usize, y: usize) -> usize {
    y * image_size + x
}

fn checked_image_bytes(image_size: usize) -> Result<usize, Box<dyn std::error::Error>> {
    image_size
        .checked_mul(image_size)
        .ok_or_else(|| "image byte count overflow".into())
}

fn averaged_byte(sum: u64, count: u64) -> u8 {
    if count == 0 {
        return 0;
    }
    let value = sum.saturating_add(count / 2) / count;
    value.min(u64::from(u8::MAX)) as u8
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

fn write_model(path: &Path, model: &LocalTableModel) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut bytes = Vec::new();
    bytes.extend_from_slice(MODEL_MAGIC);
    bytes.extend_from_slice(&checked_u32(model.image_size, "image_size")?.to_le_bytes());
    bytes.extend_from_slice(&u32::from(model.threshold).to_le_bytes());
    bytes.extend_from_slice(&checked_u32(model.timesteps, "timesteps")?.to_le_bytes());
    bytes
        .extend_from_slice(&checked_u32(CORRUPTION_KINDS.len(), "corruption count")?.to_le_bytes());
    bytes.extend_from_slice(&checked_u32(MASK_COUNT, "mask count")?.to_le_bytes());
    bytes.push(model.global_mean);
    bytes.extend_from_slice(&model.global_values);
    bytes.extend_from_slice(&model.conditional_values);
    bytes.extend_from_slice(&model.condition_blend_shifts);
    fs::write(path, bytes)?;
    Ok(())
}

fn checked_u32(value: usize, label: &str) -> Result<u32, Box<dyn std::error::Error>> {
    u32::try_from(value).map_err(|_| format!("{label} exceeds u32").into())
}

fn write_trace(
    config: &Config,
    model: &LocalTableModel,
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
    number_field(&mut out, "threshold", usize::from(config.threshold), true);
    number_field(&mut out, "timesteps", config.timesteps, true);
    out.push_str("  \"corruption_kinds\":[");
    for (index, kind) in CORRUPTION_KINDS.iter().enumerate() {
        if index != 0 {
            out.push(',');
        }
        out.push('"');
        out.push_str(kind);
        out.push('"');
    }
    out.push_str("],\n");
    out.push_str("  \"condition_blend_shifts\":[");
    for (index, shift) in model.condition_blend_shifts.iter().enumerate() {
        if index != 0 {
            out.push(',');
        }
        if *shift == u8::MAX {
            out.push_str("\"copy\"");
        } else {
            out.push_str(&shift.to_string());
        }
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
        "\"pair_count\":{},\"pixel_count\":{},\"input_abs_error\":{},\"predicted_abs_error\":{},\"input_mean_abs_q8\":{},\"predicted_mean_abs_q8\":{},\"input_mean_abs\":\"{}\",\"predicted_mean_abs\":\"{}\"",
        metrics.pair_count,
        metrics.pixel_count,
        metrics.input_abs_error,
        metrics.predicted_abs_error,
        metrics.input_mean_abs_q8,
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
