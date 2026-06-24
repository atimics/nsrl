#![deny(unsafe_code)]

use std::env;
use std::fs;
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const SCHEMA: &str = "nsrl.bitmap_denoise_dataset.v1";
const CLEAN_SCHEMA: &str = "nsrl.bitmap_clean_sample.v1";
const PAIR_SCHEMA: &str = "nsrl.bitmap_denoise_pair.v1";
const CORRUPTION_KINDS: [&str; 10] = [
    "pixel-dropout",
    "salt-pepper",
    "block-mask",
    "stroke-thin",
    "stroke-thicken",
    "line-drop",
    "mixed-noise",
    "coarse-erase",
    "box-blur",
    "noise-seed",
];
const INK_THRESHOLD: u8 = 64;
const STRONG_INK_THRESHOLD: u8 = 160;
const TARGET_CLEAN_EDGE_CLEAR: usize = 3;
const TARGET_CLEAN_RADIUS_MARGIN: usize = 2;
const CLEAN_AUGMENTATIONS: [&str; 8] = [
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "flip-h",
    "flip-v",
    "transpose",
    "anti-transpose",
];
const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

#[derive(Debug, Clone)]
struct Config {
    input_manifest: PathBuf,
    out_dir: PathBuf,
    kinds: Vec<String>,
    clean_augmentations: Vec<String>,
    target_cleaning: TargetCleaning,
    target_cleaning_strength: u16,
    image_size: usize,
    corruptions_per_image: usize,
    timesteps: usize,
    eval_ratio_permille: usize,
    seed: String,
    preview_pairs: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            input_manifest: PathBuf::from(
                "data/processed/key-solomon-goetia-bitmaps-pg72679/slices/manifest.json",
            ),
            out_dir: PathBuf::from("data/processed/key-solomon-goetia-denoise-v1"),
            kinds: vec!["seal-grid-cell".to_string()],
            clean_augmentations: CLEAN_AUGMENTATIONS
                .iter()
                .map(|augmentation| (*augmentation).to_string())
                .collect(),
            target_cleaning: TargetCleaning::Raw,
            target_cleaning_strength: u16::from(u8::MAX) + 1,
            image_size: 128,
            corruptions_per_image: 8,
            timesteps: 8,
            eval_ratio_permille: 180,
            seed: "solomon-denoise-v1".to_string(),
            preview_pairs: 96,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TargetCleaning {
    Raw,
    SealWindow,
    SealStrokes,
}

impl TargetCleaning {
    fn parse(value: &str) -> Result<Self, Box<dyn std::error::Error>> {
        match value {
            "raw" => Ok(Self::Raw),
            "seal-window" => Ok(Self::SealWindow),
            "seal-strokes" => Ok(Self::SealStrokes),
            _ => Err(format!(
                "unknown target cleaning: {value}; expected raw, seal-window, or seal-strokes"
            )
            .into()),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Raw => "raw",
            Self::SealWindow => "seal-window",
            Self::SealStrokes => "seal-strokes",
        }
    }
}

#[derive(Debug, Clone)]
struct Crop {
    x: usize,
    y: usize,
    width: usize,
    height: usize,
}

#[derive(Debug, Clone)]
struct SliceRow {
    id: String,
    label: String,
    kind: String,
    source_file: String,
    crop: Crop,
    ink_128_u8: String,
    ink_256_u8: String,
}

#[derive(Debug, Clone)]
struct SelectedSlice {
    row: SliceRow,
    clean: Vec<u8>,
    split_score: [u8; 32],
}

#[derive(Debug, Clone)]
struct Coverage {
    mean_ink_q8: u64,
    coverage_gt_32_ppm: u64,
}

#[derive(Debug)]
struct Rng {
    state: u32,
}

#[derive(Debug)]
struct ContactSheet {
    width: usize,
    height: usize,
    bytes: Vec<u8>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("nsrl-build-solomon-bitmap-denoise-dataset: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args(env::args().skip(1))?;
    if config.image_size == 0
        || config.corruptions_per_image == 0
        || config.timesteps == 0
        || config.preview_pairs == 0
    {
        return Err(
            "image-size, corruptions-per-image, timesteps, and preview-pairs must be positive"
                .into(),
        );
    }
    if config.eval_ratio_permille > 900 {
        return Err("--eval-ratio-permille must be <= 900".into());
    }
    if config.target_cleaning_strength > u16::from(u8::MAX) + 1 {
        return Err("--target-cleaning-strength must be <= 256".into());
    }

    let input_manifest_path = absolutize(&config.input_manifest)?;
    let out_dir = absolutize(&config.out_dir)?;
    let source_root = input_manifest_path
        .parent()
        .ok_or("input manifest has no parent directory")?;
    let manifest_text = fs::read_to_string(&input_manifest_path)?;
    let slices = parse_slices(&manifest_text)?;
    let image_bytes = checked_image_bytes(config.image_size)?;
    let mut selected = Vec::new();
    for row in slices {
        if !config.kinds.iter().any(|kind| kind == &row.kind) {
            continue;
        }
        let clean_rel = selected_clean_path(&row, config.image_size)?;
        let clean_rel = clean_rel.strip_prefix("slices/").unwrap_or(clean_rel);
        let clean_path = source_root.join(clean_rel);
        let clean = fs::read(&clean_path)?;
        if clean.len() != image_bytes {
            return Err(format!(
                "{} has {} bytes, expected {image_bytes}",
                clean_path.display(),
                clean.len()
            )
            .into());
        }
        let clean = clean_target_image(
            &clean,
            config.image_size,
            config.target_cleaning,
            config.target_cleaning_strength,
        )?;
        let split_score = sha256(&join_seed_parts(&[&config.seed, "split", &row.id]));
        selected.push(SelectedSlice {
            row,
            clean,
            split_score,
        });
    }
    if selected.is_empty() {
        return Err(format!("no slices matched --kinds {}", config.kinds.join(",")).into());
    }
    validate_clean_augmentations(&config.clean_augmentations)?;
    selected.sort_by(|left, right| {
        left.split_score
            .cmp(&right.split_score)
            .then_with(|| left.row.id.cmp(&right.row.id))
    });

    let eval_count = eval_count(selected.len(), config.eval_ratio_permille);
    let eval_ids: Vec<String> = selected
        .iter()
        .take(eval_count)
        .map(|slice| slice.row.id.clone())
        .collect();
    let mut train = Vec::new();
    let mut eval = Vec::new();
    for slice in selected {
        if eval_ids.iter().any(|id| id == &slice.row.id) {
            eval.push(slice);
        } else {
            train.push(slice);
        }
    }

    if out_dir.exists() {
        fs::remove_dir_all(&out_dir)?;
    }
    ensure_dirs(&out_dir)?;
    let train_summary = write_split(&config, &out_dir, "train", &train)?;
    let eval_summary = write_split(&config, &out_dir, "eval", &eval)?;
    write_manifest(
        &config,
        &out_dir,
        &input_manifest_path,
        &train_summary,
        &eval_summary,
    )?;
    println!(
        "{{\"out_dir\":\"{}\",\"train\":{},\"eval\":{}}}",
        json_escape(&config.out_dir.display().to_string()),
        train_summary.to_json(),
        eval_summary.to_json(),
    );
    Ok(())
}

fn usage() {
    println!(
        "Usage: nsrl-build-solomon-bitmap-denoise-dataset [--input-manifest PATH] [--out-dir PATH] [--kinds LIST] [--clean-augmentations LIST] [--target-cleaning raw|seal-window|seal-strokes] [--target-cleaning-strength N<=256] [--image-size N] [--corruptions-per-image N] [--timesteps N] [--eval-ratio-permille N] [--seed TEXT] [--preview-pairs N]"
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
            "--input-manifest" => {
                config.input_manifest =
                    PathBuf::from(args.next().ok_or("--input-manifest requires PATH")?);
            }
            "--out-dir" => {
                config.out_dir = PathBuf::from(args.next().ok_or("--out-dir requires PATH")?);
            }
            "--kinds" => {
                config.kinds = split_list(&args.next().ok_or("--kinds requires LIST")?);
            }
            "--clean-augmentations" => {
                config.clean_augmentations =
                    split_list(&args.next().ok_or("--clean-augmentations requires LIST")?);
            }
            "--target-cleaning" => {
                config.target_cleaning =
                    TargetCleaning::parse(&args.next().ok_or("--target-cleaning requires MODE")?)?;
            }
            "--target-cleaning-strength" => {
                config.target_cleaning_strength = args
                    .next()
                    .ok_or("--target-cleaning-strength requires N")?
                    .parse()?;
            }
            "--image-size" => {
                config.image_size = parse_positive(&args.next().ok_or("--image-size requires N")?)?;
            }
            "--corruptions-per-image" => {
                config.corruptions_per_image =
                    parse_positive(&args.next().ok_or("--corruptions-per-image requires N")?)?;
            }
            "--timesteps" => {
                config.timesteps = parse_positive(&args.next().ok_or("--timesteps requires N")?)?;
            }
            "--eval-ratio-permille" => {
                config.eval_ratio_permille =
                    parse_positive(&args.next().ok_or("--eval-ratio-permille requires N")?)?;
            }
            "--seed" => {
                config.seed = args.next().ok_or("--seed requires TEXT")?;
            }
            "--preview-pairs" => {
                config.preview_pairs =
                    parse_positive(&args.next().ok_or("--preview-pairs requires N")?)?;
            }
            _ => return Err(format!("unknown option: {arg}").into()),
        }
    }
    if config.kinds.is_empty() {
        return Err("--kinds must contain at least one kind".into());
    }
    if config.clean_augmentations.is_empty() {
        return Err("--clean-augmentations must contain at least one augmentation".into());
    }
    Ok(config)
}

fn validate_clean_augmentations(
    augmentations: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    for augmentation in augmentations {
        if !CLEAN_AUGMENTATIONS
            .iter()
            .any(|known| known == augmentation)
        {
            return Err(format!("unknown clean augmentation: {augmentation}").into());
        }
    }
    Ok(())
}

fn parse_positive(value: &str) -> Result<usize, Box<dyn std::error::Error>> {
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(format!("expected positive integer, got {value:?}").into());
    }
    let parsed: usize = value.parse()?;
    if parsed == 0 {
        return Err("expected positive integer, got zero".into());
    }
    Ok(parsed)
}

fn split_list(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(str::to_string)
        .collect()
}

fn absolutize(path: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(env::current_dir()?.join(path))
    }
}

fn eval_count(selected_len: usize, eval_ratio_permille: usize) -> usize {
    let rounded = (selected_len * eval_ratio_permille + 500) / 1000;
    rounded.max(1).min(selected_len.saturating_sub(1))
}

#[derive(Debug)]
struct SplitSummary {
    clean_count: usize,
    pair_count: usize,
    clean_file: String,
    pair_input_file: String,
    pair_target_file: String,
    clean_rows: String,
    pair_rows: String,
    preview_clean: String,
    preview_input_target: String,
    clean_bytes: u64,
    pair_input_bytes: u64,
    pair_target_bytes: u64,
}

impl SplitSummary {
    fn to_json(&self) -> String {
        format!(
            "{{\"clean_count\":{},\"pair_count\":{},\"clean_file\":\"{}\",\"pair_input_file\":\"{}\",\"pair_target_file\":\"{}\",\"clean_rows\":\"{}\",\"pair_rows\":\"{}\",\"preview_clean\":\"{}\",\"preview_input_target\":\"{}\",\"clean_bytes\":{},\"pair_input_bytes\":{},\"pair_target_bytes\":{}}}",
            self.clean_count,
            self.pair_count,
            json_escape(&self.clean_file),
            json_escape(&self.pair_input_file),
            json_escape(&self.pair_target_file),
            json_escape(&self.clean_rows),
            json_escape(&self.pair_rows),
            json_escape(&self.preview_clean),
            json_escape(&self.preview_input_target),
            self.clean_bytes,
            self.pair_input_bytes,
            self.pair_target_bytes,
        )
    }
}

fn write_split(
    config: &Config,
    out_dir: &Path,
    split: &str,
    slices: &[SelectedSlice],
) -> Result<SplitSummary, Box<dyn std::error::Error>> {
    let image_bytes = checked_image_bytes(config.image_size)?;
    let clean_file_rel = format!("clean/{split}.ink{}.u8", config.image_size);
    let pair_input_file_rel = format!("pairs/{split}.input.ink{}.u8", config.image_size);
    let pair_target_file_rel = format!("pairs/{split}.target.ink{}.u8", config.image_size);
    let clean_rows_rel = format!("rows/{split}.clean.jsonl");
    let pair_rows_rel = format!("rows/{split}.pairs.jsonl");
    let preview_clean_rel = format!("preview/{split}.clean-contact.pgm");
    let preview_pair_rel = format!("preview/{split}.input-target-contact.pgm");

    let clean_path = out_dir.join(&clean_file_rel);
    let input_path = out_dir.join(&pair_input_file_rel);
    let target_path = out_dir.join(&pair_target_file_rel);
    let clean_rows_path = out_dir.join(&clean_rows_rel);
    let pair_rows_path = out_dir.join(&pair_rows_rel);

    let mut clean_bytes = Vec::new();
    let mut pair_inputs = Vec::new();
    let mut pair_targets = Vec::new();
    let mut clean_rows = String::new();
    let mut pair_rows = String::new();
    let mut preview_clean = Vec::new();
    let mut preview_pairs = Vec::new();

    let mut clean_index = 0_usize;
    for slice in slices {
        for augmentation in &config.clean_augmentations {
            let clean = transform_clean_image(&slice.clean, config.image_size, augmentation)?;
            let clean_offset = clean_bytes.len();
            clean_bytes.extend_from_slice(&clean);
            let clean_coverage = coverage(&clean);
            clean_rows.push_str(&format!(
                "{{\"schema\":\"{}\",\"split\":\"{}\",\"clean_index\":{},\"source_clean_index\":{},\"augmentation\":\"{}\",\"slice_id\":\"{}\",\"label\":\"{}\",\"kind\":\"{}\",\"source_file\":\"{}\",\"source_crop\":{},\"clean_offset\":{},\"bytes\":{},\"sha256\":\"{}\",\"mean_ink_q8\":{},\"coverage_gt_32_ppm\":{}}}\n",
                CLEAN_SCHEMA,
                json_escape(split),
                clean_index,
                clean_index / config.clean_augmentations.len(),
                json_escape(augmentation),
                json_escape(&slice.row.id),
                json_escape(&slice.row.label),
                json_escape(&slice.row.kind),
                json_escape(&slice.row.source_file),
                crop_json(&slice.row.crop),
                clean_offset,
                image_bytes,
                sha256_hex(&clean),
                clean_coverage.mean_ink_q8,
                clean_coverage.coverage_gt_32_ppm,
            ));
            if preview_clean.len() < config.preview_pairs {
                preview_clean.push(clean.clone());
            }

            for variant in 0..config.corruptions_per_image {
                let pair_index = pair_inputs.len() / image_bytes;
                let timestep = 1 + ((variant + clean_index) % config.timesteps);
                let corruption = CORRUPTION_KINDS[(variant + clean_index) % CORRUPTION_KINDS.len()];
                let corrupted = corrupt_image(
                    &clean,
                    config.image_size,
                    config.image_size,
                    corruption,
                    timestep,
                    config.timesteps,
                    &[
                        &config.seed,
                        split,
                        &slice.row.id,
                        augmentation,
                        &variant.to_string(),
                        corruption,
                        &timestep.to_string(),
                    ],
                )?;
                let pair_offset = pair_index * image_bytes;
                let input_coverage = coverage(&corrupted);
                pair_inputs.extend_from_slice(&corrupted);
                pair_targets.extend_from_slice(&clean);
                pair_rows.push_str(&format!(
                    "{{\"schema\":\"{}\",\"split\":\"{}\",\"pair_index\":{},\"clean_index\":{},\"source_clean_index\":{},\"variant\":{},\"augmentation\":\"{}\",\"slice_id\":\"{}\",\"label\":\"{}\",\"kind\":\"{}\",\"source_file\":\"{}\",\"corruption\":\"{}\",\"timestep\":{},\"timesteps\":{},\"input_offset\":{},\"target_offset\":{},\"bytes\":{},\"clean_sha256\":\"{}\",\"input_sha256\":\"{}\",\"clean_mean_ink_q8\":{},\"input_mean_ink_q8\":{}}}\n",
                    PAIR_SCHEMA,
                    json_escape(split),
                    pair_index,
                    clean_index,
                    clean_index / config.clean_augmentations.len(),
                    variant,
                    json_escape(augmentation),
                    json_escape(&slice.row.id),
                    json_escape(&slice.row.label),
                    json_escape(&slice.row.kind),
                    json_escape(&slice.row.source_file),
                    json_escape(corruption),
                    timestep,
                    config.timesteps,
                    pair_offset,
                    pair_offset,
                    image_bytes,
                    sha256_hex(&clean),
                    sha256_hex(&corrupted),
                    clean_coverage.mean_ink_q8,
                    input_coverage.mean_ink_q8,
                ));
                if preview_pairs.len() < config.preview_pairs {
                    preview_pairs.push((corrupted, clean.clone()));
                }
            }
            clean_index += 1;
        }
    }

    fs::write(&clean_path, &clean_bytes)?;
    fs::write(&input_path, &pair_inputs)?;
    fs::write(&target_path, &pair_targets)?;
    fs::write(&clean_rows_path, clean_rows)?;
    fs::write(&pair_rows_path, pair_rows)?;

    let clean_sheet = make_contact_sheet(&preview_clean, config.image_size, 8, 4);
    write_pgm(
        &out_dir.join(&preview_clean_rel),
        clean_sheet.width,
        clean_sheet.height,
        &clean_sheet.bytes,
    )?;
    let pair_sheet = make_pair_sheet(&preview_pairs, config.image_size, 4, 4);
    write_pgm(
        &out_dir.join(&preview_pair_rel),
        pair_sheet.width,
        pair_sheet.height,
        &pair_sheet.bytes,
    )?;

    Ok(SplitSummary {
        clean_count: clean_bytes.len() / image_bytes,
        pair_count: pair_inputs.len() / image_bytes,
        clean_file: clean_file_rel,
        pair_input_file: pair_input_file_rel,
        pair_target_file: pair_target_file_rel,
        clean_rows: clean_rows_rel,
        pair_rows: pair_rows_rel,
        preview_clean: preview_clean_rel,
        preview_input_target: preview_pair_rel,
        clean_bytes: fs::metadata(&clean_path)?.len(),
        pair_input_bytes: fs::metadata(&input_path)?.len(),
        pair_target_bytes: fs::metadata(&target_path)?.len(),
    })
}

fn write_manifest(
    config: &Config,
    out_dir: &Path,
    input_manifest: &Path,
    train: &SplitSummary,
    eval: &SplitSummary,
) -> Result<(), Box<dyn std::error::Error>> {
    let generated_at = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let source_manifest = relative_path(out_dir, input_manifest);
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str(&format!("  \"schema\":\"{}\",\n", SCHEMA));
    out.push_str(&format!(
        "  \"source_manifest\":\"{}\",\n",
        json_escape(&source_manifest)
    ));
    out.push_str(&format!(
        "  \"generated_at_unix_seconds\":{},\n",
        generated_at
    ));
    out.push_str("  \"generator\":\"nsrl-build-solomon-bitmap-denoise-dataset\",\n");
    out.push_str(&format!("  \"seed\":\"{}\",\n", json_escape(&config.seed)));
    out.push_str(&format!("  \"image_size\":{},\n", config.image_size));
    out.push_str("  \"channels\":1,\n");
    out.push_str(
        "  \"pixel_contract\":\"raw u8 ink mask; 0=paper/background, 255=ink, row-major\",\n",
    );
    out.push_str("  \"target\":\"predict clean ink mask from corrupted ink mask and timestep/corruption metadata\",\n");
    out.push_str("  \"selected_kinds\":[");
    for (index, kind) in config.kinds.iter().enumerate() {
        if index != 0 {
            out.push(',');
        }
        out.push('"');
        out.push_str(&json_escape(kind));
        out.push('"');
    }
    out.push_str("],\n");
    out.push_str("  \"clean_augmentations\":[");
    for (index, augmentation) in config.clean_augmentations.iter().enumerate() {
        if index != 0 {
            out.push(',');
        }
        out.push('"');
        out.push_str(&json_escape(augmentation));
        out.push('"');
    }
    out.push_str("],\n");
    out.push_str(&format!(
        "  \"target_cleaning\":\"{}\",\n",
        config.target_cleaning.as_str()
    ));
    out.push_str(&format!(
        "  \"target_cleaning_strength\":{},\n",
        config.target_cleaning_strength
    ));
    out.push_str("  \"corruption_kinds\":[");
    for (index, kind) in CORRUPTION_KINDS.iter().enumerate() {
        if index != 0 {
            out.push(',');
        }
        out.push('"');
        out.push_str(&json_escape(kind));
        out.push('"');
    }
    out.push_str("],\n");
    out.push_str(&format!(
        "  \"corruptions_per_image\":{},\n",
        config.corruptions_per_image
    ));
    out.push_str(&format!("  \"timesteps\":{},\n", config.timesteps));
    out.push_str(&format!(
        "  \"eval_ratio_permille\":{},\n",
        config.eval_ratio_permille
    ));
    out.push_str("  \"splits\":{\n");
    out.push_str(&format!("    \"train\":{},\n", train.to_json()));
    out.push_str(&format!("    \"eval\":{}\n", eval.to_json()));
    out.push_str("  }\n");
    out.push_str("}\n");
    fs::write(out_dir.join("manifest.json"), out)?;
    Ok(())
}

fn ensure_dirs(out_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    for dir in ["clean", "pairs", "rows", "preview"] {
        fs::create_dir_all(out_dir.join(dir))?;
    }
    Ok(())
}

fn selected_clean_path(
    row: &SliceRow,
    image_size: usize,
) -> Result<&str, Box<dyn std::error::Error>> {
    match image_size {
        128 => Ok(&row.ink_128_u8),
        256 => Ok(&row.ink_256_u8),
        _ => Err(format!("unsupported --image-size {image_size}; expected 128 or 256").into()),
    }
}

fn transform_clean_image(
    input: &[u8],
    image_size: usize,
    augmentation: &str,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let image_bytes = checked_image_bytes(image_size)?;
    if input.len() != image_bytes {
        return Err(format!(
            "clean transform got {} bytes, expected {image_bytes}",
            input.len()
        )
        .into());
    }
    let mut out = vec![0_u8; image_bytes];
    for y in 0..image_size {
        for x in 0..image_size {
            let (sx, sy) = transform_coords(image_size, x, y, augmentation)?;
            out[pixel_index(image_size, x, y)] = input[pixel_index(image_size, sx, sy)];
        }
    }
    Ok(out)
}

fn clean_target_image(
    image: &[u8],
    image_size: usize,
    cleaning: TargetCleaning,
    strength: u16,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    if cleaning == TargetCleaning::Raw || strength == 0 {
        return Ok(image.to_vec());
    }
    let image_bytes = checked_image_bytes(image_size)?;
    if image.len() != image_bytes {
        return Err(format!(
            "target cleaning got {} bytes, expected {image_bytes}",
            image.len()
        )
        .into());
    }
    let center = i64::try_from(image_size / 2)?;
    let radius_margin = TARGET_CLEAN_RADIUS_MARGIN.min(image_size / 2);
    let seal_radius = i64::try_from(image_size / 2 - radius_margin)?;
    let seal_radius2 = seal_radius.saturating_mul(seal_radius);
    let mut out = vec![0_u8; image_bytes];
    for y in 0..image_size {
        for x in 0..image_size {
            if x < TARGET_CLEAN_EDGE_CLEAR
                || y < TARGET_CLEAN_EDGE_CLEAR
                || x + TARGET_CLEAN_EDGE_CLEAR >= image_size
                || y + TARGET_CLEAN_EDGE_CLEAR >= image_size
            {
                continue;
            }
            let dx = i64::try_from(x)?.saturating_sub(center);
            let dy = i64::try_from(y)?.saturating_sub(center);
            let radius2 = dx.saturating_mul(dx).saturating_add(dy.saturating_mul(dy));
            if radius2 > seal_radius2 {
                continue;
            }
            let index = pixel_index(image_size, x, y);
            let value = image[index];
            if cleaning == TargetCleaning::SealWindow {
                out[index] = value;
                continue;
            }
            let neighbors = neighbor_ink_count(image, image_size, image_size, x, y);
            out[index] = if value > STRONG_INK_THRESHOLD {
                value
            } else if value > INK_THRESHOLD && neighbors >= 2 {
                value.saturating_sub(16)
            } else if value > 40 && neighbors >= 4 {
                value.saturating_sub(40)
            } else {
                0
            };
        }
    }
    if strength >= u16::from(u8::MAX) + 1 {
        return Ok(out);
    }
    let keep = (u16::from(u8::MAX) + 1).saturating_sub(strength);
    for (target, &raw) in out.iter_mut().zip(image.iter()) {
        let blended = u16::from(raw)
            .saturating_mul(keep)
            .saturating_add(u16::from(*target).saturating_mul(strength))
            .saturating_add(128)
            / 256;
        *target = u8::try_from(blended.min(u16::from(u8::MAX)))?;
    }
    Ok(out)
}

fn transform_coords(
    image_size: usize,
    x: usize,
    y: usize,
    augmentation: &str,
) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    let last = image_size.saturating_sub(1);
    match augmentation {
        "identity" => Ok((x, y)),
        "rot90" => Ok((y, last - x)),
        "rot180" => Ok((last - x, last - y)),
        "rot270" => Ok((last - y, x)),
        "flip-h" => Ok((last - x, y)),
        "flip-v" => Ok((x, last - y)),
        "transpose" => Ok((y, x)),
        "anti-transpose" => Ok((last - y, last - x)),
        _ => Err(format!("unknown clean augmentation: {augmentation}").into()),
    }
}

fn parse_slices(text: &str) -> Result<Vec<SliceRow>, Box<dyn std::error::Error>> {
    let slices_key = text
        .find("\"slices\"")
        .ok_or("manifest missing slices field")?;
    let array_start = text[slices_key..]
        .find('[')
        .ok_or("manifest slices field is not an array")?
        + slices_key;
    let mut rows = Vec::new();
    let mut depth = 0_usize;
    let mut object_start = None;
    for (offset, ch) in text[array_start..].char_indices() {
        let index = array_start + offset;
        match ch {
            '{' => {
                depth += 1;
                if depth == 1 {
                    object_start = Some(index);
                }
            }
            '}' => {
                if depth == 0 {
                    return Err("unbalanced JSON object in slices".into());
                }
                if depth == 1 {
                    let start = object_start.ok_or("slice object missing start")?;
                    rows.push(parse_slice_object(&text[start..=index])?);
                    object_start = None;
                }
                depth -= 1;
            }
            ']' if depth == 0 => break,
            _ => {}
        }
    }
    Ok(rows)
}

fn parse_slice_object(object: &str) -> Result<SliceRow, Box<dyn std::error::Error>> {
    Ok(SliceRow {
        id: json_string_field(object, "id")?,
        label: json_string_field(object, "label")?,
        kind: json_string_field(object, "kind")?,
        source_file: json_string_field(object, "source_file")?,
        crop: Crop {
            x: json_usize_field(object, "x")?,
            y: json_usize_field(object, "y")?,
            width: json_usize_field(object, "width")?,
            height: json_usize_field(object, "height")?,
        },
        ink_128_u8: json_string_field(object, "ink_128_u8")?,
        ink_256_u8: json_string_field(object, "ink_256_u8")?,
    })
}

fn json_string_field(object: &str, field: &str) -> Result<String, Box<dyn std::error::Error>> {
    let needle = format!("\"{field}\":");
    let start = object
        .find(&needle)
        .ok_or_else(|| format!("missing string field {field}"))?
        + needle.len();
    let bytes = object.as_bytes();
    let mut index = start;
    while index < bytes.len() && bytes[index].is_ascii_whitespace() {
        index += 1;
    }
    if bytes.get(index) != Some(&b'"') {
        return Err(format!("field {field} is not a string").into());
    }
    index += 1;
    let mut out = String::new();
    let mut escaped = false;
    for ch in object[index..].chars() {
        if escaped {
            out.push(match ch {
                'n' => '\n',
                'r' => '\r',
                't' => '\t',
                '"' => '"',
                '\\' => '\\',
                other => other,
            });
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == '"' {
            return Ok(out);
        } else {
            out.push(ch);
        }
    }
    Err(format!("unterminated string field {field}").into())
}

fn json_usize_field(object: &str, field: &str) -> Result<usize, Box<dyn std::error::Error>> {
    let needle = format!("\"{field}\":");
    let start = object
        .find(&needle)
        .ok_or_else(|| format!("missing numeric field {field}"))?
        + needle.len();
    let bytes = object.as_bytes();
    let mut index = start;
    while index < bytes.len() && bytes[index].is_ascii_whitespace() {
        index += 1;
    }
    let number_start = index;
    while index < bytes.len() && bytes[index].is_ascii_digit() {
        index += 1;
    }
    if index == number_start {
        return Err(format!("field {field} is not an unsigned integer").into());
    }
    Ok(object[number_start..index].parse()?)
}

fn corrupt_image(
    clean: &[u8],
    width: usize,
    height: usize,
    corruption: &str,
    timestep: usize,
    timesteps: usize,
    seed_parts: &[&str],
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut rng = Rng::new(seed_u32(seed_parts));
    match corruption {
        "pixel-dropout" => Ok(pixel_dropout(clean, timestep, timesteps, &mut rng)),
        "salt-pepper" => Ok(salt_pepper(clean, timestep, timesteps, &mut rng)),
        "block-mask" => Ok(block_mask(
            clean, width, height, timestep, timesteps, &mut rng,
        )),
        "stroke-thin" => Ok(stroke_thin(
            clean, width, height, timestep, timesteps, &mut rng,
        )),
        "stroke-thicken" => Ok(stroke_thicken(
            clean, width, height, timestep, timesteps, &mut rng,
        )),
        "line-drop" => Ok(line_drop(
            clean, width, height, timestep, timesteps, &mut rng,
        )),
        "mixed-noise" => {
            let blocked = block_mask(clean, width, height, timestep, timesteps, &mut rng);
            Ok(salt_pepper(
                &blocked,
                timestep_div2(timestep),
                timesteps,
                &mut rng,
            ))
        }
        "coarse-erase" => Ok(coarse_erase(
            clean, width, height, timestep, timesteps, &mut rng,
        )),
        "box-blur" => Ok(box_blur_corruption(
            clean, width, height, timestep, timesteps, &mut rng,
        )),
        "noise-seed" => Ok(noise_seed(width, height, &mut rng)),
        _ => Err(format!("unknown corruption kind: {corruption}").into()),
    }
}

fn noise_seed(width: usize, height: usize, rng: &mut Rng) -> Vec<u8> {
    let mut out = vec![0_u8; width.saturating_mul(height)];
    for value in &mut out {
        let roll = rng.rand_bounded(16);
        *value = if roll == 0 {
            255
        } else if roll == 1 {
            160
        } else {
            0
        };
    }
    out
}

fn pixel_dropout(clean: &[u8], timestep: usize, timesteps: usize, rng: &mut Rng) -> Vec<u8> {
    let mut out = clean.to_vec();
    let drop_num = 4 + timestep * 6;
    let speck_num = timestep;
    let den = timesteps * 64;
    for value in &mut out {
        if *value > 16 && rng.chance(drop_num, den) {
            *value = 0;
        } else if *value < 24 && rng.chance(speck_num, den * 2) {
            *value = 255;
        }
    }
    out
}

fn salt_pepper(clean: &[u8], timestep: usize, timesteps: usize, rng: &mut Rng) -> Vec<u8> {
    let mut out = clean.to_vec();
    let noise_num = 2 + timestep * 4;
    let den = timesteps * 64;
    for value in &mut out {
        if rng.chance(noise_num, den) {
            *value = if rng.chance(55, 100) { 0 } else { 255 };
        }
    }
    out
}

fn block_mask(
    clean: &[u8],
    width: usize,
    height: usize,
    timestep: usize,
    timesteps: usize,
    rng: &mut Rng,
) -> Vec<u8> {
    let mut out = clean.to_vec();
    let blocks = 1 + ceil_div(timestep * 5, timesteps);
    for _ in 0..blocks {
        let bw = 6 + rng.rand_bounded(4 + timestep * 3);
        let bh = 6 + rng.rand_bounded(4 + timestep * 3);
        let x0 = rng.rand_bounded(width.saturating_sub(bw).max(1));
        let y0 = rng.rand_bounded(height.saturating_sub(bh).max(1));
        let fill = if rng.chance(4, 5) { 0 } else { 255 };
        for y in y0..height.min(y0 + bh) {
            for x in x0..width.min(x0 + bw) {
                out[pixel_index(width, x, y)] = fill;
            }
        }
    }
    out
}

fn stroke_thin(
    clean: &[u8],
    width: usize,
    height: usize,
    timestep: usize,
    timesteps: usize,
    rng: &mut Rng,
) -> Vec<u8> {
    let mut out = clean.to_vec();
    let threshold = (5_usize).saturating_sub((timestep * 3) / timesteps).max(2);
    for y in 0..height {
        for x in 0..width {
            let index = pixel_index(width, x, y);
            if clean[index] <= 48 {
                continue;
            }
            let neighbors = neighbor_ink_count(clean, width, height, x, y);
            if neighbors <= threshold && rng.chance(1 + timestep, timesteps * 3) {
                out[index] = 0;
            }
        }
    }
    pixel_dropout(&out, timestep_div2(timestep), timesteps, rng)
}

fn stroke_thicken(
    clean: &[u8],
    width: usize,
    height: usize,
    timestep: usize,
    timesteps: usize,
    rng: &mut Rng,
) -> Vec<u8> {
    let mut out = clean.to_vec();
    let threshold = (4_usize).saturating_sub((timestep * 3) / timesteps).max(1);
    for y in 0..height {
        for x in 0..width {
            let index = pixel_index(width, x, y);
            if clean[index] > 48 {
                continue;
            }
            let neighbors = neighbor_ink_count(clean, width, height, x, y);
            if neighbors >= threshold && rng.chance(1 + timestep, timesteps * 4) {
                out[index] = u8::try_from(160 + rng.rand_bounded(96)).unwrap_or(u8::MAX);
            }
        }
    }
    salt_pepper(&out, timestep_div2(timestep), timesteps, rng)
}

fn line_drop(
    clean: &[u8],
    width: usize,
    height: usize,
    timestep: usize,
    timesteps: usize,
    rng: &mut Rng,
) -> Vec<u8> {
    let mut out = clean.to_vec();
    let lines = 1 + (timestep * 4) / timesteps;
    for _ in 0..lines {
        if rng.chance(1, 2) {
            let y = rng.rand_bounded(height);
            let thickness = 1 + rng.rand_bounded(1 + ceil_div(timestep, 3));
            for yy in y..height.min(y + thickness) {
                for x in 0..width {
                    out[pixel_index(width, x, yy)] = 0;
                }
            }
        } else {
            let x = rng.rand_bounded(width);
            let thickness = 1 + rng.rand_bounded(1 + ceil_div(timestep, 3));
            for xx in x..width.min(x + thickness) {
                for y in 0..height {
                    out[pixel_index(width, xx, y)] = 0;
                }
            }
        }
    }
    out
}

fn coarse_erase(
    clean: &[u8],
    width: usize,
    height: usize,
    timestep: usize,
    timesteps: usize,
    rng: &mut Rng,
) -> Vec<u8> {
    let mut out = clean.to_vec();
    let cell = 18_usize.saturating_sub(timestep).max(4);
    let drop_num = 1 + timestep;
    let den = timesteps * 3;
    let mut y0 = 0;
    while y0 < height {
        let mut x0 = 0;
        while x0 < width {
            if rng.chance(drop_num, den) {
                for y in y0..height.min(y0 + cell) {
                    for x in x0..width.min(x0 + cell) {
                        let index = pixel_index(width, x, y);
                        out[index] /= 3;
                    }
                }
            }
            x0 += cell;
        }
        y0 += cell;
    }
    out
}

fn box_blur_corruption(
    clean: &[u8],
    width: usize,
    height: usize,
    timestep: usize,
    timesteps: usize,
    rng: &mut Rng,
) -> Vec<u8> {
    let passes = 1 + (timestep * 3) / timesteps;
    let mut out = clean.to_vec();
    for _ in 0..passes {
        out = box_blur_once(&out, width, height);
    }
    if rng.chance(1 + timestep, timesteps * 3) {
        out = stroke_thicken(&out, width, height, timestep_div2(timestep), timesteps, rng);
    } else {
        out = salt_pepper(&out, timestep_div2(timestep), timesteps, rng);
    }
    out
}

fn box_blur_once(input: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut out = vec![0_u8; input.len()];
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0_u16;
            let mut count = 0_u16;
            for dy in [-1_i32, 0, 1] {
                for dx in [-1_i32, 0, 1] {
                    let nx = i32::try_from(x).unwrap_or(i32::MAX).saturating_add(dx);
                    let ny = i32::try_from(y).unwrap_or(i32::MAX).saturating_add(dy);
                    if nx < 0 || ny < 0 {
                        continue;
                    }
                    let ux = usize::try_from(nx).unwrap_or(usize::MAX);
                    let uy = usize::try_from(ny).unwrap_or(usize::MAX);
                    if ux >= width || uy >= height {
                        continue;
                    }
                    sum = sum.saturating_add(u16::from(input[pixel_index(width, ux, uy)]));
                    count = count.saturating_add(1);
                }
            }
            out[pixel_index(width, x, y)] = u8::try_from((sum + count / 2) / count).unwrap_or(0);
        }
    }
    out
}

fn timestep_div2(timestep: usize) -> usize {
    (timestep / 2).max(1)
}

fn neighbor_ink_count(image: &[u8], width: usize, height: usize, x: usize, y: usize) -> usize {
    let mut count = 0;
    for dy in [-1_i32, 0, 1] {
        for dx in [-1_i32, 0, 1] {
            if dx == 0 && dy == 0 {
                continue;
            }
            let nx = i32::try_from(x).unwrap_or(i32::MAX).saturating_add(dx);
            let ny = i32::try_from(y).unwrap_or(i32::MAX).saturating_add(dy);
            if nx < 0 || ny < 0 {
                continue;
            }
            let ux = usize::try_from(nx).unwrap_or(usize::MAX);
            let uy = usize::try_from(ny).unwrap_or(usize::MAX);
            if ux >= width || uy >= height {
                continue;
            }
            if image[pixel_index(width, ux, uy)] > 64 {
                count += 1;
            }
        }
    }
    count
}

fn coverage(image: &[u8]) -> Coverage {
    let mut sum = 0_u64;
    let mut over32 = 0_u64;
    for &value in image {
        sum += u64::from(value);
        if value > 32 {
            over32 += 1;
        }
    }
    let len = u64::try_from(image.len()).unwrap_or(1);
    Coverage {
        mean_ink_q8: (sum * 256) / len,
        coverage_gt_32_ppm: (over32 * 1_000_000) / len,
    }
}

fn make_contact_sheet(
    images: &[Vec<u8>],
    tile_size: usize,
    columns: usize,
    gap: usize,
) -> ContactSheet {
    let rows = images.len().div_ceil(columns);
    let width = columns * tile_size + (columns + 1) * gap;
    let height = rows * tile_size + (rows + 1) * gap;
    let mut sheet = vec![255_u8; width * height];
    for (image_index, image) in images.iter().enumerate() {
        let col = image_index % columns;
        let row = image_index / columns;
        let x0 = gap + col * (tile_size + gap);
        let y0 = gap + row * (tile_size + gap);
        for y in 0..tile_size {
            for x in 0..tile_size {
                let value = image[pixel_index(tile_size, x, y)];
                sheet[pixel_index(width, x0 + x, y0 + y)] = 255_u8.saturating_sub(value);
            }
        }
    }
    ContactSheet {
        width,
        height,
        bytes: sheet,
    }
}

fn make_pair_sheet(
    pairs: &[(Vec<u8>, Vec<u8>)],
    tile_size: usize,
    columns: usize,
    gap: usize,
) -> ContactSheet {
    let pair_width = tile_size * 2 + gap;
    let rows = pairs.len().div_ceil(columns);
    let width = columns * pair_width + (columns + 1) * gap;
    let height = rows * tile_size + (rows + 1) * gap;
    let mut sheet = vec![255_u8; width * height];
    for (pair_index, (input, target)) in pairs.iter().enumerate() {
        let col = pair_index % columns;
        let row = pair_index / columns;
        let x0 = gap + col * (pair_width + gap);
        let y0 = gap + row * (tile_size + gap);
        for (side, image) in [input, target].iter().enumerate() {
            let side_x = x0 + side * (tile_size + gap);
            for y in 0..tile_size {
                for x in 0..tile_size {
                    let value = image[pixel_index(tile_size, x, y)];
                    sheet[pixel_index(width, side_x + x, y0 + y)] = 255_u8.saturating_sub(value);
                }
            }
        }
    }
    ContactSheet {
        width,
        height,
        bytes: sheet,
    }
}

fn write_pgm(
    path: &Path,
    width: usize,
    height: usize,
    bytes: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = fs::File::create(path)?;
    write!(file, "P5\n{width} {height}\n255\n")?;
    file.write_all(bytes)?;
    Ok(())
}

fn crop_json(crop: &Crop) -> String {
    format!(
        "{{\"x\":{},\"y\":{},\"width\":{},\"height\":{}}}",
        crop.x, crop.y, crop.width, crop.height
    )
}

fn pixel_index(width: usize, x: usize, y: usize) -> usize {
    y * width + x
}

fn checked_image_bytes(image_size: usize) -> Result<usize, Box<dyn std::error::Error>> {
    image_size
        .checked_mul(image_size)
        .ok_or_else(|| "image byte count overflow".into())
}

fn ceil_div(numerator: usize, denominator: usize) -> usize {
    if denominator == 0 {
        return 0;
    }
    numerator.div_ceil(denominator)
}

impl Rng {
    fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_add(0x6d2b79f5);
        let mut value = self.state;
        value = (value ^ (value >> 15)).wrapping_mul(value | 1);
        value ^= value.wrapping_add((value ^ (value >> 7)).wrapping_mul(value | 61));
        value ^ (value >> 14)
    }

    fn rand_bounded(&mut self, max_exclusive: usize) -> usize {
        if max_exclusive <= 1 {
            return 0;
        }
        let range = 1_u64 << 32;
        let bound = u64::try_from(max_exclusive).unwrap_or(u64::MAX);
        let limit = range - (range % bound);
        loop {
            let value = u64::from(self.next_u32());
            if value < limit {
                return usize::try_from(value % bound).unwrap_or(0);
            }
        }
    }

    fn chance(&mut self, numerator: usize, denominator: usize) -> bool {
        self.rand_bounded(denominator) < numerator
    }
}

fn seed_u32(parts: &[&str]) -> u32 {
    let digest = sha256(&join_seed_parts(parts));
    u32::from_be_bytes([digest[0], digest[1], digest[2], digest[3]])
}

fn join_seed_parts(parts: &[&str]) -> Vec<u8> {
    let mut out = Vec::new();
    for (index, part) in parts.iter().enumerate() {
        if index != 0 {
            out.push(0);
        }
        out.extend_from_slice(part.as_bytes());
    }
    out
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex_bytes(&sha256(bytes))
}

fn hex_bytes(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(char::from(HEX[usize::from(byte >> 4)]));
        out.push(char::from(HEX[usize::from(byte & 15)]));
    }
    out
}

fn sha256(input: &[u8]) -> [u8; 32] {
    let mut h = [
        0x6a09e667_u32,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut message = input.to_vec();
    let bit_len = u64::try_from(message.len()).unwrap_or(0).saturating_mul(8);
    message.push(0x80);
    while (message.len() % 64) != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in message.chunks_exact(64) {
        let mut w = [0_u32; 64];
        for (index, word) in w.iter_mut().take(16).enumerate() {
            let start = index * 4;
            *word = u32::from_be_bytes([
                chunk[start],
                chunk[start + 1],
                chunk[start + 2],
                chunk[start + 3],
            ]);
        }
        for index in 16..64 {
            let s0 = w[index - 15].rotate_right(7)
                ^ w[index - 15].rotate_right(18)
                ^ (w[index - 15] >> 3);
            let s1 = w[index - 2].rotate_right(17)
                ^ w[index - 2].rotate_right(19)
                ^ (w[index - 2] >> 10);
            w[index] = w[index - 16]
                .wrapping_add(s0)
                .wrapping_add(w[index - 7])
                .wrapping_add(s1);
        }
        let mut a = h[0];
        let mut b = h[1];
        let mut c = h[2];
        let mut d = h[3];
        let mut e = h[4];
        let mut f = h[5];
        let mut g = h[6];
        let mut hh = h[7];
        for index in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA256_K[index])
                .wrapping_add(w[index]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    let mut out = [0_u8; 32];
    for (index, word) in h.iter().enumerate() {
        out[index * 4..index * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

fn relative_path(from_dir: &Path, to_path: &Path) -> String {
    let from_components = clean_components(from_dir);
    let to_components = clean_components(to_path);
    let mut common = 0;
    while common < from_components.len()
        && common < to_components.len()
        && from_components[common] == to_components[common]
    {
        common += 1;
    }
    let mut parts = Vec::new();
    for _ in common..from_components.len() {
        parts.push("..".to_string());
    }
    for component in &to_components[common..] {
        parts.push(component.clone());
    }
    if parts.is_empty() {
        ".".to_string()
    } else {
        parts.join("/")
    }
}

fn clean_components(path: &Path) -> Vec<String> {
    let mut out = Vec::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => out.push(prefix.as_os_str().to_string_lossy().to_string()),
            Component::RootDir => out.push(String::new()),
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            Component::Normal(value) => out.push(value.to_string_lossy().to_string()),
        }
    }
    out
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
