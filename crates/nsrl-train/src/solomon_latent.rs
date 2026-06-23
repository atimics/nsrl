use std::fs;
use std::path::{Path, PathBuf};

pub use nsrl_eval::{json_escape, read_gold_hashes, stable_hash_bytes, stable_hex_u32};

pub const PROMPT_SCHEMA: &str = "nsrl.solomon_prompt.v1";
pub const DEFAULT_PROMPT_SPLIT_SEED: &str = "solomon-prompt-split-v1";
pub const DEFAULT_EVAL_PERMILLE: usize = 180;
pub const MODEL_MAGIC: &[u8; 8] = b"NSRLLAT1";
pub const MODEL_CLASS_EXTENSION_MAGIC: &[u8; 8] = b"NSRLCLS1";
pub const SIGNATURE_GRID: usize = 16;
pub const SIGNATURE_BINS: usize = SIGNATURE_GRID * SIGNATURE_GRID;
const LAYOUT_INK_FLOOR: u16 = 32;
const LAYOUT_INK_MIDPOINT: u16 = 54;
const LAYOUT_INK_CEILING: u16 = 96;

#[derive(Debug, Clone)]
pub struct TextIndexRow {
    pub number: usize,
    pub primary_name: String,
    pub aliases: String,
    pub slice_id: String,
    pub label: String,
    pub source_file: String,
    pub ink_128_u8: String,
    pub signature: [u16; SIGNATURE_BINS],
    pub text: String,
    pub variant_id: String,
    pub source_lanes: String,
    pub image_features: [i16; SIGNATURE_BINS],
}

#[derive(Debug, Clone)]
pub struct PromptRecord {
    pub spirit_id: usize,
    pub text: String,
    pub source: String,
    pub bucket: usize,
    pub tier: String,
    pub cluster: String,
    pub prompt_hash: String,
}

#[derive(Debug, Clone)]
pub struct LatentTextModel {
    pub latent_dim: usize,
    pub text_feature_count: usize,
    pub text_encoder_shift: u8,
    pub image_encoder_shift: u8,
    pub decoder_shift: u8,
    pub text_weights: Vec<i8>,
    pub text_biases: Vec<i16>,
    pub image_weights: Vec<i8>,
    pub image_biases: Vec<i16>,
    pub decoder_weights: Vec<i8>,
    pub decoder_biases: [i16; SIGNATURE_BINS],
    pub class_head: Option<LatentClassHead>,
}

#[derive(Debug, Clone)]
pub struct LatentClassHead {
    pub numbers: Vec<usize>,
    pub names: Vec<String>,
    pub codes: Vec<Vec<i16>>,
    pub weights: Vec<i8>,
    pub biases: Vec<i16>,
}

impl LatentTextModel {
    pub fn encode_text(&self, features: &[i16]) -> Result<Vec<i16>, Box<dyn std::error::Error>> {
        if features.len() != self.text_feature_count {
            return Err("latent text feature count mismatch".into());
        }
        Ok(encode_latent(
            features,
            &self.text_weights,
            &self.text_biases,
            self.text_feature_count,
            self.latent_dim,
            self.text_encoder_shift,
        ))
    }

    pub fn encode_image(
        &self,
        features: &[i16; SIGNATURE_BINS],
    ) -> Result<Vec<i16>, Box<dyn std::error::Error>> {
        Ok(encode_latent(
            features,
            &self.image_weights,
            &self.image_biases,
            SIGNATURE_BINS,
            self.latent_dim,
            self.image_encoder_shift,
        ))
    }

    pub fn decode_signature(&self, latent: &[i16]) -> [u16; SIGNATURE_BINS] {
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

    pub fn predict_class(&self, features: &[i16]) -> Option<(usize, i64)> {
        self.class_head
            .as_ref()
            .map(|head| head.predict(features, self.text_feature_count))
    }
}

impl LatentClassHead {
    pub fn predict(&self, features: &[i16], feature_count: usize) -> (usize, i64) {
        let active_features: Vec<(usize, i16)> = features
            .iter()
            .copied()
            .enumerate()
            .take(feature_count)
            .filter(|(_, value)| *value != 0)
            .collect();
        let mut best_index = 0_usize;
        let mut best_score = i64::MIN;
        for class_index in 0..self.numbers.len() {
            let mut score = i64::from(self.biases[class_index]);
            for &(feature, value) in &active_features {
                let weight = self.weights[class_index * feature_count + feature];
                score = score.saturating_add(i64::from(weight) * i64::from(value));
            }
            if score > best_score
                || (score == best_score && self.numbers[class_index] < self.numbers[best_index])
            {
                best_score = score;
                best_index = class_index;
            }
        }
        (best_index, best_score)
    }
}

pub fn read_text_index_rows(path: &Path) -> Result<Vec<TextIndexRow>, Box<dyn std::error::Error>> {
    let text = fs::read_to_string(path)?;
    let mut rows = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        if line_index == 0 || line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 9 {
            return Err(format!(
                "{} line {} has {} fields, expected at least 9",
                path.display(),
                line_index + 1,
                fields.len()
            )
            .into());
        }
        let signature = sharpen_signature(&parse_signature(fields[7], path, line_index + 1)?);
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
        rows.push(TextIndexRow {
            number: fields[0].parse()?,
            primary_name: fields[1].to_string(),
            aliases: fields[2].to_string(),
            slice_id: fields[3].to_string(),
            label: fields[4].to_string(),
            source_file: fields[5].to_string(),
            ink_128_u8: fields[6].to_string(),
            signature,
            text: fields[8].to_string(),
            variant_id,
            source_lanes,
            image_features: image_features(&signature)?,
        });
    }
    rows.sort_by(|left, right| {
        left.number
            .cmp(&right.number)
            .then_with(|| left.variant_id.cmp(&right.variant_id))
    });
    if rows.is_empty() {
        return Err(format!("{} has no rows", path.display()).into());
    }
    Ok(rows)
}

pub fn read_prompt_records(
    path: &Path,
    split_seed: &str,
) -> Result<Vec<PromptRecord>, Box<dyn std::error::Error>> {
    let text = fs::read_to_string(path)?;
    let mut prompts = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let schema = json_string_field(line, "schema")?;
        if let Some(schema) = schema
            && schema != PROMPT_SCHEMA
        {
            return Err(format!(
                "{} line {} has unsupported schema {}",
                path.display(),
                line_index + 1,
                schema
            )
            .into());
        }
        let spirit_id = json_usize_field(line, "spirit_id")?.ok_or_else(|| {
            format!(
                "{} line {} missing spirit_id",
                path.display(),
                line_index + 1
            )
        })?;
        let text = json_string_field(line, "text")?
            .ok_or_else(|| format!("{} line {} missing text", path.display(), line_index + 1))?;
        let source = json_string_field(line, "source")?.unwrap_or_else(|| "canonical".to_string());
        let tier =
            json_string_field(line, "tier")?.unwrap_or_else(|| "tier-paraphrase".to_string());
        let cluster = json_string_field(line, "cluster")?.unwrap_or_else(|| {
            format!(
                "{}:{}",
                spirit_id,
                json_string_field_lossy(line, "source", "canonical")
            )
        });
        let computed_bucket = prompt_bucket(split_seed, &text);
        if let Some(bucket) = json_usize_field(line, "bucket")?
            && bucket != computed_bucket
        {
            return Err(format!(
                "{} line {} bucket {} does not match computed {}",
                path.display(),
                line_index + 1,
                bucket,
                computed_bucket
            )
            .into());
        }
        let computed_hash = prompt_hash(split_seed, &text);
        if let Some(record_hash) = json_string_field(line, "prompt_hash")?
            && record_hash != computed_hash
        {
            return Err(format!(
                "{} line {} prompt_hash {} does not match computed {}",
                path.display(),
                line_index + 1,
                record_hash,
                computed_hash
            )
            .into());
        }
        prompts.push(PromptRecord {
            spirit_id,
            text,
            source,
            bucket: computed_bucket,
            tier,
            cluster,
            prompt_hash: computed_hash,
        });
    }
    if prompts.is_empty() {
        return Err(format!("{} has no prompt rows", path.display()).into());
    }
    Ok(prompts)
}

pub fn default_gold_path(prompts_path: &Path) -> PathBuf {
    prompts_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("gold.tsv")
}

pub fn read_latent_model(path: &Path) -> Result<LatentTextModel, Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    read_latent_model_bytes(&bytes, path)
}

fn read_latent_model_bytes(
    bytes: &[u8],
    path: &Path,
) -> Result<LatentTextModel, Box<dyn std::error::Error>> {
    let mut cursor = ByteCursor { bytes, offset: 0 };
    let magic = cursor.read_bytes(MODEL_MAGIC.len())?;
    if magic != MODEL_MAGIC {
        return Err(format!("{} is not an NSRLLAT1 latent model", path.display()).into());
    }
    let latent_dim = usize::try_from(cursor.read_u32()?)?;
    let text_feature_count = usize::try_from(cursor.read_u32()?)?;
    let signature_bins = usize::try_from(cursor.read_u32()?)?;
    let text_encoder_shift = u8::try_from(cursor.read_u32()?)?;
    let image_encoder_shift = u8::try_from(cursor.read_u32()?)?;
    let decoder_shift = u8::try_from(cursor.read_u32()?)?;
    let signature_grid = usize::try_from(cursor.read_u32()?)?;
    if latent_dim == 0
        || text_feature_count == 0
        || signature_bins != SIGNATURE_BINS
        || signature_grid != SIGNATURE_GRID
    {
        return Err(format!("{} has incompatible latent dimensions", path.display()).into());
    }
    let text_weight_count = latent_dim
        .checked_mul(text_feature_count)
        .ok_or("latent text weight count overflow")?;
    let image_weight_count = latent_dim
        .checked_mul(SIGNATURE_BINS)
        .ok_or("latent image weight count overflow")?;
    let decoder_weight_count = SIGNATURE_BINS
        .checked_mul(latent_dim)
        .ok_or("latent decoder weight count overflow")?;
    let text_weights = cursor.read_i8_vec(text_weight_count)?;
    let text_biases = cursor.read_i16_vec(latent_dim)?;
    let image_weights = cursor.read_i8_vec(image_weight_count)?;
    let image_biases = cursor.read_i16_vec(latent_dim)?;
    let decoder_weights = cursor.read_i8_vec(decoder_weight_count)?;
    let decoder_bias_vec = cursor.read_i16_vec(SIGNATURE_BINS)?;
    let mut decoder_biases = [0_i16; SIGNATURE_BINS];
    decoder_biases.copy_from_slice(&decoder_bias_vec);
    let class_head = if cursor.remaining() == 0 {
        None
    } else {
        let extension_magic = cursor.read_bytes(MODEL_CLASS_EXTENSION_MAGIC.len())?;
        if extension_magic != MODEL_CLASS_EXTENSION_MAGIC {
            return Err(
                format!("{} has unsupported latent model extension", path.display()).into(),
            );
        }
        let class_count = usize::try_from(cursor.read_u32()?)?;
        if class_count == 0 {
            return Err(format!("{} has empty class extension", path.display()).into());
        }
        let mut numbers = Vec::with_capacity(class_count);
        let mut names = Vec::with_capacity(class_count);
        let mut codes = Vec::with_capacity(class_count);
        for _ in 0..class_count {
            numbers.push(usize::try_from(cursor.read_u32()?)?);
            let name_len = usize::try_from(cursor.read_u32()?)?;
            names.push(String::from_utf8(cursor.read_bytes(name_len)?.to_vec())?);
            codes.push(cursor.read_i16_vec(latent_dim)?);
        }
        let class_weight_count = class_count
            .checked_mul(text_feature_count)
            .ok_or("latent class weight count overflow")?;
        let weights = cursor.read_i8_vec(class_weight_count)?;
        let biases = cursor.read_i16_vec(class_count)?;
        if cursor.remaining() != 0 {
            return Err(format!("{} has trailing latent model bytes", path.display()).into());
        }
        Some(LatentClassHead {
            numbers,
            names,
            codes,
            weights,
            biases,
        })
    };
    Ok(LatentTextModel {
        latent_dim,
        text_feature_count,
        text_encoder_shift,
        image_encoder_shift,
        decoder_shift,
        text_weights,
        text_biases,
        image_weights,
        image_biases,
        decoder_weights,
        decoder_biases,
        class_head,
    })
}

pub fn text_features(text: &str, feature_count: usize) -> Vec<i16> {
    let mut features = vec![0_i16; feature_count];
    let tokens = tokenize_text(text);
    if !tokens.is_empty() && tokens.len() <= 4 {
        add_text_feature(&mut features, "whole", &tokens.join(" "), 0, 320);
    }
    for (position, token) in tokens.iter().enumerate() {
        add_text_feature(&mut features, "tok", token, position, 72);
        if let Some(next) = tokens.get(position + 1) {
            add_text_feature(
                &mut features,
                "bi",
                &format!("{token} {next}"),
                position,
                96,
            );
        }
        if let (Some(next), Some(third)) = (tokens.get(position + 1), tokens.get(position + 2)) {
            add_text_feature(
                &mut features,
                "tri",
                &format!("{token} {next} {third}"),
                position,
                112,
            );
        }
    }
    let content = content_tokens(&tokens);
    if !content.is_empty() && content.len() <= 5 {
        add_text_feature(&mut features, "cwhole", &content.join(" "), 0, 336);
        add_text_feature(
            &mut features,
            "cset",
            &sorted_feature_key(&content.iter().map(String::as_str).collect::<Vec<_>>()),
            0,
            336,
        );
    }
    for (position, token) in content.iter().enumerate() {
        add_text_feature(&mut features, "ctok", token, position, 128);
        if let Some(next) = content.get(position + 1) {
            add_text_feature(
                &mut features,
                "cbi",
                &format!("{token} {next}"),
                position,
                160,
            );
        }
        if let (Some(next), Some(third)) = (content.get(position + 1), content.get(position + 2)) {
            add_text_feature(
                &mut features,
                "ctri",
                &format!("{token} {next} {third}"),
                position,
                176,
            );
        }
        let window_end = (position + 16).min(content.len());
        for right in (position + 1)..window_end {
            add_text_feature(
                &mut features,
                "skip2",
                &format!("{token} {}", content[right]),
                position,
                176,
            );
            add_text_feature(
                &mut features,
                "pair",
                &sorted_feature_key(&[token.as_str(), content[right].as_str()]),
                position,
                192,
            );
            for third in (right + 1)..window_end {
                add_text_feature(
                    &mut features,
                    "triple",
                    &sorted_feature_key(&[
                        token.as_str(),
                        content[right].as_str(),
                        content[third].as_str(),
                    ]),
                    position,
                    192,
                );
            }
        }
    }
    features
}

pub fn prompt_bucket(split_seed: &str, text: &str) -> usize {
    usize::try_from(hash_parts(&[split_seed, text]) % 1000).unwrap_or(0)
}

pub fn prompt_hash(split_seed: &str, text: &str) -> String {
    stable_hex_u32(hash_parts(&[split_seed, text]))
}

pub fn prompt_partition_bucket(prompt: &PromptRecord, split_seed: &str) -> usize {
    if prompt.tier == "tier-cluster-holdout" {
        usize::try_from(hash_parts(&[split_seed, "cluster", &prompt.cluster]) % 1000).unwrap_or(0)
    } else {
        prompt.bucket
    }
}

pub fn dot_i16(left: &[i16], right: &[i16]) -> i64 {
    left.iter()
        .zip(right.iter())
        .fold(0_i64, |acc, (&left, &right)| {
            acc.saturating_add(i64::from(left) * i64::from(right))
        })
}

pub fn signature_abs_error(left: &[u16; SIGNATURE_BINS], right: &[u16; SIGNATURE_BINS]) -> u64 {
    left.iter()
        .zip(right.iter())
        .map(|(&left, &right)| abs_diff_u16(left, right))
        .sum()
}

pub fn latent_abs_error(left: &[i16], right: &[i16]) -> u64 {
    left.iter()
        .zip(right.iter())
        .map(|(&left, &right)| abs_diff_i16(left, right))
        .sum()
}

pub fn mean_q8(total: u64, count: u64) -> u64 {
    if count == 0 {
        return 0;
    }
    ((u128::from(total) * 256_u128) / u128::from(count)) as u64
}

fn image_features(
    signature: &[u16; SIGNATURE_BINS],
) -> Result<[i16; SIGNATURE_BINS], Box<dyn std::error::Error>> {
    let mut features = [0_i16; SIGNATURE_BINS];
    for (out, &value) in features.iter_mut().zip(signature.iter()) {
        *out = i16::try_from(value)?.saturating_sub(64).clamp(-511, 511);
    }
    Ok(features)
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

fn sharpen_signature(input: &[u16; SIGNATURE_BINS]) -> [u16; SIGNATURE_BINS] {
    let mut out = [0_u16; SIGNATURE_BINS];
    for (target, &value) in out.iter_mut().zip(input.iter()) {
        *target = sharpen_signature_value(value);
    }
    out
}

fn sharpen_signature_value(value: u16) -> u16 {
    if value <= LAYOUT_INK_FLOOR {
        return 0;
    }
    if value >= LAYOUT_INK_CEILING {
        return 255;
    }
    if value >= LAYOUT_INK_MIDPOINT { 255 } else { 0 }
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

fn tokenize_text(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for byte in text.bytes() {
        if byte.is_ascii_alphanumeric() {
            current.push(char::from(byte.to_ascii_lowercase()));
        } else if !current.is_empty() {
            push_normalized_token(&mut tokens, &mut current);
        }
    }
    if !current.is_empty() {
        push_normalized_token(&mut tokens, &mut current);
    }
    tokens
}

fn push_normalized_token(tokens: &mut Vec<String>, current: &mut String) {
    let normalized = normalize_token(current);
    current.clear();
    if normalized.len() >= 2 {
        tokens.push(normalized);
    }
}

fn normalize_token(token: &str) -> String {
    match token {
        "teach" | "teacher" | "teaches" | "teacheth" | "teaching" => {
            return "teach".to_string();
        }
        "know" | "knows" | "known" | "knowing" | "knoweth" | "knowledge" => {
            return "know".to_string();
        }
        "make" | "makes" | "maketh" | "making" => return "make".to_string(),
        "discover" | "discovers" | "discovereth" | "discovering" => {
            return "discover".to_string();
        }
        "produce" | "produces" | "produceth" | "producing" => return "produce".to_string(),
        "answer" | "answers" | "answereth" | "answering" => return "answer".to_string(),
        "virtue" | "virtues" => return "virtue".to_string(),
        "water" | "waters" => return "water".to_string(),
        "rush" | "rushing" | "rushings" => return "rush".to_string(),
        "herb" | "herbs" => return "herb".to_string(),
        "stone" | "stones" => return "stone".to_string(),
        "science" | "sciences" => return "science".to_string(),
        _ => {}
    }
    if token.len() > 5 && token.ends_with("eth") {
        token[..token.len() - 3].to_string()
    } else if token.len() > 5 && token.ends_with("ing") {
        token[..token.len() - 3].to_string()
    } else if token.len() > 4 && token.ends_with("es") {
        token[..token.len() - 2].to_string()
    } else if token.len() > 3 && token.ends_with('s') {
        token[..token.len() - 1].to_string()
    } else {
        token.to_string()
    }
}

fn content_tokens(tokens: &[String]) -> Vec<String> {
    tokens
        .iter()
        .filter(|token| token.len() >= 3 && !is_stopword(token))
        .cloned()
        .collect()
}

fn is_stopword(token: &str) -> bool {
    matches!(
        token,
        "a" | "about"
            | "after"
            | "again"
            | "all"
            | "also"
            | "an"
            | "and"
            | "any"
            | "are"
            | "as"
            | "at"
            | "be"
            | "before"
            | "both"
            | "but"
            | "by"
            | "can"
            | "etc"
            | "for"
            | "from"
            | "great"
            | "have"
            | "he"
            | "her"
            | "him"
            | "his"
            | "in"
            | "is"
            | "it"
            | "man"
            | "many"
            | "men"
            | "must"
            | "of"
            | "or"
            | "order"
            | "seal"
            | "shall"
            | "she"
            | "spirit"
            | "spirits"
            | "the"
            | "this"
            | "thou"
            | "to"
            | "unto"
            | "upon"
            | "which"
            | "who"
            | "will"
            | "with"
    )
}

fn sorted_feature_key(parts: &[&str]) -> String {
    let mut parts = parts.to_vec();
    parts.sort_unstable();
    parts.join("\u{0}")
}

fn add_text_feature(
    features: &mut [i16],
    namespace: &str,
    text: &str,
    position: usize,
    base_value: i16,
) {
    if text.len() < 2 || features.is_empty() {
        return;
    }
    let hash = hash_parts(&[namespace, text]);
    let bin = usize::try_from(hash).unwrap_or(0) % features.len();
    let length = i16::try_from(text.len().min(28)).unwrap_or(28);
    let position_bonus = i16::try_from(position % 7).unwrap_or(0).saturating_mul(5);
    let value = base_value
        .saturating_add(length.saturating_mul(6))
        .saturating_add(position_bonus)
        .min(384);
    let signed = if hash & 0x8000_0000 == 0 {
        value
    } else {
        -value
    };
    features[bin] = features[bin].saturating_add(signed).clamp(-511, 511);
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

fn json_string_field(line: &str, key: &str) -> Result<Option<String>, Box<dyn std::error::Error>> {
    let Some(start) = json_field_value_start(line, key) else {
        return Ok(None);
    };
    let mut chars = line[start..].char_indices();
    let Some((_, first)) = chars.next() else {
        return Ok(None);
    };
    if first != '"' {
        return Err(format!("json field {key} is not a string").into());
    }
    let mut out = String::new();
    let mut escaped = false;
    let mut unicode = String::new();
    let mut unicode_left = 0_usize;
    for (_, ch) in chars {
        if unicode_left != 0 {
            unicode.push(ch);
            unicode_left = unicode_left.saturating_sub(1);
            if unicode_left == 0 {
                let code = u32::from_str_radix(&unicode, 16)?;
                if let Some(decoded) = char::from_u32(code) {
                    out.push(decoded);
                }
                unicode.clear();
            }
            continue;
        }
        if escaped {
            match ch {
                '"' => out.push('"'),
                '\\' => out.push('\\'),
                '/' => out.push('/'),
                'b' => out.push('\u{0008}'),
                'f' => out.push('\u{000c}'),
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                'u' => unicode_left = 4,
                other => out.push(other),
            }
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == '"' {
            return Ok(Some(out));
        } else {
            out.push(ch);
        }
    }
    Err(format!("json field {key} has unterminated string").into())
}

fn json_string_field_lossy(line: &str, key: &str, fallback: &str) -> String {
    json_string_field(line, key)
        .ok()
        .flatten()
        .unwrap_or_else(|| fallback.to_string())
}

fn json_usize_field(line: &str, key: &str) -> Result<Option<usize>, Box<dyn std::error::Error>> {
    let Some(start) = json_field_value_start(line, key) else {
        return Ok(None);
    };
    let rest = line[start..].trim_start();
    let digits: String = rest.chars().take_while(|ch| ch.is_ascii_digit()).collect();
    if digits.is_empty() {
        return Err(format!("json field {key} is not an integer").into());
    }
    Ok(Some(digits.parse()?))
}

fn json_field_value_start(line: &str, key: &str) -> Option<usize> {
    let needle = format!("\"{key}\"");
    let key_start = line.find(&needle)?;
    let after_key = key_start + needle.len();
    let colon_offset = line[after_key..].find(':')?;
    Some(after_key + colon_offset + 1)
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

struct ByteCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> ByteCursor<'a> {
    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.offset)
    }

    fn read_bytes(&mut self, len: usize) -> Result<&'a [u8], Box<dyn std::error::Error>> {
        let end = self.offset.checked_add(len).ok_or("byte cursor overflow")?;
        if end > self.bytes.len() {
            return Err("unexpected end of latent model".into());
        }
        let out = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(out)
    }

    fn read_u32(&mut self) -> Result<u32, Box<dyn std::error::Error>> {
        let bytes = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_i16(&mut self) -> Result<i16, Box<dyn std::error::Error>> {
        let bytes = self.read_bytes(2)?;
        Ok(i16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_i8_vec(&mut self, len: usize) -> Result<Vec<i8>, Box<dyn std::error::Error>> {
        Ok(self
            .read_bytes(len)?
            .iter()
            .map(|&value| value as i8)
            .collect())
    }

    fn read_i16_vec(&mut self, len: usize) -> Result<Vec<i16>, Box<dyn std::error::Error>> {
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            out.push(self.read_i16()?);
        }
        Ok(out)
    }
}
