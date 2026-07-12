use wasm_bindgen::prelude::*;

const SCHEMA: &str = "nsrl.web_solomon_sigil.v1";
const MODEL_MAGIC_TCH: &[u8; 8] = b"NSRLTCH\n";
const KERNEL: usize = 9;
const HIDDEN_CHANNELS: usize = 8;
const POSITION_FEATURE_CHANNELS: usize = 6;
const HIDDEN_KERNELS: [[i8; KERNEL]; HIDDEN_CHANNELS] = [
    [1, 1, 1, 1, 0, 1, 1, 1, 1],
    [0, 1, 0, 1, 0, 1, 0, 1, 0],
    [1, 0, 1, 0, 0, 0, 1, 0, 1],
    [-1, 0, 1, -2, 0, 2, -1, 0, 1],
    [-1, -2, -1, 0, 0, 0, 1, 2, 1],
    [-2, -1, 0, -1, 0, 1, 0, 1, 2],
    [0, 1, 2, -1, 0, 1, -2, -1, 0],
    [-1, 2, -1, 2, 0, 2, -1, 2, -1],
];
// 8x8 grid used only for the browser candidate-scoring heuristic
// (sample_signature / text_signature_distance). This is independent of the
// model and does not affect conditioning.
const SIGNATURE_GRID: usize = 8;
const SIGNATURE_BINS: usize = SIGNATURE_GRID * SIGNATURE_GRID;
// 16x16 grid used for the model's text-conditioning features. These must match
// the native trainer/sampler (nsrl-bitmap-multichannel-denoise /
// nsrl-bitmap-sample) byte-for-byte or the learned weights produce garbage.
const TEXT_FEATURE_CHANNELS: usize = 16;
const TEXT_SIGNATURE_GRID: usize = 16;
const TEXT_SIGNATURE_BINS: usize = TEXT_SIGNATURE_GRID * TEXT_SIGNATURE_GRID;
const QUALITY_RING_BUCKETS: usize = 32;
const QUALITY_BUCKET_STEPS: i64 = 4;
const INK_THRESHOLD: u8 = 64;
const STRONG_INK_THRESHOLD: u8 = 160;
const SPECK_INK_THRESHOLD: u8 = 96;
const TEXT_WEIGHT: i64 = 96;
const MAX_BROWSER_CANDIDATES: usize = 32;
const MAX_BROWSER_PASSES: usize = 12;
const MAX_BROWSER_CONDITIONS: usize = 64;
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

#[wasm_bindgen]
pub struct SolomonSampler {
    model: MultichannelModel,
    active_conditions: Vec<usize>,
    targets: Vec<TextTarget>,
}

#[wasm_bindgen]
impl SolomonSampler {
    #[wasm_bindgen(constructor)]
    pub fn new(model_bytes: &[u8], text_index_tsv: &str) -> Result<SolomonSampler, JsValue> {
        let model = read_text_conditioned_model(model_bytes).map_err(js_error)?;
        let active_conditions = generation_conditions(&model);
        if active_conditions.is_empty() {
            return Err(JsValue::from_str(
                "Solomon sampler model has no active conditions",
            ));
        }
        let targets = read_text_targets(text_index_tsv).map_err(js_error)?;
        Ok(Self {
            model,
            active_conditions,
            targets,
        })
    }

    pub fn model_card(&self) -> String {
        format!(
            "{{\"schema\":\"{}\",\"model_format\":\"NSRLTCH\",\"image_size\":{},\"timesteps\":{},\"layers\":{},\"targets\":{},\"active_conditions\":{}}}",
            SCHEMA,
            self.model.image_size,
            self.model.timesteps,
            self.model.layers.len(),
            self.targets.len(),
            self.active_conditions.len(),
        )
    }

    pub fn sample(
        &self,
        prompt: &str,
        seed: &str,
        candidate_multiplier: usize,
        passes: usize,
    ) -> Result<SolomonSample, JsValue> {
        self.sample_with_condition_limit(
            prompt,
            seed,
            candidate_multiplier,
            passes,
            self.active_conditions.len(),
        )
    }

    pub fn sample_fast(
        &self,
        prompt: &str,
        seed: &str,
        candidate_multiplier: usize,
        passes: usize,
        condition_limit: usize,
    ) -> Result<SolomonSample, JsValue> {
        self.sample_with_condition_limit(
            prompt,
            seed,
            candidate_multiplier,
            passes,
            condition_limit.min(MAX_BROWSER_CONDITIONS),
        )
    }
}

impl SolomonSampler {
    fn sample_with_condition_limit(
        &self,
        prompt: &str,
        seed: &str,
        candidate_multiplier: usize,
        passes: usize,
        condition_limit: usize,
    ) -> Result<SolomonSample, JsValue> {
        let candidate_multiplier = candidate_multiplier.clamp(1, MAX_BROWSER_CANDIDATES);
        let passes = passes.clamp(1, MAX_BROWSER_PASSES);
        let condition_limit = condition_limit.max(1);
        let used_condition_count = condition_limit.min(self.active_conditions.len());
        let active_conditions = &self.active_conditions[..used_condition_count];
        let target = select_text_target(prompt, &self.targets).map_err(js_error)?;
        let conditioning = build_text_conditioning(
            &target.text_signature,
            &target.text_stats,
            self.model.image_size,
        )
        .map_err(js_error)?;
        let mut best: Option<Candidate> = None;
        for source_index in 0..candidate_multiplier {
            let candidate = sample_candidate(
                &self.model,
                active_conditions,
                source_index,
                seed,
                passes,
                &conditioning,
                &target.score_signature,
            )
            .map_err(js_error)?;
            let replace = best
                .as_ref()
                .map(|known| {
                    candidate.score > known.score
                        || (candidate.score == known.score
                            && candidate.source_index < known.source_index)
                })
                .unwrap_or(true);
            if replace {
                best = Some(candidate);
            }
        }
        let best =
            best.ok_or_else(|| JsValue::from_str("Solomon sampler produced no candidates"))?;
        let rgba = ink_to_rgba(&best.image);
        let metadata = format!(
            "{{\"schema\":\"{}\",\"model_format\":\"NSRLTCH\",\"prompt\":\"{}\",\"seed\":\"{}\",\"candidate_multiplier\":{},\"passes\":{},\"active_conditions\":{},\"used_conditions\":{},\"source_index\":{},\"score\":{},\"quality_score\":{},\"text_distance\":{},\"target_number\":{},\"target_name\":\"{}\",\"target_score\":{},\"width\":{},\"height\":{}}}",
            SCHEMA,
            json_escape(prompt),
            json_escape(seed),
            candidate_multiplier,
            passes,
            self.active_conditions.len(),
            used_condition_count,
            best.source_index,
            best.score,
            best.quality.total_score,
            best.text_distance,
            target.number,
            json_escape(&target.name),
            target.score,
            self.model.image_size,
            self.model.image_size,
        );
        Ok(SolomonSample {
            width: self.model.image_size,
            height: self.model.image_size,
            rgba,
            metadata,
        })
    }
}

#[wasm_bindgen]
pub struct SolomonSample {
    width: usize,
    height: usize,
    rgba: Vec<u8>,
    metadata: String,
}

#[wasm_bindgen]
impl SolomonSample {
    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn rgba(&self) -> Vec<u8> {
        self.rgba.clone()
    }

    pub fn metadata_json(&self) -> String {
        self.metadata.clone()
    }
}

#[derive(Debug)]
struct MultichannelLayer {
    condition_weights: Vec<Vec<i8>>,
    condition_biases: Vec<i16>,
    condition_blend_shifts: Vec<u8>,
}

#[derive(Debug)]
struct MultichannelModel {
    image_size: usize,
    timesteps: usize,
    hidden_shift: u8,
    output_shift: u8,
    hidden_channels: usize,
    corruption_count: usize,
    layers: Vec<MultichannelLayer>,
}

#[derive(Debug)]
struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

#[derive(Debug, Clone)]
struct TextTarget {
    number: usize,
    name: String,
    aliases: String,
    row_text: String,
    // 16x16 signature fed to the model's text-conditioning features.
    text_signature: [u16; TEXT_SIGNATURE_BINS],
    text_stats: SignatureStats,
    // 8x8 downsample used only by the browser candidate-scoring heuristic.
    score_signature: [u16; SIGNATURE_BINS],
    score: i64,
}

#[derive(Debug, Clone, Copy)]
struct SignatureStats {
    global_mean: i16,
    row_means: [i16; TEXT_SIGNATURE_GRID],
    col_means: [i16; TEXT_SIGNATURE_GRID],
}

/// Per-pixel text-feature cache for one target signature. Channels 1..=14 depend
/// only on the target signature and pixel position, so they are computed once
/// per sample; channels 0 and 15 depend on the live (denoising) pixel value and
/// are recomputed per pixel from `centers` and `global_mean`.
#[derive(Debug)]
struct TextConditioning {
    static_features: Vec<[i16; TEXT_FEATURE_CHANNELS]>,
    centers: Vec<i16>,
    global_mean: i16,
}

#[derive(Debug, Clone)]
struct Candidate {
    source_index: usize,
    score: i64,
    text_distance: i64,
    quality: SampleQuality,
    image: Vec<u8>,
}

#[derive(Debug, Clone, Copy)]
struct SampleQuality {
    total_score: i64,
}

#[derive(Debug)]
struct Rng {
    state: u32,
}

fn read_text_conditioned_model(bytes: &[u8]) -> Result<MultichannelModel, String> {
    let mut cursor = Cursor { bytes, offset: 0 };
    let magic = cursor.read_bytes(MODEL_MAGIC_TCH.len())?.to_vec();
    if magic.as_slice() != MODEL_MAGIC_TCH {
        return Err("Solomon web sampler requires an NSRLTCH model".to_string());
    }
    let model = read_multichannel_model(&mut cursor)?;
    if cursor.offset != cursor.bytes.len() {
        return Err(format!(
            "Solomon model has {} trailing bytes",
            cursor.bytes.len() - cursor.offset
        ));
    }
    Ok(model)
}

fn read_multichannel_model(cursor: &mut Cursor<'_>) -> Result<MultichannelModel, String> {
    let image_size = usize::try_from(cursor.read_u32()?).map_err(|error| error.to_string())?;
    let timesteps = usize::try_from(cursor.read_u32()?).map_err(|error| error.to_string())?;
    let hidden_shift = u8::try_from(cursor.read_u32()?).map_err(|error| error.to_string())?;
    let output_shift = u8::try_from(cursor.read_u32()?).map_err(|error| error.to_string())?;
    let hidden_channels = usize::try_from(cursor.read_u32()?).map_err(|error| error.to_string())?;
    let max_channels = HIDDEN_KERNELS.len() + POSITION_FEATURE_CHANNELS + TEXT_FEATURE_CHANNELS;
    if hidden_channels == 0 || hidden_channels > max_channels {
        return Err(format!(
            "unsupported Solomon hidden channel count: {hidden_channels}; max {max_channels}"
        ));
    }
    let corruption_count =
        usize::try_from(cursor.read_u32()?).map_err(|error| error.to_string())?;
    let layer_count = usize::try_from(cursor.read_u32()?).map_err(|error| error.to_string())?;
    let condition_count = corruption_count
        .checked_mul(timesteps)
        .ok_or_else(|| "Solomon condition count overflow".to_string())?;
    let mut layers = Vec::with_capacity(layer_count);
    for _ in 0..layer_count {
        let mut condition_weights = Vec::with_capacity(condition_count);
        for _ in 0..condition_count {
            let mut weights = Vec::with_capacity(hidden_channels);
            for _ in 0..hidden_channels {
                weights.push(cursor.read_i8()?);
            }
            condition_weights.push(weights);
        }
        let mut condition_biases = Vec::with_capacity(condition_count);
        for _ in 0..condition_count {
            condition_biases.push(cursor.read_i16()?);
        }
        let condition_blend_shifts = cursor.read_bytes(condition_count)?.to_vec();
        layers.push(MultichannelLayer {
            condition_weights,
            condition_biases,
            condition_blend_shifts,
        });
    }
    Ok(MultichannelModel {
        image_size,
        timesteps,
        hidden_shift,
        output_shift,
        hidden_channels,
        corruption_count,
        layers,
    })
}

impl Cursor<'_> {
    fn read_bytes(&mut self, count: usize) -> Result<&[u8], String> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or_else(|| "Solomon cursor overflow".to_string())?;
        if end > self.bytes.len() {
            return Err("unexpected end of Solomon model".to_string());
        }
        let start = self.offset;
        self.offset = end;
        Ok(&self.bytes[start..end])
    }

    fn read_u32(&mut self) -> Result<u32, String> {
        let bytes = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_i16(&mut self) -> Result<i16, String> {
        let bytes = self.read_bytes(2)?;
        Ok(i16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_i8(&mut self) -> Result<i8, String> {
        Ok(self.read_bytes(1)?[0] as i8)
    }
}

fn read_text_targets(text: &str) -> Result<Vec<TextTarget>, String> {
    let mut targets = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        if line_index == 0 || line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 9 {
            return Err(format!(
                "Solomon text index line {} has {} fields, expected 9",
                line_index + 1,
                fields.len()
            ));
        }
        let text_signature = parse_text_signature(fields[7], line_index + 1)?;
        targets.push(TextTarget {
            number: fields[0]
                .parse()
                .map_err(|error: std::num::ParseIntError| error.to_string())?,
            name: fields[1].to_string(),
            aliases: fields[2].to_string(),
            row_text: fields[8].to_string(),
            text_stats: signature_stats(&text_signature),
            score_signature: downsample_text_signature(&text_signature),
            text_signature,
            score: 0,
        });
    }
    targets.sort_by_key(|target| target.number);
    if targets.is_empty() {
        return Err("Solomon text index has no target rows".to_string());
    }
    Ok(targets)
}

fn parse_text_signature(
    text: &str,
    line_number: usize,
) -> Result<[u16; TEXT_SIGNATURE_BINS], String> {
    let mut signature = [0_u16; TEXT_SIGNATURE_BINS];
    let parts: Vec<&str> = text.split(',').collect();
    if parts.len() != TEXT_SIGNATURE_BINS {
        return Err(format!(
            "Solomon text index line {line_number} has {} signature bins, expected {TEXT_SIGNATURE_BINS}",
            parts.len()
        ));
    }
    for (index, part) in parts.iter().enumerate() {
        signature[index] = part
            .parse()
            .map_err(|error: std::num::ParseIntError| error.to_string())?;
    }
    Ok(signature)
}

/// Average each 2x2 block of the 16x16 signature down to the 8x8 grid used by
/// the browser candidate-scoring heuristic.
fn downsample_text_signature(signature: &[u16; TEXT_SIGNATURE_BINS]) -> [u16; SIGNATURE_BINS] {
    let mut out = [0_u16; SIGNATURE_BINS];
    let ratio = TEXT_SIGNATURE_GRID / SIGNATURE_GRID;
    for gy in 0..SIGNATURE_GRID {
        for gx in 0..SIGNATURE_GRID {
            let mut sum = 0_u32;
            let mut count = 0_u32;
            for dy in 0..ratio {
                for dx in 0..ratio {
                    let sx = gx * ratio + dx;
                    let sy = gy * ratio + dy;
                    sum = sum.saturating_add(u32::from(signature[sy * TEXT_SIGNATURE_GRID + sx]));
                    count += 1;
                }
            }
            out[gy * SIGNATURE_GRID + gx] = u16::try_from(sum / count.max(1)).unwrap_or(u16::MAX);
        }
    }
    out
}

fn select_text_target(prompt: &str, targets: &[TextTarget]) -> Result<TextTarget, String> {
    let prompt_tokens = unique_tokens(tokenize_text(prompt));
    if prompt_tokens.is_empty() {
        return Err("type at least one word before sampling a Solomon sigil".to_string());
    }
    let mut best: Option<TextTarget> = None;
    for target in targets {
        let score = text_match_score(&prompt_tokens, target);
        let replace = best
            .as_ref()
            .map(|known| {
                score > known.score || (score == known.score && target.number < known.number)
            })
            .unwrap_or(true);
        if replace {
            let mut next = target.clone();
            next.score = score;
            best = Some(next);
        }
    }
    best.ok_or_else(|| "Solomon text index has no target rows".to_string())
}

fn text_match_score(prompt_tokens: &[String], target: &TextTarget) -> i64 {
    let mut score = 0_i64;
    let row_tokens = unique_tokens(tokenize_text(&format!(
        "{} {} {}",
        target.name,
        target.aliases.replace('|', " "),
        target.row_text
    )));
    for token in prompt_tokens {
        if row_tokens.iter().any(|row_token| row_token == token) {
            score = score.saturating_add(200);
            score = score.saturating_add(i64::try_from(token.len()).unwrap_or(0) * 30);
        }
    }
    for alias in target
        .aliases
        .split('|')
        .chain(std::iter::once(target.name.as_str()))
    {
        let alias_tokens = unique_tokens(tokenize_text(alias));
        if alias_tokens.is_empty() {
            continue;
        }
        if alias_tokens
            .iter()
            .all(|alias_token| prompt_tokens.iter().any(|token| token == alias_token))
        {
            score = score.saturating_add(8_000);
            score = score.saturating_add(i64::try_from(alias_tokens.len()).unwrap_or(0) * 400);
        }
    }
    score
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

fn unique_tokens(tokens: Vec<String>) -> Vec<String> {
    let mut unique = Vec::new();
    for token in tokens {
        if token.len() < 2 {
            continue;
        }
        if !unique.iter().any(|known| known == &token) {
            unique.push(token);
        }
    }
    unique
}

fn active_conditions(model: &MultichannelModel) -> Vec<usize> {
    let condition_count = model.corruption_count * model.timesteps;
    let mut conditions = Vec::new();
    for condition in 0..condition_count {
        if model.layers.iter().any(|layer| {
            layer
                .condition_blend_shifts
                .get(condition)
                .copied()
                .unwrap_or(u8::MAX)
                != u8::MAX
        }) {
            conditions.push(condition);
        }
    }
    conditions.sort_by(|left, right| {
        let left_timestep = left % model.timesteps;
        let right_timestep = right % model.timesteps;
        right_timestep
            .cmp(&left_timestep)
            .then_with(|| left.cmp(right))
    });
    conditions
}

fn generation_conditions(model: &MultichannelModel) -> Vec<usize> {
    let conditions = active_conditions(model);
    let Some(noise_kind) = CORRUPTION_KINDS
        .iter()
        .position(|&kind| kind == "noise-seed")
    else {
        return conditions;
    };
    if noise_kind >= model.corruption_count {
        return conditions;
    }
    let noise_conditions: Vec<usize> = conditions
        .iter()
        .copied()
        .filter(|condition| condition / model.timesteps == noise_kind)
        .collect();
    if noise_conditions.is_empty() {
        conditions
    } else {
        let mut scheduled = noise_conditions;
        scheduled.extend(
            conditions
                .iter()
                .copied()
                .filter(|condition| condition / model.timesteps != noise_kind),
        );
        scheduled
    }
}

fn sample_candidate(
    model: &MultichannelModel,
    conditions: &[usize],
    source_index: usize,
    seed: &str,
    passes: usize,
    conditioning: &TextConditioning,
    score_signature: &[u16; SIGNATURE_BINS],
) -> Result<Candidate, String> {
    let mut image = initial_noise(model.image_size, source_index, seed)?;
    for _ in 0..passes {
        for &condition in conditions {
            image = apply_model_condition(model, condition, &image, conditioning)?;
        }
    }
    let quality = score_sample(&image, model.image_size)?;
    let signature = sample_signature(&image, model.image_size)?;
    let text_distance = text_signature_distance(&signature, score_signature);
    let score = quality
        .total_score
        .saturating_sub(text_distance.saturating_mul(TEXT_WEIGHT));
    Ok(Candidate {
        source_index,
        score,
        text_distance,
        quality,
        image,
    })
}

fn initial_noise(image_size: usize, sample_index: usize, seed: &str) -> Result<Vec<u8>, String> {
    let image_bytes = checked_image_bytes(image_size)?;
    let init_seed = hash_seed(&[seed, &sample_index.to_string(), "noise"]);
    let mut rng = Rng::new(init_seed);
    let mut image = vec![0_u8; image_bytes];
    for value in &mut image {
        let roll = rng.rand_bounded(16);
        *value = if roll == 0 {
            255
        } else if roll == 1 {
            160
        } else {
            0
        };
    }
    Ok(image)
}

fn apply_model_condition(
    model: &MultichannelModel,
    condition: usize,
    input: &[u8],
    conditioning: &TextConditioning,
) -> Result<Vec<u8>, String> {
    let mut image = input.to_vec();
    for layer in &model.layers {
        image = apply_multichannel_layer_condition(model, layer, condition, &image, conditioning)?;
    }
    Ok(image)
}

fn apply_multichannel_layer_condition(
    model: &MultichannelModel,
    layer: &MultichannelLayer,
    condition: usize,
    input: &[u8],
    conditioning: &TextConditioning,
) -> Result<Vec<u8>, String> {
    let image_bytes = checked_image_bytes(model.image_size)?;
    if input.len() != image_bytes {
        return Err(format!(
            "Solomon input image has {} bytes, expected {image_bytes}",
            input.len()
        ));
    }
    let blend_shift = layer
        .condition_blend_shifts
        .get(condition)
        .copied()
        .unwrap_or(u8::MAX);
    let mut out = vec![0_u8; input.len()];
    let mut features = vec![0_i16; model.hidden_channels];
    for y in 0..model.image_size {
        for x in 0..model.image_size {
            let index = pixel_index(model.image_size, x, y);
            conditioned_features(
                input,
                model.image_size,
                x,
                y,
                index,
                model.hidden_shift,
                conditioning,
                &mut features,
            );
            let raw = predict_multichannel_raw_pixel(
                layer,
                model.output_shift,
                condition,
                input[index],
                &features,
            );
            out[index] = blend_pixel(input[index], raw, blend_shift);
        }
    }
    Ok(out)
}

#[expect(
    clippy::too_many_arguments,
    reason = "arguments mirror the serialized native/WASM conditioning contract"
)]
fn conditioned_features(
    input: &[u8],
    image_size: usize,
    x: usize,
    y: usize,
    pixel: usize,
    hidden_shift: u8,
    conditioning: &TextConditioning,
    out: &mut [i16],
) {
    let image_channels = out.len().min(HIDDEN_CHANNELS);
    hidden_features(
        input,
        image_size,
        x,
        y,
        hidden_shift,
        &mut out[..image_channels],
    );
    let mut offset = HIDDEN_CHANNELS;
    if out.len() >= offset + POSITION_FEATURE_CHANNELS {
        position_features(
            image_size,
            x,
            y,
            &mut out[offset..offset + POSITION_FEATURE_CHANNELS],
        );
        offset += POSITION_FEATURE_CHANNELS;
    }
    if out.len() < offset + TEXT_FEATURE_CHANNELS {
        return;
    }
    let (Some(static_features), Some(&center)) = (
        conditioning.static_features.get(pixel),
        conditioning.centers.get(pixel),
    ) else {
        return;
    };
    let input_center = i16::from(input[pixel]);
    let text = &mut out[offset..offset + TEXT_FEATURE_CHANNELS];
    text.copy_from_slice(static_features);
    // Channels 0 and 15 depend on the live (denoising) pixel value; the rest are
    // cached. Mirrors nsrl-bitmap-sample::conditioned_features.
    text[0] = center.saturating_sub(input_center).clamp(-511, 511);
    text[15] = center
        .saturating_sub(input_center)
        .saturating_add(center.saturating_sub(conditioning.global_mean))
        .saturating_mul(2)
        .clamp(-511, 511);
}

/// Precompute the per-pixel text features that do not depend on the evolving
/// image. Channels 0 and 15 are filled with `input_center = 0` and recomputed
/// per pixel at use time. Mirrors nsrl-bitmap-sample::build_feature_cache.
fn build_text_conditioning(
    signature: &[u16; TEXT_SIGNATURE_BINS],
    stats: &SignatureStats,
    image_size: usize,
) -> Result<TextConditioning, String> {
    let image_bytes = checked_image_bytes(image_size)?;
    let mut centers = Vec::with_capacity(image_bytes);
    let mut static_features = Vec::with_capacity(image_bytes);
    for y in 0..image_size {
        for x in 0..image_size {
            centers.push(interpolated_text_signature(signature, image_size, x, y));
            let mut features = [0_i16; TEXT_FEATURE_CHANNELS];
            text_signature_features(signature, stats, image_size, x, y, 0, &mut features);
            static_features.push(features);
        }
    }
    Ok(TextConditioning {
        static_features,
        centers,
        global_mean: stats.global_mean,
    })
}

fn interpolated_text_signature(
    signature: &[u16; TEXT_SIGNATURE_BINS],
    image_size: usize,
    x: usize,
    y: usize,
) -> i16 {
    if image_size <= 1 {
        return i16::try_from(signature[0].min(255)).unwrap_or(0);
    }
    let grid_max = TEXT_SIGNATURE_GRID - 1;
    let scale = 256_usize;
    let sx = x.saturating_mul(grid_max).saturating_mul(scale) / (image_size - 1);
    let sy = y.saturating_mul(grid_max).saturating_mul(scale) / (image_size - 1);
    let x0 = (sx / scale).min(grid_max);
    let y0 = (sy / scale).min(grid_max);
    let x1 = (x0 + 1).min(grid_max);
    let y1 = (y0 + 1).min(grid_max);
    let wx = i64::try_from(sx % scale).unwrap_or(0);
    let wy = i64::try_from(sy % scale).unwrap_or(0);
    let ix = i64::try_from(scale).unwrap_or(256).saturating_sub(wx);
    let iy = i64::try_from(scale).unwrap_or(256).saturating_sub(wy);
    let at = |xx: usize, yy: usize| -> i64 {
        i64::from(signature[yy * TEXT_SIGNATURE_GRID + xx].min(255))
    };
    let weighted = at(x0, y0)
        .saturating_mul(ix)
        .saturating_mul(iy)
        .saturating_add(at(x1, y0).saturating_mul(wx).saturating_mul(iy))
        .saturating_add(at(x0, y1).saturating_mul(ix).saturating_mul(wy))
        .saturating_add(at(x1, y1).saturating_mul(wx).saturating_mul(wy));
    i16::try_from((weighted + 32_768) >> 16).unwrap_or(0)
}

#[allow(clippy::too_many_arguments)]
fn text_signature_features(
    signature: &[u16; TEXT_SIGNATURE_BINS],
    stats: &SignatureStats,
    image_size: usize,
    x: usize,
    y: usize,
    input_center: i16,
    out: &mut [i16],
) {
    if out.len() < TEXT_FEATURE_CHANNELS {
        return;
    }
    let step = (image_size / TEXT_SIGNATURE_GRID.max(1)).max(1);
    let center = interpolated_text_signature(signature, image_size, x, y);
    let left = interpolated_text_signature(signature, image_size, x.saturating_sub(step), y);
    let right = interpolated_text_signature(
        signature,
        image_size,
        (x + step).min(image_size.saturating_sub(1)),
        y,
    );
    let up = interpolated_text_signature(signature, image_size, x, y.saturating_sub(step));
    let down = interpolated_text_signature(
        signature,
        image_size,
        x,
        (y + step).min(image_size.saturating_sub(1)),
    );
    let up_left = interpolated_text_signature(
        signature,
        image_size,
        x.saturating_sub(step),
        y.saturating_sub(step),
    );
    let down_right = interpolated_text_signature(
        signature,
        image_size,
        (x + step).min(image_size.saturating_sub(1)),
        (y + step).min(image_size.saturating_sub(1)),
    );
    let neighbor_mean =
        i16::try_from((i32::from(left) + i32::from(right) + i32::from(up) + i32::from(down)) / 4)
            .unwrap_or(0);
    let grid_x = signature_grid_coord(image_size, x);
    let grid_y = signature_grid_coord(image_size, y);
    let global_mean = stats.global_mean;
    let row_mean = stats.row_means[grid_y];
    let col_mean = stats.col_means[grid_x];
    let horizontal_curve = left
        .saturating_add(right)
        .saturating_sub(center.saturating_mul(2));
    let vertical_curve = up
        .saturating_add(down)
        .saturating_sub(center.saturating_mul(2));
    out[0] = center.saturating_sub(input_center).clamp(-511, 511);
    out[1] = center.saturating_sub(64).clamp(-511, 511);
    out[2] = center.saturating_sub(32).clamp(-511, 511);
    out[3] = center.saturating_sub(48).saturating_mul(4).clamp(-511, 511);
    out[4] = right
        .saturating_sub(left)
        .saturating_mul(2)
        .clamp(-511, 511);
    out[5] = down.saturating_sub(up).saturating_mul(2).clamp(-511, 511);
    out[6] = down_right
        .saturating_sub(up_left)
        .saturating_mul(2)
        .clamp(-511, 511);
    out[7] = center
        .saturating_sub(neighbor_mean)
        .saturating_mul(4)
        .clamp(-511, 511);
    out[8] = center
        .saturating_sub(global_mean)
        .saturating_mul(4)
        .clamp(-511, 511);
    out[9] = center
        .saturating_sub(row_mean)
        .saturating_mul(4)
        .clamp(-511, 511);
    out[10] = center
        .saturating_sub(col_mean)
        .saturating_mul(4)
        .clamp(-511, 511);
    out[11] = horizontal_curve.saturating_mul(4).clamp(-511, 511);
    out[12] = vertical_curve.saturating_mul(4).clamp(-511, 511);
    out[13] = if center >= 160 { 256 } else { -256 };
    out[14] = if center <= 32 { -256 } else { 256 };
    out[15] = center
        .saturating_sub(input_center)
        .saturating_add(center.saturating_sub(global_mean))
        .saturating_mul(2)
        .clamp(-511, 511);
}

fn signature_grid_coord(image_size: usize, value: usize) -> usize {
    if image_size <= 1 {
        return 0;
    }
    value
        .saturating_mul(TEXT_SIGNATURE_GRID)
        .checked_div(image_size)
        .unwrap_or(0)
        .min(TEXT_SIGNATURE_GRID - 1)
}

fn signature_stats(signature: &[u16; TEXT_SIGNATURE_BINS]) -> SignatureStats {
    let global_mean = signature_global_mean(signature);
    let mut row_means = [0_i16; TEXT_SIGNATURE_GRID];
    let mut col_means = [0_i16; TEXT_SIGNATURE_GRID];
    for (row, mean) in row_means.iter_mut().enumerate() {
        *mean = signature_row_mean(signature, row);
    }
    for (col, mean) in col_means.iter_mut().enumerate() {
        *mean = signature_col_mean(signature, col);
    }
    SignatureStats {
        global_mean,
        row_means,
        col_means,
    }
}

fn signature_global_mean(signature: &[u16; TEXT_SIGNATURE_BINS]) -> i16 {
    let total: u32 = signature
        .iter()
        .map(|&value| u32::from(value.min(255)))
        .sum();
    i16::try_from(total / u32::try_from(TEXT_SIGNATURE_BINS).unwrap_or(1)).unwrap_or(0)
}

fn signature_row_mean(signature: &[u16; TEXT_SIGNATURE_BINS], row: usize) -> i16 {
    let row = row.min(TEXT_SIGNATURE_GRID - 1);
    let start = row * TEXT_SIGNATURE_GRID;
    let total: u32 = signature[start..start + TEXT_SIGNATURE_GRID]
        .iter()
        .map(|&value| u32::from(value.min(255)))
        .sum();
    i16::try_from(total / u32::try_from(TEXT_SIGNATURE_GRID).unwrap_or(1)).unwrap_or(0)
}

fn signature_col_mean(signature: &[u16; TEXT_SIGNATURE_BINS], col: usize) -> i16 {
    let col = col.min(TEXT_SIGNATURE_GRID - 1);
    let mut total = 0_u32;
    for row in 0..TEXT_SIGNATURE_GRID {
        total = total.saturating_add(u32::from(
            signature[row * TEXT_SIGNATURE_GRID + col].min(255),
        ));
    }
    i16::try_from(total / u32::try_from(TEXT_SIGNATURE_GRID).unwrap_or(1)).unwrap_or(0)
}

fn position_features(image_size: usize, x: usize, y: usize, out: &mut [i16]) {
    if out.len() < POSITION_FEATURE_CHANNELS || image_size == 0 {
        return;
    }
    let size = i64::try_from(image_size).unwrap_or(1).max(1);
    let dx = i64::try_from(x)
        .unwrap_or(0)
        .saturating_mul(2)
        .saturating_add(1)
        .saturating_sub(size);
    let dy = i64::try_from(y)
        .unwrap_or(0)
        .saturating_mul(2)
        .saturating_add(1)
        .saturating_sub(size);
    let radius = integer_sqrt_u64(
        u64::try_from(dx.saturating_mul(dx).saturating_add(dy.saturating_mul(dy))).unwrap_or(0),
    );
    let radius_i64 = i64::try_from(radius).unwrap_or(i64::MAX);
    let outer_radius = size.saturating_sub(18).max(1);
    let inner_radius = size.saturating_sub(38).max(1);
    let ring_width = (size / 12).max(6);

    out[0] = clamp_i16(dx.saturating_mul(256) / size);
    out[1] = clamp_i16(dy.saturating_mul(256) / size);
    out[2] = clamp_i16(radius_i64.saturating_mul(384) / size - 192);
    out[3] = clamp_i16(triangular_peak(radius_i64, outer_radius, ring_width));
    out[4] = clamp_i16(triangular_peak(radius_i64, inner_radius, ring_width));
    out[5] = if radius_i64 <= outer_radius {
        192
    } else {
        -192
    };
}

fn triangular_peak(value: i64, center: i64, width: i64) -> i64 {
    let distance = abs_i64(value.saturating_sub(center));
    if distance >= width {
        return 0;
    }
    255 - distance.saturating_mul(255) / width.max(1)
}

fn hidden_features(
    input: &[u8],
    image_size: usize,
    x: usize,
    y: usize,
    hidden_shift: u8,
    out: &mut [i16],
) {
    let mut local = [0_i16; KERNEL];
    local_features(input, image_size, x, y, &mut local);
    for (channel, kernel) in HIDDEN_KERNELS.iter().take(out.len()).enumerate() {
        let mut acc = 0_i64;
        for (weight, feature) in kernel.iter().zip(local.iter()) {
            acc = acc.saturating_add(i64::from(*weight) * i64::from(*feature));
        }
        out[channel] = signed_round_shift(acc, hidden_shift).clamp(-511, 511) as i16;
    }
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

fn predict_multichannel_raw_pixel(
    layer: &MultichannelLayer,
    output_shift: u8,
    condition: usize,
    input_center: u8,
    features: &[i16],
) -> u8 {
    let mut acc = i64::from(*layer.condition_biases.get(condition).unwrap_or(&0));
    if let Some(weights) = layer.condition_weights.get(condition) {
        for (weight, feature) in weights.iter().zip(features.iter()) {
            acc = acc.saturating_add(i64::from(*weight) * i64::from(*feature));
        }
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

fn score_sample(image: &[u8], image_size: usize) -> Result<SampleQuality, String> {
    let image_bytes = checked_image_bytes(image_size)?;
    if image.len() != image_bytes {
        return Err(format!(
            "Solomon sample has {} bytes, expected {image_bytes}",
            image.len()
        ));
    }
    let center = i64::try_from(image_size / 2).map_err(|error| error.to_string())?;
    let outer_radius = i64::try_from(image_size / 2)
        .map_err(|error| error.to_string())?
        .saturating_sub(10);
    let inner_radius = outer_radius.saturating_sub(11);
    let outer_low = outer_radius.saturating_sub(5);
    let outer_high = outer_radius.saturating_add(5);
    let inner_low = inner_radius.saturating_sub(4);
    let inner_high = inner_radius.saturating_add(4);
    let center_radius = inner_radius.saturating_sub(7);
    let outer_low2 = outer_low.saturating_mul(outer_low);
    let outer_high2 = outer_high.saturating_mul(outer_high);
    let inner_low2 = inner_low.saturating_mul(inner_low);
    let inner_high2 = inner_high.saturating_mul(inner_high);
    let center_radius2 = center_radius.saturating_mul(center_radius);
    let outside_radius2 = outer_high.saturating_add(6).saturating_pow(2);

    let mut ink_sum = 0_i64;
    let mut outer_ring_sum = 0_i64;
    let mut inner_ring_sum = 0_i64;
    let mut center_sum = 0_i64;
    let mut center_count = 0_i64;
    let mut border_ink = 0_i64;
    let mut outside_ink_sum = 0_i64;
    let mut edge_sum = 0_i64;
    let mut strong_ink_sum = 0_i64;
    let mut soft_ink_sum = 0_i64;
    let mut ring_buckets = [0_i64; QUALITY_RING_BUCKETS];

    for y in 0..image_size {
        for x in 0..image_size {
            let value_u8 = image[pixel_index(image_size, x, y)];
            let value = i64::from(value_u8);
            ink_sum = ink_sum.saturating_add(value);
            let dx = i64::try_from(x).map_err(|error| error.to_string())? - center;
            let dy = i64::try_from(y).map_err(|error| error.to_string())? - center;
            let radius2 = dx.saturating_mul(dx).saturating_add(dy.saturating_mul(dy));
            if radius2 <= center_radius2 {
                center_sum = center_sum.saturating_add(value);
                center_count = center_count.saturating_add(1);
            }
            if value_u8 > STRONG_INK_THRESHOLD {
                strong_ink_sum = strong_ink_sum.saturating_add(value);
            } else if value_u8 > 0 && value_u8 <= STRONG_INK_THRESHOLD {
                soft_ink_sum = soft_ink_sum.saturating_add(value);
            }
            if value_u8 > INK_THRESHOLD {
                if x < 3 || y < 3 || x + 3 >= image_size || y + 3 >= image_size {
                    border_ink = border_ink.saturating_add(1);
                }
                if radius2 > outside_radius2 {
                    outside_ink_sum = outside_ink_sum.saturating_add(value);
                }
                if radius2 >= outer_low2 && radius2 <= outer_high2 {
                    outer_ring_sum = outer_ring_sum.saturating_add(value);
                    let bucket = angle_bucket(dx, dy);
                    ring_buckets[bucket] = ring_buckets[bucket].saturating_add(value);
                }
                if radius2 >= inner_low2 && radius2 <= inner_high2 {
                    inner_ring_sum = inner_ring_sum.saturating_add(value);
                    let bucket = angle_bucket(dx, dy);
                    ring_buckets[bucket] = ring_buckets[bucket].saturating_add(value);
                }
            }
            if x + 1 < image_size {
                let diff = abs_diff_u8(value_u8, image[pixel_index(image_size, x + 1, y)]);
                if diff > INK_THRESHOLD {
                    edge_sum = edge_sum.saturating_add(i64::from(diff));
                }
            }
            if y + 1 < image_size {
                let diff = abs_diff_u8(value_u8, image[pixel_index(image_size, x, y + 1)]);
                if diff > INK_THRESHOLD {
                    edge_sum = edge_sum.saturating_add(i64::from(diff));
                }
            }
        }
    }

    let mut ring_hit_count = 0_i64;
    let mut ring_bucket_total = 0_i64;
    for &bucket_sum in &ring_buckets {
        ring_bucket_total = ring_bucket_total.saturating_add(bucket_sum);
        if bucket_sum >= 256 {
            ring_hit_count = ring_hit_count.saturating_add(1);
        }
    }
    let ring_mean = ring_bucket_total / i64::try_from(QUALITY_RING_BUCKETS).unwrap_or(1);
    let mut ring_balance_penalty = 0_i64;
    for &bucket_sum in &ring_buckets {
        ring_balance_penalty =
            ring_balance_penalty.saturating_add(abs_i64(bucket_sum.saturating_sub(ring_mean)));
    }
    ring_balance_penalty /= 8;

    let mut speck_count = 0_i64;
    for y in 0..image_size {
        for x in 0..image_size {
            if image[pixel_index(image_size, x, y)] <= SPECK_INK_THRESHOLD {
                continue;
            }
            if neighbor_ink_count(image, image_size, x, y) <= 1 {
                speck_count = speck_count.saturating_add(1);
            }
        }
    }

    let target_ink_sum = i64::try_from(image_bytes).unwrap_or(0).saturating_mul(47);
    let target_center_sum = center_count.saturating_mul(38);
    let density_penalty = abs_i64(ink_sum.saturating_sub(target_ink_sum)) / 3;
    let center_density_error = abs_i64(center_sum.saturating_sub(target_center_sum));
    let ring_coverage_score = ring_hit_count.saturating_mul(900);
    let ring_miss_penalty = (i64::try_from(QUALITY_RING_BUCKETS).unwrap_or(0) - ring_hit_count)
        .max(0)
        .saturating_mul(550);
    let ring_score = outer_ring_sum
        .saturating_mul(3)
        .saturating_add(inner_ring_sum.saturating_mul(2))
        .saturating_add(ring_coverage_score)
        .saturating_sub(ring_miss_penalty)
        .saturating_sub(ring_balance_penalty);
    let stroke_score = edge_sum
        .saturating_mul(2)
        .saturating_add(strong_ink_sum / 4)
        .saturating_sub(soft_ink_sum / 8);
    let interior_score = target_center_sum
        .saturating_sub(center_density_error)
        .max(0)
        .saturating_mul(6);
    let border_penalty = border_ink.saturating_mul(550);
    let outside_penalty = outside_ink_sum / 2;
    let speck_penalty = speck_count.saturating_mul(900);
    let total_score = ring_score
        .saturating_add(stroke_score)
        .saturating_add(interior_score)
        .saturating_sub(density_penalty)
        .saturating_sub(border_penalty)
        .saturating_sub(outside_penalty)
        .saturating_sub(speck_penalty);

    Ok(SampleQuality { total_score })
}

fn sample_signature(image: &[u8], image_size: usize) -> Result<[u16; SIGNATURE_BINS], String> {
    let image_bytes = checked_image_bytes(image_size)?;
    if image.len() != image_bytes {
        return Err(format!(
            "Solomon sample has {} bytes, expected {image_bytes}",
            image.len()
        ));
    }
    let mut sums = [0_u32; SIGNATURE_BINS];
    let mut counts = [0_u32; SIGNATURE_BINS];
    for y in 0..image_size {
        let bin_y = y * SIGNATURE_GRID / image_size;
        for x in 0..image_size {
            let bin_x = x * SIGNATURE_GRID / image_size;
            let bin = bin_y * SIGNATURE_GRID + bin_x;
            sums[bin] = sums[bin].saturating_add(u32::from(image[pixel_index(image_size, x, y)]));
            counts[bin] = counts[bin].saturating_add(1);
        }
    }
    let mut signature = [0_u16; SIGNATURE_BINS];
    for index in 0..SIGNATURE_BINS {
        if counts[index] != 0 {
            signature[index] =
                u16::try_from(sums[index].saturating_add(counts[index] / 2) / counts[index])
                    .map_err(|error| error.to_string())?;
        }
    }
    Ok(signature)
}

fn text_signature_distance(
    candidate: &[u16; SIGNATURE_BINS],
    target: &[u16; SIGNATURE_BINS],
) -> i64 {
    let mut best = i64::MAX;
    for variant in 0..8 {
        let mut distance = 0_i64;
        for y in 0..SIGNATURE_GRID {
            for x in 0..SIGNATURE_GRID {
                let (sx, sy) = transform_coords(SIGNATURE_GRID, x, y, variant);
                let left = candidate[pixel_index(SIGNATURE_GRID, x, y)];
                let right = target[pixel_index(SIGNATURE_GRID, sx, sy)];
                distance = distance.saturating_add(i64::from(abs_diff_u16(left, right)));
            }
        }
        best = best.min(distance);
    }
    best
}

fn neighbor_ink_count(image: &[u8], image_size: usize, x: usize, y: usize) -> u8 {
    let mut count = 0_u8;
    for dy in 0..3 {
        for dx in 0..3 {
            if dx == 1 && dy == 1 {
                continue;
            }
            let nx = x.checked_add(dx).and_then(|value| value.checked_sub(1));
            let ny = y.checked_add(dy).and_then(|value| value.checked_sub(1));
            if let (Some(nx), Some(ny)) = (nx, ny)
                && nx < image_size
                && ny < image_size
            {
                let value = image[pixel_index(image_size, nx, ny)];
                if value > INK_THRESHOLD {
                    count = count.saturating_add(1);
                }
            }
        }
    }
    count
}

fn angle_bucket(dx: i64, dy: i64) -> usize {
    let ax = abs_i64(dx);
    let ay = abs_i64(dy);
    if ax == 0 && ay == 0 {
        return 0;
    }
    let (octant, major, minor) = if dx >= 0 && dy >= 0 {
        if ax >= ay {
            (0_usize, ax, ay)
        } else {
            (1_usize, ay, ax)
        }
    } else if dx < 0 && dy >= 0 {
        if ay >= ax {
            (2_usize, ay, ax)
        } else {
            (3_usize, ax, ay)
        }
    } else if dx < 0 && dy < 0 {
        if ax >= ay {
            (4_usize, ax, ay)
        } else {
            (5_usize, ay, ax)
        }
    } else if ay >= ax {
        (6_usize, ay, ax)
    } else {
        (7_usize, ax, ay)
    };
    let local = if major == 0 {
        0
    } else {
        ((minor.saturating_mul(QUALITY_BUCKET_STEPS)) / major).min(QUALITY_BUCKET_STEPS - 1)
    };
    (octant * usize::try_from(QUALITY_BUCKET_STEPS).unwrap_or(1)
        + usize::try_from(local).unwrap_or(0))
        % QUALITY_RING_BUCKETS
}

fn transform_coords(image_size: usize, x: usize, y: usize, variant: usize) -> (usize, usize) {
    let last = image_size - 1;
    match variant % 8 {
        0 => (x, y),
        1 => (last - x, y),
        2 => (x, last - y),
        3 => (last - x, last - y),
        4 => (y, x),
        5 => (last - y, x),
        6 => (y, last - x),
        _ => (last - y, last - x),
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

fn integer_sqrt_u64(value: u64) -> u64 {
    if value <= 1 {
        return value;
    }
    let mut low = 1_u64;
    let mut high = value.min(1_u64 << 32);
    while low <= high {
        let mid = low + ((high - low) / 2);
        let square = u128::from(mid) * u128::from(mid);
        if square == u128::from(value) {
            return mid;
        }
        if square < u128::from(value) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    high
}

fn clamp_i16(value: i64) -> i16 {
    value.clamp(-511, 511) as i16
}

fn pixel_index(image_size: usize, x: usize, y: usize) -> usize {
    y * image_size + x
}

fn checked_image_bytes(image_size: usize) -> Result<usize, String> {
    image_size
        .checked_mul(image_size)
        .ok_or_else(|| "Solomon image byte count overflow".to_string())
}

fn ink_to_rgba(image: &[u8]) -> Vec<u8> {
    let mut rgba = Vec::with_capacity(image.len() * 4);
    let paper = [239_u16, 232_u16, 210_u16];
    let ink = [18_u16, 18_u16, 15_u16];
    for &value in image {
        let alpha = u16::from(value);
        for channel in 0..3 {
            let blended = (paper[channel] * (255 - alpha) + ink[channel] * alpha) / 255;
            rgba.push(u8::try_from(blended).unwrap_or(u8::MAX));
        }
        rgba.push(255);
    }
    rgba
}

fn hash_seed(parts: &[&str]) -> u32 {
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

impl Rng {
    fn new(seed: u32) -> Self {
        Self { state: seed | 1 }
    }

    fn next_u32(&mut self) -> u32 {
        let mut value = self.state;
        value ^= value << 13;
        value ^= value >> 17;
        value ^= value << 5;
        self.state = value;
        value
    }

    fn rand_bounded(&mut self, max_exclusive: u32) -> u32 {
        if max_exclusive <= 1 {
            return 0;
        }
        let range = 1_u64 << 32;
        let bound = u64::from(max_exclusive);
        let limit = range - (range % bound);
        loop {
            let value = u64::from(self.next_u32());
            if value < limit {
                return (value % bound) as u32;
            }
        }
    }
}

fn abs_i64(value: i64) -> i64 {
    if value >= 0 {
        value
    } else {
        value.saturating_neg()
    }
}

fn abs_diff_u8(left: u8, right: u8) -> u8 {
    left.abs_diff(right)
}

fn abs_diff_u16(left: u16, right: u16) -> u16 {
    left.abs_diff(right)
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

fn js_error(error: String) -> JsValue {
    JsValue::from_str(&error)
}

#[cfg(test)]
mod parity_tests {
    use super::*;

    fn load() -> (MultichannelModel, Vec<TextTarget>) {
        let root = env!("CARGO_MANIFEST_DIR");
        let model_bytes =
            std::fs::read(format!("{root}/../../web/assets/solomon-model.nsrltch")).unwrap();
        let model = read_text_conditioned_model(&model_bytes).unwrap();
        let tsv = std::fs::read_to_string(format!(
            "{root}/../../web/assets/solomon-spirit-text-signatures.tsv"
        ))
        .unwrap();
        let targets = read_text_targets(&tsv).unwrap();
        (model, targets)
    }

    #[test]
    fn loads_30_channel_model_and_256_bin_signatures() {
        let (model, targets) = load();
        assert_eq!(
            model.hidden_channels,
            HIDDEN_CHANNELS + POSITION_FEATURE_CHANNELS + TEXT_FEATURE_CHANNELS,
            "expected 30-channel model"
        );
        assert!(!targets.is_empty());
        // 256-bin signature parsed in full; 8x8 downsample populated for scoring.
        assert!(targets[0].text_signature.iter().any(|&v| v > 0));
        assert!(targets[0].score_signature.iter().any(|&v| v > 0));
    }

    #[test]
    fn renders_nonblank_text_conditioned_sigil() {
        let (model, targets) = load();
        let conditions = generation_conditions(&model);
        assert!(!conditions.is_empty());
        let target = &targets[0];
        let conditioning =
            build_text_conditioning(&target.text_signature, &target.text_stats, model.image_size)
                .unwrap();
        let candidate = sample_candidate(
            &model,
            &conditions,
            0,
            "parity-seed",
            4,
            &conditioning,
            &target.score_signature,
        )
        .unwrap();
        let ink = candidate
            .image
            .iter()
            .filter(|&&v| v > INK_THRESHOLD)
            .count();
        // Not blank (the regression we are guarding against) and not fully saturated.
        assert!(ink > 50, "sigil has too little ink: {ink}");
        assert!(
            ink < candidate.image.len() * 9 / 10,
            "sigil is over-saturated: {ink}"
        );
    }

    #[test]
    fn different_targets_produce_different_sigils() {
        let (model, targets) = load();
        let conditions = generation_conditions(&model);
        let render = |target: &TextTarget| {
            let conditioning = build_text_conditioning(
                &target.text_signature,
                &target.text_stats,
                model.image_size,
            )
            .unwrap();
            sample_candidate(
                &model,
                &conditions,
                0,
                "parity-seed",
                4,
                &conditioning,
                &target.score_signature,
            )
            .unwrap()
            .image
        };
        let a = render(&targets[0]);
        let b = render(&targets[targets.len() / 2]);
        let differing = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
        assert!(
            differing > 100,
            "text conditioning had no effect: {differing} differing pixels"
        );
    }

    fn load_model_at(path: &std::path::Path) -> MultichannelModel {
        read_text_conditioned_model(&std::fs::read(path).unwrap()).unwrap()
    }

    // Blur proxy: pixels-per-thousand in the mid-gray band [INK_THRESHOLD, STRONG_INK_THRESHOLD].
    // Lower = crisper (more pixels snapped to paper/ink extremes). Integer permille
    // keeps this in the no-float contract enforced by check-no-floats.sh.
    fn blur_permille(image: &[u8]) -> u64 {
        if image.is_empty() {
            return 0;
        }
        let mid = image
            .iter()
            .filter(|&&v| v > INK_THRESHOLD && v <= STRONG_INK_THRESHOLD)
            .count() as u64;
        mid.saturating_mul(1000) / image.len() as u64
    }

    #[test]
    #[ignore] // set NSRL_SCALED_MODEL and run explicitly to compare crispness
    fn compare_scaled_seal_crispness() {
        let root = env!("CARGO_MANIFEST_DIR");
        let tsv = std::fs::read_to_string(format!(
            "{root}/../../web/assets/solomon-spirit-text-signatures.tsv"
        ))
        .unwrap();
        let targets = read_text_targets(&tsv).unwrap();
        let target = &targets[0];
        let render = |model: &MultichannelModel| {
            let cond = build_text_conditioning(
                &target.text_signature,
                &target.text_stats,
                model.image_size,
            )
            .unwrap();
            let conditions = generation_conditions(model);
            sample_candidate(
                model,
                &conditions,
                0,
                "crisp-cmp",
                4,
                &cond,
                &target.score_signature,
            )
            .unwrap()
            .image
        };
        let deployed_path =
            std::path::PathBuf::from(format!("{root}/../../web/assets/solomon-model.nsrltch"));
        let Some(scaled_path) = std::env::var_os("NSRL_SCALED_MODEL").map(std::path::PathBuf::from)
        else {
            eprintln!("skipped: set NSRL_SCALED_MODEL to a scaled .nsrltch artifact");
            return;
        };
        assert!(
            scaled_path.is_file(),
            "NSRL_SCALED_MODEL does not exist: {}",
            scaled_path.display()
        );
        let deployed = load_model_at(&deployed_path);
        let scaled = load_model_at(&scaled_path);
        let bd = blur_permille(&render(&deployed));
        let bs = blur_permille(&render(&scaled));
        println!(
            "DEPLOYED layers={} blur_permille={bd}",
            deployed.layers.len()
        );
        println!("SCALED   layers={} blur_permille={bs}", scaled.layers.len());
        let delta = bs as i64 - bd as i64;
        println!(
            "VERDICT: scaled is {} ({delta:+} permille blur change)",
            if bs < bd { "CRISPER" } else { "NOT crisper" },
        );
    }
}
