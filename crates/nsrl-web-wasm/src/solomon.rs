use wasm_bindgen::prelude::*;

const SCHEMA: &str = "nsrl.web_solomon_sigil.v1";
const MODEL_MAGIC_TCH: &[u8; 8] = b"NSRLTCH\n";
const KERNEL: usize = 9;
const HIDDEN_CHANNELS: usize = 8;
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
const SIGNATURE_GRID: usize = 8;
const SIGNATURE_BINS: usize = SIGNATURE_GRID * SIGNATURE_GRID;
const QUALITY_RING_BUCKETS: usize = 32;
const QUALITY_BUCKET_STEPS: i64 = 4;
const INK_THRESHOLD: u8 = 64;
const STRONG_INK_THRESHOLD: u8 = 160;
const SPECK_INK_THRESHOLD: u8 = 96;
const TEXT_WEIGHT: i64 = 96;
const MAX_BROWSER_CANDIDATES: usize = 32;
const MAX_BROWSER_PASSES: usize = 12;

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
        let active_conditions = active_conditions(&model);
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
        let candidate_multiplier = candidate_multiplier.clamp(1, MAX_BROWSER_CANDIDATES);
        let passes = passes.clamp(1, MAX_BROWSER_PASSES);
        let target = select_text_target(prompt, &self.targets).map_err(js_error)?;
        let mut best: Option<Candidate> = None;
        for source_index in 0..candidate_multiplier {
            let candidate = sample_candidate(
                &self.model,
                &self.active_conditions,
                source_index,
                seed,
                passes,
                &target.signature,
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
            "{{\"schema\":\"{}\",\"model_format\":\"NSRLTCH\",\"prompt\":\"{}\",\"seed\":\"{}\",\"candidate_multiplier\":{},\"passes\":{},\"source_index\":{},\"score\":{},\"quality_score\":{},\"text_distance\":{},\"target_number\":{},\"target_name\":\"{}\",\"target_score\":{},\"width\":{},\"height\":{}}}",
            SCHEMA,
            json_escape(prompt),
            json_escape(seed),
            candidate_multiplier,
            passes,
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
    signature: [u16; SIGNATURE_BINS],
    score: i64,
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
    let max_channels = HIDDEN_KERNELS.len() + 2;
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
        targets.push(TextTarget {
            number: fields[0]
                .parse()
                .map_err(|error: std::num::ParseIntError| error.to_string())?,
            name: fields[1].to_string(),
            aliases: fields[2].to_string(),
            row_text: fields[8].to_string(),
            signature: parse_text_signature(fields[7], line_index + 1)?,
            score: 0,
        });
    }
    targets.sort_by(|left, right| left.number.cmp(&right.number));
    if targets.is_empty() {
        return Err("Solomon text index has no target rows".to_string());
    }
    Ok(targets)
}

fn parse_text_signature(text: &str, line_number: usize) -> Result<[u16; SIGNATURE_BINS], String> {
    let mut signature = [0_u16; SIGNATURE_BINS];
    let parts: Vec<&str> = text.split(',').collect();
    if parts.len() != SIGNATURE_BINS {
        return Err(format!(
            "Solomon text index line {line_number} has {} signature bins, expected {SIGNATURE_BINS}",
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

fn sample_candidate(
    model: &MultichannelModel,
    conditions: &[usize],
    source_index: usize,
    seed: &str,
    passes: usize,
    target_signature: &[u16; SIGNATURE_BINS],
) -> Result<Candidate, String> {
    let mut image = initial_seal_prior(model.image_size, source_index, seed)?;
    for _ in 0..passes {
        for &condition in conditions {
            image = apply_model_condition(model, condition, &image, target_signature)?;
        }
    }
    let quality = score_sample(&image, model.image_size)?;
    let signature = sample_signature(&image, model.image_size)?;
    let text_distance = text_signature_distance(&signature, target_signature);
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

fn initial_seal_prior(
    image_size: usize,
    sample_index: usize,
    seed: &str,
) -> Result<Vec<u8>, String> {
    let image_bytes = checked_image_bytes(image_size)?;
    let init_seed = hash_seed(&[seed, &sample_index.to_string(), "seal-prior"]);
    let mut rng = Rng::new(init_seed);
    let mut image = vec![0_u8; image_bytes];
    let center = i32::try_from(image_size / 2).map_err(|error| error.to_string())?;
    let jitter_x = i32::try_from(rng.rand_bounded(9)).map_err(|error| error.to_string())? - 4;
    let jitter_y = i32::try_from(rng.rand_bounded(9)).map_err(|error| error.to_string())? - 4;
    let cx = center + jitter_x;
    let cy = center + jitter_y;
    let base_radius = i32::try_from(image_size / 2).map_err(|error| error.to_string())? - 13;
    let outer =
        base_radius + i32::try_from(rng.rand_bounded(7)).map_err(|error| error.to_string())? - 3;
    draw_circle(&mut image, image_size, cx, cy, outer, 2, 210)?;
    draw_circle(
        &mut image,
        image_size,
        cx,
        cy,
        outer - 8 - i32::try_from(rng.rand_bounded(4)).map_err(|error| error.to_string())?,
        1,
        190,
    )?;

    let stroke_count =
        5 + usize::try_from(rng.rand_bounded(5)).map_err(|error| error.to_string())?;
    let span = outer - 8;
    for _ in 0..stroke_count {
        let x0 = cx + random_signed(&mut rng, span)?;
        let y0 = cy + random_signed(&mut rng, span)?;
        let x1 = cx + random_signed(&mut rng, span)?;
        let y1 = cy + random_signed(&mut rng, span)?;
        let thickness =
            1 + i32::try_from(rng.rand_bounded(2)).map_err(|error| error.to_string())?;
        draw_line(&mut image, image_size, x0, y0, x1, y1, thickness, 230)?;
    }

    for _ in 0..3 {
        let arm = 12 + i32::try_from(rng.rand_bounded(18)).map_err(|error| error.to_string())?;
        let x = cx + random_signed(&mut rng, span / 2)?;
        let y = cy + random_signed(&mut rng, span / 2)?;
        draw_line(&mut image, image_size, x - arm, y, x + arm, y, 1, 240)?;
        draw_line(&mut image, image_size, x, y - arm, x, y + arm, 1, 240)?;
    }

    for value in &mut image {
        if *value > 0 && rng.chance(1, 18) {
            *value = value.saturating_sub(80);
        } else if *value == 0 && rng.chance(1, 96) {
            *value = 160;
        }
    }
    Ok(image)
}

fn apply_model_condition(
    model: &MultichannelModel,
    condition: usize,
    input: &[u8],
    target_signature: &[u16; SIGNATURE_BINS],
) -> Result<Vec<u8>, String> {
    let mut image = input.to_vec();
    for layer in &model.layers {
        image =
            apply_multichannel_layer_condition(model, layer, condition, &image, target_signature)?;
    }
    Ok(image)
}

fn apply_multichannel_layer_condition(
    model: &MultichannelModel,
    layer: &MultichannelLayer,
    condition: usize,
    input: &[u8],
    target_signature: &[u16; SIGNATURE_BINS],
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
                model.hidden_shift,
                target_signature,
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

fn conditioned_features(
    input: &[u8],
    image_size: usize,
    x: usize,
    y: usize,
    hidden_shift: u8,
    target_signature: &[u16; SIGNATURE_BINS],
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
    if out.len() <= HIDDEN_CHANNELS {
        return;
    }
    let bin_x = x * SIGNATURE_GRID / image_size;
    let bin_y = y * SIGNATURE_GRID / image_size;
    let target = i16::try_from(target_signature[bin_y * SIGNATURE_GRID + bin_x]).unwrap_or(0);
    let input_center = i16::from(input[pixel_index(image_size, x, y)]);
    out[HIDDEN_CHANNELS] = target.saturating_sub(input_center).clamp(-511, 511);
    if out.len() > HIDDEN_CHANNELS + 1 {
        out[HIDDEN_CHANNELS + 1] = target.saturating_sub(64).clamp(-511, 511);
    }
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
                    ring_buckets[angle_bucket(dx, dy)] =
                        ring_buckets[angle_bucket(dx, dy)].saturating_add(value);
                }
                if radius2 >= inner_low2 && radius2 <= inner_high2 {
                    inner_ring_sum = inner_ring_sum.saturating_add(value);
                    ring_buckets[angle_bucket(dx, dy)] =
                        ring_buckets[angle_bucket(dx, dy)].saturating_add(value);
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

fn draw_circle(
    image: &mut [u8],
    image_size: usize,
    cx: i32,
    cy: i32,
    radius: i32,
    thickness: i32,
    value: u8,
) -> Result<(), String> {
    for offset in 0..thickness {
        let r = radius.saturating_sub(offset);
        if r <= 0 {
            continue;
        }
        let mut x = r;
        let mut y = 0_i32;
        let mut error = 1_i32 - x;
        while x >= y {
            plot_circle_points(image, image_size, cx, cy, x, y, value)?;
            y += 1;
            if error < 0 {
                error += 2 * y + 1;
            } else {
                x -= 1;
                error += 2 * (y - x) + 1;
            }
        }
    }
    Ok(())
}

fn plot_circle_points(
    image: &mut [u8],
    image_size: usize,
    cx: i32,
    cy: i32,
    x: i32,
    y: i32,
    value: u8,
) -> Result<(), String> {
    for (px, py) in [
        (cx + x, cy + y),
        (cx + y, cy + x),
        (cx - y, cy + x),
        (cx - x, cy + y),
        (cx - x, cy - y),
        (cx - y, cy - x),
        (cx + y, cy - x),
        (cx + x, cy - y),
    ] {
        draw_point(image, image_size, px, py, 1, value)?;
    }
    Ok(())
}

fn draw_line(
    image: &mut [u8],
    image_size: usize,
    mut x0: i32,
    mut y0: i32,
    x1: i32,
    y1: i32,
    thickness: i32,
    value: u8,
) -> Result<(), String> {
    let dx = abs_i32(x1 - x0);
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -abs_i32(y1 - y0);
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut error = dx + dy;
    loop {
        draw_point(image, image_size, x0, y0, thickness, value)?;
        if x0 == x1 && y0 == y1 {
            break;
        }
        let doubled = error.saturating_mul(2);
        if doubled >= dy {
            error += dy;
            x0 += sx;
        }
        if doubled <= dx {
            error += dx;
            y0 += sy;
        }
    }
    Ok(())
}

fn draw_point(
    image: &mut [u8],
    image_size: usize,
    x: i32,
    y: i32,
    thickness: i32,
    value: u8,
) -> Result<(), String> {
    let radius = thickness.saturating_sub(1);
    for yy in y - radius..=y + radius {
        for xx in x - radius..=x + radius {
            if xx < 0 || yy < 0 {
                continue;
            }
            let ux = usize::try_from(xx).map_err(|error| error.to_string())?;
            let uy = usize::try_from(yy).map_err(|error| error.to_string())?;
            if ux >= image_size || uy >= image_size {
                continue;
            }
            let index = pixel_index(image_size, ux, uy);
            image[index] = image[index].max(value);
        }
    }
    Ok(())
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
            if let (Some(nx), Some(ny)) = (nx, ny) {
                if nx < image_size && ny < image_size {
                    let value = image[pixel_index(image_size, nx, ny)];
                    if value > INK_THRESHOLD {
                        count = count.saturating_add(1);
                    }
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

fn random_signed(rng: &mut Rng, magnitude: i32) -> Result<i32, String> {
    let span = u32::try_from(magnitude.saturating_mul(2).saturating_add(1))
        .map_err(|error| error.to_string())?;
    Ok(i32::try_from(rng.rand_bounded(span)).map_err(|error| error.to_string())? - magnitude)
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

    fn chance(&mut self, numerator: u32, denominator: u32) -> bool {
        self.rand_bounded(denominator) < numerator
    }
}

fn abs_i32(value: i32) -> i32 {
    if value >= 0 {
        value
    } else {
        value.saturating_neg()
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
    if left >= right {
        left - right
    } else {
        right - left
    }
}

fn abs_diff_u16(left: u16, right: u16) -> u16 {
    if left >= right {
        left - right
    } else {
        right - left
    }
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
