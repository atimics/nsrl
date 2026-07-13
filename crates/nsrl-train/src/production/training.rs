use core::ops::Range;

use nsrl_core::{
    GatedMlpI16Params, GatedMlpWorkspace, LinearAttentionWorkspace, RmsNormBackwardWorkspace,
    SelfAttentionI16Params, base2_softmax_i32_q15, gated_activation_backward_i16_q15,
    gated_mlp_i16_q15_checked, hard_silu_derivative_q15, hard_silu_q15,
    linear_attention_i16_q15_checked, rms_norm_backward_i16_q15_checked, rms_norm_i16_q15_checked,
    round_shift_rhu_i64, saturate_i8, saturate_i16,
};
use nsrl_corpus::subword::{BOS_TOKEN_ID, EOS_TOKEN_ID};

use super::{
    FNV_OFFSET, FNV_PRIME, PRODUCTION_RMS_EPSILON, ProductionModelV1, TrainError, checked_product,
    fnv1a, linear_params, scales,
};

const OPTIMIZER_MAGIC: &[u8; 8] = b"NSRLPO2\n";
const OPTIMIZER_VERSION: u32 = 2;
const GROUP_NAMES: [&str; 13] = [
    "embeddings",
    "attention_rms",
    "mlp_rms",
    "final_rms",
    "q",
    "k",
    "v",
    "o",
    "up",
    "gate",
    "down",
    "output",
    "bias",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductionFullTrainConfig {
    pub context_tokens: usize,
    pub max_windows: usize,
    pub epochs: usize,
    pub matrix_learning_rate_shift: u8,
    pub q_learning_rate_shift: Option<u8>,
    pub k_learning_rate_shift: Option<u8>,
    pub v_learning_rate_shift: Option<u8>,
    pub o_learning_rate_shift: Option<u8>,
    pub up_learning_rate_shift: Option<u8>,
    pub gate_learning_rate_shift: Option<u8>,
    pub down_learning_rate_shift: Option<u8>,
    pub vector_learning_rate_shift: u8,
    pub embedding_learning_rate_shift: u8,
    pub output_learning_rate_shift: u8,
    pub output_backward_shift: Option<u8>,
    pub batch_windows: usize,
    pub max_optimizer_steps: usize,
    pub evaluation_windows: usize,
}

impl Default for ProductionFullTrainConfig {
    fn default() -> Self {
        Self {
            context_tokens: 4,
            max_windows: 8,
            epochs: 2,
            matrix_learning_rate_shift: 16,
            q_learning_rate_shift: None,
            k_learning_rate_shift: None,
            v_learning_rate_shift: None,
            o_learning_rate_shift: None,
            up_learning_rate_shift: None,
            gate_learning_rate_shift: None,
            down_learning_rate_shift: None,
            vector_learning_rate_shift: 10,
            embedding_learning_rate_shift: 4,
            output_learning_rate_shift: 24,
            output_backward_shift: None,
            batch_windows: 4,
            max_optimizer_steps: usize::MAX,
            evaluation_windows: usize::MAX,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionOptimizerStateV2 {
    pub tokenizer_hash: u64,
    pub token_stream_hash: u64,
    pub bound_model_hash: u64,
    pub step: u64,
    pub schedule_hash: u64,
    pub next_epoch: u64,
    pub next_window: u64,
    pub residuals: Vec<i64>,
}

impl ProductionOptimizerStateV2 {
    pub fn new(
        model: &ProductionModelV1,
        token_stream_hash: u64,
        config: ProductionFullTrainConfig,
    ) -> Self {
        Self {
            tokenizer_hash: model.tokenizer_hash,
            token_stream_hash,
            bound_model_hash: model.model_hash(),
            step: 0,
            schedule_hash: schedule_hash(config),
            next_epoch: 0,
            next_window: 0,
            residuals: vec![0; model.parameter_count()],
        }
    }

    pub fn validate_binding(
        &self,
        model: &ProductionModelV1,
        token_stream_hash: u64,
        config: ProductionFullTrainConfig,
    ) -> Result<(), TrainError> {
        if self.tokenizer_hash != model.tokenizer_hash
            || self.token_stream_hash != token_stream_hash
            || self.bound_model_hash != model.model_hash()
            || self.schedule_hash != schedule_hash(config)
            || self.residuals.len() != model.parameter_count()
        {
            return Err(TrainError::InvalidModel(
                "production optimizer binding mismatch",
            ));
        }
        Ok(())
    }

    pub fn state_hash(&self) -> u64 {
        let mut hash = FNV_OFFSET;
        let mut write = |bytes: &[u8]| {
            for &byte in bytes {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(FNV_PRIME);
            }
        };
        write(OPTIMIZER_MAGIC);
        write(&OPTIMIZER_VERSION.to_le_bytes());
        write(&self.tokenizer_hash.to_le_bytes());
        write(&self.token_stream_hash.to_le_bytes());
        write(&self.bound_model_hash.to_le_bytes());
        write(&self.step.to_le_bytes());
        write(&self.schedule_hash.to_le_bytes());
        write(&self.next_epoch.to_le_bytes());
        write(&self.next_window.to_le_bytes());
        write(&(self.residuals.len() as u64).to_le_bytes());
        for residual in &self.residuals {
            write(&residual.to_le_bytes());
        }
        hash
    }

    pub fn try_to_bytes(&self) -> Result<Vec<u8>, TrainError> {
        if self.tokenizer_hash == 0
            || self.token_stream_hash == 0
            || self.bound_model_hash == 0
            || self.residuals.is_empty()
        {
            return Err(TrainError::InvalidModel(
                "invalid production optimizer state",
            ));
        }
        let mut bytes = self.bytes_without_checksum();
        bytes.extend_from_slice(&fnv1a(&bytes).to_le_bytes());
        Ok(bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TrainError> {
        if bytes.len() < 84 || &bytes[..8] != OPTIMIZER_MAGIC {
            return Err(TrainError::InvalidModel(
                "bad production optimizer artifact",
            ));
        }
        let checksum_offset = bytes.len() - 8;
        let expected = u64::from_le_bytes(
            bytes[checksum_offset..]
                .try_into()
                .map_err(|_| TrainError::InvalidModel("bad production optimizer checksum"))?,
        );
        if fnv1a(&bytes[..checksum_offset]) != expected {
            return Err(TrainError::InvalidModel(
                "bad production optimizer checksum",
            ));
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if version != OPTIMIZER_VERSION {
            return Err(TrainError::InvalidModel(
                "unsupported production optimizer version",
            ));
        }
        let residual_count = usize::try_from(u64::from_le_bytes(bytes[68..76].try_into().unwrap()))
            .map_err(|_| TrainError::InvalidModel("production optimizer residual overflow"))?;
        if bytes.len() != 84_usize.saturating_add(residual_count.saturating_mul(8)) {
            return Err(TrainError::InvalidModel(
                "production optimizer residual length mismatch",
            ));
        }
        let residuals = bytes[76..checksum_offset]
            .chunks_exact(8)
            .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        Ok(Self {
            tokenizer_hash: u64::from_le_bytes(bytes[12..20].try_into().unwrap()),
            token_stream_hash: u64::from_le_bytes(bytes[20..28].try_into().unwrap()),
            bound_model_hash: u64::from_le_bytes(bytes[28..36].try_into().unwrap()),
            step: u64::from_le_bytes(bytes[36..44].try_into().unwrap()),
            schedule_hash: u64::from_le_bytes(bytes[44..52].try_into().unwrap()),
            next_epoch: u64::from_le_bytes(bytes[52..60].try_into().unwrap()),
            next_window: u64::from_le_bytes(bytes[60..68].try_into().unwrap()),
            residuals,
        })
    }

    fn bytes_without_checksum(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(84 + self.residuals.len() * 8);
        bytes.extend_from_slice(OPTIMIZER_MAGIC);
        bytes.extend_from_slice(&OPTIMIZER_VERSION.to_le_bytes());
        bytes.extend_from_slice(&self.tokenizer_hash.to_le_bytes());
        bytes.extend_from_slice(&self.token_stream_hash.to_le_bytes());
        bytes.extend_from_slice(&self.bound_model_hash.to_le_bytes());
        bytes.extend_from_slice(&self.step.to_le_bytes());
        bytes.extend_from_slice(&self.schedule_hash.to_le_bytes());
        bytes.extend_from_slice(&self.next_epoch.to_le_bytes());
        bytes.extend_from_slice(&self.next_window.to_le_bytes());
        bytes.extend_from_slice(&(self.residuals.len() as u64).to_le_bytes());
        for residual in &self.residuals {
            bytes.extend_from_slice(&residual.to_le_bytes());
        }
        bytes
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductionFullTrainTrace {
    pub profile: &'static str,
    pub parameter_count: usize,
    pub tokenizer_hash: u64,
    pub token_stream_hash: u64,
    pub context_tokens: usize,
    pub windows: usize,
    pub epochs: usize,
    pub initial_mistakes: usize,
    pub final_mistakes: usize,
    pub optimizer_steps: usize,
    pub total_optimizer_step: u64,
    pub batch_windows: usize,
    pub evaluation_windows: usize,
    pub start_epoch: u64,
    pub start_window: u64,
    pub next_epoch: u64,
    pub next_window: u64,
    pub schedule_complete: bool,
    pub learning_rate_shifts: [u8; 13],
    pub output_backward_shift: u8,
    pub gradient_saturation_count: usize,
    pub weight_saturation_count: usize,
    pub movement_l1: [u64; 13],
    pub gradient_nonzero_count: [u64; 13],
    pub residual_carry_count: [u64; 13],
    pub update_nonzero_count: [u64; 13],
    pub saturation_by_group: [u64; 13],
    pub backward_ste_rescue_count: u64,
    pub backward_quantization_count: u64,
    pub initial_model_hash: u64,
    pub final_model_hash: u64,
    pub optimizer_state_hash: u64,
}

impl ProductionFullTrainTrace {
    pub fn to_json_line(self) -> String {
        let movement = GROUP_NAMES
            .iter()
            .zip(self.movement_l1)
            .map(|(name, value)| format!("\"{name}\":{value}"))
            .collect::<Vec<_>>()
            .join(",");
        let moved = GROUP_NAMES
            .iter()
            .zip(self.movement_l1)
            .filter(|(_, value)| *value > 0)
            .map(|(name, _)| format!("\"{name}\""))
            .collect::<Vec<_>>()
            .join(",");
        let render_groups = |values: [u64; 13]| {
            GROUP_NAMES
                .iter()
                .zip(values)
                .map(|(name, value)| format!("\"{name}\":{value}"))
                .collect::<Vec<_>>()
                .join(",")
        };
        let learning_rate_shifts = GROUP_NAMES
            .iter()
            .zip(self.learning_rate_shifts)
            .map(|(name, value)| format!("\"{name}\":{value}"))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            concat!(
                "{{\"schema\":\"nsrl.production_full_train_smoke.v1\",",
                "\"profile\":\"{}\",\"parameter_count\":{},",
                "\"bindings\":{{\"tokenizer_hash\":\"0x{:016x}\",\"token_stream_hash\":\"0x{:016x}\"}},",
                "\"training\":{{\"optimizer\":\"integer_residual_sgd\",\"backward\":\"full_quantized_straight_through\",\"context_tokens\":{},\"windows\":{},\"evaluation_windows\":{},\"epochs\":{},\"batch_windows\":{},\"optimizer_steps\":{},\"total_optimizer_step\":{},\"initial_mistakes\":{},\"final_mistakes\":{},\"learning_rate_shifts\":{{{}}},\"output_backward_shift\":{}}},",
                "\"cursor\":{{\"start_epoch\":{},\"start_window\":{},\"next_epoch\":{},\"next_window\":{},\"schedule_complete\":{}}},",
                "\"movement_l1\":{{{}}},\"moved_parameter_groups\":[{}],",
                "\"diagnostics\":{{\"gradient_nonzero_count\":{{{}}},\"residual_carry_count\":{{{}}},\"update_nonzero_count\":{{{}}},\"saturation_by_group\":{{{}}},\"backward_ste_rescue_count\":{},\"backward_quantization_count\":{},\"backward_ste_rescue_per_million\":{}}},",
                "\"health\":{{\"gradient_saturation_count\":{},\"weight_saturation_count\":{}}},",
                "\"hashes\":{{\"initial_model\":\"0x{:016x}\",\"final_model\":\"0x{:016x}\",\"optimizer_state\":\"0x{:016x}\"}},",
                "\"gates\":{{\"all_parameter_groups_moved\":{},\"model_hash_changed\":{},\"resumable_optimizer_state\":true,\"batched_residual_updates\":true}},",
                "\"known_non_claims\":[\"bounded_full_backward_smoke_not_scaling_run\",\"straight_through_backward_derivatives\",\"not_open_generation_quality\"]}}\n"
            ),
            self.profile,
            self.parameter_count,
            self.tokenizer_hash,
            self.token_stream_hash,
            self.context_tokens,
            self.windows,
            self.evaluation_windows,
            self.epochs,
            self.batch_windows,
            self.optimizer_steps,
            self.total_optimizer_step,
            self.initial_mistakes,
            self.final_mistakes,
            learning_rate_shifts,
            self.output_backward_shift,
            self.start_epoch,
            self.start_window,
            self.next_epoch,
            self.next_window,
            self.schedule_complete,
            movement,
            moved,
            render_groups(self.gradient_nonzero_count),
            render_groups(self.residual_carry_count),
            render_groups(self.update_nonzero_count),
            render_groups(self.saturation_by_group),
            self.backward_ste_rescue_count,
            self.backward_quantization_count,
            self.backward_ste_rescue_count.saturating_mul(1_000_000)
                / self.backward_quantization_count.max(1),
            self.gradient_saturation_count,
            self.weight_saturation_count,
            self.initial_model_hash,
            self.final_model_hash,
            self.optimizer_state_hash,
            self.movement_l1.iter().all(|&value| value > 0),
            self.initial_model_hash != self.final_model_hash,
        )
    }
}

#[derive(Clone)]
struct LayerCache {
    input: Vec<i16>,
    attention_input: Vec<i16>,
    q: Vec<i16>,
    k: Vec<i16>,
    v: Vec<i16>,
    context: Vec<i16>,
    attention_residual: Vec<i16>,
    mlp_input: Vec<i16>,
    up: Vec<i16>,
    gate: Vec<i16>,
    gated: Vec<i16>,
}

struct ForwardCache {
    layers: Vec<LayerCache>,
    final_hidden: Vec<i16>,
    features: Vec<i16>,
    probabilities: Vec<i16>,
}

#[derive(Default)]
struct UpdateStats {
    movement: [u64; 13],
    gradient_nonzero: [u64; 13],
    residual_carry: [u64; 13],
    update_nonzero: [u64; 13],
    saturation_by_group: [u64; 13],
    backward_ste_rescue: u64,
    backward_quantization: u64,
    gradient_saturation: usize,
    weight_saturation: usize,
}

struct ParameterRanges {
    embeddings: Range<usize>,
    attention_rms: Range<usize>,
    mlp_rms: Range<usize>,
    final_rms: Range<usize>,
    q: Range<usize>,
    k: Range<usize>,
    v: Range<usize>,
    o: Range<usize>,
    up: Range<usize>,
    gate: Range<usize>,
    down: Range<usize>,
    output: Range<usize>,
    bias: Range<usize>,
}

impl ParameterRanges {
    fn new(model: &ProductionModelV1) -> Self {
        let mut cursor = 0;
        let mut take = |len: usize| {
            let range = cursor..cursor + len;
            cursor += len;
            range
        };
        let ranges = Self {
            embeddings: take(model.embeddings.len()),
            attention_rms: take(model.attention_rms_weights.len()),
            mlp_rms: take(model.mlp_rms_weights.len()),
            final_rms: take(model.final_rms_weights.len()),
            q: take(model.q_weights.len()),
            k: take(model.k_weights.len()),
            v: take(model.v_weights.len()),
            o: take(model.o_weights.len()),
            up: take(model.up_weights.len()),
            gate: take(model.gate_weights.len()),
            down: take(model.down_weights.len()),
            output: take(model.output_weights.len()),
            bias: take(model.output_bias_q8.len()),
        };
        debug_assert_eq!(cursor, model.parameter_count());
        ranges
    }
}

pub fn train_production_full_smoke(
    model: &mut ProductionModelV1,
    tokens: &[u32],
    token_stream_hash: u64,
    config: ProductionFullTrainConfig,
    state: Option<ProductionOptimizerStateV2>,
) -> Result<(ProductionFullTrainTrace, ProductionOptimizerStateV2), TrainError> {
    model.validate()?;
    if config.context_tokens == 0
        || config.context_tokens > model.config.context_tokens
        || config.max_windows == 0
        || config.epochs == 0
        || config.batch_windows == 0
        || config.max_optimizer_steps == 0
        || config.evaluation_windows == 0
        || config.output_backward_shift.is_some_and(|shift| shift > 30)
        || effective_learning_rate_shifts(config)
            .iter()
            .any(|&shift| shift > 62)
    {
        return Err(TrainError::InvalidConfig);
    }
    let windows = document_windows(tokens, config.context_tokens, config.max_windows);
    if windows.is_empty() {
        return Err(TrainError::InvalidConfig);
    }
    let initial_model_hash = model.model_hash();
    let evaluation_windows = config.evaluation_windows.min(windows.len());
    let initial_mistakes = evaluate_mistakes(model, &windows[..evaluation_windows])?;
    let mut state =
        state.unwrap_or_else(|| ProductionOptimizerStateV2::new(model, token_stream_hash, config));
    state.validate_binding(model, token_stream_hash, config)?;
    if state.next_epoch > config.epochs as u64
        || state.next_window > windows.len() as u64
        || (state.next_epoch == config.epochs as u64 && state.next_window != 0)
    {
        return Err(TrainError::InvalidModel(
            "production optimizer cursor mismatch",
        ));
    }
    let start_epoch = state.next_epoch;
    let start_window = state.next_window;
    let ranges = ParameterRanges::new(model);
    let mut stats = UpdateStats::default();
    let mut optimizer_steps = 0_usize;
    while state.next_epoch < config.epochs as u64 && optimizer_steps < config.max_optimizer_steps {
        let batch_start = state.next_window as usize;
        let batch_end = batch_start
            .saturating_add(config.batch_windows)
            .min(windows.len());
        for (index, (context, target)) in windows[batch_start..batch_end].iter().enumerate() {
            let cache = forward_cache(model, context)?;
            backward_update(
                model,
                context,
                *target as usize,
                cache,
                config,
                &ranges,
                &mut state.residuals,
                index + 1 == batch_end - batch_start,
                &mut stats,
            )?;
        }
        state.step = state.step.checked_add(1).ok_or(TrainError::InvalidConfig)?;
        optimizer_steps = optimizer_steps.saturating_add(1);
        if batch_end == windows.len() {
            state.next_epoch = state.next_epoch.saturating_add(1);
            state.next_window = 0;
        } else {
            state.next_window = batch_end as u64;
        }
    }
    state.bound_model_hash = model.model_hash();
    let final_mistakes = evaluate_mistakes(model, &windows[..evaluation_windows])?;
    let trace = ProductionFullTrainTrace {
        profile: model.config.profile_id().unwrap_or("custom"),
        parameter_count: model.parameter_count(),
        tokenizer_hash: model.tokenizer_hash,
        token_stream_hash,
        context_tokens: config.context_tokens,
        windows: windows.len(),
        epochs: config.epochs,
        initial_mistakes,
        final_mistakes,
        optimizer_steps,
        total_optimizer_step: state.step,
        batch_windows: config.batch_windows,
        evaluation_windows,
        start_epoch,
        start_window,
        next_epoch: state.next_epoch,
        next_window: state.next_window,
        schedule_complete: state.next_epoch == config.epochs as u64,
        learning_rate_shifts: effective_learning_rate_shifts(config),
        output_backward_shift: config
            .output_backward_shift
            .unwrap_or(model.scales.output_shift),
        gradient_saturation_count: stats.gradient_saturation,
        weight_saturation_count: stats.weight_saturation,
        movement_l1: stats.movement,
        gradient_nonzero_count: stats.gradient_nonzero,
        residual_carry_count: stats.residual_carry,
        update_nonzero_count: stats.update_nonzero,
        saturation_by_group: stats.saturation_by_group,
        backward_ste_rescue_count: stats.backward_ste_rescue,
        backward_quantization_count: stats.backward_quantization,
        initial_model_hash,
        final_model_hash: model.model_hash(),
        optimizer_state_hash: state.state_hash(),
    };
    Ok((trace, state))
}

fn forward_cache(
    model: &ProductionModelV1,
    context_tokens: &[u32],
) -> Result<ForwardCache, TrainError> {
    let config = model.config;
    if context_tokens.is_empty()
        || context_tokens.len() > config.context_tokens
        || context_tokens
            .iter()
            .any(|&token| token as usize >= config.vocab_size)
    {
        return Err(TrainError::InvalidConfig);
    }
    let mut hidden = Vec::with_capacity(context_tokens.len() * config.d_model);
    for &token in context_tokens {
        let start = token as usize * config.d_model;
        hidden.extend_from_slice(&model.embeddings[start..start + config.d_model]);
    }
    let seq_len = context_tokens.len();
    let total = checked_product(seq_len, config.d_model)?;
    let matrix = checked_product(config.d_model, config.d_model)?;
    let up_matrix = checked_product(config.d_model, config.hidden_dim)?;
    let down_matrix = checked_product(config.hidden_dim, config.d_model)?;
    let qkv_scales = scales(config.d_model, model.scales.qkv_shift);
    let o_scales = scales(config.d_model, model.scales.o_shift);
    let up_scales = scales(config.hidden_dim, model.scales.up_shift);
    let gate_scales = scales(config.hidden_dim, model.scales.gate_shift);
    let down_scales = scales(config.d_model, model.scales.down_shift);
    let mut layers = Vec::with_capacity(config.layers);
    for layer in 0..config.layers {
        let input = hidden.clone();
        let rms = layer * config.d_model..(layer + 1) * config.d_model;
        let attention_input = rms_rows(
            &hidden,
            &model.attention_rms_weights[rms.clone()],
            config.d_model,
        )?;
        let range = layer * matrix..(layer + 1) * matrix;
        let params = SelfAttentionI16Params {
            q: linear_params(
                &model.q_weights[range.clone()],
                &qkv_scales,
                config.d_model,
                config.d_model,
            ),
            k: linear_params(
                &model.k_weights[range.clone()],
                &qkv_scales,
                config.d_model,
                config.d_model,
            ),
            v: linear_params(
                &model.v_weights[range.clone()],
                &qkv_scales,
                config.d_model,
                config.d_model,
            ),
            o: linear_params(
                &model.o_weights[range],
                &o_scales,
                config.d_model,
                config.d_model,
            ),
            seq_len,
            d_model: config.d_model,
            heads: config.heads,
            causal: true,
        };
        let head_dim = config.d_model / config.heads;
        let mut q = vec![0_i16; total];
        let mut k = vec![0_i16; total];
        let mut v = vec![0_i16; total];
        let mut attention_context = vec![0_i16; total];
        let mut state_kv = vec![0_i64; config.heads * head_dim * head_dim];
        let mut key_sums = vec![0_i64; config.heads * head_dim];
        let mut attention_output = vec![0_i16; total];
        linear_attention_i16_q15_checked(
            &attention_input,
            params,
            LinearAttentionWorkspace {
                q: &mut q,
                k: &mut k,
                v: &mut v,
                context: &mut attention_context,
                state_kv: &mut state_kv,
                key_sums: &mut key_sums,
            },
            &mut attention_output,
        )
        .ok_or(TrainError::CoreRejected("production_training_attention"))?;
        let attention_residual = add_rows(&hidden, &attention_output);
        let mlp_input = rms_rows(
            &attention_residual,
            &model.mlp_rms_weights[rms],
            config.d_model,
        )?;
        let up_range = layer * up_matrix..(layer + 1) * up_matrix;
        let down_range = layer * down_matrix..(layer + 1) * down_matrix;
        let mlp_params = GatedMlpI16Params {
            up: linear_params(
                &model.up_weights[up_range.clone()],
                &up_scales,
                config.d_model,
                config.hidden_dim,
            ),
            gate: linear_params(
                &model.gate_weights[up_range],
                &gate_scales,
                config.d_model,
                config.hidden_dim,
            ),
            down: linear_params(
                &model.down_weights[down_range],
                &down_scales,
                config.hidden_dim,
                config.d_model,
            ),
            seq_len,
            d_model: config.d_model,
            hidden_dim: config.hidden_dim,
        };
        let mut up = vec![0_i16; seq_len * config.hidden_dim];
        let mut gate = vec![0_i16; seq_len * config.hidden_dim];
        let mut gated = vec![0_i16; seq_len * config.hidden_dim];
        let mut mlp_output = vec![0_i16; total];
        gated_mlp_i16_q15_checked(
            &mlp_input,
            mlp_params,
            GatedMlpWorkspace {
                up: &mut up,
                gate: &mut gate,
                gated: &mut gated,
            },
            &mut mlp_output,
        )
        .ok_or(TrainError::CoreRejected("production_training_mlp"))?;
        hidden = add_rows(&attention_residual, &mlp_output);
        layers.push(LayerCache {
            input,
            attention_input,
            q,
            k,
            v,
            context: attention_context,
            attention_residual,
            mlp_input,
            up,
            gate,
            gated,
        });
    }
    let final_hidden = hidden;
    let last = (seq_len - 1) * config.d_model;
    let mut features = vec![0_i16; config.d_model];
    rms_norm_i16_q15_checked(
        &final_hidden[last..last + config.d_model],
        &model.final_rms_weights,
        PRODUCTION_RMS_EPSILON,
        &mut features,
    )
    .ok_or(TrainError::CoreRejected("production_training_final_rms"))?;
    let logits = output_logits(model, &features);
    let mut probabilities = vec![0_i16; config.vocab_size];
    base2_softmax_i32_q15(&logits, &mut probabilities)
        .ok_or(TrainError::CoreRejected("production_training_softmax"))?;
    Ok(ForwardCache {
        layers,
        final_hidden,
        features,
        probabilities,
    })
}

#[allow(clippy::too_many_arguments)]
fn backward_update(
    model: &mut ProductionModelV1,
    context_tokens: &[u32],
    target: usize,
    cache: ForwardCache,
    config: ProductionFullTrainConfig,
    ranges: &ParameterRanges,
    residuals: &mut [i64],
    apply_updates: bool,
    stats: &mut UpdateStats,
) -> Result<(), TrainError> {
    let c = model.config;
    let seq_len = context_tokens.len();
    let total = seq_len * c.d_model;
    let mut grad_logits = cache.probabilities;
    grad_logits[target] = grad_logits[target].saturating_sub(i16::MAX);
    let mut grad_features = vec![0_i16; c.d_model];
    for (dim, grad_feature) in grad_features.iter_mut().enumerate() {
        let mut acc = 0_i64;
        for (token, &grad) in grad_logits.iter().enumerate() {
            acc = acc.saturating_add(
                i64::from(grad) * i64::from(model.output_weights[token * c.d_model + dim]),
            );
        }
        *grad_feature = saturate_i16(quantized_nonzero(
            acc,
            config
                .output_backward_shift
                .unwrap_or(model.scales.output_shift),
            stats,
        ));
    }
    for (token, &grad) in grad_logits.iter().enumerate() {
        update_i32(
            &mut model.output_bias_q8[token],
            i64::from(grad),
            config.vector_learning_rate_shift,
            &mut residuals[ranges.bias.start + token],
            apply_updates,
            12,
            stats,
        );
        for (dim, &feature) in cache.features.iter().enumerate() {
            update_i16(
                &mut model.output_weights[token * c.d_model + dim],
                i64::from(grad) * i64::from(feature),
                config.output_learning_rate_shift,
                &mut residuals[ranges.output.start + token * c.d_model + dim],
                apply_updates,
                11,
                stats,
            );
        }
    }
    let final_start = total - c.d_model;
    let mut grad_hidden = vec![0_i16; total];
    let mut final_gamma_grad = vec![0_i64; c.d_model];
    rms_backward_row(
        &cache.final_hidden[final_start..],
        &model.final_rms_weights,
        &grad_features,
        &mut grad_hidden[final_start..],
        &mut final_gamma_grad,
        stats,
    )?;
    update_i16_slice(
        &mut model.final_rms_weights,
        &final_gamma_grad,
        vector_group_shift(config.vector_learning_rate_shift, 3),
        &mut residuals[ranges.final_rms.clone()],
        apply_updates,
        3,
        stats,
    );

    let matrix = c.d_model * c.d_model;
    let up_matrix = c.d_model * c.hidden_dim;
    let down_matrix = c.hidden_dim * c.d_model;
    for layer in (0..c.layers).rev() {
        let item = &cache.layers[layer];
        let mut grad_attention_residual = grad_hidden.clone();
        let grad_mlp_output = grad_hidden;
        let down_range = layer * down_matrix..(layer + 1) * down_matrix;
        let down_residual_range =
            ranges.down.start + down_range.start..ranges.down.start + down_range.end;
        let mut grad_gated = linear_backward_input(
            &grad_mlp_output,
            &model.down_weights[down_range.clone()],
            c.hidden_dim,
            c.d_model,
            model.scales.down_shift,
            stats,
        );
        let mut grad_up = vec![0_i16; seq_len * c.hidden_dim];
        let mut grad_gate = vec![0_i16; seq_len * c.hidden_dim];
        for index in 0..grad_gated.len() {
            stats.backward_quantization = stats.backward_quantization.saturating_add(2);
            let grad = gated_activation_backward_i16_q15(
                item.up[index],
                item.gate[index],
                grad_gated[index],
            );
            let up_numerator =
                i64::from(grad_gated[index]) * i64::from(hard_silu_q15(item.gate[index]));
            grad_up[index] = if grad.up == 0 && up_numerator != 0 {
                stats.backward_ste_rescue = stats.backward_ste_rescue.saturating_add(1);
                saturate_i16(up_numerator.signum())
            } else {
                grad.up
            };
            let gate_numerator = i64::from(grad_gated[index])
                * i64::from(item.up[index])
                * i64::from(hard_silu_derivative_q15(item.gate[index]));
            grad_gate[index] = if grad.gate == 0 && gate_numerator != 0 {
                stats.backward_ste_rescue = stats.backward_ste_rescue.saturating_add(1);
                saturate_i16(gate_numerator.signum())
            } else {
                grad.gate
            };
        }
        let up_range = layer * up_matrix..(layer + 1) * up_matrix;
        let up_residual_range = ranges.up.start + up_range.start..ranges.up.start + up_range.end;
        let gate_residual_range =
            ranges.gate.start + up_range.start..ranges.gate.start + up_range.end;
        let up_input = linear_backward_input(
            &grad_up,
            &model.up_weights[up_range.clone()],
            c.d_model,
            c.hidden_dim,
            model.scales.up_shift,
            stats,
        );
        let gate_input = linear_backward_input(
            &grad_gate,
            &model.gate_weights[up_range.clone()],
            c.d_model,
            c.hidden_dim,
            model.scales.gate_shift,
            stats,
        );
        let grad_mlp_input = add_rows(&up_input, &gate_input);
        let rms_range = layer * c.d_model..(layer + 1) * c.d_model;
        let mlp_rms_residual_range =
            ranges.mlp_rms.start + rms_range.start..ranges.mlp_rms.start + rms_range.end;
        let attention_rms_residual_range = ranges.attention_rms.start + rms_range.start
            ..ranges.attention_rms.start + rms_range.end;
        let mut grad_mlp_residual = vec![0_i16; total];
        let mut mlp_gamma_grad = vec![0_i64; c.d_model];
        rms_backward_rows(
            &item.attention_residual,
            &model.mlp_rms_weights[rms_range.clone()],
            &grad_mlp_input,
            c.d_model,
            &mut grad_mlp_residual,
            &mut mlp_gamma_grad,
            stats,
        )?;
        add_rows_in_place(&mut grad_attention_residual, &grad_mlp_residual, stats);

        update_i8_matrix_rows(
            &item.gated,
            &grad_mlp_output,
            &mut model.down_weights[down_range],
            c.hidden_dim,
            c.d_model,
            matrix_learning_rate_shift(config, 10),
            &mut residuals[down_residual_range],
            apply_updates,
            10,
            stats,
        );
        update_i8_matrix_rows(
            &item.mlp_input,
            &grad_up,
            &mut model.up_weights[up_range.clone()],
            c.d_model,
            c.hidden_dim,
            matrix_learning_rate_shift(config, 8),
            &mut residuals[up_residual_range],
            apply_updates,
            8,
            stats,
        );
        update_i8_matrix_rows(
            &item.mlp_input,
            &grad_gate,
            &mut model.gate_weights[up_range],
            c.d_model,
            c.hidden_dim,
            matrix_learning_rate_shift(config, 9),
            &mut residuals[gate_residual_range],
            apply_updates,
            9,
            stats,
        );
        update_i16_slice(
            &mut model.mlp_rms_weights[rms_range.clone()],
            &mlp_gamma_grad,
            vector_group_shift(config.vector_learning_rate_shift, 2),
            &mut residuals[mlp_rms_residual_range],
            apply_updates,
            2,
            stats,
        );

        let attention_range = layer * matrix..(layer + 1) * matrix;
        let q_residual_range =
            ranges.q.start + attention_range.start..ranges.q.start + attention_range.end;
        let k_residual_range =
            ranges.k.start + attention_range.start..ranges.k.start + attention_range.end;
        let v_residual_range =
            ranges.v.start + attention_range.start..ranges.v.start + attention_range.end;
        let o_residual_range =
            ranges.o.start + attention_range.start..ranges.o.start + attention_range.end;
        let grad_context = linear_backward_input(
            &grad_attention_residual,
            &model.o_weights[attention_range.clone()],
            c.d_model,
            c.d_model,
            model.scales.o_shift,
            stats,
        );
        let grad_context = scale_gradient(&grad_context, 8, stats);
        let (grad_q, grad_k, grad_v) = linear_attention_backward(
            c.d_model,
            c.heads,
            &item.q,
            &item.k,
            &item.v,
            &item.context,
            &grad_context,
            stats,
        )?;
        let q_input = linear_backward_input(
            &grad_q,
            &model.q_weights[attention_range.clone()],
            c.d_model,
            c.d_model,
            model.scales.qkv_shift,
            stats,
        );
        let k_input = linear_backward_input(
            &grad_k,
            &model.k_weights[attention_range.clone()],
            c.d_model,
            c.d_model,
            model.scales.qkv_shift,
            stats,
        );
        let v_input = linear_backward_input(
            &grad_v,
            &model.v_weights[attention_range.clone()],
            c.d_model,
            c.d_model,
            model.scales.qkv_shift,
            stats,
        );
        let grad_attention_input = add_rows(&add_rows(&q_input, &k_input), &v_input);
        let mut grad_input_norm = vec![0_i16; total];
        let mut attention_gamma_grad = vec![0_i64; c.d_model];
        rms_backward_rows(
            &item.input,
            &model.attention_rms_weights[rms_range.clone()],
            &grad_attention_input,
            c.d_model,
            &mut grad_input_norm,
            &mut attention_gamma_grad,
            stats,
        )?;
        grad_hidden = grad_attention_residual.clone();
        add_rows_in_place(&mut grad_hidden, &grad_input_norm, stats);

        update_i8_matrix_rows(
            &item.context,
            &grad_attention_residual,
            &mut model.o_weights[attention_range.clone()],
            c.d_model,
            c.d_model,
            matrix_learning_rate_shift(config, 7),
            &mut residuals[o_residual_range],
            apply_updates,
            7,
            stats,
        );
        update_i8_matrix_rows(
            &item.attention_input,
            &grad_q,
            &mut model.q_weights[attention_range.clone()],
            c.d_model,
            c.d_model,
            matrix_learning_rate_shift(config, 4),
            &mut residuals[q_residual_range],
            apply_updates,
            4,
            stats,
        );
        update_i8_matrix_rows(
            &item.attention_input,
            &grad_k,
            &mut model.k_weights[attention_range.clone()],
            c.d_model,
            c.d_model,
            matrix_learning_rate_shift(config, 5),
            &mut residuals[k_residual_range],
            apply_updates,
            5,
            stats,
        );
        update_i8_matrix_rows(
            &item.attention_input,
            &grad_v,
            &mut model.v_weights[attention_range],
            c.d_model,
            c.d_model,
            matrix_learning_rate_shift(config, 6),
            &mut residuals[v_residual_range],
            apply_updates,
            6,
            stats,
        );
        update_i16_slice(
            &mut model.attention_rms_weights[rms_range],
            &attention_gamma_grad,
            vector_group_shift(config.vector_learning_rate_shift, 1),
            &mut residuals[attention_rms_residual_range],
            apply_updates,
            1,
            stats,
        );
        grad_gated.clear();
    }
    for (row, &token) in context_tokens.iter().enumerate() {
        let embedding =
            &mut model.embeddings[token as usize * c.d_model..(token as usize + 1) * c.d_model];
        for dim in 0..c.d_model {
            update_i16(
                &mut embedding[dim],
                i64::from(grad_hidden[row * c.d_model + dim]),
                config.embedding_learning_rate_shift,
                &mut residuals[ranges.embeddings.start + token as usize * c.d_model + dim],
                apply_updates,
                0,
                stats,
            );
        }
    }
    Ok(())
}

fn linear_backward_input(
    grad: &[i16],
    weights: &[i8],
    input_dim: usize,
    output_dim: usize,
    shift: u8,
    stats: &mut UpdateStats,
) -> Vec<i16> {
    let rows = grad.len() / output_dim;
    let mut result = vec![0_i16; rows * input_dim];
    for row in 0..rows {
        for input in 0..input_dim {
            let mut acc = 0_i64;
            for output in 0..output_dim {
                acc = acc.saturating_add(
                    i64::from(grad[row * output_dim + output])
                        * i64::from(weights[output * input_dim + input]),
                );
            }
            let wide = quantized_nonzero(acc, shift, stats);
            result[row * input_dim + input] = saturate_i16(wide);
            stats.gradient_saturation +=
                usize::from(wide < i64::from(i16::MIN) || wide > i64::from(i16::MAX));
        }
    }
    result
}

fn scale_gradient(values: &[i16], left_shift: u8, stats: &mut UpdateStats) -> Vec<i16> {
    values
        .iter()
        .map(|&value| {
            let wide = i64::from(value) << left_shift;
            stats.gradient_saturation = stats.gradient_saturation.saturating_add(usize::from(
                wide < i64::from(i16::MIN) || wide > i64::from(i16::MAX),
            ));
            saturate_i16(wide)
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn update_i8_matrix_rows(
    input: &[i16],
    grad: &[i16],
    weights: &mut [i8],
    input_dim: usize,
    output_dim: usize,
    shift: u8,
    residuals: &mut [i64],
    apply_updates: bool,
    group: usize,
    stats: &mut UpdateStats,
) {
    let rows = input.len() / input_dim;
    for output in 0..output_dim {
        for input_index in 0..input_dim {
            let mut gradient = 0_i64;
            for row in 0..rows {
                gradient = gradient.saturating_add(
                    i64::from(grad[row * output_dim + output])
                        * i64::from(input[row * input_dim + input_index]),
                );
            }
            let index = output * input_dim + input_index;
            stats.gradient_nonzero[group] =
                stats.gradient_nonzero[group].saturating_add(u64::from(gradient != 0));
            residuals[index] = residuals[index].saturating_add(gradient);
            if !apply_updates {
                continue;
            }
            let update = round_shift_rhu_i64(residuals[index], shift);
            stats.residual_carry[group] = stats.residual_carry[group]
                .saturating_add(u64::from(update == 0 && residuals[index] != 0));
            let previous = weights[index];
            let wide = i64::from(previous).saturating_sub(update);
            let next = saturate_i8(wide);
            let saturated = wide < i64::from(i8::MIN) || wide > i64::from(i8::MAX);
            stats.weight_saturation += usize::from(saturated);
            stats.saturation_by_group[group] =
                stats.saturation_by_group[group].saturating_add(u64::from(saturated));
            stats.update_nonzero[group] =
                stats.update_nonzero[group].saturating_add(u64::from(next != previous));
            stats.movement[group] = stats.movement[group]
                .saturating_add((i64::from(next) - i64::from(previous)).unsigned_abs());
            weights[index] = next;
            consume_residual(&mut residuals[index], update, shift);
        }
    }
}

fn rms_backward_rows(
    input: &[i16],
    weights: &[i16],
    grad: &[i16],
    d_model: usize,
    output: &mut [i16],
    gamma_grad: &mut [i64],
    stats: &mut UpdateStats,
) -> Result<(), TrainError> {
    for row in 0..input.len() / d_model {
        let range = row * d_model..(row + 1) * d_model;
        rms_backward_row(
            &input[range.clone()],
            weights,
            &grad[range.clone()],
            &mut output[range],
            gamma_grad,
            stats,
        )?;
    }
    Ok(())
}

fn rms_backward_row(
    input: &[i16],
    weights: &[i16],
    grad: &[i16],
    output: &mut [i16],
    gamma_grad: &mut [i64],
    stats: &mut UpdateStats,
) -> Result<(), TrainError> {
    let mut normalized = vec![0_i32; input.len()];
    let mut scaled = vec![0_i32; input.len()];
    stats.gradient_saturation = stats.gradient_saturation.saturating_add(
        rms_norm_backward_i16_q15_checked(
            input,
            weights,
            grad,
            PRODUCTION_RMS_EPSILON,
            RmsNormBackwardWorkspace {
                normalized_q15: &mut normalized,
                scaled_grad_q15: &mut scaled,
            },
            output,
            gamma_grad,
        )
        .ok_or(TrainError::CoreRejected("production_training_rms_backward"))?,
    );
    Ok(())
}

type AttentionGradients = (Vec<i16>, Vec<i16>, Vec<i16>);

#[allow(clippy::too_many_arguments)]
fn linear_attention_backward(
    d_model: usize,
    heads: usize,
    q: &[i16],
    k: &[i16],
    v: &[i16],
    context: &[i16],
    grad_context: &[i16],
    stats: &mut UpdateStats,
) -> Result<AttentionGradients, TrainError> {
    let seq_len = q.len() / d_model;
    let head_dim = d_model / heads;
    let state_len = head_dim * head_dim;
    let mut gq = vec![0_i64; q.len()];
    let mut gk = vec![0_i64; q.len()];
    let mut gv = vec![0_i64; q.len()];
    for head in 0..heads {
        let offset = head * head_dim;
        let mut prefixes = vec![0_i64; seq_len * state_len];
        let mut prefix_sums = vec![0_i64; seq_len * head_dim];
        let mut denominators = vec![0_i64; seq_len];
        let mut state = vec![0_i64; state_len];
        let mut sums = vec![0_i64; head_dim];
        for token in 0..seq_len {
            let base = token * d_model + offset;
            for kd in 0..head_dim {
                let phi = i64::from(k[base + kd]) + 32769;
                sums[kd] = sums[kd].saturating_add(phi);
                for vd in 0..head_dim {
                    state[kd * head_dim + vd] =
                        state[kd * head_dim + vd].saturating_add(phi * i64::from(v[base + vd]));
                }
            }
            denominators[token] = (0..head_dim)
                .map(|d| (i64::from(q[base + d]) + 32769) * sums[d])
                .sum();
            if denominators[token] <= 0 {
                return Err(TrainError::CoreRejected(
                    "production_training_attention_denominator",
                ));
            }
            prefixes[token * state_len..(token + 1) * state_len].copy_from_slice(&state);
            prefix_sums[token * head_dim..(token + 1) * head_dim].copy_from_slice(&sums);
        }
        let mut grad_state = vec![0_i64; state_len];
        let mut grad_sums = vec![0_i64; head_dim];
        for token in (0..seq_len).rev() {
            let base = token * d_model + offset;
            let denominator = denominators[token];
            let dot_grad_context = (0..head_dim).fold(0_i64, |acc, vd| {
                acc.saturating_add(
                    i64::from(grad_context[base + vd]) * i64::from(context[base + vd]),
                )
            });
            for kd in 0..head_dim {
                let mut numerator = 0_i64;
                for vd in 0..head_dim {
                    numerator = numerator.saturating_add(
                        i64::from(grad_context[base + vd])
                            * prefixes[token * state_len + kd * head_dim + vd],
                    );
                }
                numerator = numerator.saturating_sub(
                    dot_grad_context.saturating_mul(prefix_sums[token * head_dim + kd]),
                );
                gq[base + kd] =
                    gq[base + kd].saturating_add(round_ratio(numerator, denominator, stats)?);
            }
            for kd in 0..head_dim {
                let phi_q = i64::from(q[base + kd]) + 32769;
                for vd in 0..head_dim {
                    let product = i64::from(grad_context[base + vd])
                        .saturating_mul(phi_q)
                        .saturating_mul(1_i64 << 15);
                    grad_state[kd * head_dim + vd] = grad_state[kd * head_dim + vd]
                        .saturating_add(round_ratio(product, denominator, stats)?);
                }
                grad_sums[kd] = grad_sums[kd].saturating_add(round_ratio(
                    dot_grad_context.saturating_mul(phi_q).saturating_neg(),
                    denominator,
                    stats,
                )?);
            }
            for kd in 0..head_dim {
                let phi_k = i64::from(k[base + kd]) + 32769;
                let mut key_grad = 0_i64;
                for vd in 0..head_dim {
                    let sg = grad_state[kd * head_dim + vd];
                    gv[base + vd] = gv[base + vd].saturating_add(quantized_nonzero(
                        sg.saturating_mul(phi_k),
                        15,
                        stats,
                    ));
                    key_grad = key_grad.saturating_add(quantized_nonzero(
                        sg.saturating_mul(i64::from(v[base + vd])),
                        15,
                        stats,
                    ));
                }
                gk[base + kd] = gk[base + kd]
                    .saturating_add(key_grad)
                    .saturating_add(grad_sums[kd]);
            }
        }
    }
    Ok((
        gq.into_iter().map(saturate_i16).collect(),
        gk.into_iter().map(saturate_i16).collect(),
        gv.into_iter().map(saturate_i16).collect(),
    ))
}

fn round_ratio(
    numerator: i64,
    denominator: i64,
    stats: &mut UpdateStats,
) -> Result<i64, TrainError> {
    stats.backward_quantization = stats.backward_quantization.saturating_add(1);
    if denominator <= 0 {
        return Err(TrainError::InvalidConfig);
    }
    let half = denominator / 2;
    let rounded = if numerator >= 0 {
        numerator.checked_add(half).map(|v| v / denominator)
    } else {
        numerator
            .checked_neg()
            .and_then(|v| v.checked_add(half))
            .map(|v| -(v / denominator))
    }
    .ok_or(TrainError::CoreRejected("production_training_ratio"))?;
    Ok(if rounded == 0 && numerator != 0 {
        stats.backward_ste_rescue = stats.backward_ste_rescue.saturating_add(1);
        numerator.signum()
    } else {
        rounded
    })
}

fn update_i16_slice(
    values: &mut [i16],
    gradients: &[i64],
    shift: u8,
    residuals: &mut [i64],
    apply_updates: bool,
    group: usize,
    stats: &mut UpdateStats,
) {
    for ((value, &gradient), residual) in values.iter_mut().zip(gradients).zip(residuals) {
        update_i16(
            value,
            gradient,
            shift,
            residual,
            apply_updates,
            group,
            stats,
        );
    }
}
fn update_i16(
    value: &mut i16,
    gradient: i64,
    shift: u8,
    residual: &mut i64,
    apply_updates: bool,
    group: usize,
    stats: &mut UpdateStats,
) {
    stats.gradient_nonzero[group] =
        stats.gradient_nonzero[group].saturating_add(u64::from(gradient != 0));
    *residual = residual.saturating_add(gradient);
    if !apply_updates {
        return;
    }
    let update = round_shift_rhu_i64(*residual, shift);
    stats.residual_carry[group] =
        stats.residual_carry[group].saturating_add(u64::from(update == 0 && *residual != 0));
    let previous = *value;
    let wide = i64::from(previous).saturating_sub(update);
    let next = saturate_i16(wide);
    let saturated = wide < i64::from(i16::MIN) || wide > i64::from(i16::MAX);
    stats.weight_saturation += usize::from(saturated);
    stats.saturation_by_group[group] =
        stats.saturation_by_group[group].saturating_add(u64::from(saturated));
    stats.update_nonzero[group] =
        stats.update_nonzero[group].saturating_add(u64::from(next != previous));
    stats.movement[group] = stats.movement[group]
        .saturating_add((i64::from(next) - i64::from(previous)).unsigned_abs());
    *value = next;
    consume_residual(residual, update, shift);
}
fn update_i32(
    value: &mut i32,
    gradient: i64,
    shift: u8,
    residual: &mut i64,
    apply_updates: bool,
    group: usize,
    stats: &mut UpdateStats,
) {
    stats.gradient_nonzero[group] =
        stats.gradient_nonzero[group].saturating_add(u64::from(gradient != 0));
    *residual = residual.saturating_add(gradient);
    if !apply_updates {
        return;
    }
    let update = round_shift_rhu_i64(*residual, shift);
    stats.residual_carry[group] =
        stats.residual_carry[group].saturating_add(u64::from(update == 0 && *residual != 0));
    let previous = *value;
    let wide = i64::from(previous).saturating_sub(update);
    let next = wide.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
    let saturated = wide < i64::from(i32::MIN) || wide > i64::from(i32::MAX);
    stats.weight_saturation += usize::from(saturated);
    stats.saturation_by_group[group] =
        stats.saturation_by_group[group].saturating_add(u64::from(saturated));
    stats.update_nonzero[group] =
        stats.update_nonzero[group].saturating_add(u64::from(next != previous));
    stats.movement[group] = stats.movement[group]
        .saturating_add((i64::from(next) - i64::from(previous)).unsigned_abs());
    *value = next;
    consume_residual(residual, update, shift);
}

fn consume_residual(residual: &mut i64, update: i64, shift: u8) {
    let consumed = i128::from(update) * (1_i128 << shift);
    let remaining = i128::from(*residual) - consumed;
    *residual = remaining.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64;
}
fn add_rows(left: &[i16], right: &[i16]) -> Vec<i16> {
    left.iter()
        .zip(right)
        .map(|(&a, &b)| saturate_i16(i64::from(a) + i64::from(b)))
        .collect()
}
fn add_rows_in_place(left: &mut [i16], right: &[i16], stats: &mut UpdateStats) {
    for (a, &b) in left.iter_mut().zip(right) {
        let wide = i64::from(*a) + i64::from(b);
        stats.gradient_saturation +=
            usize::from(wide < i64::from(i16::MIN) || wide > i64::from(i16::MAX));
        *a = saturate_i16(wide);
    }
}
fn rms_rows(input: &[i16], weights: &[i16], d_model: usize) -> Result<Vec<i16>, TrainError> {
    let mut out = vec![0_i16; input.len()];
    for (a, b) in input
        .chunks_exact(d_model)
        .zip(out.chunks_exact_mut(d_model))
    {
        rms_norm_i16_q15_checked(a, weights, PRODUCTION_RMS_EPSILON, b)
            .ok_or(TrainError::CoreRejected("production_training_rms"))?;
    }
    Ok(out)
}
fn output_logits(model: &ProductionModelV1, features: &[i16]) -> Vec<i32> {
    (0..model.config.vocab_size)
        .map(|token| {
            let start = token * model.config.d_model;
            let acc = features.iter().enumerate().fold(0_i64, |sum, (d, &f)| {
                sum.saturating_add(i64::from(f) * i64::from(model.output_weights[start + d]))
            });
            ((acc >> model.scales.output_shift).clamp(i64::from(i32::MIN), i64::from(i32::MAX))
                as i32)
                .saturating_add(model.output_bias_q8[token])
        })
        .collect()
}
fn evaluate_mistakes(
    model: &ProductionModelV1,
    windows: &[(Vec<u32>, u32)],
) -> Result<usize, TrainError> {
    let mut n = 0;
    for (context, target) in windows {
        let cache = forward_cache(model, context)?;
        let predicted = cache
            .probabilities
            .iter()
            .enumerate()
            .max_by_key(|&(i, v)| (*v, core::cmp::Reverse(i)))
            .map(|(i, _)| i)
            .unwrap_or(0);
        n += usize::from(predicted != *target as usize);
    }
    Ok(n)
}
fn document_windows(tokens: &[u32], context: usize, max: usize) -> Vec<(Vec<u32>, u32)> {
    let mut windows = Vec::new();
    let mut doc = Vec::new();
    let mut active = false;
    for &token in tokens {
        if token == BOS_TOKEN_ID {
            doc.clear();
            active = true
        } else if token == EOS_TOKEN_ID {
            if active && doc.len() > context {
                for start in 0..doc.len() - context {
                    windows.push((doc[start..start + context].to_vec(), doc[start + context]));
                    if windows.len() >= max {
                        return windows;
                    }
                }
            }
            doc.clear();
            active = false
        } else if active {
            doc.push(token)
        }
    }
    windows
}
fn schedule_hash(c: ProductionFullTrainConfig) -> u64 {
    let mut bytes = Vec::new();
    for value in [c.context_tokens, c.max_windows, c.epochs, c.batch_windows] {
        bytes.extend_from_slice(&(value as u64).to_le_bytes())
    }
    if projection_shift_overrides(c).iter().all(Option::is_none)
        && c.output_backward_shift.is_none()
    {
        bytes.extend_from_slice(&[
            c.matrix_learning_rate_shift,
            c.vector_learning_rate_shift,
            c.embedding_learning_rate_shift,
            c.output_learning_rate_shift,
        ]);
    } else {
        bytes.extend_from_slice(&effective_learning_rate_shifts(c));
        bytes.push(c.output_backward_shift.unwrap_or(u8::MAX));
    }
    bytes.iter().fold(FNV_OFFSET, |mut hash, &byte| {
        hash ^= u64::from(byte);
        hash.wrapping_mul(FNV_PRIME)
    })
}

fn projection_shift_overrides(config: ProductionFullTrainConfig) -> [Option<u8>; 7] {
    [
        config.q_learning_rate_shift,
        config.k_learning_rate_shift,
        config.v_learning_rate_shift,
        config.o_learning_rate_shift,
        config.up_learning_rate_shift,
        config.gate_learning_rate_shift,
        config.down_learning_rate_shift,
    ]
}

fn quantized_nonzero(value: i64, shift: u8, stats: &mut UpdateStats) -> i64 {
    stats.backward_quantization = stats.backward_quantization.saturating_add(1);
    let rounded = round_shift_rhu_i64(value, shift);
    if rounded == 0 && value != 0 {
        stats.backward_ste_rescue = stats.backward_ste_rescue.saturating_add(1);
        value.signum()
    } else {
        rounded
    }
}

fn matrix_group_shift(base: u8, group: usize) -> u8 {
    match group {
        4 | 7..=9 => base,
        5 => base.saturating_add(4).min(62),
        6 => base.saturating_add(8).min(62),
        10 => base.saturating_sub(8),
        _ => base,
    }
}

fn matrix_learning_rate_shift(config: ProductionFullTrainConfig, group: usize) -> u8 {
    let explicit = match group {
        4 => config.q_learning_rate_shift,
        5 => config.k_learning_rate_shift,
        6 => config.v_learning_rate_shift,
        7 => config.o_learning_rate_shift,
        8 => config.up_learning_rate_shift,
        9 => config.gate_learning_rate_shift,
        10 => config.down_learning_rate_shift,
        _ => None,
    };
    explicit.unwrap_or_else(|| matrix_group_shift(config.matrix_learning_rate_shift, group))
}

fn effective_learning_rate_shifts(config: ProductionFullTrainConfig) -> [u8; 13] {
    [
        config.embedding_learning_rate_shift,
        vector_group_shift(config.vector_learning_rate_shift, 1),
        vector_group_shift(config.vector_learning_rate_shift, 2),
        vector_group_shift(config.vector_learning_rate_shift, 3),
        matrix_learning_rate_shift(config, 4),
        matrix_learning_rate_shift(config, 5),
        matrix_learning_rate_shift(config, 6),
        matrix_learning_rate_shift(config, 7),
        matrix_learning_rate_shift(config, 8),
        matrix_learning_rate_shift(config, 9),
        matrix_learning_rate_shift(config, 10),
        config.output_learning_rate_shift,
        config.vector_learning_rate_shift,
    ]
}

fn vector_group_shift(base: u8, group: usize) -> u8 {
    match group {
        1 | 3 => base.saturating_sub(4),
        2 => base.saturating_sub(10),
        _ => base,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        FNV_OFFSET, FNV_PRIME, ProductionFullTrainConfig, UpdateStats,
        effective_learning_rate_shifts, quantized_nonzero, schedule_hash,
    };

    #[test]
    fn default_effective_learning_rate_shifts_preserve_legacy_schedule() {
        let config = ProductionFullTrainConfig::default();
        assert_eq!(
            effective_learning_rate_shifts(config),
            [4, 6, 0, 6, 16, 20, 24, 16, 16, 16, 8, 24, 10]
        );
        let mut legacy_bytes = Vec::new();
        for value in [
            config.context_tokens,
            config.max_windows,
            config.epochs,
            config.batch_windows,
        ] {
            legacy_bytes.extend_from_slice(&(value as u64).to_le_bytes());
        }
        legacy_bytes.extend_from_slice(&[16, 10, 4, 24]);
        let legacy_hash = legacy_bytes.iter().fold(FNV_OFFSET, |mut hash, &byte| {
            hash ^= u64::from(byte);
            hash.wrapping_mul(FNV_PRIME)
        });
        assert_eq!(schedule_hash(config), legacy_hash);
    }

    #[test]
    fn projection_overrides_are_independent_and_bind_the_optimizer_schedule() {
        let base = ProductionFullTrainConfig::default();
        let tuned = ProductionFullTrainConfig {
            q_learning_rate_shift: Some(21),
            k_learning_rate_shift: Some(30),
            down_learning_rate_shift: Some(24),
            ..base
        };
        assert_eq!(
            effective_learning_rate_shifts(tuned),
            [4, 6, 0, 6, 21, 30, 24, 16, 16, 16, 24, 24, 10]
        );
        assert_ne!(schedule_hash(base), schedule_hash(tuned));
        let backward_tuned = ProductionFullTrainConfig {
            output_backward_shift: Some(8),
            ..base
        };
        assert_ne!(schedule_hash(base), schedule_hash(backward_tuned));
    }

    #[test]
    fn output_backward_dead_zone_rescue_preserves_nonzero_signal() {
        let mut stats = UpdateStats::default();
        assert_eq!(quantized_nonzero(1, 12, &mut stats), 1);
        assert_eq!(quantized_nonzero(-1, 12, &mut stats), -1);
        assert_eq!(stats.backward_quantization, 2);
        assert_eq!(stats.backward_ste_rescue, 2);
    }
}
