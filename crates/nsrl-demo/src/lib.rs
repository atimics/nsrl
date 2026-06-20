use nsrl_core::{
    AttentionResidualWorkspace, FixedScale, GatedMlpI16Params, GatedMlpResidualWorkspace,
    GatedMlpWorkspace, LOGIT_FRAC_BITS, LinearI16I8Params, PreNormAttentionResidualWorkspace,
    PreNormGatedMlpResidualWorkspace, Q15_SHIFT, SelfAttentionI16Params, SelfAttentionWorkspace,
    base2_softmax_i32_q15, linear_i16_i8_i16_per_channel_checked,
    prenorm_attention_residual_block_i16_q15_checked,
    prenorm_gated_mlp_residual_block_i16_q15_checked, sqrt_power_of_four_shift,
};

pub mod bench;

pub use bench::{
    BenchmarkPreset, BenchmarkSuiteTrace, BenchmarkTrace, run_benchmark, run_benchmark_suite,
    run_benchmark_suite_with_linear_kernel, run_benchmark_with_linear_kernel,
};

pub const SCHEMA: &str = "nsrl.forward_trace.v1";
pub const AUTHORITY: &str = "integer_runtime_determinism";
pub const SEQ_LEN: usize = 4;
pub const D_MODEL: usize = 4;
pub const HEADS: usize = 1;
pub const HIDDEN_DIM: usize = 8;
pub const VOCAB: usize = 8;
pub const RMS_EPS: u64 = 1;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

const VOCAB_LABELS: [&str; VOCAB] = ["<pad>", "a", "b", "c", "d", "e", "f", "g"];

const EMBEDDINGS_Q15: [[i16; D_MODEL]; VOCAB] = [
    [512, -256, 128, -64],
    [2048, -1024, 768, 256],
    [-1536, 1792, -512, 1024],
    [1024, 512, -1792, 768],
    [-2048, -768, 1536, -512],
    [768, -2048, -1024, 1536],
    [-512, 1024, 2048, -1536],
    [1536, 768, -256, -2048],
];

const IDENTITY_4: [i8; D_MODEL * D_MODEL] = [
    1, 0, 0, 0, //
    0, 1, 0, 0, //
    0, 0, 1, 0, //
    0, 0, 0, 1,
];

const OUTPUT_HEAD_WEIGHTS: [i8; VOCAB * D_MODEL] = [
    1, 0, 0, 0, //
    1, -1, 1, 0, //
    -1, 1, 0, 1, //
    0, 1, -1, 1, //
    -1, 0, 1, -1, //
    1, 1, -1, -1, //
    0, -1, 1, 1, //
    -1, -1, 0, 1,
];

const MLP_UP_WEIGHTS: [i8; HIDDEN_DIM * D_MODEL] = [
    1, 0, 1, 0, //
    0, 1, 0, 1, //
    1, -1, 0, 1, //
    -1, 0, 1, 1, //
    1, 1, -1, 0, //
    0, -1, 1, -1, //
    -1, 1, 1, 0, //
    1, 0, -1, 1,
];

const MLP_GATE_WEIGHTS: [i8; HIDDEN_DIM * D_MODEL] = [
    1, 0, 0, 0, //
    0, 1, 0, 0, //
    0, 0, 1, 0, //
    0, 0, 0, 1, //
    1, 1, 0, 0, //
    0, 1, 1, 0, //
    0, 0, 1, 1, //
    1, 0, 0, -1,
];

const MLP_DOWN_WEIGHTS: [i8; D_MODEL * HIDDEN_DIM] = [
    1, 0, 1, -1, 0, 1, -1, 1, //
    0, 1, -1, 0, 1, -1, 1, 0, //
    1, -1, 0, 1, -1, 0, 1, -1, //
    -1, 1, 1, 0, 0, 1, -1, 1,
];

const QK_SCALES: [FixedScale; D_MODEL] = [FixedScale {
    multiplier: 1,
    right_shift: 8,
}; D_MODEL];
const V_SCALES: [FixedScale; D_MODEL] = [FixedScale {
    multiplier: 1,
    right_shift: 4,
}; D_MODEL];
const O_SCALES: [FixedScale; D_MODEL] = [FixedScale {
    multiplier: 1,
    right_shift: 1,
}; D_MODEL];
const OUTPUT_HEAD_SCALES: [FixedScale; VOCAB] = [FixedScale {
    multiplier: 1,
    right_shift: 4,
}; VOCAB];
const MLP_UP_SCALES: [FixedScale; HIDDEN_DIM] = [FixedScale {
    multiplier: 1,
    right_shift: 5,
}; HIDDEN_DIM];
const MLP_GATE_SCALES: [FixedScale; HIDDEN_DIM] = [FixedScale {
    multiplier: 1,
    right_shift: 3,
}; HIDDEN_DIM];
const MLP_DOWN_SCALES: [FixedScale; D_MODEL] = [FixedScale {
    multiplier: 1,
    right_shift: 2,
}; D_MODEL];
const RMS_WEIGHTS_Q15: [i16; D_MODEL] = [16_384; D_MODEL];

const KNOWN_NON_CLAIMS: [&str; 4] = [
    "does_not_run_standard_gpt_or_llama_checkpoints",
    "does_not_claim_language_model_quality",
    "does_not_use_euler_softmax_compatibility",
    "does_not_prove_training_convergence",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForwardTrace {
    pub model_hash: u64,
    pub input_text: String,
    pub input_ids: Vec<usize>,
    pub attention_logits_q8: Vec<i32>,
    pub attention_probabilities_q15: Vec<i16>,
    pub attention_softmax_sum_q15: u64,
    pub attention_residual_saturation_count: usize,
    pub mlp_residual_saturation_count: usize,
    pub residual_saturation_count: usize,
    pub layer_stats: Vec<LayerStats>,
    pub output_logits_q15: Vec<i16>,
    pub chosen_id: usize,
    pub output_hash: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerStats {
    pub name: &'static str,
    pub hash: u64,
    pub min: i16,
    pub max: i16,
    pub max_abs: i16,
    pub saturation_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DemoError {
    CoreRejected(&'static str),
    EmptyOutputHead,
}

impl core::fmt::Display for DemoError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::CoreRejected(stage) => write!(f, "nsrl-core rejected demo stage: {stage}"),
            Self::EmptyOutputHead => write!(f, "output head produced no logits"),
        }
    }
}

impl std::error::Error for DemoError {}

pub fn run_forward(input: &str) -> Result<ForwardTrace, DemoError> {
    let input_ids = encode_input_ids(input);
    let input_activations = embed_input(&input_ids);
    let total = SEQ_LEN * D_MODEL;

    let mut normalized = vec![0_i16; total];
    let mut q = vec![0_i16; total];
    let mut k = vec![0_i16; total];
    let mut v = vec![0_i16; total];
    let mut context = vec![0_i16; total];
    let mut logits_q8 = vec![0_i32; SEQ_LEN];
    let mut probabilities_q15 = vec![0_i16; SEQ_LEN];
    let mut attention_output = vec![0_i16; total];
    let mut attention_residual_output = vec![0_i16; total];

    let params = SelfAttentionI16Params {
        q: qk_params(),
        k: qk_params(),
        v: v_params(),
        o: o_params(),
        seq_len: SEQ_LEN,
        d_model: D_MODEL,
        heads: HEADS,
        causal: true,
    };

    let workspace = PreNormAttentionResidualWorkspace {
        normalized: &mut normalized,
        residual: AttentionResidualWorkspace {
            attention: SelfAttentionWorkspace {
                q: &mut q,
                k: &mut k,
                v: &mut v,
                context: &mut context,
                logits_q8: &mut logits_q8,
                probabilities_q15: &mut probabilities_q15,
            },
            attention_output: &mut attention_output,
        },
    };

    let attention_residual_saturation_count = prenorm_attention_residual_block_i16_q15_checked(
        &input_activations,
        &RMS_WEIGHTS_Q15,
        RMS_EPS,
        params,
        workspace,
        &mut attention_residual_output,
    )
    .ok_or(DemoError::CoreRejected("prenorm_attention_residual_block"))?;

    let hidden_total = SEQ_LEN * HIDDEN_DIM;
    let mut mlp_normalized = vec![0_i16; total];
    let mut mlp_up = vec![0_i16; hidden_total];
    let mut mlp_gate = vec![0_i16; hidden_total];
    let mut mlp_gated = vec![0_i16; hidden_total];
    let mut mlp_output = vec![0_i16; total];
    let mut transformer_output = vec![0_i16; total];

    let mlp_residual_saturation_count = prenorm_gated_mlp_residual_block_i16_q15_checked(
        &attention_residual_output,
        &RMS_WEIGHTS_Q15,
        RMS_EPS,
        mlp_params(),
        PreNormGatedMlpResidualWorkspace {
            normalized: &mut mlp_normalized,
            residual: GatedMlpResidualWorkspace {
                mlp: GatedMlpWorkspace {
                    up: &mut mlp_up,
                    gate: &mut mlp_gate,
                    gated: &mut mlp_gated,
                },
                mlp_output: &mut mlp_output,
            },
        },
        &mut transformer_output,
    )
    .ok_or(DemoError::CoreRejected("prenorm_gated_mlp_residual_block"))?;

    let last_token_start = (SEQ_LEN - 1) * D_MODEL;
    let last_token = &transformer_output[last_token_start..last_token_start + D_MODEL];
    let mut output_logits_q15 = vec![0_i16; VOCAB];
    linear_i16_i8_i16_per_channel_checked(last_token, output_head_params(), &mut output_logits_q15)
        .ok_or(DemoError::CoreRejected("output_head"))?;

    let (chosen_id, _) = output_logits_q15
        .iter()
        .enumerate()
        .max_by_key(|&(id, &logit)| (logit, core::cmp::Reverse(id)))
        .ok_or(DemoError::EmptyOutputHead)?;

    let mut attention_probabilities_q15 = vec![0_i16; SEQ_LEN];
    let attention_softmax_sum_q15 =
        base2_softmax_i32_q15(&logits_q8, &mut attention_probabilities_q15)
            .ok_or(DemoError::CoreRejected("attention_preview_softmax"))?;
    let output_hash = hash_i16_slice(&transformer_output)
        ^ hash_i16_slice(last_token).rotate_left(17)
        ^ hash_i16_slice(&output_logits_q15).rotate_left(31);

    Ok(ForwardTrace {
        model_hash: model_hash(),
        input_text: input.to_owned(),
        input_ids,
        attention_logits_q8: logits_q8,
        attention_probabilities_q15,
        attention_softmax_sum_q15,
        attention_residual_saturation_count,
        mlp_residual_saturation_count,
        residual_saturation_count: attention_residual_saturation_count
            + mlp_residual_saturation_count,
        layer_stats: vec![
            layer_stats("input_q15", &input_activations),
            layer_stats("attention_rms_norm_q15", &normalized),
            layer_stats("q_projection_q15", &q),
            layer_stats("k_projection_q15", &k),
            layer_stats("v_projection_q15", &v),
            layer_stats("attention_context_q15", &context),
            layer_stats("attention_output_q15", &attention_output),
            layer_stats("attention_residual_output_q15", &attention_residual_output),
            layer_stats("mlp_rms_norm_q15", &mlp_normalized),
            layer_stats("mlp_up_q15", &mlp_up),
            layer_stats("mlp_gate_q15", &mlp_gate),
            layer_stats("mlp_hard_silu_gated_q15", &mlp_gated),
            layer_stats("mlp_output_q15", &mlp_output),
            layer_stats("transformer_output_q15", &transformer_output),
            layer_stats("output_logits_q15", &output_logits_q15),
        ],
        output_logits_q15,
        chosen_id,
        output_hash,
    })
}

impl ForwardTrace {
    pub fn to_json_line(&self) -> String {
        let mut out = String::new();
        out.push('{');
        push_string_field(&mut out, "schema", SCHEMA);
        comma(&mut out);
        push_string_field(&mut out, "authority", AUTHORITY);
        comma(&mut out);
        out.push_str("\"arithmetic\":{");
        push_usize_field(&mut out, "residual_scale_q", 15);
        comma(&mut out);
        push_string_field(&mut out, "rounding", "branchless_round_half_up");
        comma(&mut out);
        push_usize_field(&mut out, "seq_len", SEQ_LEN);
        comma(&mut out);
        push_usize_field(&mut out, "d_model", D_MODEL);
        comma(&mut out);
        push_usize_field(&mut out, "hidden_dim", HIDDEN_DIM);
        comma(&mut out);
        push_usize_field(&mut out, "heads", HEADS);
        comma(&mut out);
        push_usize_field(&mut out, "head_dim", D_MODEL / HEADS);
        comma(&mut out);
        push_usize_field(
            &mut out,
            "sqrt_head_shift",
            usize::from(sqrt_power_of_four_shift(D_MODEL / HEADS).unwrap_or(0)),
        );
        comma(&mut out);
        push_usize_field(&mut out, "logit_frac_bits", usize::from(LOGIT_FRAC_BITS));
        comma(&mut out);
        push_usize_field(&mut out, "probability_shift", usize::from(Q15_SHIFT));
        comma(&mut out);
        push_string_field(&mut out, "attention_temperature", "native_log2");
        comma(&mut out);
        push_string_field(&mut out, "mlp_activation", "hard_silu_shift2_q15");
        out.push('}');
        comma(&mut out);
        push_hash_field(&mut out, "model_hash", self.model_hash);
        comma(&mut out);
        out.push_str("\"input\":{");
        push_string_field(&mut out, "text", &self.input_text);
        comma(&mut out);
        push_usize_array_field(&mut out, "ids", &self.input_ids);
        out.push('}');
        comma(&mut out);
        out.push_str("\"attention_preview\":{");
        push_usize_field(&mut out, "token_index", SEQ_LEN - 1);
        comma(&mut out);
        push_i32_array_field(&mut out, "logits_q8", &self.attention_logits_q8);
        comma(&mut out);
        push_i16_array_field(
            &mut out,
            "probabilities_q15",
            &self.attention_probabilities_q15,
        );
        comma(&mut out);
        push_u64_field(
            &mut out,
            "pre_normalization_sum_q15",
            self.attention_softmax_sum_q15,
        );
        out.push('}');
        comma(&mut out);
        push_usize_field(
            &mut out,
            "attention_residual_saturation_count",
            self.attention_residual_saturation_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "mlp_residual_saturation_count",
            self.mlp_residual_saturation_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "residual_saturation_count",
            self.residual_saturation_count,
        );
        comma(&mut out);
        out.push_str("\"layer_stats\":[");
        for (index, stats) in self.layer_stats.iter().enumerate() {
            if index != 0 {
                comma(&mut out);
            }
            stats.push_json(&mut out);
        }
        out.push(']');
        comma(&mut out);
        out.push_str("\"output\":{");
        push_i16_array_field(&mut out, "logits_q15", &self.output_logits_q15);
        comma(&mut out);
        push_usize_field(&mut out, "chosen_id", self.chosen_id);
        comma(&mut out);
        push_string_field(&mut out, "chosen_token", VOCAB_LABELS[self.chosen_id]);
        comma(&mut out);
        push_hash_field(&mut out, "output_hash", self.output_hash);
        out.push('}');
        comma(&mut out);
        push_string_array_field(&mut out, "known_non_claims", &KNOWN_NON_CLAIMS);
        out.push('}');
        out.push('\n');
        out
    }
}

impl LayerStats {
    fn push_json(&self, out: &mut String) {
        out.push('{');
        push_string_field(out, "name", self.name);
        comma(out);
        push_hash_field(out, "hash", self.hash);
        comma(out);
        push_i16_field(out, "min", self.min);
        comma(out);
        push_i16_field(out, "max", self.max);
        comma(out);
        push_i16_field(out, "max_abs", self.max_abs);
        comma(out);
        push_usize_field(out, "saturation_count", self.saturation_count);
        out.push('}');
    }
}

fn qk_params() -> LinearI16I8Params<'static> {
    LinearI16I8Params {
        weights: &IDENTITY_4,
        bias: None,
        scales: &QK_SCALES,
        input_dim: D_MODEL,
        output_dim: D_MODEL,
    }
}

fn v_params() -> LinearI16I8Params<'static> {
    LinearI16I8Params {
        weights: &IDENTITY_4,
        bias: None,
        scales: &V_SCALES,
        input_dim: D_MODEL,
        output_dim: D_MODEL,
    }
}

fn o_params() -> LinearI16I8Params<'static> {
    LinearI16I8Params {
        weights: &IDENTITY_4,
        bias: None,
        scales: &O_SCALES,
        input_dim: D_MODEL,
        output_dim: D_MODEL,
    }
}

fn output_head_params() -> LinearI16I8Params<'static> {
    LinearI16I8Params {
        weights: &OUTPUT_HEAD_WEIGHTS,
        bias: None,
        scales: &OUTPUT_HEAD_SCALES,
        input_dim: D_MODEL,
        output_dim: VOCAB,
    }
}

fn mlp_params() -> GatedMlpI16Params<'static> {
    GatedMlpI16Params {
        up: LinearI16I8Params {
            weights: &MLP_UP_WEIGHTS,
            bias: None,
            scales: &MLP_UP_SCALES,
            input_dim: D_MODEL,
            output_dim: HIDDEN_DIM,
        },
        gate: LinearI16I8Params {
            weights: &MLP_GATE_WEIGHTS,
            bias: None,
            scales: &MLP_GATE_SCALES,
            input_dim: D_MODEL,
            output_dim: HIDDEN_DIM,
        },
        down: LinearI16I8Params {
            weights: &MLP_DOWN_WEIGHTS,
            bias: None,
            scales: &MLP_DOWN_SCALES,
            input_dim: HIDDEN_DIM,
            output_dim: D_MODEL,
        },
        seq_len: SEQ_LEN,
        d_model: D_MODEL,
        hidden_dim: HIDDEN_DIM,
    }
}

fn encode_input_ids(input: &str) -> Vec<usize> {
    let mut ids = vec![0_usize; SEQ_LEN];
    for (slot, byte) in ids.iter_mut().zip(input.bytes()) {
        *slot = match byte {
            b' ' => 0,
            b'a'..=b'g' => usize::from(byte - b'a' + 1),
            b'A'..=b'G' => usize::from(byte - b'A' + 1),
            other => usize::from(other) & (VOCAB - 1),
        };
    }
    ids
}

fn embed_input(ids: &[usize]) -> Vec<i16> {
    let mut activations = Vec::with_capacity(SEQ_LEN * D_MODEL);
    for &id in ids.iter().take(SEQ_LEN) {
        activations.extend_from_slice(&EMBEDDINGS_Q15[id]);
    }
    activations
}

fn layer_stats(name: &'static str, values: &[i16]) -> LayerStats {
    let mut min = i16::MAX;
    let mut max = i16::MIN;
    let mut max_abs = 0_i16;
    let mut saturation_count = 0_usize;

    for &value in values {
        min = min.min(value);
        max = max.max(value);
        let abs = if value == i16::MIN {
            i16::MAX
        } else {
            value.abs()
        };
        max_abs = max_abs.max(abs);
        if value == i16::MIN || value == i16::MAX {
            saturation_count += 1;
        }
    }

    LayerStats {
        name,
        hash: hash_i16_slice(values),
        min,
        max,
        max_abs,
        saturation_count,
    }
}

fn model_hash() -> u64 {
    let mut hasher = StableHasher::new();
    hasher.update_str(SCHEMA);
    hasher.update_str(AUTHORITY);
    hasher.update_usize(SEQ_LEN);
    hasher.update_usize(D_MODEL);
    hasher.update_usize(HEADS);
    hasher.update_usize(HIDDEN_DIM);
    hasher.update_u64(RMS_EPS);
    for row in EMBEDDINGS_Q15 {
        hasher.update_i16_slice(&row);
    }
    hasher.update_i8_slice(&IDENTITY_4);
    hasher.update_i8_slice(&OUTPUT_HEAD_WEIGHTS);
    hasher.update_i8_slice(&MLP_UP_WEIGHTS);
    hasher.update_i8_slice(&MLP_GATE_WEIGHTS);
    hasher.update_i8_slice(&MLP_DOWN_WEIGHTS);
    for scale in QK_SCALES {
        hasher.update_i32(scale.multiplier);
        hasher.update_u8(scale.right_shift);
    }
    for scale in V_SCALES {
        hasher.update_i32(scale.multiplier);
        hasher.update_u8(scale.right_shift);
    }
    for scale in O_SCALES {
        hasher.update_i32(scale.multiplier);
        hasher.update_u8(scale.right_shift);
    }
    for scale in OUTPUT_HEAD_SCALES {
        hasher.update_i32(scale.multiplier);
        hasher.update_u8(scale.right_shift);
    }
    for scale in MLP_UP_SCALES {
        hasher.update_i32(scale.multiplier);
        hasher.update_u8(scale.right_shift);
    }
    for scale in MLP_GATE_SCALES {
        hasher.update_i32(scale.multiplier);
        hasher.update_u8(scale.right_shift);
    }
    for scale in MLP_DOWN_SCALES {
        hasher.update_i32(scale.multiplier);
        hasher.update_u8(scale.right_shift);
    }
    hasher.update_i16_slice(&RMS_WEIGHTS_Q15);
    hasher.finish()
}

fn hash_i16_slice(values: &[i16]) -> u64 {
    let mut hasher = StableHasher::new();
    hasher.update_i16_slice(values);
    hasher.finish()
}

struct StableHasher(u64);

impl StableHasher {
    fn new() -> Self {
        Self(FNV_OFFSET)
    }

    fn finish(self) -> u64 {
        self.0
    }

    fn update_u8(&mut self, value: u8) {
        self.0 ^= u64::from(value);
        self.0 = self.0.wrapping_mul(FNV_PRIME);
    }

    fn update_bytes(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.update_u8(byte);
        }
    }

    fn update_str(&mut self, value: &str) {
        self.update_bytes(value.as_bytes());
        self.update_u8(0xff);
    }

    fn update_usize(&mut self, value: usize) {
        self.update_bytes(&(value as u64).to_le_bytes());
    }

    fn update_u64(&mut self, value: u64) {
        self.update_bytes(&value.to_le_bytes());
    }

    fn update_i32(&mut self, value: i32) {
        self.update_bytes(&value.to_le_bytes());
    }

    fn update_i16_slice(&mut self, values: &[i16]) {
        self.update_usize(values.len());
        for &value in values {
            self.update_bytes(&value.to_le_bytes());
        }
    }

    fn update_i8_slice(&mut self, values: &[i8]) {
        self.update_usize(values.len());
        for &value in values {
            self.update_u8(value as u8);
        }
    }
}

fn comma(out: &mut String) {
    out.push(',');
}

fn push_string_field(out: &mut String, name: &str, value: &str) {
    push_quoted(out, name);
    out.push(':');
    push_quoted(out, value);
}

fn push_string_array_field(out: &mut String, name: &str, values: &[&str]) {
    push_quoted(out, name);
    out.push_str(":[");
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            comma(out);
        }
        push_quoted(out, value);
    }
    out.push(']');
}

fn push_hash_field(out: &mut String, name: &str, value: u64) {
    push_quoted(out, name);
    out.push(':');
    push_quoted(out, &format!("0x{value:016x}"));
}

fn push_usize_field(out: &mut String, name: &str, value: usize) {
    push_quoted(out, name);
    out.push(':');
    out.push_str(&value.to_string());
}

fn push_u64_field(out: &mut String, name: &str, value: u64) {
    push_quoted(out, name);
    out.push(':');
    out.push_str(&value.to_string());
}

fn push_i16_field(out: &mut String, name: &str, value: i16) {
    push_quoted(out, name);
    out.push(':');
    out.push_str(&value.to_string());
}

fn push_usize_array_field(out: &mut String, name: &str, values: &[usize]) {
    push_quoted(out, name);
    out.push(':');
    push_array(out, values.iter().map(usize::to_string));
}

fn push_i16_array_field(out: &mut String, name: &str, values: &[i16]) {
    push_quoted(out, name);
    out.push(':');
    push_array(out, values.iter().map(i16::to_string));
}

fn push_i32_array_field(out: &mut String, name: &str, values: &[i32]) {
    push_quoted(out, name);
    out.push(':');
    push_array(out, values.iter().map(i32::to_string));
}

fn push_array(out: &mut String, values: impl Iterator<Item = String>) {
    out.push('[');
    for (index, value) in values.enumerate() {
        if index != 0 {
            comma(out);
        }
        out.push_str(&value);
    }
    out.push(']');
}

fn push_quoted(out: &mut String, value: &str) {
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch.is_control() => out.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => out.push(ch),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_trace_is_byte_stable() {
        let left = run_forward("abba").expect("forward").to_json_line();
        let right = run_forward("abba").expect("forward").to_json_line();

        assert_eq!(left, right);
    }

    #[test]
    fn forward_trace_carries_core_contract_fields() {
        let trace = run_forward("abba").expect("forward");
        let json = trace.to_json_line();

        assert!(json.contains("\"schema\":\"nsrl.forward_trace.v1\""));
        assert!(json.contains("\"authority\":\"integer_runtime_determinism\""));
        assert!(json.contains("\"attention_temperature\":\"native_log2\""));
        assert!(json.contains("\"mlp_activation\":\"hard_silu_shift2_q15\""));
        assert!(json.contains("\"mlp_residual_saturation_count\":0"));
        assert!(json.contains("\"known_non_claims\""));
        assert_eq!(trace.input_ids.len(), SEQ_LEN);
        assert_eq!(trace.output_logits_q15.len(), VOCAB);
        assert_eq!(trace.attention_logits_q8.len(), SEQ_LEN);
        assert_eq!(trace.attention_probabilities_q15.len(), SEQ_LEN);
        assert!(
            trace
                .layer_stats
                .iter()
                .any(|stats| stats.name == "mlp_hard_silu_gated_q15")
        );
        assert!(
            trace
                .layer_stats
                .iter()
                .all(|stats| stats.saturation_count == 0)
        );
    }

    #[test]
    fn input_encoding_pads_or_truncates_to_fixed_context() {
        assert_eq!(encode_input_ids(""), [0, 0, 0, 0]);
        assert_eq!(encode_input_ids("abcdef"), [1, 2, 3, 4]);
        assert_eq!(encode_input_ids("A G"), [1, 0, 7, 0]);
    }
}
