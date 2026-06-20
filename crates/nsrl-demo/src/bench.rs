use std::time::Instant;

use nsrl_core::{
    AttentionResidualWorkspace, FixedScale, GatedMlpI16Params, GatedMlpResidualWorkspace,
    GatedMlpWorkspace, LOGIT_FRAC_BITS, LinearI16I8Params, LinearKernel,
    PreNormAttentionResidualWorkspace, PreNormGatedMlpResidualWorkspace, Q15_SHIFT,
    SelfAttentionI16Params, SelfAttentionWorkspace,
    prenorm_attention_residual_block_i16_q15_with_linear_kernel_checked,
    prenorm_gated_mlp_residual_block_i16_q15_with_linear_kernel_checked, sqrt_power_of_four_shift,
};

const BENCH_SCHEMA: &str = "nsrl.benchmark_trace.v1";
const BENCH_SUITE_SCHEMA: &str = "nsrl.benchmark_suite.v1";
const AUTHORITY: &str = "integer_runtime_determinism";
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
const RMS_EPS: u64 = 1;

const KNOWN_NON_CLAIMS: [&str; 4] = [
    "not_a_training_run",
    "does_not_claim_language_model_quality",
    "does_not_measure_cross_machine_determinism",
    "does_not_run_standard_gpt_or_llama_checkpoints",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BenchmarkPreset {
    pub name: &'static str,
    pub seq_len: usize,
    pub d_model: usize,
    pub heads: usize,
    pub hidden_dim: usize,
    pub blocks: usize,
    pub seed: u64,
}

impl BenchmarkPreset {
    pub fn bench_1m() -> Self {
        Self {
            name: "bench-1m",
            seq_len: 128,
            d_model: 128,
            heads: 2,
            hidden_dim: 512,
            blocks: 4,
            seed: 0x4e53_524c_5f31_4d21,
        }
    }

    fn head_dim(self) -> usize {
        self.d_model / self.heads
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchmarkTrace {
    pub preset: BenchmarkPreset,
    pub linear_kernel: LinearKernel,
    pub model_hash: u64,
    pub input_hash: u64,
    pub output_hash: u64,
    pub elapsed_micros: u128,
    pub tokens_per_second_x100: u128,
    pub i8_weight_count: usize,
    pub parameter_bytes: usize,
    pub workspace_bytes: usize,
    pub attention_residual_saturation_count: usize,
    pub mlp_residual_saturation_count: usize,
    pub residual_saturation_count: usize,
    pub final_max_abs: i16,
    pub final_saturation_count: usize,
    pub blocks: Vec<BlockBenchmarkTrace>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchmarkSuiteTrace {
    pub preset: BenchmarkPreset,
    pub linear_kernel: LinearKernel,
    pub warmup_runs: usize,
    pub measured_runs: usize,
    pub elapsed_micros: Vec<u128>,
    pub min_elapsed_micros: u128,
    pub median_elapsed_micros: u128,
    pub mean_elapsed_micros: u128,
    pub max_elapsed_micros: u128,
    pub median_tokens_per_second_x100: u128,
    pub model_hash: u64,
    pub input_hash: u64,
    pub output_hash: u64,
    pub i8_weight_count: usize,
    pub parameter_bytes: usize,
    pub workspace_bytes: usize,
    pub residual_saturation_count: usize,
    pub final_saturation_count: usize,
    pub final_max_abs: i16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockBenchmarkTrace {
    pub index: usize,
    pub attention_residual_saturation_count: usize,
    pub mlp_residual_saturation_count: usize,
    pub output_hash: u64,
    pub max_abs: i16,
    pub saturation_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BenchmarkError {
    InvalidPreset,
    InvalidRunCount,
    UnstableHash,
    CoreRejected(&'static str),
}

impl core::fmt::Display for BenchmarkError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidPreset => write!(f, "invalid benchmark preset"),
            Self::InvalidRunCount => write!(f, "benchmark measured run count must be nonzero"),
            Self::UnstableHash => write!(
                f,
                "benchmark produced unstable input, model, or output hashes"
            ),
            Self::CoreRejected(stage) => write!(f, "nsrl-core rejected benchmark stage: {stage}"),
        }
    }
}

impl std::error::Error for BenchmarkError {}

pub fn run_benchmark_suite(
    preset: BenchmarkPreset,
    warmup_runs: usize,
    measured_runs: usize,
) -> Result<BenchmarkSuiteTrace, BenchmarkError> {
    run_benchmark_suite_with_linear_kernel(
        preset,
        warmup_runs,
        measured_runs,
        LinearKernel::GenericI8,
    )
}

pub fn run_benchmark_suite_with_linear_kernel(
    preset: BenchmarkPreset,
    warmup_runs: usize,
    measured_runs: usize,
    linear_kernel: LinearKernel,
) -> Result<BenchmarkSuiteTrace, BenchmarkError> {
    if measured_runs == 0 {
        return Err(BenchmarkError::InvalidRunCount);
    }

    for _ in 0..warmup_runs {
        let _ = run_benchmark_with_linear_kernel(preset, linear_kernel)?;
    }

    let mut traces = Vec::with_capacity(measured_runs);
    for _ in 0..measured_runs {
        traces.push(run_benchmark_with_linear_kernel(preset, linear_kernel)?);
    }

    let first = traces.first().ok_or(BenchmarkError::InvalidRunCount)?;
    if traces.iter().any(|trace| {
        trace.model_hash != first.model_hash
            || trace.input_hash != first.input_hash
            || trace.output_hash != first.output_hash
    }) {
        return Err(BenchmarkError::UnstableHash);
    }

    let elapsed_micros = traces
        .iter()
        .map(|trace| trace.elapsed_micros)
        .collect::<Vec<_>>();
    let mut sorted_elapsed = elapsed_micros.clone();
    sorted_elapsed.sort_unstable();

    let min_elapsed_micros = sorted_elapsed[0];
    let median_elapsed_micros = sorted_elapsed[sorted_elapsed.len() / 2];
    let max_elapsed_micros = sorted_elapsed[sorted_elapsed.len() - 1];
    let mean_elapsed_micros =
        sorted_elapsed.iter().sum::<u128>() / u128::from(sorted_elapsed.len() as u64);
    let tokens = (preset.seq_len as u128) * (preset.blocks as u128);
    let median_tokens_per_second_x100 = if median_elapsed_micros == 0 {
        0
    } else {
        tokens * 100_000_000 / median_elapsed_micros
    };

    Ok(BenchmarkSuiteTrace {
        preset,
        linear_kernel,
        warmup_runs,
        measured_runs,
        elapsed_micros,
        min_elapsed_micros,
        median_elapsed_micros,
        mean_elapsed_micros,
        max_elapsed_micros,
        median_tokens_per_second_x100,
        model_hash: first.model_hash,
        input_hash: first.input_hash,
        output_hash: first.output_hash,
        i8_weight_count: first.i8_weight_count,
        parameter_bytes: first.parameter_bytes,
        workspace_bytes: first.workspace_bytes,
        residual_saturation_count: traces
            .iter()
            .map(|trace| trace.residual_saturation_count)
            .max()
            .unwrap_or(0),
        final_saturation_count: traces
            .iter()
            .map(|trace| trace.final_saturation_count)
            .max()
            .unwrap_or(0),
        final_max_abs: traces
            .iter()
            .map(|trace| trace.final_max_abs)
            .max()
            .unwrap_or(0),
    })
}

pub fn run_benchmark(preset: BenchmarkPreset) -> Result<BenchmarkTrace, BenchmarkError> {
    run_benchmark_with_linear_kernel(preset, LinearKernel::GenericI8)
}

pub fn run_benchmark_with_linear_kernel(
    preset: BenchmarkPreset,
    linear_kernel: LinearKernel,
) -> Result<BenchmarkTrace, BenchmarkError> {
    validate_preset(preset)?;

    let weights = generate_model(preset);
    let total = preset
        .seq_len
        .checked_mul(preset.d_model)
        .ok_or(BenchmarkError::InvalidPreset)?;
    let hidden_total = preset
        .seq_len
        .checked_mul(preset.hidden_dim)
        .ok_or(BenchmarkError::InvalidPreset)?;

    let mut current = generate_input(preset, total);
    let input_hash = hash_i16_slice(&current);
    let mut attention_residual = vec![0_i16; total];
    let mut next = vec![0_i16; total];

    let mut attention_norm = vec![0_i16; total];
    let mut q = vec![0_i16; total];
    let mut k = vec![0_i16; total];
    let mut v = vec![0_i16; total];
    let mut context = vec![0_i16; total];
    let mut logits_q8 = vec![0_i32; preset.seq_len];
    let mut probabilities_q15 = vec![0_i16; preset.seq_len];
    let mut attention_output = vec![0_i16; total];

    let mut mlp_norm = vec![0_i16; total];
    let mut mlp_up = vec![0_i16; hidden_total];
    let mut mlp_gate = vec![0_i16; hidden_total];
    let mut mlp_gated = vec![0_i16; hidden_total];
    let mut mlp_output = vec![0_i16; total];

    let rms_weights_q15 = vec![8_192_i16; preset.d_model];
    let workspace_bytes = bytes_i16(&current)
        + bytes_i16(&attention_residual)
        + bytes_i16(&next)
        + bytes_i16(&attention_norm)
        + bytes_i16(&q)
        + bytes_i16(&k)
        + bytes_i16(&v)
        + bytes_i16(&context)
        + bytes_i32(&logits_q8)
        + bytes_i16(&probabilities_q15)
        + bytes_i16(&attention_output)
        + bytes_i16(&mlp_norm)
        + bytes_i16(&mlp_up)
        + bytes_i16(&mlp_gate)
        + bytes_i16(&mlp_gated)
        + bytes_i16(&mlp_output)
        + bytes_i16(&rms_weights_q15);

    let started = Instant::now();
    let mut block_traces = Vec::with_capacity(preset.blocks);
    let mut attention_residual_saturation_count = 0_usize;
    let mut mlp_residual_saturation_count = 0_usize;

    for (index, block) in weights.blocks.iter().enumerate() {
        let attention_saturation =
            prenorm_attention_residual_block_i16_q15_with_linear_kernel_checked(
                &current,
                &rms_weights_q15,
                RMS_EPS,
                block.attention_params(preset),
                PreNormAttentionResidualWorkspace {
                    normalized: &mut attention_norm,
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
                },
                &mut attention_residual,
                linear_kernel,
            )
            .ok_or(BenchmarkError::CoreRejected("attention"))?;

        let mlp_saturation = prenorm_gated_mlp_residual_block_i16_q15_with_linear_kernel_checked(
            &attention_residual,
            &rms_weights_q15,
            RMS_EPS,
            block.mlp_params(preset),
            PreNormGatedMlpResidualWorkspace {
                normalized: &mut mlp_norm,
                residual: GatedMlpResidualWorkspace {
                    mlp: GatedMlpWorkspace {
                        up: &mut mlp_up,
                        gate: &mut mlp_gate,
                        gated: &mut mlp_gated,
                    },
                    mlp_output: &mut mlp_output,
                },
            },
            &mut next,
            linear_kernel,
        )
        .ok_or(BenchmarkError::CoreRejected("mlp"))?;

        attention_residual_saturation_count += attention_saturation;
        mlp_residual_saturation_count += mlp_saturation;

        let stats = tensor_health(&next);
        block_traces.push(BlockBenchmarkTrace {
            index,
            attention_residual_saturation_count: attention_saturation,
            mlp_residual_saturation_count: mlp_saturation,
            output_hash: hash_i16_slice(&next),
            max_abs: stats.max_abs,
            saturation_count: stats.saturation_count,
        });

        core::mem::swap(&mut current, &mut next);
    }

    let elapsed_micros = started.elapsed().as_micros();
    let tokens = (preset.seq_len as u128) * (preset.blocks as u128);
    let tokens_per_second_x100 = if elapsed_micros == 0 {
        0
    } else {
        tokens * 100_000_000 / elapsed_micros
    };
    let final_stats = tensor_health(&current);

    Ok(BenchmarkTrace {
        preset,
        linear_kernel,
        model_hash: weights.model_hash,
        input_hash,
        output_hash: hash_i16_slice(&current),
        elapsed_micros,
        tokens_per_second_x100,
        i8_weight_count: weights.i8_weight_count,
        parameter_bytes: weights.parameter_bytes + bytes_i16(&rms_weights_q15),
        workspace_bytes,
        attention_residual_saturation_count,
        mlp_residual_saturation_count,
        residual_saturation_count: attention_residual_saturation_count
            + mlp_residual_saturation_count,
        final_max_abs: final_stats.max_abs,
        final_saturation_count: final_stats.saturation_count,
        blocks: block_traces,
    })
}

impl BenchmarkSuiteTrace {
    pub fn to_json_line(&self) -> String {
        let mut out = String::new();
        out.push('{');
        push_string_field(&mut out, "schema", BENCH_SUITE_SCHEMA);
        comma(&mut out);
        push_string_field(&mut out, "authority", AUTHORITY);
        comma(&mut out);
        push_string_field(&mut out, "preset", self.preset.name);
        comma(&mut out);
        push_string_field(
            &mut out,
            "linear_kernel",
            linear_kernel_name(self.linear_kernel),
        );
        comma(&mut out);
        out.push_str("\"runs\":{");
        push_usize_field(&mut out, "warmup", self.warmup_runs);
        comma(&mut out);
        push_usize_field(&mut out, "measured", self.measured_runs);
        out.push('}');
        comma(&mut out);
        out.push_str("\"model\":{");
        push_usize_field(&mut out, "blocks", self.preset.blocks);
        comma(&mut out);
        push_usize_field(&mut out, "seq_len", self.preset.seq_len);
        comma(&mut out);
        push_usize_field(&mut out, "d_model", self.preset.d_model);
        comma(&mut out);
        push_usize_field(&mut out, "heads", self.preset.heads);
        comma(&mut out);
        push_usize_field(&mut out, "head_dim", self.preset.head_dim());
        comma(&mut out);
        push_usize_field(&mut out, "hidden_dim", self.preset.hidden_dim);
        comma(&mut out);
        push_usize_field(&mut out, "i8_weight_count", self.i8_weight_count);
        comma(&mut out);
        push_hash_field(&mut out, "model_hash", self.model_hash);
        out.push('}');
        comma(&mut out);
        out.push_str("\"memory\":{");
        push_usize_field(&mut out, "parameter_bytes", self.parameter_bytes);
        comma(&mut out);
        push_usize_field(&mut out, "workspace_bytes", self.workspace_bytes);
        out.push('}');
        comma(&mut out);
        out.push_str("\"runtime\":{");
        push_u128_array_field(&mut out, "elapsed_micros", &self.elapsed_micros);
        comma(&mut out);
        push_u128_field(&mut out, "min_elapsed_micros", self.min_elapsed_micros);
        comma(&mut out);
        push_u128_field(
            &mut out,
            "median_elapsed_micros",
            self.median_elapsed_micros,
        );
        comma(&mut out);
        push_u128_field(&mut out, "mean_elapsed_micros", self.mean_elapsed_micros);
        comma(&mut out);
        push_u128_field(&mut out, "max_elapsed_micros", self.max_elapsed_micros);
        comma(&mut out);
        push_u128_field(
            &mut out,
            "median_tokens_per_second_x100",
            self.median_tokens_per_second_x100,
        );
        out.push('}');
        comma(&mut out);
        out.push_str("\"saturation\":{");
        push_usize_field(&mut out, "total_residual", self.residual_saturation_count);
        comma(&mut out);
        push_usize_field(&mut out, "final_tensor", self.final_saturation_count);
        out.push('}');
        comma(&mut out);
        push_hash_field(&mut out, "input_hash", self.input_hash);
        comma(&mut out);
        push_hash_field(&mut out, "output_hash", self.output_hash);
        comma(&mut out);
        push_i16_field(&mut out, "final_max_abs", self.final_max_abs);
        comma(&mut out);
        push_string_array_field(&mut out, "known_non_claims", &KNOWN_NON_CLAIMS);
        out.push('}');
        out.push('\n');
        out
    }
}

impl BenchmarkTrace {
    pub fn to_json_line(&self) -> String {
        let mut out = String::new();
        out.push('{');
        push_string_field(&mut out, "schema", BENCH_SCHEMA);
        comma(&mut out);
        push_string_field(&mut out, "authority", AUTHORITY);
        comma(&mut out);
        push_string_field(&mut out, "preset", self.preset.name);
        comma(&mut out);
        push_string_field(
            &mut out,
            "linear_kernel",
            linear_kernel_name(self.linear_kernel),
        );
        comma(&mut out);
        out.push_str("\"arithmetic\":{");
        push_usize_field(&mut out, "residual_scale_q", 15);
        comma(&mut out);
        push_string_field(&mut out, "rounding", "branchless_round_half_up");
        comma(&mut out);
        push_string_field(&mut out, "attention_temperature", "native_log2");
        comma(&mut out);
        push_string_field(&mut out, "mlp_activation", "hard_silu_shift2_q15");
        comma(&mut out);
        push_usize_field(&mut out, "logit_frac_bits", usize::from(LOGIT_FRAC_BITS));
        comma(&mut out);
        push_usize_field(&mut out, "probability_shift", usize::from(Q15_SHIFT));
        out.push('}');
        comma(&mut out);
        out.push_str("\"model\":{");
        push_usize_field(&mut out, "blocks", self.preset.blocks);
        comma(&mut out);
        push_usize_field(&mut out, "seq_len", self.preset.seq_len);
        comma(&mut out);
        push_usize_field(&mut out, "d_model", self.preset.d_model);
        comma(&mut out);
        push_usize_field(&mut out, "heads", self.preset.heads);
        comma(&mut out);
        push_usize_field(&mut out, "head_dim", self.preset.head_dim());
        comma(&mut out);
        push_usize_field(
            &mut out,
            "sqrt_head_shift",
            usize::from(sqrt_power_of_four_shift(self.preset.head_dim()).unwrap_or(0)),
        );
        comma(&mut out);
        push_usize_field(&mut out, "hidden_dim", self.preset.hidden_dim);
        comma(&mut out);
        push_usize_field(&mut out, "i8_weight_count", self.i8_weight_count);
        comma(&mut out);
        push_hash_field(&mut out, "model_hash", self.model_hash);
        out.push('}');
        comma(&mut out);
        out.push_str("\"memory\":{");
        push_usize_field(&mut out, "parameter_bytes", self.parameter_bytes);
        comma(&mut out);
        push_usize_field(&mut out, "workspace_bytes", self.workspace_bytes);
        out.push('}');
        comma(&mut out);
        out.push_str("\"runtime\":{");
        push_u128_field(&mut out, "elapsed_micros", self.elapsed_micros);
        comma(&mut out);
        push_u128_field(
            &mut out,
            "tokens_per_second_x100",
            self.tokens_per_second_x100,
        );
        out.push('}');
        comma(&mut out);
        out.push_str("\"saturation\":{");
        push_usize_field(
            &mut out,
            "attention_residual",
            self.attention_residual_saturation_count,
        );
        comma(&mut out);
        push_usize_field(&mut out, "mlp_residual", self.mlp_residual_saturation_count);
        comma(&mut out);
        push_usize_field(&mut out, "total_residual", self.residual_saturation_count);
        comma(&mut out);
        push_usize_field(&mut out, "final_tensor", self.final_saturation_count);
        out.push('}');
        comma(&mut out);
        out.push_str("\"blocks\":[");
        for (index, block) in self.blocks.iter().enumerate() {
            if index != 0 {
                comma(&mut out);
            }
            block.push_json(&mut out);
        }
        out.push(']');
        comma(&mut out);
        push_hash_field(&mut out, "input_hash", self.input_hash);
        comma(&mut out);
        push_hash_field(&mut out, "output_hash", self.output_hash);
        comma(&mut out);
        push_i16_field(&mut out, "final_max_abs", self.final_max_abs);
        comma(&mut out);
        push_string_array_field(&mut out, "known_non_claims", &KNOWN_NON_CLAIMS);
        out.push('}');
        out.push('\n');
        out
    }
}

impl BlockBenchmarkTrace {
    fn push_json(&self, out: &mut String) {
        out.push('{');
        push_usize_field(out, "index", self.index);
        comma(out);
        push_usize_field(
            out,
            "attention_residual_saturation_count",
            self.attention_residual_saturation_count,
        );
        comma(out);
        push_usize_field(
            out,
            "mlp_residual_saturation_count",
            self.mlp_residual_saturation_count,
        );
        comma(out);
        push_hash_field(out, "output_hash", self.output_hash);
        comma(out);
        push_i16_field(out, "max_abs", self.max_abs);
        comma(out);
        push_usize_field(out, "saturation_count", self.saturation_count);
        out.push('}');
    }
}

struct GeneratedModel {
    blocks: Vec<BlockWeights>,
    model_hash: u64,
    i8_weight_count: usize,
    parameter_bytes: usize,
}

struct BlockWeights {
    q: Vec<i8>,
    k: Vec<i8>,
    v: Vec<i8>,
    o: Vec<i8>,
    up: Vec<i8>,
    gate: Vec<i8>,
    down: Vec<i8>,
    qk_scales: Vec<FixedScale>,
    v_scales: Vec<FixedScale>,
    o_scales: Vec<FixedScale>,
    up_scales: Vec<FixedScale>,
    gate_scales: Vec<FixedScale>,
    down_scales: Vec<FixedScale>,
}

impl BlockWeights {
    fn attention_params(&self, preset: BenchmarkPreset) -> SelfAttentionI16Params<'_> {
        SelfAttentionI16Params {
            q: LinearI16I8Params {
                weights: &self.q,
                bias: None,
                scales: &self.qk_scales,
                input_dim: preset.d_model,
                output_dim: preset.d_model,
            },
            k: LinearI16I8Params {
                weights: &self.k,
                bias: None,
                scales: &self.qk_scales,
                input_dim: preset.d_model,
                output_dim: preset.d_model,
            },
            v: LinearI16I8Params {
                weights: &self.v,
                bias: None,
                scales: &self.v_scales,
                input_dim: preset.d_model,
                output_dim: preset.d_model,
            },
            o: LinearI16I8Params {
                weights: &self.o,
                bias: None,
                scales: &self.o_scales,
                input_dim: preset.d_model,
                output_dim: preset.d_model,
            },
            seq_len: preset.seq_len,
            d_model: preset.d_model,
            heads: preset.heads,
            causal: true,
        }
    }

    fn mlp_params(&self, preset: BenchmarkPreset) -> GatedMlpI16Params<'_> {
        GatedMlpI16Params {
            up: LinearI16I8Params {
                weights: &self.up,
                bias: None,
                scales: &self.up_scales,
                input_dim: preset.d_model,
                output_dim: preset.hidden_dim,
            },
            gate: LinearI16I8Params {
                weights: &self.gate,
                bias: None,
                scales: &self.gate_scales,
                input_dim: preset.d_model,
                output_dim: preset.hidden_dim,
            },
            down: LinearI16I8Params {
                weights: &self.down,
                bias: None,
                scales: &self.down_scales,
                input_dim: preset.hidden_dim,
                output_dim: preset.d_model,
            },
            seq_len: preset.seq_len,
            d_model: preset.d_model,
            hidden_dim: preset.hidden_dim,
        }
    }
}

fn validate_preset(preset: BenchmarkPreset) -> Result<(), BenchmarkError> {
    if preset.seq_len == 0
        || preset.d_model == 0
        || preset.heads == 0
        || preset.hidden_dim == 0
        || preset.blocks == 0
        || preset.d_model % preset.heads != 0
        || sqrt_power_of_four_shift(preset.head_dim()).is_none()
    {
        return Err(BenchmarkError::InvalidPreset);
    }

    Ok(())
}

fn generate_model(preset: BenchmarkPreset) -> GeneratedModel {
    let mut rng = SplitMix64::new(preset.seed);
    let mut blocks = Vec::with_capacity(preset.blocks);
    let mut model_hash = StableHasher::new();
    model_hash.update_str(BENCH_SCHEMA);
    model_hash.update_str(preset.name);
    model_hash.update_usize(preset.seq_len);
    model_hash.update_usize(preset.d_model);
    model_hash.update_usize(preset.heads);
    model_hash.update_usize(preset.hidden_dim);
    model_hash.update_usize(preset.blocks);

    let mut i8_weight_count = 0_usize;
    let mut parameter_bytes = 0_usize;

    for _ in 0..preset.blocks {
        let d_square = preset.d_model * preset.d_model;
        let up_len = preset.hidden_dim * preset.d_model;
        let down_len = preset.d_model * preset.hidden_dim;
        let block = BlockWeights {
            q: ternary_weights(&mut rng, d_square),
            k: ternary_weights(&mut rng, d_square),
            v: ternary_weights(&mut rng, d_square),
            o: ternary_weights(&mut rng, d_square),
            up: ternary_weights(&mut rng, up_len),
            gate: ternary_weights(&mut rng, up_len),
            down: ternary_weights(&mut rng, down_len),
            qk_scales: vec![
                FixedScale {
                    multiplier: 1,
                    right_shift: 10,
                };
                preset.d_model
            ],
            v_scales: vec![
                FixedScale {
                    multiplier: 1,
                    right_shift: 7,
                };
                preset.d_model
            ],
            o_scales: vec![
                FixedScale {
                    multiplier: 1,
                    right_shift: 5,
                };
                preset.d_model
            ],
            up_scales: vec![
                FixedScale {
                    multiplier: 1,
                    right_shift: 9,
                };
                preset.hidden_dim
            ],
            gate_scales: vec![
                FixedScale {
                    multiplier: 1,
                    right_shift: 8,
                };
                preset.hidden_dim
            ],
            down_scales: vec![
                FixedScale {
                    multiplier: 1,
                    right_shift: 5,
                };
                preset.d_model
            ],
        };

        i8_weight_count += block.q.len()
            + block.k.len()
            + block.v.len()
            + block.o.len()
            + block.up.len()
            + block.gate.len()
            + block.down.len();
        parameter_bytes += bytes_i8(&block.q)
            + bytes_i8(&block.k)
            + bytes_i8(&block.v)
            + bytes_i8(&block.o)
            + bytes_i8(&block.up)
            + bytes_i8(&block.gate)
            + bytes_i8(&block.down)
            + bytes_scale(&block.qk_scales)
            + bytes_scale(&block.v_scales)
            + bytes_scale(&block.o_scales)
            + bytes_scale(&block.up_scales)
            + bytes_scale(&block.gate_scales)
            + bytes_scale(&block.down_scales);
        model_hash.update_i8_slice(&block.q);
        model_hash.update_i8_slice(&block.k);
        model_hash.update_i8_slice(&block.v);
        model_hash.update_i8_slice(&block.o);
        model_hash.update_i8_slice(&block.up);
        model_hash.update_i8_slice(&block.gate);
        model_hash.update_i8_slice(&block.down);
        update_scales_hash(&mut model_hash, &block.qk_scales);
        update_scales_hash(&mut model_hash, &block.v_scales);
        update_scales_hash(&mut model_hash, &block.o_scales);
        update_scales_hash(&mut model_hash, &block.up_scales);
        update_scales_hash(&mut model_hash, &block.gate_scales);
        update_scales_hash(&mut model_hash, &block.down_scales);
        blocks.push(block);
    }

    GeneratedModel {
        blocks,
        model_hash: model_hash.finish(),
        i8_weight_count,
        parameter_bytes,
    }
}

fn generate_input(preset: BenchmarkPreset, len: usize) -> Vec<i16> {
    let mut rng = SplitMix64::new(preset.seed ^ 0xa17e_5711_6d5a_1234);
    (0..len)
        .map(|index| {
            let token = (index / preset.d_model) as i16;
            let channel = (index % preset.d_model) as i16;
            let noise = ((rng.next() >> 53) as i16) - 512;
            ((token & 31) - 16) * 16 + ((channel & 15) - 8) * 8 + noise
        })
        .collect()
}

fn ternary_weights(rng: &mut SplitMix64, len: usize) -> Vec<i8> {
    (0..len)
        .map(|_| match rng.next() & 7 {
            0 => -1,
            1 => 1,
            _ => 0,
        })
        .collect()
}

#[derive(Debug, Clone, Copy)]
struct TensorHealth {
    max_abs: i16,
    saturation_count: usize,
}

fn tensor_health(values: &[i16]) -> TensorHealth {
    let mut max_abs = 0_i16;
    let mut saturation_count = 0_usize;
    for &value in values {
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
    TensorHealth {
        max_abs,
        saturation_count,
    }
}

fn bytes_i8(values: &[i8]) -> usize {
    core::mem::size_of_val(values)
}

fn bytes_i16(values: &[i16]) -> usize {
    core::mem::size_of_val(values)
}

fn bytes_i32(values: &[i32]) -> usize {
    core::mem::size_of_val(values)
}

fn bytes_scale(values: &[FixedScale]) -> usize {
    core::mem::size_of_val(values)
}

fn linear_kernel_name(kernel: LinearKernel) -> &'static str {
    match kernel {
        LinearKernel::GenericI8 => "generic_i8",
        LinearKernel::Ternary => "ternary",
    }
}

fn hash_i16_slice(values: &[i16]) -> u64 {
    let mut hasher = StableHasher::new();
    hasher.update_i16_slice(values);
    hasher.finish()
}

fn update_scales_hash(hasher: &mut StableHasher, scales: &[FixedScale]) {
    hasher.update_usize(scales.len());
    for scale in scales {
        hasher.update_i32(scale.multiplier);
        hasher.update_u8(scale.right_shift);
    }
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
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

fn push_u128_field(out: &mut String, name: &str, value: u128) {
    push_quoted(out, name);
    out.push(':');
    out.push_str(&value.to_string());
}

fn push_u128_array_field(out: &mut String, name: &str, values: &[u128]) {
    push_quoted(out, name);
    out.push(':');
    out.push('[');
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            comma(out);
        }
        out.push_str(&value.to_string());
    }
    out.push(']');
}

fn push_i16_field(out: &mut String, name: &str, value: i16) {
    push_quoted(out, name);
    out.push(':');
    out.push_str(&value.to_string());
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
