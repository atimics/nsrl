#![deny(unsafe_code)]

use std::time::Instant;

use nsrl_core::{
    FixedScale, LinearAttentionState, LinearAttentionStepWorkspace, LinearAttentionWorkspace,
    LinearI16I8Params, SelfAttentionI16Params, SelfAttentionWorkspace,
    clear_linear_attention_state_checked, linear_attention_i16_q15_checked,
    linear_attention_state_lengths, linear_attention_step_i16_q15_checked,
    self_attention_i16_q15_checked,
};

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Debug, Clone, Copy)]
struct Case {
    name: &'static str,
    seq_len: usize,
    d_model: usize,
    heads: usize,
    repeat: usize,
}

fn main() {
    for case in [
        Case {
            name: "seq128_d32_h2",
            seq_len: 128,
            d_model: 32,
            heads: 2,
            repeat: 40,
        },
        Case {
            name: "seq128_d128_h2",
            seq_len: 128,
            d_model: 128,
            heads: 2,
            repeat: 20,
        },
        Case {
            name: "seq512_d32_h2",
            seq_len: 512,
            d_model: 32,
            heads: 2,
            repeat: 8,
        },
    ] {
        let trace = run_case(case).expect("linear attention microbench");
        println!("{trace}");
    }
}

fn run_case(case: Case) -> Option<String> {
    let total = case.seq_len.checked_mul(case.d_model)?;
    let head_dim = case.d_model / case.heads;
    let input = generate_input(total, 0x5153_524c ^ case.d_model as u64);
    let weights = generate_weights(
        case.d_model.checked_mul(case.d_model)?,
        0x4c41_5454 ^ total as u64,
    );
    let scales = vec![
        FixedScale {
            multiplier: 1,
            right_shift: 4,
        };
        case.d_model
    ];
    let linear = LinearI16I8Params {
        weights: &weights,
        bias: None,
        scales: &scales,
        input_dim: case.d_model,
        output_dim: case.d_model,
    };
    let params = SelfAttentionI16Params {
        q: linear,
        k: linear,
        v: linear,
        o: linear,
        seq_len: case.seq_len,
        d_model: case.d_model,
        heads: case.heads,
        causal: true,
    };

    let mut softmax = SoftmaxBuffers::new(case.seq_len, case.d_model);
    let mut linear_buffers = LinearBuffers::new(case.seq_len, case.d_model, case.heads);
    let mut incremental_buffers = IncrementalLinearBuffers::new(case.d_model, case.heads)?;
    let mut softmax_output = vec![0_i16; total];
    let mut linear_output = vec![0_i16; total];
    let mut incremental_output = vec![0_i16; total];
    let mut rescan_output = vec![0_i16; total];

    for _ in 0..3 {
        run_softmax_attention(&input, params, &mut softmax, &mut softmax_output)?;
        run_linear_attention(&input, params, &mut linear_buffers, &mut linear_output)?;
        run_incremental_linear_attention(
            &input,
            params,
            &mut incremental_buffers,
            &mut incremental_output,
        )?;
    }

    let mut softmax_times = Vec::with_capacity(case.repeat);
    for _ in 0..case.repeat {
        let started = Instant::now();
        run_softmax_attention(&input, params, &mut softmax, &mut softmax_output)?;
        softmax_times.push(started.elapsed().as_nanos());
    }

    let mut linear_times = Vec::with_capacity(case.repeat);
    for _ in 0..case.repeat {
        let started = Instant::now();
        run_linear_attention(&input, params, &mut linear_buffers, &mut linear_output)?;
        linear_times.push(started.elapsed().as_nanos());
    }

    let mut incremental_times = Vec::with_capacity(case.repeat);
    for _ in 0..case.repeat {
        let started = Instant::now();
        run_incremental_linear_attention(
            &input,
            params,
            &mut incremental_buffers,
            &mut incremental_output,
        )?;
        incremental_times.push(started.elapsed().as_nanos());
    }

    let generation_repeat = (case.repeat / 5).max(3);
    let mut rescan_times = Vec::with_capacity(generation_repeat);
    for _ in 0..generation_repeat {
        let started = Instant::now();
        run_rescan_linear_generation(&input, params, &mut linear_buffers, &mut rescan_output)?;
        rescan_times.push(started.elapsed().as_nanos());
    }

    softmax_times.sort_unstable();
    linear_times.sort_unstable();
    incremental_times.sort_unstable();
    rescan_times.sort_unstable();
    let softmax_median_nanos = softmax_times[softmax_times.len() / 2];
    let linear_median_nanos = linear_times[linear_times.len() / 2];
    let incremental_median_nanos = incremental_times[incremental_times.len() / 2];
    let rescan_median_nanos = rescan_times[rescan_times.len() / 2];
    let speedup_x100 = if linear_median_nanos == 0 {
        0
    } else {
        softmax_median_nanos.saturating_mul(100) / linear_median_nanos
    };
    let rescan_to_incremental_speedup_x100 = if incremental_median_nanos == 0 {
        0
    } else {
        rescan_median_nanos.saturating_mul(100) / incremental_median_nanos
    };

    Some(format!(
        "{{\"schema\":\"nsrl.linear_attention_microbench.v2\",\"case\":\"{}\",\"seq_len\":{},\"d_model\":{},\"heads\":{},\"head_dim\":{},\"repeat\":{},\"generation_repeat\":{},\"softmax_median_micros\":{},\"linear_median_micros\":{},\"incremental_linear_median_micros\":{},\"rescan_generation_median_micros\":{},\"softmax_to_linear_speedup_x100\":{},\"rescan_to_incremental_speedup_x100\":{},\"softmax_workspace_bytes\":{},\"linear_workspace_bytes\":{},\"incremental_workspace_bytes\":{},\"linear_state_bytes\":{},\"linear_key_sum_bytes\":{},\"softmax_output_hash\":\"0x{:016x}\",\"linear_output_hash\":\"0x{:016x}\",\"incremental_output_hash\":\"0x{:016x}\",\"rescan_output_hash\":\"0x{:016x}\"}}",
        case.name,
        case.seq_len,
        case.d_model,
        case.heads,
        head_dim,
        case.repeat,
        generation_repeat,
        softmax_median_nanos / 1_000,
        linear_median_nanos / 1_000,
        incremental_median_nanos / 1_000,
        rescan_median_nanos / 1_000,
        speedup_x100,
        rescan_to_incremental_speedup_x100,
        softmax.workspace_bytes(),
        linear_buffers.workspace_bytes(),
        incremental_buffers.workspace_bytes(),
        linear_buffers.state_kv.len() * core::mem::size_of::<i64>(),
        linear_buffers.key_sums.len() * core::mem::size_of::<i64>(),
        hash_i16(&softmax_output),
        hash_i16(&linear_output),
        hash_i16(&incremental_output),
        hash_i16(&rescan_output),
    ))
}

fn run_softmax_attention(
    input: &[i16],
    params: SelfAttentionI16Params<'_>,
    buffers: &mut SoftmaxBuffers,
    output: &mut [i16],
) -> Option<()> {
    let workspace = SelfAttentionWorkspace {
        q: &mut buffers.q,
        k: &mut buffers.k,
        v: &mut buffers.v,
        context: &mut buffers.context,
        logits_q8: &mut buffers.logits_q8,
        probabilities_q15: &mut buffers.probabilities_q15,
    };
    self_attention_i16_q15_checked(input, params, workspace, output)
}

fn run_linear_attention(
    input: &[i16],
    params: SelfAttentionI16Params<'_>,
    buffers: &mut LinearBuffers,
    output: &mut [i16],
) -> Option<()> {
    let workspace = LinearAttentionWorkspace {
        q: &mut buffers.q,
        k: &mut buffers.k,
        v: &mut buffers.v,
        context: &mut buffers.context,
        state_kv: &mut buffers.state_kv,
        key_sums: &mut buffers.key_sums,
    };
    linear_attention_i16_q15_checked(input, params, workspace, output)
}

fn run_rescan_linear_generation(
    input: &[i16],
    params: SelfAttentionI16Params<'_>,
    buffers: &mut LinearBuffers,
    output: &mut [i16],
) -> Option<()> {
    let mut scratch = vec![0_i16; output.len()];
    for token in 0..params.seq_len {
        let prefix_len = token.checked_add(1)?;
        let prefix_total = prefix_len.checked_mul(params.d_model)?;
        let prefix_params = SelfAttentionI16Params {
            seq_len: prefix_len,
            ..params
        };
        let workspace = LinearAttentionWorkspace {
            q: &mut buffers.q[..prefix_total],
            k: &mut buffers.k[..prefix_total],
            v: &mut buffers.v[..prefix_total],
            context: &mut buffers.context[..prefix_total],
            state_kv: &mut buffers.state_kv,
            key_sums: &mut buffers.key_sums,
        };
        linear_attention_i16_q15_checked(
            &input[..prefix_total],
            prefix_params,
            workspace,
            &mut scratch[..prefix_total],
        )?;
        let row_start = token.checked_mul(params.d_model)?;
        let row_end = row_start.checked_add(params.d_model)?;
        output[row_start..row_end].copy_from_slice(&scratch[row_start..row_end]);
    }
    Some(())
}

fn run_incremental_linear_attention(
    input: &[i16],
    params: SelfAttentionI16Params<'_>,
    buffers: &mut IncrementalLinearBuffers,
    output: &mut [i16],
) -> Option<()> {
    if input.len() != params.seq_len.checked_mul(params.d_model)?
        || output.len() != input.len()
        || !params.causal
    {
        return None;
    }
    clear_linear_attention_state_checked(
        params.d_model,
        params.heads,
        LinearAttentionState {
            state_kv: &mut buffers.state_kv,
            key_sums: &mut buffers.key_sums,
        },
    )?;
    for token in 0..params.seq_len {
        let row_start = token.checked_mul(params.d_model)?;
        let row_end = row_start.checked_add(params.d_model)?;
        linear_attention_step_i16_q15_checked(
            &input[row_start..row_end],
            params,
            LinearAttentionStepWorkspace {
                q: &mut buffers.q,
                k: &mut buffers.k,
                v: &mut buffers.v,
                context: &mut buffers.context,
            },
            LinearAttentionState {
                state_kv: &mut buffers.state_kv,
                key_sums: &mut buffers.key_sums,
            },
            &mut output[row_start..row_end],
        )?;
    }
    Some(())
}

struct SoftmaxBuffers {
    q: Vec<i16>,
    k: Vec<i16>,
    v: Vec<i16>,
    context: Vec<i16>,
    logits_q8: Vec<i32>,
    probabilities_q15: Vec<i16>,
}

impl SoftmaxBuffers {
    fn new(seq_len: usize, d_model: usize) -> Self {
        let total = seq_len * d_model;
        Self {
            q: vec![0; total],
            k: vec![0; total],
            v: vec![0; total],
            context: vec![0; total],
            logits_q8: vec![0; seq_len],
            probabilities_q15: vec![0; seq_len],
        }
    }

    fn workspace_bytes(&self) -> usize {
        bytes_i16(&self.q)
            + bytes_i16(&self.k)
            + bytes_i16(&self.v)
            + bytes_i16(&self.context)
            + self.logits_q8.len() * core::mem::size_of::<i32>()
            + bytes_i16(&self.probabilities_q15)
    }
}

struct LinearBuffers {
    q: Vec<i16>,
    k: Vec<i16>,
    v: Vec<i16>,
    context: Vec<i16>,
    state_kv: Vec<i64>,
    key_sums: Vec<i64>,
}

impl LinearBuffers {
    fn new(seq_len: usize, d_model: usize, heads: usize) -> Self {
        let total = seq_len * d_model;
        let head_dim = d_model / heads;
        Self {
            q: vec![0; total],
            k: vec![0; total],
            v: vec![0; total],
            context: vec![0; total],
            state_kv: vec![0; heads * head_dim * head_dim],
            key_sums: vec![0; heads * head_dim],
        }
    }

    fn workspace_bytes(&self) -> usize {
        bytes_i16(&self.q)
            + bytes_i16(&self.k)
            + bytes_i16(&self.v)
            + bytes_i16(&self.context)
            + self.state_kv.len() * core::mem::size_of::<i64>()
            + self.key_sums.len() * core::mem::size_of::<i64>()
    }
}

struct IncrementalLinearBuffers {
    q: Vec<i16>,
    k: Vec<i16>,
    v: Vec<i16>,
    context: Vec<i16>,
    state_kv: Vec<i64>,
    key_sums: Vec<i64>,
}

impl IncrementalLinearBuffers {
    fn new(d_model: usize, heads: usize) -> Option<Self> {
        let (state_len, key_sum_len) = linear_attention_state_lengths(d_model, heads)?;
        Some(Self {
            q: vec![0; d_model],
            k: vec![0; d_model],
            v: vec![0; d_model],
            context: vec![0; d_model],
            state_kv: vec![0; state_len],
            key_sums: vec![0; key_sum_len],
        })
    }

    fn workspace_bytes(&self) -> usize {
        bytes_i16(&self.q)
            + bytes_i16(&self.k)
            + bytes_i16(&self.v)
            + bytes_i16(&self.context)
            + self.state_kv.len() * core::mem::size_of::<i64>()
            + self.key_sums.len() * core::mem::size_of::<i64>()
    }
}

fn generate_input(len: usize, seed: u64) -> Vec<i16> {
    let mut state = seed;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state = xorshift64(state);
        out.push(((state >> 16) as i16 % 4096).clamp(-4096, 4095));
    }
    out
}

fn generate_weights(len: usize, seed: u64) -> Vec<i8> {
    let mut state = seed;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state = xorshift64(state);
        out.push(((state >> 24) as i8 % 5).clamp(-2, 2));
    }
    out
}

fn xorshift64(mut value: u64) -> u64 {
    value ^= value << 13;
    value ^= value >> 7;
    value ^= value << 17;
    value
}

fn bytes_i16(values: &[i16]) -> usize {
    values.len() * core::mem::size_of::<i16>()
}

fn hash_i16(values: &[i16]) -> u64 {
    let mut hash = FNV_OFFSET;
    for &value in values {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    hash
}
