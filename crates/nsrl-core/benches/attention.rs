//! Criterion benchmarks for the attention hot path.
//!
//! These exercise the same integer kernels used in production forward passes.
//! Run with:
//!
//! ```bash
//! cargo bench -p nsrl-core --bench attention
//! ```

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use nsrl_core::{
    attention_dot_q_k_i16_i32_checked, attention_weight_v_i16_q15_checked, base2_softmax_i32_q15,
};

fn bench_qk_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention/qk_dot");
    for &head_dim in &[64, 128, 256] {
        let query: Vec<i16> = (0..head_dim)
            .map(|i| ((i as i16 * 127) % 16384) - 8192)
            .collect();
        let key: Vec<i16> = (0..head_dim)
            .map(|i| ((i as i16 * 103) % 16384) - 8192)
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(head_dim),
            &(&query, &key),
            |b, (q, k)| {
                b.iter(|| attention_dot_q_k_i16_i32_checked(black_box(q), black_box(k)));
            },
        );
    }
    group.finish();
}

fn bench_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention/base2_softmax");
    for &vocab_size in &[256, 2048, 8192] {
        // Fill with Q8 logits roughly centred around zero.
        let logits: Vec<i32> = (0..vocab_size)
            .map(|i| ((i as i32 * 7919) % 4096) - 2048)
            .collect();
        let mut output = vec![0_i16; vocab_size];

        group.bench_with_input(
            BenchmarkId::from_parameter(vocab_size),
            &vocab_size,
            |b, &vocab_size| {
                b.iter(|| {
                    base2_softmax_i32_q15(
                        black_box(&logits[..vocab_size]),
                        black_box(&mut output[..vocab_size]),
                    )
                });
            },
        );
    }
    group.finish();
}

fn bench_weighted_value_agg(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention/weighted_value_agg");
    for &seq_len in &[64, 256, 1024] {
        let value_dim = 128;
        let probs: Vec<i16> = vec![(i16::MAX / seq_len as i16).max(1); seq_len];
        let values: Vec<i16> = (0..seq_len * value_dim)
            .map(|i| ((i as i32 * 73) % 65536 - 32768) as i16)
            .collect();
        let mut output = vec![0_i16; value_dim];

        group.bench_with_input(BenchmarkId::from_parameter(seq_len), &seq_len, |b, _| {
            b.iter(|| {
                attention_weight_v_i16_q15_checked(
                    black_box(&probs),
                    black_box(&values),
                    black_box(value_dim),
                    black_box(&mut output),
                )
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_qk_dot,
    bench_softmax,
    bench_weighted_value_agg
);
criterion_main!(benches);
