//! Criterion benchmarks for the integer linear (fully-connected) layer.
//!
//! Run with:
//!
//! ```bash
//! cargo bench -p nsrl-core --bench linear
//! ```

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use nsrl_core::{
    FixedScale, LinearI16I8Params, LinearKernel, linear_i16_i8_i16_per_channel_with_kernel_checked,
};

fn bench_linear_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear/forward_i16_i8_i16");

    for &(d_in, d_out) in &[(128, 256), (256, 256), (256, 1024), (1024, 4096)] {
        let weights: Vec<i8> = (0..d_out * d_in)
            .map(|i| (i as i8).wrapping_mul(13))
            .collect();
        let scales: Vec<FixedScale> = vec![
            FixedScale {
                multiplier: 1,
                right_shift: 2
            };
            d_out
        ];
        let bias: Vec<i32> = vec![0_i32; d_out];
        let input: Vec<i16> = (0..d_in)
            .map(|i| ((i as i32 * 113) % 65536 - 32768) as i16)
            .collect();
        let mut output = vec![0_i16; d_out];

        let params = LinearI16I8Params {
            weights: &weights,
            scales: &scales,
            bias: Some(&bias),
            input_dim: d_in,
            output_dim: d_out,
        };

        group.bench_with_input(
            BenchmarkId::new("generic_i8", format!("{d_in}x{d_out}")),
            &params,
            |b, params| {
                b.iter(|| {
                    linear_i16_i8_i16_per_channel_with_kernel_checked(
                        black_box(&input),
                        *black_box(params),
                        black_box(&mut output),
                        LinearKernel::GenericI8,
                    )
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_linear_forward);
criterion_main!(benches);
