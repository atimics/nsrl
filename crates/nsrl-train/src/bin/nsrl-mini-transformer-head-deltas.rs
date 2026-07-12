#![deny(unsafe_code)]

use std::env;
use std::fs;
use std::path::PathBuf;

use nsrl_train::{
    MINI_TRANSFORMER_ARCHITECTURE_PROFILE, MINI_TRANSFORMER_D_MODEL, MINI_TRANSFORMER_HEADS,
    MiniTransformerMlpModel,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("nsrl-mini-transformer-head-deltas: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let mut model_path = None;
    let mut output_path = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model_path = Some(PathBuf::from(required(&mut args, "--model")?)),
            "--out" => output_path = Some(PathBuf::from(required(&mut args, "--out")?)),
            "--help" | "-h" => {
                println!("Usage: nsrl-mini-transformer-head-deltas --model PATH --out PATH");
                return Ok(());
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }
    let model =
        MiniTransformerMlpModel::from_bytes(&fs::read(model_path.ok_or("--model is required")?)?)?;
    let initial = MiniTransformerMlpModel::new_initial_with_seq_len_and_layers(
        model.context_seq_len,
        model.transformer_layers(),
    )?;
    let head_dim = MINI_TRANSFORMER_D_MODEL / MINI_TRANSFORMER_HEADS;
    let mut rows = Vec::new();
    for head in 0..MINI_TRANSFORMER_HEADS {
        rows.push(head_delta(&initial, &model, head, head_dim));
    }
    let heads_json = rows
        .iter()
        .enumerate()
        .map(|(head, delta)| {
            format!(
                "{{\"head\":{},\"q\":{},\"k\":{},\"v\":{},\"o\":{},\"total\":{}}}",
                head,
                delta[0],
                delta[1],
                delta[2],
                delta[3],
                delta.iter().sum::<u64>()
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    fs::write(
        output_path.ok_or("--out is required")?,
        format!(
            "{{\"schema\":\"nsrl.mini_transformer_head_deltas.v1\",\"architecture_profile\":\"{}\",\"d_model\":{},\"heads\":{},\"head_dim\":{},\"layers\":{},\"head_deltas\":[{}],\"all_heads_updated\":{}}}\n",
            MINI_TRANSFORMER_ARCHITECTURE_PROFILE,
            MINI_TRANSFORMER_D_MODEL,
            MINI_TRANSFORMER_HEADS,
            head_dim,
            model.transformer_layers(),
            heads_json,
            rows.iter().all(|delta| delta.iter().sum::<u64>() > 0),
        ),
    )?;
    Ok(())
}

fn head_delta(
    initial: &MiniTransformerMlpModel,
    model: &MiniTransformerMlpModel,
    head: usize,
    head_dim: usize,
) -> [u64; 4] {
    let mut output = [0_u64; 4];
    let matrices = [
        (&initial.q_weights, &model.q_weights),
        (&initial.k_weights, &model.k_weights),
        (&initial.v_weights, &model.v_weights),
    ];
    let matrix_len = MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_D_MODEL;
    for (matrix_index, (before, after)) in matrices.into_iter().enumerate() {
        for layer in 0..model.transformer_layers() {
            let layer_start = layer * matrix_len;
            for row in head * head_dim..(head + 1) * head_dim {
                let start = layer_start + row * MINI_TRANSFORMER_D_MODEL;
                for index in start..start + MINI_TRANSFORMER_D_MODEL {
                    output[matrix_index] +=
                        (i16::from(after[index]) - i16::from(before[index])).unsigned_abs() as u64;
                }
            }
        }
    }
    for layer in 0..model.transformer_layers() {
        let layer_start = layer * matrix_len;
        for row in 0..MINI_TRANSFORMER_D_MODEL {
            for column in head * head_dim..(head + 1) * head_dim {
                let index = layer_start + row * MINI_TRANSFORMER_D_MODEL + column;
                output[3] += (i16::from(model.o_weights[index])
                    - i16::from(initial.o_weights[index]))
                .unsigned_abs() as u64;
            }
        }
    }
    output
}

fn required(
    args: &mut impl Iterator<Item = String>,
    option: &'static str,
) -> Result<String, Box<dyn std::error::Error>> {
    args.next()
        .ok_or_else(|| format!("{option} requires a value").into())
}
