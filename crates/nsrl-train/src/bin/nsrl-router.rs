#![deny(unsafe_code)]

use std::env;
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

use nsrl_core::base2_softmax_i32_q15;

const MAGIC: &[u8; 8] = b"NSRLRT1\n";
const FEATURE_COUNT: usize = 41;
const HIDDEN_DIM: usize = 16;
const OUTPUT_DIM: usize = 3;
const OUTPUT_LOGIT_SHIFT: u8 = 8;
const OUTPUT_WEIGHT_GRAD_SHIFT: u8 = 27;
const INPUT_WEIGHT_GRAD_SHIFT: u8 = 25;
const OUTPUT_BIAS_GRAD_SHIFT: u8 = 24;
const HIDDEN_BIAS_GRAD_SHIFT: u8 = 18;
const UTILITY_LOSS_LOGIT_SHIFT: u8 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RouterObjective {
    HardLabel,
    UtilitySoft,
    ExpectedRegret,
}

impl RouterObjective {
    fn as_str(self) -> &'static str {
        match self {
            Self::HardLabel => "hard_label",
            Self::UtilitySoft => "utility_soft",
            Self::ExpectedRegret => "expected_regret",
        }
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("nsrl-router: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let command = args.next().ok_or("expected train or eval")?;
    match command.as_str() {
        "train" => run_train(args),
        "eval" => run_eval(args),
        "--help" | "-h" => {
            print_help();
            Ok(())
        }
        _ => Err("expected train or eval".into()),
    }
}

fn run_train(mut args: impl Iterator<Item = String>) -> Result<(), Box<dyn std::error::Error>> {
    let mut train_path = None;
    let mut calibration_path = None;
    let mut model_out = None;
    let mut trace_out = None;
    let mut predictions_out = None;
    let mut feature_spec = String::from("0-40");
    let mut epochs = 64_usize;
    let mut seed = 1_u64;
    let mut objective = RouterObjective::HardLabel;
    let mut regret_gradient_shift = 3_u8;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--train" => train_path = Some(PathBuf::from(required(&mut args, "--train")?)),
            "--calibration" => {
                calibration_path = Some(PathBuf::from(required(&mut args, "--calibration")?))
            }
            "--model-out" => model_out = Some(PathBuf::from(required(&mut args, "--model-out")?)),
            "--trace" => trace_out = Some(PathBuf::from(required(&mut args, "--trace")?)),
            "--predictions-out" => {
                predictions_out = Some(PathBuf::from(required(&mut args, "--predictions-out")?))
            }
            "--features" => feature_spec = required(&mut args, "--features")?,
            "--epochs" => epochs = required(&mut args, "--epochs")?.parse()?,
            "--seed" => seed = required(&mut args, "--seed")?.parse()?,
            "--objective" => {
                objective = match required(&mut args, "--objective")?.as_str() {
                    "hard-label" => RouterObjective::HardLabel,
                    "utility-soft" => RouterObjective::UtilitySoft,
                    "expected-regret" => RouterObjective::ExpectedRegret,
                    _ => {
                        return Err(
                            "--objective requires hard-label, utility-soft, or expected-regret"
                                .into(),
                        );
                    }
                }
            }
            "--regret-gradient-shift" => {
                regret_gradient_shift = required(&mut args, "--regret-gradient-shift")?.parse()?
            }
            other => return Err(format!("unknown train argument: {other}").into()),
        }
    }
    if epochs == 0 || regret_gradient_shift > 30 {
        return Err("--epochs must be positive and --regret-gradient-shift at most 30".into());
    }
    let train_path = train_path.ok_or("--train is required")?;
    let calibration_path = calibration_path.ok_or("--calibration is required")?;
    let model_out = model_out.ok_or("--model-out is required")?;
    let trace_out = trace_out.ok_or("--trace is required")?;
    let feature_indices = parse_feature_indices(&feature_spec)?;
    let train_rows = read_dataset(&train_path)?;
    let calibration_rows = read_dataset(&calibration_path)?;
    let mut model = RouterModel::new(feature_indices, seed);
    let initial_train = evaluate(&model, &train_rows)?;
    let initial_calibration = evaluate(&model, &calibration_rows)?;
    let update = train(
        &mut model,
        &train_rows,
        epochs,
        objective,
        regret_gradient_shift,
    )?;
    let final_train = evaluate(&model, &train_rows)?;
    let final_calibration = evaluate(&model, &calibration_rows)?;
    fs::write(&model_out, model.to_bytes()?)?;
    fs::write(
        &trace_out,
        training_trace_json(
            &model,
            epochs,
            objective,
            regret_gradient_shift,
            update,
            initial_train,
            final_train,
            initial_calibration,
            final_calibration,
        ),
    )?;
    if let Some(path) = predictions_out {
        write_predictions(&path, &model, &calibration_rows)?;
    }
    Ok(())
}

fn run_eval(mut args: impl Iterator<Item = String>) -> Result<(), Box<dyn std::error::Error>> {
    let mut data_path = None;
    let mut model_path = None;
    let mut trace_out = None;
    let mut predictions_out = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--data" => data_path = Some(PathBuf::from(required(&mut args, "--data")?)),
            "--model" => model_path = Some(PathBuf::from(required(&mut args, "--model")?)),
            "--trace" => trace_out = Some(PathBuf::from(required(&mut args, "--trace")?)),
            "--predictions-out" => {
                predictions_out = Some(PathBuf::from(required(&mut args, "--predictions-out")?))
            }
            other => return Err(format!("unknown eval argument: {other}").into()),
        }
    }
    let data_path = data_path.ok_or("--data is required")?;
    let model_path = model_path.ok_or("--model is required")?;
    let trace_out = trace_out.ok_or("--trace is required")?;
    let model = RouterModel::from_bytes(&fs::read(model_path)?)?;
    let rows = read_dataset(&data_path)?;
    let metrics = evaluate(&model, &rows)?;
    fs::write(&trace_out, eval_trace_json(&model, metrics))?;
    if let Some(path) = predictions_out {
        write_predictions(&path, &model, &rows)?;
    }
    Ok(())
}

fn print_help() {
    println!("Usage:");
    println!(
        "  nsrl-router train --train DATA.tsv --calibration DATA.tsv --features 0-23|24-40|0-40 --epochs N --seed N [--objective hard-label|utility-soft|expected-regret] [--regret-gradient-shift N] --model-out PATH --trace PATH [--predictions-out PATH]"
    );
    println!(
        "  nsrl-router eval --data DATA.tsv --model PATH --trace PATH [--predictions-out PATH]"
    );
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RouterRow {
    sample_id: String,
    target: usize,
    features_q15: [i16; FEATURE_COUNT],
    child_losses_q15: [usize; OUTPUT_DIM],
}

fn read_dataset(path: &PathBuf) -> Result<Vec<RouterRow>, Box<dyn std::error::Error>> {
    let reader = BufReader::new(fs::File::open(path)?);
    let mut rows = Vec::new();
    for (line_index, line) in reader.lines().enumerate() {
        let line = line?;
        if line_index == 0 && line == "sample_id\ttarget\tfeatures_q15\toracle_child_losses_q15" {
            continue;
        }
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != 4 || fields[0].is_empty() {
            return Err(format!("invalid router row at line {}", line_index + 1).into());
        }
        let target = fields[1].parse::<usize>()?;
        if target >= OUTPUT_DIM {
            return Err("router target is out of range".into());
        }
        let features = parse_i16_csv(fields[2])?;
        let losses = parse_usize_csv(fields[3])?;
        if features.len() != FEATURE_COUNT || losses.len() != OUTPUT_DIM {
            return Err("router row has the wrong feature/loss shape".into());
        }
        let mut features_q15 = [0_i16; FEATURE_COUNT];
        features_q15.copy_from_slice(&features);
        let mut child_losses_q15 = [0_usize; OUTPUT_DIM];
        child_losses_q15.copy_from_slice(&losses);
        rows.push(RouterRow {
            sample_id: fields[0].to_string(),
            target,
            features_q15,
            child_losses_q15,
        });
    }
    if rows.is_empty() {
        return Err("router dataset contains no rows".into());
    }
    Ok(rows)
}

fn parse_i16_csv(value: &str) -> Result<Vec<i16>, Box<dyn std::error::Error>> {
    value
        .split(',')
        .map(|item| item.parse::<i16>().map_err(Into::into))
        .collect()
}

fn parse_usize_csv(value: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    value
        .split(',')
        .map(|item| item.parse::<usize>().map_err(Into::into))
        .collect()
}

fn parse_feature_indices(value: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let (start, end) = value
        .split_once('-')
        .ok_or("--features requires START-END")?;
    let start = start.parse::<usize>()?;
    let end = end.parse::<usize>()?;
    if start > end || end >= FEATURE_COUNT {
        return Err("feature range is invalid".into());
    }
    Ok((start..=end).collect())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RouterModel {
    feature_indices: Vec<usize>,
    seed: u64,
    input_weights: Vec<i8>,
    hidden_bias_q15: [i32; HIDDEN_DIM],
    output_weights: [i8; HIDDEN_DIM * OUTPUT_DIM],
    output_bias_q8: [i32; OUTPUT_DIM],
}

impl RouterModel {
    fn new(feature_indices: Vec<usize>, seed: u64) -> Self {
        let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
        let mut next_weight = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 32) % 9) as i8 - 4
        };
        let input_weights = (0..feature_indices.len() * HIDDEN_DIM)
            .map(|_| next_weight())
            .collect();
        let mut output_weights = [0_i8; HIDDEN_DIM * OUTPUT_DIM];
        for value in &mut output_weights {
            *value = next_weight();
        }
        Self {
            feature_indices,
            seed,
            input_weights,
            hidden_bias_q15: [0; HIDDEN_DIM],
            output_weights,
            output_bias_q8: [0; OUTPUT_DIM],
        }
    }

    fn forward(&self, features: &[i16; FEATURE_COUNT]) -> Result<RouterForward, &'static str> {
        let mut hidden = [0_i16; HIDDEN_DIM];
        for (hidden_index, hidden_value) in hidden.iter_mut().enumerate() {
            let mut acc = i64::from(self.hidden_bias_q15[hidden_index]);
            for (local_index, &feature_index) in self.feature_indices.iter().enumerate() {
                let weight =
                    self.input_weights[hidden_index * self.feature_indices.len() + local_index];
                acc = acc
                    .checked_add((i64::from(features[feature_index]) * i64::from(weight)) >> 7)
                    .ok_or("hidden accumulation overflow")?;
            }
            *hidden_value = acc.clamp(0, i64::from(i16::MAX)) as i16;
        }
        let mut logits_q8 = [0_i32; OUTPUT_DIM];
        for output_index in 0..OUTPUT_DIM {
            let mut acc = i64::from(self.output_bias_q8[output_index]);
            for (hidden_index, &hidden_value) in hidden.iter().enumerate() {
                let weight = self.output_weights[output_index * HIDDEN_DIM + hidden_index];
                acc = acc
                    .checked_add(
                        (i64::from(hidden_value) * i64::from(weight)) >> OUTPUT_LOGIT_SHIFT,
                    )
                    .ok_or("output accumulation overflow")?;
            }
            logits_q8[output_index] = acc.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
        }
        let mut probabilities_q15 = [0_i16; OUTPUT_DIM];
        base2_softmax_i32_q15(&logits_q8, &mut probabilities_q15).ok_or("router softmax failed")?;
        Ok(RouterForward {
            hidden,
            logits_q8,
            probabilities_q15,
        })
    }

    fn to_bytes(&self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(self.feature_indices.len() as u32).to_le_bytes());
        out.extend_from_slice(&(HIDDEN_DIM as u32).to_le_bytes());
        out.extend_from_slice(&(OUTPUT_DIM as u32).to_le_bytes());
        out.extend_from_slice(&self.seed.to_le_bytes());
        for &index in &self.feature_indices {
            out.push(u8::try_from(index)?);
        }
        out.extend(
            self.input_weights
                .iter()
                .map(|value| value.to_le_bytes()[0]),
        );
        for &value in &self.hidden_bias_q15 {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out.extend(
            self.output_weights
                .iter()
                .map(|value| value.to_le_bytes()[0]),
        );
        for &value in &self.output_bias_q8 {
            out.extend_from_slice(&value.to_le_bytes());
        }
        Ok(out)
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        let mut cursor = 0_usize;
        if take(bytes, &mut cursor, MAGIC.len())? != MAGIC {
            return Err("invalid router artifact magic".into());
        }
        let input_dim = read_u32(bytes, &mut cursor)? as usize;
        let hidden_dim = read_u32(bytes, &mut cursor)? as usize;
        let output_dim = read_u32(bytes, &mut cursor)? as usize;
        let seed = read_u64(bytes, &mut cursor)?;
        if input_dim == 0
            || input_dim > FEATURE_COUNT
            || hidden_dim != HIDDEN_DIM
            || output_dim != OUTPUT_DIM
        {
            return Err("invalid router artifact shape".into());
        }
        let feature_indices = take(bytes, &mut cursor, input_dim)?
            .iter()
            .map(|&value| usize::from(value))
            .collect::<Vec<_>>();
        if feature_indices.iter().any(|&index| index >= FEATURE_COUNT) {
            return Err("invalid router feature index".into());
        }
        let input_weights = take(bytes, &mut cursor, input_dim * HIDDEN_DIM)?
            .iter()
            .map(|&value| i8::from_le_bytes([value]))
            .collect::<Vec<_>>();
        let mut hidden_bias_q15 = [0_i32; HIDDEN_DIM];
        for value in &mut hidden_bias_q15 {
            *value = read_i32(bytes, &mut cursor)?;
        }
        let mut output_weights = [0_i8; HIDDEN_DIM * OUTPUT_DIM];
        for value in &mut output_weights {
            *value = i8::from_le_bytes([take(bytes, &mut cursor, 1)?[0]]);
        }
        let mut output_bias_q8 = [0_i32; OUTPUT_DIM];
        for value in &mut output_bias_q8 {
            *value = read_i32(bytes, &mut cursor)?;
        }
        if cursor != bytes.len() {
            return Err("router artifact has trailing bytes".into());
        }
        Ok(Self {
            feature_indices,
            seed,
            input_weights,
            hidden_bias_q15,
            output_weights,
            output_bias_q8,
        })
    }

    fn model_hash(&self) -> Result<u64, Box<dyn std::error::Error>> {
        Ok(fnv64(&self.to_bytes()?))
    }
}

struct RouterForward {
    hidden: [i16; HIDDEN_DIM],
    logits_q8: [i32; OUTPUT_DIM],
    probabilities_q15: [i16; OUTPUT_DIM],
}

#[derive(Debug, Clone, Copy, Default)]
struct RouterUpdateStats {
    input_weight_updates: usize,
    output_weight_updates: usize,
    hidden_bias_updates: usize,
    output_bias_updates: usize,
    saturation_count: usize,
}

fn train(
    model: &mut RouterModel,
    rows: &[RouterRow],
    epochs: usize,
    objective: RouterObjective,
    regret_gradient_shift: u8,
) -> Result<RouterUpdateStats, Box<dyn std::error::Error>> {
    let mut input_carry = vec![0_i64; model.input_weights.len()];
    let mut output_carry = [0_i64; HIDDEN_DIM * OUTPUT_DIM];
    let mut hidden_bias_carry = [0_i64; HIDDEN_DIM];
    let mut output_bias_carry = [0_i64; OUTPUT_DIM];
    let mut stats = RouterUpdateStats::default();
    let mut label_counts = [0_usize; OUTPUT_DIM];
    for row in rows {
        label_counts[row.target] += 1;
    }
    let max_label_count = *label_counts.iter().max().ok_or("no router labels")?;
    for _ in 0..epochs {
        for row in rows {
            let forward = model.forward(&row.features_q15)?;
            let soft_targets = match objective {
                RouterObjective::HardLabel | RouterObjective::ExpectedRegret => None,
                RouterObjective::UtilitySoft => Some(utility_target_probabilities(row)?),
            };
            let errors_q15 = if objective == RouterObjective::ExpectedRegret {
                expected_regret_errors_q15(row, &forward.probabilities_q15, regret_gradient_shift)?
            } else {
                let mut errors = [0_i32; OUTPUT_DIM];
                for output_index in 0..OUTPUT_DIM {
                    let target = soft_targets.as_ref().map_or_else(
                        || {
                            if output_index == row.target {
                                i32::from(i16::MAX)
                            } else {
                                0
                            }
                        },
                        |targets| i32::from(targets[output_index]),
                    );
                    let raw_error = i32::from(forward.probabilities_q15[output_index]) - target;
                    errors[output_index] = match objective {
                        RouterObjective::HardLabel => {
                            let class_weight =
                                max_label_count.div_ceil(label_counts[row.target].max(1));
                            raw_error.saturating_mul(class_weight as i32)
                        }
                        RouterObjective::UtilitySoft => raw_error,
                        RouterObjective::ExpectedRegret => unreachable!(),
                    };
                }
                errors
            };
            let output_weights_before = model.output_weights;
            let mut hidden_grad_q15 = [0_i64; HIDDEN_DIM];
            for hidden_index in 0..HIDDEN_DIM {
                if forward.hidden[hidden_index] <= 0 {
                    continue;
                }
                let mut acc = 0_i64;
                for output_index in 0..OUTPUT_DIM {
                    acc += i64::from(errors_q15[output_index])
                        * i64::from(
                            output_weights_before[output_index * HIDDEN_DIM + hidden_index],
                        );
                }
                hidden_grad_q15[hidden_index] = acc >> 7;
            }

            for output_index in 0..OUTPUT_DIM {
                for hidden_index in 0..HIDDEN_DIM {
                    let index = output_index * HIDDEN_DIM + hidden_index;
                    output_carry[index] += i64::from(errors_q15[output_index])
                        * i64::from(forward.hidden[hidden_index]);
                    apply_i8_gradient(
                        &mut model.output_weights[index],
                        &mut output_carry[index],
                        OUTPUT_WEIGHT_GRAD_SHIFT,
                        &mut stats.output_weight_updates,
                        &mut stats.saturation_count,
                    );
                }
                output_bias_carry[output_index] += i64::from(errors_q15[output_index]);
                apply_i32_gradient(
                    &mut model.output_bias_q8[output_index],
                    &mut output_bias_carry[output_index],
                    OUTPUT_BIAS_GRAD_SHIFT,
                    &mut stats.output_bias_updates,
                );
            }

            for hidden_index in 0..HIDDEN_DIM {
                for (local_index, &feature_index) in model.feature_indices.iter().enumerate() {
                    let index = hidden_index * model.feature_indices.len() + local_index;
                    input_carry[index] +=
                        hidden_grad_q15[hidden_index] * i64::from(row.features_q15[feature_index]);
                    apply_i8_gradient(
                        &mut model.input_weights[index],
                        &mut input_carry[index],
                        INPUT_WEIGHT_GRAD_SHIFT,
                        &mut stats.input_weight_updates,
                        &mut stats.saturation_count,
                    );
                }
                hidden_bias_carry[hidden_index] += hidden_grad_q15[hidden_index];
                apply_i32_gradient(
                    &mut model.hidden_bias_q15[hidden_index],
                    &mut hidden_bias_carry[hidden_index],
                    HIDDEN_BIAS_GRAD_SHIFT,
                    &mut stats.hidden_bias_updates,
                );
            }
        }
    }
    Ok(stats)
}

fn utility_target_probabilities(
    row: &RouterRow,
) -> Result<[i16; OUTPUT_DIM], Box<dyn std::error::Error>> {
    let best_loss = *row
        .child_losses_q15
        .iter()
        .min()
        .ok_or("router row has no child losses")?;
    let mut utility_logits_q8 = [0_i32; OUTPUT_DIM];
    for output_index in 0..OUTPUT_DIM {
        let regret = row.child_losses_q15[output_index].saturating_sub(best_loss);
        utility_logits_q8[output_index] = -i32::try_from(regret >> UTILITY_LOSS_LOGIT_SHIFT)?;
    }
    let mut probabilities_q15 = [0_i16; OUTPUT_DIM];
    base2_softmax_i32_q15(&utility_logits_q8, &mut probabilities_q15)
        .ok_or("utility target softmax failed")?;
    Ok(probabilities_q15)
}

fn expected_regret_errors_q15(
    row: &RouterRow,
    probabilities_q15: &[i16; OUTPUT_DIM],
    gradient_shift: u8,
) -> Result<[i32; OUTPUT_DIM], Box<dyn std::error::Error>> {
    let probability_sum = probabilities_q15
        .iter()
        .map(|&probability| i64::from(probability.max(0)))
        .sum::<i64>();
    if probability_sum == 0 {
        return Err("expected-regret probabilities sum to zero".into());
    }
    let weighted_loss = probabilities_q15
        .iter()
        .zip(row.child_losses_q15.iter())
        .map(|(&probability, &loss)| i64::from(probability.max(0)).saturating_mul(loss as i64))
        .sum::<i64>();
    let expected_loss = weighted_loss.saturating_add(probability_sum / 2) / probability_sum;
    let mut errors = [0_i32; OUTPUT_DIM];
    for output_index in 0..OUTPUT_DIM {
        let difference = (row.child_losses_q15[output_index] as i64).saturating_sub(expected_loss);
        let scaled = i64::from(probabilities_q15[output_index].max(0)).saturating_mul(difference)
            >> gradient_shift;
        errors[output_index] = scaled.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
    }
    Ok(errors)
}

fn apply_i8_gradient(
    weight: &mut i8,
    carry: &mut i64,
    shift: u8,
    updates: &mut usize,
    saturation_count: &mut usize,
) {
    let threshold = 1_i64 << shift;
    while *carry >= threshold {
        if *weight == i8::MIN {
            *saturation_count += 1;
        } else {
            *weight -= 1;
            *updates += 1;
        }
        *carry -= threshold;
    }
    while *carry <= -threshold {
        if *weight == i8::MAX {
            *saturation_count += 1;
        } else {
            *weight += 1;
            *updates += 1;
        }
        *carry += threshold;
    }
}

fn apply_i32_gradient(weight: &mut i32, carry: &mut i64, shift: u8, updates: &mut usize) {
    let threshold = 1_i64 << shift;
    while *carry >= threshold {
        *weight = weight.saturating_sub(1);
        *carry -= threshold;
        *updates += 1;
    }
    while *carry <= -threshold {
        *weight = weight.saturating_add(1);
        *carry += threshold;
        *updates += 1;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RouterMetrics {
    rows: usize,
    correct: usize,
    accuracy_per_mille: usize,
    top2_oracle_coverage_per_mille: usize,
    mean_probability_error_q15: usize,
    mean_regret_q15: usize,
    label_counts: [usize; OUTPUT_DIM],
    predicted_counts: [usize; OUTPUT_DIM],
}

fn evaluate(
    model: &RouterModel,
    rows: &[RouterRow],
) -> Result<RouterMetrics, Box<dyn std::error::Error>> {
    let mut correct = 0_usize;
    let mut top2 = 0_usize;
    let mut probability_error = 0_usize;
    let mut regret = 0_usize;
    let mut label_counts = [0_usize; OUTPUT_DIM];
    let mut predicted_counts = [0_usize; OUTPUT_DIM];
    for row in rows {
        let forward = model.forward(&row.features_q15)?;
        let ranked = ranked_indices(&forward.logits_q8);
        let predicted = ranked[0];
        correct += usize::from(predicted == row.target);
        top2 += usize::from(ranked[0] == row.target || ranked[1] == row.target);
        probability_error += usize::try_from(
            i32::from(i16::MAX) - i32::from(forward.probabilities_q15[row.target]),
        )?;
        regret += row.child_losses_q15[predicted]
            .saturating_sub(*row.child_losses_q15.iter().min().ok_or("no child losses")?);
        label_counts[row.target] += 1;
        predicted_counts[predicted] += 1;
    }
    Ok(RouterMetrics {
        rows: rows.len(),
        correct,
        accuracy_per_mille: correct * 1000 / rows.len(),
        top2_oracle_coverage_per_mille: top2 * 1000 / rows.len(),
        mean_probability_error_q15: probability_error / rows.len(),
        mean_regret_q15: regret / rows.len(),
        label_counts,
        predicted_counts,
    })
}

fn ranked_indices(logits: &[i32; OUTPUT_DIM]) -> [usize; OUTPUT_DIM] {
    let mut indices = [0, 1, 2];
    indices.sort_by_key(|&index| (core::cmp::Reverse(logits[index]), index));
    indices
}

fn write_predictions(
    path: &PathBuf,
    model: &RouterModel,
    rows: &[RouterRow],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut out = BufWriter::new(fs::File::create(path)?);
    writeln!(
        out,
        "sample_id\ttarget\tpredicted\ttop2\tprobabilities_q15\tchild_losses_q15"
    )?;
    for row in rows {
        let forward = model.forward(&row.features_q15)?;
        let ranked = ranked_indices(&forward.logits_q8);
        writeln!(
            out,
            "{}\t{}\t{}\t{},{}\t{},{},{}\t{},{},{}",
            row.sample_id,
            row.target,
            ranked[0],
            ranked[0],
            ranked[1],
            forward.probabilities_q15[0],
            forward.probabilities_q15[1],
            forward.probabilities_q15[2],
            row.child_losses_q15[0],
            row.child_losses_q15[1],
            row.child_losses_q15[2]
        )?;
    }
    out.flush()?;
    Ok(())
}

fn training_trace_json(
    model: &RouterModel,
    epochs: usize,
    objective: RouterObjective,
    regret_gradient_shift: u8,
    update: RouterUpdateStats,
    initial_train: RouterMetrics,
    final_train: RouterMetrics,
    initial_calibration: RouterMetrics,
    final_calibration: RouterMetrics,
) -> String {
    format!(
        "{{\"schema\":\"nsrl.router_training.v2\",\"model\":{{\"artifact\":\"NSRLRT1\",\"model_hash\":\"0x{:016x}\",\"seed\":{},\"input_dim\":{},\"hidden_dim\":{},\"output_dim\":{},\"feature_indices\":[{}]}},\"training\":{{\"epochs\":{},\"optimizer\":\"integer_error_feedback_sgd\",\"objective\":\"{}\",\"class_balanced\":{},\"utility_loss_logit_shift\":{},\"regret_gradient_shift\":{},\"output_logit_shift\":{},\"output_weight_grad_shift\":{},\"input_weight_grad_shift\":{},\"updates\":{{\"input_weights\":{},\"output_weights\":{},\"hidden_bias\":{},\"output_bias\":{},\"saturation_count\":{}}}}},\"initial_train\":{},\"final_train\":{},\"initial_calibration\":{},\"final_calibration\":{}}}\n",
        model.model_hash().unwrap_or(0),
        model.seed,
        model.feature_indices.len(),
        HIDDEN_DIM,
        OUTPUT_DIM,
        model
            .feature_indices
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(","),
        epochs,
        objective.as_str(),
        objective == RouterObjective::HardLabel,
        UTILITY_LOSS_LOGIT_SHIFT,
        regret_gradient_shift,
        OUTPUT_LOGIT_SHIFT,
        OUTPUT_WEIGHT_GRAD_SHIFT,
        INPUT_WEIGHT_GRAD_SHIFT,
        update.input_weight_updates,
        update.output_weight_updates,
        update.hidden_bias_updates,
        update.output_bias_updates,
        update.saturation_count,
        metrics_json(initial_train),
        metrics_json(final_train),
        metrics_json(initial_calibration),
        metrics_json(final_calibration),
    )
}

fn eval_trace_json(model: &RouterModel, metrics: RouterMetrics) -> String {
    format!(
        "{{\"schema\":\"nsrl.router_eval.v1\",\"model_hash\":\"0x{:016x}\",\"metrics\":{}}}\n",
        model.model_hash().unwrap_or(0),
        metrics_json(metrics)
    )
}

fn metrics_json(metrics: RouterMetrics) -> String {
    format!(
        "{{\"rows\":{},\"correct\":{},\"accuracy_per_mille\":{},\"top2_oracle_coverage_per_mille\":{},\"mean_probability_error_q15\":{},\"mean_regret_q15\":{},\"label_counts\":[{},{},{}],\"predicted_counts\":[{},{},{}]}}",
        metrics.rows,
        metrics.correct,
        metrics.accuracy_per_mille,
        metrics.top2_oracle_coverage_per_mille,
        metrics.mean_probability_error_q15,
        metrics.mean_regret_q15,
        metrics.label_counts[0],
        metrics.label_counts[1],
        metrics.label_counts[2],
        metrics.predicted_counts[0],
        metrics.predicted_counts[1],
        metrics.predicted_counts[2]
    )
}

fn required(
    args: &mut impl Iterator<Item = String>,
    option: &'static str,
) -> Result<String, Box<dyn std::error::Error>> {
    args.next()
        .ok_or_else(|| format!("{option} requires a value").into())
}

fn take<'a>(
    bytes: &'a [u8],
    cursor: &mut usize,
    count: usize,
) -> Result<&'a [u8], Box<dyn std::error::Error>> {
    let end = cursor
        .checked_add(count)
        .ok_or("artifact length overflow")?;
    let value = bytes.get(*cursor..end).ok_or("truncated router artifact")?;
    *cursor = end;
    Ok(value)
}

fn read_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32, Box<dyn std::error::Error>> {
    Ok(u32::from_le_bytes(take(bytes, cursor, 4)?.try_into()?))
}

fn read_u64(bytes: &[u8], cursor: &mut usize) -> Result<u64, Box<dyn std::error::Error>> {
    Ok(u64::from_le_bytes(take(bytes, cursor, 8)?.try_into()?))
}

fn read_i32(bytes: &[u8], cursor: &mut usize) -> Result<i32, Box<dyn std::error::Error>> {
    Ok(i32::from_le_bytes(take(bytes, cursor, 4)?.try_into()?))
}

fn fnv64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_rows() -> Vec<RouterRow> {
        (0..12)
            .map(|index| {
                let target = index % 3;
                let mut features_q15 = [0_i16; FEATURE_COUNT];
                features_q15[target] = i16::MAX;
                RouterRow {
                    sample_id: format!("row-{index}"),
                    target,
                    features_q15,
                    child_losses_q15: core::array::from_fn(|candidate| {
                        if candidate == target { 100 } else { 1000 }
                    }),
                }
            })
            .collect()
    }

    #[test]
    fn artifact_round_trip_is_exact() {
        let model = RouterModel::new((0..32).collect(), 7);
        let bytes = model.to_bytes().expect("bytes");
        assert_eq!(RouterModel::from_bytes(&bytes).expect("model"), model);
    }

    #[test]
    fn training_is_deterministic_and_moves_weights() {
        let rows = fixture_rows();
        let mut left = RouterModel::new((0..32).collect(), 11);
        let mut right = left.clone();
        let before = left.model_hash().expect("before");
        let left_stats = train(&mut left, &rows, 16, RouterObjective::HardLabel, 3).expect("left");
        let right_stats =
            train(&mut right, &rows, 16, RouterObjective::HardLabel, 3).expect("right");
        assert_eq!(left, right);
        assert_eq!(
            left_stats.output_weight_updates,
            right_stats.output_weight_updates
        );
        assert_ne!(left.model_hash().expect("after"), before);
        assert!(left_stats.output_weight_updates > 0);
        assert_eq!(evaluate(&left, &rows).expect("eval").rows, rows.len());
    }

    #[test]
    fn utility_soft_targets_preserve_order_and_near_ties() {
        let mut row = fixture_rows().remove(0);
        row.child_losses_q15 = [1000, 1008, 4000];
        let probabilities = utility_target_probabilities(&row).expect("targets");
        assert!(probabilities[0] > probabilities[1]);
        assert!(probabilities[1] > probabilities[2]);
        assert!(i32::from(probabilities[0]) - i32::from(probabilities[1]) < 4096);
    }

    #[test]
    fn expected_regret_gradient_preserves_one_unit_utility_differences() {
        let mut row = fixture_rows().remove(0);
        row.child_losses_q15 = [1000, 1001, 1002];
        let probabilities = [10_923, 10_922, 10_922];
        let errors =
            expected_regret_errors_q15(&row, &probabilities, 3).expect("expected regret errors");

        assert!(errors[0] < 0);
        assert_eq!(errors[1], 0);
        assert!(errors[2] > 0);
        assert!(errors.iter().any(|&error| error != 0));
    }
}
