#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use nsrl_eval::q22::{
    Q22Manifest, Q22TrainingRecord, load_q22_encoded_training_records, load_q22_manifest,
};
use nsrl_eval::{stable_hash_bytes, stable_hex_u32};

const MODEL_MAGIC: &[u8; 8] = b"NSRLQ221";
const MODEL_VERSION: u32 = 1;
const MODEL_SCHEMA: &str = "nsrl.q22_integer_class_head.v1";
const TRAIN_TRACE_SCHEMA: &str = "nsrl.q22_integer_class_head_train.v1";
const PREDICT_TRACE_SCHEMA: &str = "nsrl.q22_integer_class_head_predict.v1";
const FEATURE_COUNT: usize = 8_192;
const DEFAULT_EPOCHS: usize = 4;
const OPERATIONS: [&str; 5] = [
    "quantity.add",
    "quantity.multiply",
    "quantity.add-rational",
    "quantity.convert",
    "quantity.solve-linear",
];

#[derive(Debug, Clone, PartialEq, Eq)]
struct Q22ClassHead {
    seed: u64,
    epochs: usize,
    train_records: usize,
    encoded_dataset_sha256: String,
    weights: Vec<i32>,
    biases: [i32; OPERATIONS.len()],
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BlindInput {
    id: String,
    input: String,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("nsrl-q22-proposer: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("train") => train_command(args),
        Some("predict") => predict_command(args),
        Some("inspect") => inspect_command(args),
        Some("--help" | "-h") | None => {
            usage();
            Ok(())
        }
        Some(other) => Err(format!("unknown command: {other}").into()),
    }
}

fn train_command(mut args: impl Iterator<Item = String>) -> Result<(), Box<dyn std::error::Error>> {
    let mut manifest_path = None;
    let mut encoded_path = None;
    let mut model_out = None;
    let mut trace_out = None;
    let mut seed = None;
    let mut epochs = DEFAULT_EPOCHS;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--manifest" => manifest_path = Some(path_arg(&mut args, "--manifest")?),
            "--encoded" => encoded_path = Some(path_arg(&mut args, "--encoded")?),
            "--model-out" => model_out = Some(path_arg(&mut args, "--model-out")?),
            "--trace-out" => trace_out = Some(path_arg(&mut args, "--trace-out")?),
            "--seed" => seed = Some(value_arg(&mut args, "--seed")?.parse::<u64>()?),
            "--epochs" => epochs = value_arg(&mut args, "--epochs")?.parse::<usize>()?,
            other => return Err(format!("unknown train argument: {other}").into()),
        }
    }
    let manifest = load_q22_manifest(&manifest_path.ok_or("train requires --manifest PATH")?)?;
    let records = load_q22_encoded_training_records(
        &encoded_path.ok_or("train requires --encoded PATH")?,
        &manifest,
    )?;
    let seed = seed.ok_or("train requires --seed N")?;
    let (model, mistakes_by_epoch) = train_model(&records, &manifest, seed, epochs)?;
    let model_bytes = model.to_bytes()?;
    let model_hash = stable_hex_u32(stable_hash_bytes(&model_bytes));
    let train_exact = records
        .iter()
        .filter(|record| {
            Some(model.predict(&record.input)) == operation_index(&record.model_request)
        })
        .count();
    fs::write(
        model_out.ok_or("train requires --model-out PATH")?,
        &model_bytes,
    )?;
    let trace = format!(
        "{{\"schema\":\"{TRAIN_TRACE_SCHEMA}\",\"model_schema\":\"{MODEL_SCHEMA}\",\"seed\":{seed},\"epochs\":{epochs},\"feature_count\":{FEATURE_COUNT},\"train_records\":{},\"encoded_dataset_sha256\":\"{}\",\"mistakes_by_epoch\":{},\"train_exact\":{train_exact},\"train_exact_rate_ppm\":{},\"model_hash\":\"{model_hash}\",\"integer_only\":true}}\n",
        records.len(),
        manifest.encoded_dataset_sha256,
        json_usize_array(&mistakes_by_epoch),
        train_exact.saturating_mul(1_000_000) / records.len().max(1),
    );
    write_or_print(trace_out.as_ref(), &trace)?;
    Ok(())
}

fn predict_command(
    mut args: impl Iterator<Item = String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut manifest_path = None;
    let mut model_path = None;
    let mut inputs_path = None;
    let mut predictions_out = None;
    let mut trace_out = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--manifest" => manifest_path = Some(path_arg(&mut args, "--manifest")?),
            "--model" => model_path = Some(path_arg(&mut args, "--model")?),
            "--inputs" => inputs_path = Some(path_arg(&mut args, "--inputs")?),
            "--predictions-out" => {
                predictions_out = Some(path_arg(&mut args, "--predictions-out")?)
            }
            "--trace-out" => trace_out = Some(path_arg(&mut args, "--trace-out")?),
            other => return Err(format!("unknown predict argument: {other}").into()),
        }
    }
    let manifest = load_q22_manifest(&manifest_path.ok_or("predict requires --manifest PATH")?)?;
    let model_bytes = fs::read(model_path.ok_or("predict requires --model PATH")?)?;
    let model_hash = stable_hex_u32(stable_hash_bytes(&model_bytes));
    let model = Q22ClassHead::from_bytes(&model_bytes)?;
    if model.encoded_dataset_sha256 != manifest.encoded_dataset_sha256
        || model.train_records != manifest.train_records
    {
        return Err("model does not bind the frozen Q22 training dataset".into());
    }
    let inputs = load_blind_inputs(
        &inputs_path.ok_or("predict requires --inputs PATH")?,
        manifest.eval_records,
    )?;
    let mut predictions = String::from("id\tmodel_request\n");
    for row in &inputs {
        predictions.push_str(&row.id);
        predictions.push('\t');
        predictions.push_str(OPERATIONS[model.predict(&row.input)]);
        predictions.push('\n');
    }
    fs::write(
        predictions_out.ok_or("predict requires --predictions-out PATH")?,
        predictions,
    )?;
    let trace = format!(
        "{{\"schema\":\"{PREDICT_TRACE_SCHEMA}\",\"model_schema\":\"{MODEL_SCHEMA}\",\"seed\":{},\"records\":{},\"model_hash\":\"{model_hash}\",\"blind_input_only\":true}}\n",
        model.seed,
        inputs.len(),
    );
    write_or_print(trace_out.as_ref(), &trace)?;
    Ok(())
}

fn inspect_command(
    mut args: impl Iterator<Item = String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let model_path = if args.next().as_deref() == Some("--model") {
        path_arg(&mut args, "--model")?
    } else {
        return Err("inspect requires --model PATH".into());
    };
    if args.next().is_some() {
        return Err("inspect accepts only --model PATH".into());
    }
    let bytes = fs::read(model_path)?;
    let model = Q22ClassHead::from_bytes(&bytes)?;
    println!(
        "{{\"schema\":\"{MODEL_SCHEMA}\",\"seed\":{},\"epochs\":{},\"feature_count\":{FEATURE_COUNT},\"train_records\":{},\"encoded_dataset_sha256\":\"{}\",\"model_hash\":\"{}\",\"integer_only\":true}}",
        model.seed,
        model.epochs,
        model.train_records,
        model.encoded_dataset_sha256,
        stable_hex_u32(stable_hash_bytes(&bytes)),
    );
    Ok(())
}

fn train_model(
    records: &[Q22TrainingRecord],
    manifest: &Q22Manifest,
    seed: u64,
    epochs: usize,
) -> Result<(Q22ClassHead, Vec<usize>), String> {
    if records.is_empty() || epochs == 0 || records.len() != manifest.train_records {
        return Err("invalid Q22 training configuration".to_string());
    }
    let mut model = Q22ClassHead {
        seed,
        epochs,
        train_records: records.len(),
        encoded_dataset_sha256: manifest.encoded_dataset_sha256.clone(),
        weights: vec![0; FEATURE_COUNT * OPERATIONS.len()],
        biases: [0; OPERATIONS.len()],
    };
    let examples = records
        .iter()
        .map(|record| {
            operation_index(&record.model_request)
                .map(|label| (features(&record.input), label))
                .ok_or_else(|| format!("unsupported Q22 operation {}", record.model_request))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut mistakes_by_epoch = Vec::with_capacity(epochs);
    for epoch in 0..epochs {
        let mut order = (0..examples.len()).collect::<Vec<_>>();
        shuffle(&mut order, seed, epoch);
        let mut mistakes = 0_usize;
        for index in order {
            let (feature_rows, label) = &examples[index];
            let predicted = model.predict_features(feature_rows);
            if predicted == *label {
                continue;
            }
            mistakes += 1;
            model.update(*label, feature_rows, 1)?;
            model.update(predicted, feature_rows, -1)?;
            model.biases[*label] = model.biases[*label].saturating_add(1);
            model.biases[predicted] = model.biases[predicted].saturating_sub(1);
        }
        mistakes_by_epoch.push(mistakes);
    }
    Ok((model, mistakes_by_epoch))
}

impl Q22ClassHead {
    fn predict(&self, input: &str) -> usize {
        self.predict_features(&features(input))
    }

    fn predict_features(&self, feature_rows: &[(usize, i32)]) -> usize {
        (0..OPERATIONS.len())
            .max_by_key(|&label| (self.score(label, feature_rows), std::cmp::Reverse(label)))
            .unwrap_or(0)
    }

    fn score(&self, label: usize, feature_rows: &[(usize, i32)]) -> i64 {
        let offset = label * FEATURE_COUNT;
        feature_rows
            .iter()
            .fold(i64::from(self.biases[label]), |score, &(feature, value)| {
                score.saturating_add(
                    i64::from(self.weights[offset + feature]).saturating_mul(i64::from(value)),
                )
            })
    }

    fn update(
        &mut self,
        label: usize,
        feature_rows: &[(usize, i32)],
        direction: i32,
    ) -> Result<(), String> {
        let offset = label
            .checked_mul(FEATURE_COUNT)
            .ok_or_else(|| "Q22 class-head offset overflow".to_string())?;
        for &(feature, value) in feature_rows {
            let delta = direction
                .checked_mul(value)
                .ok_or_else(|| "Q22 class-head update overflow".to_string())?;
            self.weights[offset + feature] = self.weights[offset + feature]
                .checked_add(delta)
                .ok_or_else(|| "Q22 class-head weight overflow".to_string())?;
        }
        Ok(())
    }

    fn to_bytes(&self) -> Result<Vec<u8>, String> {
        self.validate()?;
        let mut bytes = Vec::with_capacity(128 + self.weights.len() * 4);
        bytes.extend_from_slice(MODEL_MAGIC);
        push_u32(&mut bytes, MODEL_VERSION);
        bytes.extend_from_slice(&self.seed.to_le_bytes());
        push_u32(&mut bytes, usize_to_u32(self.epochs, "epochs")?);
        push_u32(&mut bytes, usize_to_u32(FEATURE_COUNT, "feature_count")?);
        push_u32(&mut bytes, usize_to_u32(OPERATIONS.len(), "class_count")?);
        push_u32(
            &mut bytes,
            usize_to_u32(self.train_records, "train_records")?,
        );
        bytes.extend_from_slice(self.encoded_dataset_sha256.as_bytes());
        for &weight in &self.weights {
            bytes.extend_from_slice(&weight.to_le_bytes());
        }
        for bias in self.biases {
            bytes.extend_from_slice(&bias.to_le_bytes());
        }
        Ok(bytes)
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let mut offset = 0_usize;
        if take(bytes, &mut offset, MODEL_MAGIC.len())? != MODEL_MAGIC {
            return Err("bad Q22 class-head magic".to_string());
        }
        if read_u32(bytes, &mut offset)? != MODEL_VERSION {
            return Err("unsupported Q22 class-head version".to_string());
        }
        let seed = u64::from_le_bytes(
            take(bytes, &mut offset, 8)?
                .try_into()
                .map_err(|_| "bad Q22 seed")?,
        );
        let epochs = read_u32(bytes, &mut offset)? as usize;
        if read_u32(bytes, &mut offset)? as usize != FEATURE_COUNT
            || read_u32(bytes, &mut offset)? as usize != OPERATIONS.len()
        {
            return Err("Q22 class-head geometry mismatch".to_string());
        }
        let train_records = read_u32(bytes, &mut offset)? as usize;
        let encoded_dataset_sha256 = String::from_utf8(take(bytes, &mut offset, 64)?.to_vec())
            .map_err(|_| "bad Q22 encoded dataset SHA-256")?;
        let mut weights = Vec::with_capacity(FEATURE_COUNT * OPERATIONS.len());
        for _ in 0..FEATURE_COUNT * OPERATIONS.len() {
            weights.push(i32::from_le_bytes(
                take(bytes, &mut offset, 4)?
                    .try_into()
                    .map_err(|_| "bad Q22 weight")?,
            ));
        }
        let mut biases = [0_i32; OPERATIONS.len()];
        for bias in &mut biases {
            *bias = i32::from_le_bytes(
                take(bytes, &mut offset, 4)?
                    .try_into()
                    .map_err(|_| "bad Q22 bias")?,
            );
        }
        if offset != bytes.len() {
            return Err("trailing Q22 class-head bytes".to_string());
        }
        let model = Self {
            seed,
            epochs,
            train_records,
            encoded_dataset_sha256,
            weights,
            biases,
        };
        model.validate()?;
        Ok(model)
    }

    fn validate(&self) -> Result<(), String> {
        if self.epochs == 0
            || self.train_records == 0
            || self.encoded_dataset_sha256.len() != 64
            || !self
                .encoded_dataset_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
            || self.weights.len() != FEATURE_COUNT * OPERATIONS.len()
        {
            return Err("invalid Q22 class-head model".to_string());
        }
        Ok(())
    }
}

fn features(input: &str) -> Vec<(usize, i32)> {
    let normalized = input.trim().to_ascii_lowercase();
    let mut accum = BTreeMap::<usize, i32>::new();
    let first = normalized.split_ascii_whitespace().next().unwrap_or("");
    add_feature(&mut accum, format!("first:{first}").as_bytes(), 8);
    for (position, token) in normalized.split_ascii_whitespace().enumerate() {
        add_feature(&mut accum, format!("token:{token}").as_bytes(), 2);
        add_feature(
            &mut accum,
            format!("position:{position}:{token}").as_bytes(),
            3,
        );
    }
    let bytes = normalized.as_bytes();
    for order in 1..=4 {
        for window in bytes.windows(order) {
            let mut feature = Vec::with_capacity(order + 1);
            feature.push(order as u8);
            feature.extend_from_slice(window);
            add_feature(&mut accum, &feature, 1);
        }
    }
    accum.into_iter().collect()
}

fn add_feature(accum: &mut BTreeMap<usize, i32>, value: &[u8], weight: i32) {
    let index = (fnv1a64(value) as usize) % FEATURE_COUNT;
    let entry = accum.entry(index).or_insert(0);
    *entry = entry.saturating_add(weight);
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn shuffle(order: &mut [usize], seed: u64, epoch: usize) {
    let mut state = seed
        .wrapping_add((epoch as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
        .wrapping_add(0xa076_1d64_78bd_642f);
    for index in (1..order.len()).rev() {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let draw = state.wrapping_mul(0x2545_f491_4f6c_dd1d);
        order.swap(index, (draw as usize) % (index + 1));
    }
}

fn operation_index(value: &str) -> Option<usize> {
    OPERATIONS.iter().position(|&operation| operation == value)
}

fn load_blind_inputs(path: &Path, expected_records: usize) -> Result<Vec<BlindInput>, String> {
    let input = fs::read_to_string(path)
        .map_err(|error| format!("cannot read blinded inputs {}: {error}", path.display()))?;
    if !input.ends_with('\n') {
        return Err("blinded Q22 inputs must end with one newline".to_string());
    }
    let mut lines = input.lines();
    if lines.next() != Some("id\tinput") {
        return Err("blinded Q22 input header is invalid".to_string());
    }
    let mut ids = BTreeSet::new();
    let mut rows = Vec::new();
    for (offset, line) in lines.enumerate() {
        let (id, input) = line
            .split_once('\t')
            .ok_or_else(|| format!("blinded Q22 row {} requires two fields", offset + 2))?;
        if input.contains('\t')
            || input.is_empty()
            || !id.starts_with("quantity-request/promotion/")
            || !ids.insert(id.to_string())
        {
            return Err(format!("invalid blinded Q22 row {}", offset + 2));
        }
        rows.push(BlindInput {
            id: id.to_string(),
            input: input.to_string(),
        });
    }
    if rows.len() != expected_records {
        return Err(format!(
            "blinded Q22 inputs have {} records, expected {expected_records}",
            rows.len()
        ));
    }
    Ok(rows)
}

fn take<'a>(bytes: &'a [u8], offset: &mut usize, count: usize) -> Result<&'a [u8], String> {
    let end = offset
        .checked_add(count)
        .ok_or_else(|| "Q22 model offset overflow".to_string())?;
    let value = bytes
        .get(*offset..end)
        .ok_or_else(|| "truncated Q22 class-head model".to_string())?;
    *offset = end;
    Ok(value)
}

fn read_u32(bytes: &[u8], offset: &mut usize) -> Result<u32, String> {
    Ok(u32::from_le_bytes(
        take(bytes, offset, 4)?
            .try_into()
            .map_err(|_| "bad Q22 u32")?,
    ))
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn usize_to_u32(value: usize, field: &str) -> Result<u32, String> {
    u32::try_from(value).map_err(|_| format!("{field} exceeds u32"))
}

fn value_arg(
    args: &mut impl Iterator<Item = String>,
    option: &'static str,
) -> Result<String, Box<dyn std::error::Error>> {
    args.next()
        .ok_or_else(|| format!("{option} requires a value").into())
}

fn path_arg(
    args: &mut impl Iterator<Item = String>,
    option: &'static str,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    Ok(PathBuf::from(value_arg(args, option)?))
}

fn write_or_print(path: Option<&PathBuf>, value: &str) -> Result<(), std::io::Error> {
    if let Some(path) = path {
        fs::write(path, value)
    } else {
        print!("{value}");
        Ok(())
    }
}

fn json_usize_array(values: &[usize]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn usage() {
    println!(
        "Usage:\n  nsrl-q22-proposer train --manifest PATH --encoded PATH --seed N [--epochs N] --model-out PATH [--trace-out PATH]\n  nsrl-q22-proposer predict --manifest PATH --model PATH --inputs BLINDED.tsv --predictions-out PATH [--trace-out PATH]\n  nsrl-q22-proposer inspect --model PATH\n\nTrain reads only the frozen Solomon Q22 encoding. Predict accepts only the two-column blinded ID/input surface."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(input: &str, operation: &str) -> Q22TrainingRecord {
        Q22TrainingRecord {
            input: input.to_string(),
            model_request: operation.to_string(),
        }
    }

    #[test]
    fn integer_head_learns_operation_routing_and_round_trips() {
        let records = vec![
            record("add 2 3", "quantity.add"),
            record("add -5 8", "quantity.add"),
            record("multiply 2 3", "quantity.multiply"),
            record("multiply -5 8", "quantity.multiply"),
            record("add-rational 1/2 1/3", "quantity.add-rational"),
            record("add-rational -2/5 7/9", "quantity.add-rational"),
            record("convert 2 kg-to-g", "quantity.convert"),
            record("convert 3 m-to-cm", "quantity.convert"),
            record("solve 2*x+1=5", "quantity.solve-linear"),
            record("solve 7*x+-2=33", "quantity.solve-linear"),
        ];
        let manifest = Q22Manifest {
            dataset_sha256: "a".repeat(64),
            eval_sha256: "b".repeat(64),
            encoded_dataset_sha256: "c".repeat(64),
            train_records: records.len(),
            eval_records: 2,
        };
        let (model, _) = train_model(&records, &manifest, 3, 6).expect("train");
        for row in &records {
            assert_eq!(
                model.predict(&row.input),
                operation_index(&row.model_request).unwrap()
            );
        }
        let bytes = model.to_bytes().expect("encode");
        assert_eq!(Q22ClassHead::from_bytes(&bytes).expect("decode"), model);
        assert!(Q22ClassHead::from_bytes(&bytes[..bytes.len() - 1]).is_err());
    }

    #[test]
    fn features_and_shuffle_are_deterministic_and_seeded() {
        assert_eq!(features("add 2 3"), features(" add 2 3 "));
        let mut first = (0..20).collect::<Vec<_>>();
        let mut replay = first.clone();
        let mut other = first.clone();
        shuffle(&mut first, 1, 0);
        shuffle(&mut replay, 1, 0);
        shuffle(&mut other, 2, 0);
        assert_eq!(first, replay);
        assert_ne!(first, other);
    }
}
