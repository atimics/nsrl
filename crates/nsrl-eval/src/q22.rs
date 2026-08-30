use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use serde_json::Value;

pub const Q22_MANIFEST_SCHEMA: &str = "nsrl.q22_shared_task_manifest.v1";
pub const Q22_CONTRACT_ID: &str = "zero-solomon.q22-operation.v1";
pub const Q22_ENCODING_ID: &str = "solomon.q22-operation.byte-v1";
pub const Q22_VERIFIER_ID: &str = "verifier://solomon/q22-operation/v1";
pub const Q22_MANIFEST_HEADER: &str = "schema\tcontract\tdataset_sha256\teval_sha256\tencoding\tencoded_dataset_sha256\tverifier\ttrain_records\teval_records";
const TRAINING_SCHEMA: &str = "zero.faculty_operation_request.v1";
const PREVIOUS_SUMMARY: &str = "quantity channel has no prior committed result";
const EVAL_HEADER: &str =
    "id\tdomain\tprevious_summary\tinput\tmodel_request\trequest\tartifact\tsummary";
const PREDICTION_HEADER: &str = "id\tmodel_request";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Q22Manifest {
    pub dataset_sha256: String,
    pub eval_sha256: String,
    pub encoded_dataset_sha256: String,
    pub train_records: usize,
    pub eval_records: usize,
}

impl Q22Manifest {
    pub fn to_json_line(&self) -> String {
        format!(
            "{{\"schema\":\"{Q22_MANIFEST_SCHEMA}\",\"contract\":\"{Q22_CONTRACT_ID}\",\"dataset_sha256\":\"{}\",\"eval_sha256\":\"{}\",\"encoding\":\"{Q22_ENCODING_ID}\",\"encoded_dataset_sha256\":\"{}\",\"verifier\":\"{Q22_VERIFIER_ID}\",\"train_records\":{},\"eval_records\":{},\"valid\":true}}\n",
            self.dataset_sha256,
            self.eval_sha256,
            self.encoded_dataset_sha256,
            self.train_records,
            self.eval_records
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Q22Check {
    pub eval_sha256: String,
    pub cases: usize,
    pub operation_exact: usize,
}

impl Q22Check {
    pub fn operation_exact_rate_ppm(&self) -> usize {
        self.operation_exact.saturating_mul(1_000_000) / self.cases.max(1)
    }

    pub fn to_json_line(&self) -> String {
        format!(
            "{{\"schema\":\"nsrl.q22_shared_task_check.v1\",\"contract\":\"{Q22_CONTRACT_ID}\",\"eval_sha256\":\"{}\",\"cases\":{},\"operation_exact\":{},\"operation_exact_rate_ppm\":{},\"valid\":true}}\n",
            self.eval_sha256,
            self.cases,
            self.operation_exact,
            self.operation_exact_rate_ppm()
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TaskRecord {
    input: String,
    model_request: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Expected {
    model_request: String,
    request: String,
    artifact: String,
}

pub fn load_q22_manifest(path: &Path) -> Result<Q22Manifest, String> {
    let input = fs::read_to_string(path)
        .map_err(|error| format!("cannot read Q22 manifest {}: {error}", path.display()))?;
    let lines = input.lines().collect::<Vec<_>>();
    if lines.len() != 2 || lines[0] != Q22_MANIFEST_HEADER {
        return Err("Q22 manifest must contain the exact header and one row".to_string());
    }
    let fields = lines[1].split('\t').collect::<Vec<_>>();
    if fields.len() != 9
        || fields[0] != Q22_MANIFEST_SCHEMA
        || fields[1] != Q22_CONTRACT_ID
        || fields[4] != Q22_ENCODING_ID
        || fields[6] != Q22_VERIFIER_ID
    {
        return Err("Q22 manifest identity is invalid".to_string());
    }
    validate_sha256(fields[2], "dataset_sha256")?;
    validate_sha256(fields[3], "eval_sha256")?;
    validate_sha256(fields[5], "encoded_dataset_sha256")?;
    Ok(Q22Manifest {
        dataset_sha256: fields[2].to_string(),
        eval_sha256: fields[3].to_string(),
        encoded_dataset_sha256: fields[5].to_string(),
        train_records: positive_usize(fields[7], "train_records")?,
        eval_records: positive_usize(fields[8], "eval_records")?,
    })
}

pub fn encode_q22_dataset(path: &Path, manifest: &Q22Manifest) -> Result<Vec<u8>, String> {
    let input = fs::read(path)
        .map_err(|error| format!("cannot read Q22 dataset {}: {error}", path.display()))?;
    require_hash(&input, &manifest.dataset_sha256, "Q22 dataset")?;
    let records = parse_training_records(&input, manifest.train_records)?;
    let mut encoded = Vec::new();
    for record in records {
        encoded.extend_from_slice(b"[q22]\ninput: ");
        encoded.extend_from_slice(record.input.as_bytes());
        encoded.extend_from_slice(b"\noperation: ");
        encoded.extend_from_slice(record.model_request.as_bytes());
        encoded.extend_from_slice(b"\n[/q22]\n");
    }
    require_hash(
        &encoded,
        &manifest.encoded_dataset_sha256,
        "Q22 Solomon encoding",
    )?;
    Ok(encoded)
}

pub fn check_q22_predictions(
    eval_path: &Path,
    prediction_path: &Path,
    manifest: &Q22Manifest,
) -> Result<Q22Check, String> {
    let evaluation = fs::read(eval_path).map_err(|error| {
        format!(
            "cannot read Q22 evaluation {}: {error}",
            eval_path.display()
        )
    })?;
    require_hash(&evaluation, &manifest.eval_sha256, "Q22 evaluation")?;
    let expected = parse_evaluation(&evaluation, manifest.eval_records)?;
    let predictions = parse_predictions(prediction_path, manifest.eval_records)?;
    if expected.keys().collect::<BTreeSet<_>>() != predictions.keys().collect::<BTreeSet<_>>() {
        return Err("Q22 predictions must cover exactly the frozen evaluation IDs".to_string());
    }
    let operation_exact = expected
        .iter()
        .filter(|(id, operation)| predictions.get(*id) == Some(*operation))
        .count();
    Ok(Q22Check {
        eval_sha256: manifest.eval_sha256.clone(),
        cases: expected.len(),
        operation_exact,
    })
}

fn parse_training_records(input: &[u8], expected_count: usize) -> Result<Vec<TaskRecord>, String> {
    let text = std::str::from_utf8(input).map_err(|_| "Q22 dataset must be UTF-8 JSONL")?;
    if !text.ends_with('\n') {
        return Err("Q22 dataset must end with one newline".to_string());
    }
    let mut records = Vec::new();
    let mut ids = BTreeSet::new();
    for (index, line) in text.lines().enumerate() {
        let value: Value = serde_json::from_str(line)
            .map_err(|error| format!("invalid Q22 JSONL row {}: {error}", index + 1))?;
        let id = string_field(&value, "id", index + 1)?;
        let input = string_field(&value, "input", index + 1)?;
        let model_request = string_field(&value, "model_request", index + 1)?;
        if string_field(&value, "schema", index + 1)? != TRAINING_SCHEMA
            || string_field(&value, "split", index + 1)? != "train"
            || string_field(&value, "domain", index + 1)? != "quantity"
            || string_field(&value, "previous_summary", index + 1)? != PREVIOUS_SUMMARY
            || !id.starts_with("quantity-request/train/")
        {
            return Err(format!(
                "Q22 training row {} has invalid identity",
                index + 1
            ));
        }
        let expected = expected_for_input(&input)?;
        validate_task_fields(&value, index + 1, &expected, &model_request)?;
        if !ids.insert(id.clone()) {
            return Err(format!("duplicate Q22 training ID {id}"));
        }
        records.push(TaskRecord {
            input,
            model_request,
        });
    }
    if records.len() != expected_count {
        return Err(format!(
            "Q22 dataset has {} records, expected {expected_count}",
            records.len()
        ));
    }
    Ok(records)
}

fn parse_evaluation(
    input: &[u8],
    expected_count: usize,
) -> Result<BTreeMap<String, String>, String> {
    let text = std::str::from_utf8(input).map_err(|_| "Q22 evaluation must be UTF-8 TSV")?;
    if !text.ends_with('\n') {
        return Err("Q22 evaluation must end with one newline".to_string());
    }
    let mut lines = text.lines();
    if lines.next() != Some(EVAL_HEADER) {
        return Err("Q22 evaluation header is invalid".to_string());
    }
    let mut rows = BTreeMap::new();
    for (offset, line) in lines.enumerate() {
        let line_number = offset + 2;
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != 8
            || fields[1] != "quantity"
            || fields[2] != PREVIOUS_SUMMARY
            || !fields[0].starts_with("quantity-request/promotion/")
        {
            return Err(format!("invalid Q22 evaluation row {line_number}"));
        }
        let expected = expected_for_input(fields[3])?;
        if fields[4] != expected.model_request
            || fields[5] != expected.request
            || fields[6] != expected.artifact
            || fields[7] != format!("kernel committed {}", expected.artifact)
        {
            return Err(format!("Q22 kernel rejected evaluation row {line_number}"));
        }
        if rows
            .insert(fields[0].to_string(), fields[4].to_string())
            .is_some()
        {
            return Err(format!("duplicate Q22 evaluation ID {}", fields[0]));
        }
    }
    if rows.len() != expected_count {
        return Err(format!(
            "Q22 evaluation has {} records, expected {expected_count}",
            rows.len()
        ));
    }
    Ok(rows)
}

fn parse_predictions(
    path: &Path,
    expected_count: usize,
) -> Result<BTreeMap<String, String>, String> {
    let input = fs::read_to_string(path)
        .map_err(|error| format!("cannot read Q22 predictions {}: {error}", path.display()))?;
    let mut lines = input.lines();
    if lines.next() != Some(PREDICTION_HEADER) {
        return Err("Q22 prediction header is invalid".to_string());
    }
    let mut rows = BTreeMap::new();
    for (offset, line) in lines.enumerate() {
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != 2 || !is_operation(fields[1]) {
            return Err(format!("invalid Q22 prediction row {}", offset + 2));
        }
        if rows
            .insert(fields[0].to_string(), fields[1].to_string())
            .is_some()
        {
            return Err(format!("duplicate Q22 prediction ID {}", fields[0]));
        }
    }
    if rows.len() != expected_count {
        return Err(format!(
            "Q22 predictions have {} records, expected {expected_count}",
            rows.len()
        ));
    }
    Ok(rows)
}

fn validate_task_fields(
    value: &Value,
    line: usize,
    expected: &Expected,
    model_request: &str,
) -> Result<(), String> {
    if model_request != expected.model_request
        || string_field(value, "request", line)? != expected.request
        || string_field(value, "artifact", line)? != expected.artifact
        || string_field(value, "summary", line)?
            != format!("kernel committed {}", expected.artifact)
    {
        return Err(format!("Q22 kernel rejected training row {line}"));
    }
    Ok(())
}

fn expected_for_input(input: &str) -> Result<Expected, String> {
    let parts = input.split_whitespace().collect::<Vec<_>>();
    match parts.as_slice() {
        ["add", a, b] => {
            let a = integer(a, input)?;
            let b = integer(b, input)?;
            expected("add", format!("{a} {b}"), format!("result {}", a + b))
        }
        ["multiply", a, b] => {
            let a = integer(a, input)?;
            let b = integer(b, input)?;
            expected("multiply", format!("{a} {b}"), format!("result {}", a * b))
        }
        ["add-rational", left, right] => {
            let (a, b) = fraction(left, input)?;
            let (c, d) = fraction(right, input)?;
            let artifact = rational(a * d + c * b, b * d)?;
            expected(
                "add-rational",
                format!("{left} {right}"),
                format!("result {artifact}"),
            )
        }
        [
            "convert",
            value,
            conversion @ ("m-to-cm" | "cm-to-mm" | "kg-to-g"),
        ] => {
            let value = positive_integer(value, input)?;
            let (factor, unit) = match *conversion {
                "m-to-cm" => (100, "cm"),
                "cm-to-mm" => (10, "mm"),
                _ => (1000, "g"),
            };
            expected(
                "convert",
                format!("{value} {conversion}"),
                format!("result {} {unit}", value * factor),
            )
        }
        _ if input.starts_with("solve ") => expected_solve(input),
        _ => Err(format!("invalid Q22 input {input}")),
    }
}

fn expected_solve(input: &str) -> Result<Expected, String> {
    let equation = input
        .strip_prefix("solve ")
        .ok_or_else(|| format!("invalid Q22 input {input}"))?;
    let (coefficient, tail) = equation
        .split_once("*x+")
        .ok_or_else(|| format!("invalid Q22 input {input}"))?;
    let (offset, result) = tail
        .split_once('=')
        .ok_or_else(|| format!("invalid Q22 input {input}"))?;
    let coefficient = positive_integer(coefficient, input)?;
    let offset = integer(offset, input)?;
    let result = integer(result, input)?;
    let numerator = result - offset;
    if numerator % coefficient != 0 {
        return Err(format!("non-integral Q22 solution {input}"));
    }
    expected(
        "solve-linear",
        format!("{coefficient} {offset} {result}"),
        format!("x {}", numerator / coefficient),
    )
}

fn expected(operation: &str, arguments: String, artifact: String) -> Result<Expected, String> {
    let model_request = format!("quantity.{operation}");
    Ok(Expected {
        request: format!("{model_request} {arguments}"),
        model_request,
        artifact,
    })
}

fn fraction(value: &str, input: &str) -> Result<(i64, i64), String> {
    let (numerator, denominator) = value
        .split_once('/')
        .ok_or_else(|| format!("invalid Q22 fraction in {input}"))?;
    Ok((
        integer(numerator, input)?,
        positive_integer(denominator, input)?,
    ))
}

fn rational(mut numerator: i64, mut denominator: i64) -> Result<String, String> {
    if denominator == 0 {
        return Err("zero Q22 denominator".to_string());
    }
    if denominator < 0 {
        numerator = -numerator;
        denominator = -denominator;
    }
    let divisor = gcd(numerator, denominator);
    numerator /= divisor;
    denominator /= divisor;
    Ok(if denominator == 1 {
        numerator.to_string()
    } else {
        format!("{numerator}/{denominator}")
    })
}

fn gcd(mut a: i64, mut b: i64) -> i64 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.max(1)
}

fn integer(value: &str, input: &str) -> Result<i64, String> {
    value
        .parse::<i64>()
        .map_err(|_| format!("invalid integer in Q22 input {input}"))
}

fn positive_integer(value: &str, input: &str) -> Result<i64, String> {
    let value = integer(value, input)?;
    if value <= 0 {
        return Err(format!("expected positive integer in Q22 input {input}"));
    }
    Ok(value)
}

fn is_operation(value: &str) -> bool {
    matches!(
        value,
        "quantity.add"
            | "quantity.multiply"
            | "quantity.add-rational"
            | "quantity.convert"
            | "quantity.solve-linear"
    )
}

fn string_field(value: &Value, field: &str, line: usize) -> Result<String, String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .ok_or_else(|| format!("Q22 JSONL row {line} requires string field {field}"))
}

fn positive_usize(value: &str, field: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("invalid {field}"))?;
    if parsed == 0 {
        return Err(format!("{field} must be positive"));
    }
    Ok(parsed)
}

fn validate_sha256(value: &str, field: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(format!("{field} must be a lowercase SHA-256 digest"));
    }
    Ok(())
}

fn require_hash(bytes: &[u8], expected: &str, label: &str) -> Result<(), String> {
    let actual = sha256_hex(bytes);
    if actual != expected {
        return Err(format!(
            "{label} SHA-256 mismatch: expected {expected}, got {actual}"
        ));
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for byte in sha256(bytes) {
        out.push(char::from(HEX[usize::from(byte >> 4)]));
        out.push(char::from(HEX[usize::from(byte & 15)]));
    }
    out
}

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

fn sha256(input: &[u8]) -> [u8; 32] {
    let mut h = [
        0x6a09e667_u32,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut message = input.to_vec();
    let bit_len = u64::try_from(message.len()).unwrap_or(0).saturating_mul(8);
    message.push(0x80);
    while (message.len() % 64) != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in message.chunks_exact(64) {
        let mut w = [0_u32; 64];
        for (index, word) in w.iter_mut().take(16).enumerate() {
            let start = index * 4;
            *word = u32::from_be_bytes([
                chunk[start],
                chunk[start + 1],
                chunk[start + 2],
                chunk[start + 3],
            ]);
        }
        for index in 16..64 {
            let s0 = w[index - 15].rotate_right(7)
                ^ w[index - 15].rotate_right(18)
                ^ (w[index - 15] >> 3);
            let s1 = w[index - 2].rotate_right(17)
                ^ w[index - 2].rotate_right(19)
                ^ (w[index - 2] >> 10);
            w[index] = w[index - 16]
                .wrapping_add(s0)
                .wrapping_add(w[index - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d) = (h[0], h[1], h[2], h[3]);
        let (mut e, mut f, mut g, mut hh) = (h[4], h[5], h[6], h[7]);
        for index in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA256_K[index])
                .wrapping_add(w[index]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    let mut out = [0_u8; 32];
    for (index, word) in h.iter().enumerate() {
        out[index * 4..index * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_covers_all_five_q22_operations() {
        let cases = [
            ("add -2 7", "quantity.add", "result 5"),
            ("multiply -3 9", "quantity.multiply", "result -27"),
            (
                "add-rational -5/19 -22/9",
                "quantity.add-rational",
                "result -463/171",
            ),
            ("convert 12 kg-to-g", "quantity.convert", "result 12000 g"),
            ("solve 7*x+-2=33", "quantity.solve-linear", "x 5"),
        ];
        for (input, operation, artifact) in cases {
            let expected = expected_for_input(input).expect("valid task");
            assert_eq!(expected.model_request, operation);
            assert_eq!(expected.artifact, artifact);
        }
    }

    #[test]
    fn sha256_matches_the_standard_vector() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn prediction_metric_is_integer_and_exact() {
        let check = Q22Check {
            eval_sha256: "a".repeat(64),
            cases: 5,
            operation_exact: 4,
        };
        assert_eq!(check.operation_exact_rate_ppm(), 800_000);
        assert!(check.to_json_line().contains("\"valid\":true"));
    }
}
