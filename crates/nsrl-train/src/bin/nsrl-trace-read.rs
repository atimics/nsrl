#![deny(unsafe_code)]
#![cfg_attr(test, allow(dead_code))]

use std::env;
use std::fs;
use std::io::{self, Read, Write};
use std::path::PathBuf;

use nsrl_train::{
    MINI_TRANSFORMER_BINARY_ADAPTIVE_SHIFT_RECORD_LEN,
    MINI_TRANSFORMER_BINARY_FINAL_SUMMARY_RECORD_LEN,
    MINI_TRANSFORMER_BINARY_STEP_SAMPLE_RECORD_LEN, MINI_TRANSFORMER_BINARY_TAG_ADAPTIVE_SHIFT,
    MINI_TRANSFORMER_BINARY_TAG_FINAL_SUMMARY, MINI_TRANSFORMER_BINARY_TAG_STEP_SAMPLE,
    MINI_TRANSFORMER_BINARY_TRACE_HEADER_LEN, MINI_TRANSFORMER_BINARY_TRACE_MAGIC,
    MINI_TRANSFORMER_BINARY_TRACE_SCHEMA, MINI_TRANSFORMER_BINARY_TRACE_SCHEMA_ID,
    MINI_TRANSFORMER_BINARY_TRACE_VERSION,
};

#[cfg(not(test))]
fn main() {
    if let Err(error) = run() {
        eprintln!("nsrl-trace-read: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
fn main() {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputFormat {
    Table,
    Json,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StepLimit {
    All,
    Count(usize),
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut trace_path = None;
    let mut output_format = OutputFormat::Table;
    let mut step_limit = StepLimit::Count(16);
    let mut step_limit_explicit = false;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--trace" => {
                trace_path = Some(PathBuf::from(
                    args.next().ok_or("--trace requires a following path")?,
                ));
            }
            "--format" => {
                output_format =
                    parse_output_format(&args.next().ok_or("--format requires table or json")?)?;
                if output_format == OutputFormat::Json && !step_limit_explicit {
                    step_limit = StepLimit::All;
                }
            }
            "--max-step-records" => {
                step_limit = parse_step_limit(
                    &args
                        .next()
                        .ok_or("--max-step-records requires an integer or all")?,
                )?;
                step_limit_explicit = true;
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other if !other.starts_with("--") && trace_path.is_none() => {
                trace_path = Some(PathBuf::from(other));
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    let bytes = if let Some(path) = trace_path {
        fs::read(path)?
    } else {
        let mut bytes = Vec::new();
        io::stdin().read_to_end(&mut bytes)?;
        bytes
    };
    let trace = decode_trace(&bytes)?;

    match output_format {
        OutputFormat::Table => write_table(&mut io::stdout(), &trace, step_limit)?,
        OutputFormat::Json => write_json(&mut io::stdout(), &trace, step_limit)?,
    }

    Ok(())
}

fn parse_output_format(value: &str) -> Result<OutputFormat, Box<dyn std::error::Error>> {
    match value {
        "table" | "pretty" | "text" => Ok(OutputFormat::Table),
        "json" => Ok(OutputFormat::Json),
        _ => Err("--format requires table or json".into()),
    }
}

fn parse_step_limit(value: &str) -> Result<StepLimit, Box<dyn std::error::Error>> {
    match value {
        "all" => Ok(StepLimit::All),
        _ => Ok(StepLimit::Count(value.parse()?)),
    }
}

fn print_help() {
    println!(
        "Usage: nsrl-trace-read [TRACE_PATH|--trace TRACE_PATH] [--format table|json] [--max-step-records N|all]"
    );
    println!();
    println!("Decodes NSRL mini-transformer binary traces off the training hot path.");
    println!("Without a trace path, reads binary trace bytes from stdin.");
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BinaryTrace {
    header: TraceHeader,
    steps: Vec<StepRecord>,
    adaptive_shift_events: Vec<AdaptiveShiftRecord>,
    summary: FinalSummary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TraceHeader {
    version: u8,
    schema_id: u8,
    reserved: u16,
    initial_model_hash: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StepRecord {
    update_index: u32,
    window_start: u32,
    first_token: u8,
    last_token: u8,
    target_token: u8,
    predicted_token_before: u8,
    predicted_token_after: u8,
    target_probability_before_q15: i16,
    target_probability_after_q15: i16,
    residual_saturation_count: u16,
    component_saturation_count: u16,
    zero_delta_count: u16,
    attention_delta_l1: u32,
    total_delta_l1: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AdaptiveShiftRecord {
    batch_index: u32,
    component_code: u8,
    reason_code: u8,
    previous_shift: u8,
    next_shift: u8,
    delta: i8,
    observation_batches: u16,
    rejected_batches: u16,
    saturation_count: u16,
    zero_delta_count: u16,
    weight_delta_l1: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FinalSummary {
    trace_detail_code: u8,
    tokenizer_code: u8,
    attention_code: u8,
    position_code: u8,
    config_flags: u16,
    epochs: u32,
    seq_len: u32,
    stride: u32,
    window_offset: u32,
    max_windows: Option<u32>,
    batch_windows: u32,
    learning_rate: i32,
    output_learning_rate_shift: u8,
    mlp_learning_rate_shift: u8,
    embedding_learning_rate_shift: u8,
    attention_learning_rate_shift: u8,
    attention_q_learning_rate_shift: u8,
    attention_qk_learning_rate_shift: u8,
    token_count: u64,
    token_hash: u64,
    window_hash: u64,
    windows: u64,
    examined_windows: u64,
    updates: u64,
    accepted_batch_count: u64,
    rejected_batch_count: u64,
    rollback_count: u64,
    rejected_window_count: u64,
    loss_regression_rejected_batch_count: u64,
    final_invalid_forward_count: u64,
    initial_total_error: u64,
    final_total_error: u64,
    initial_probability_error_q15: u64,
    final_probability_error_q15: u64,
    probability_error_delta_q15: i64,
    initial_mistakes: u64,
    final_mistakes: u64,
    final_accuracy_per_mille: u16,
    output_head_saturation_count: u64,
    mlp_saturation_count: u64,
    embedding_saturation_count: u64,
    attention_saturation_count: u64,
    residual_saturation_count: u64,
    output_head_zero_delta_count: u64,
    mlp_zero_delta_count: u64,
    embedding_zero_delta_count: u64,
    attention_zero_delta_count: u64,
    output_head_delta_l1: u64,
    mlp_delta_l1: u64,
    embedding_delta_l1: u64,
    attention_delta_l1: u64,
    attention_q_delta_l1: u64,
    attention_k_delta_l1: u64,
    attention_v_delta_l1: u64,
    attention_o_delta_l1: u64,
    adaptive_rule_shift_adjustment_count: u64,
    adaptive_rule_update_count: u64,
    adaptive_rule_event_count: u64,
    adaptive_holographic_shift_adjustment_count: u64,
    adaptive_holographic_update_count: u64,
    adaptive_holographic_hash: u64,
    adaptive_attention_shift_adjustment_count: u64,
    adaptive_attention_holographic_update_count: u64,
    adaptive_attention_holographic_hash: u64,
    final_output_learning_rate_shift: u8,
    final_mlp_learning_rate_shift: u8,
    final_embedding_learning_rate_shift: u8,
    final_attention_learning_rate_shift: u8,
    final_attention_q_learning_rate_shift: u8,
    final_attention_qk_learning_rate_shift: u8,
    initial_model_hash: u64,
    final_model_hash: u64,
    initial_embedding_hash: u64,
    final_embedding_hash: u64,
    initial_output_head_hash: u64,
    final_output_head_hash: u64,
    initial_mlp_hash: u64,
    final_mlp_hash: u64,
    initial_attention_hash: u64,
    final_attention_hash: u64,
    initial_attention_q_hash: u64,
    final_attention_q_hash: u64,
    initial_attention_k_hash: u64,
    final_attention_k_hash: u64,
    initial_attention_v_hash: u64,
    final_attention_v_hash: u64,
    initial_attention_o_hash: u64,
    final_attention_o_hash: u64,
    final_logits_hash: u64,
}

fn decode_trace(bytes: &[u8]) -> Result<BinaryTrace, Box<dyn std::error::Error>> {
    if bytes.len() < MINI_TRANSFORMER_BINARY_TRACE_HEADER_LEN {
        return Err("trace is shorter than the binary header".into());
    }
    if &bytes[..4] != MINI_TRANSFORMER_BINARY_TRACE_MAGIC {
        return Err("trace magic does not match NSRL".into());
    }
    let header = TraceHeader {
        version: bytes[4],
        schema_id: bytes[5],
        reserved: read_u16_at(bytes, 6)?,
        initial_model_hash: read_u64_at(bytes, 8)?,
    };
    if header.version != MINI_TRANSFORMER_BINARY_TRACE_VERSION {
        return Err(format!("unsupported trace version: {}", header.version).into());
    }
    if header.schema_id != MINI_TRANSFORMER_BINARY_TRACE_SCHEMA_ID {
        return Err(format!("unsupported trace schema id: {}", header.schema_id).into());
    }

    let mut offset = MINI_TRANSFORMER_BINARY_TRACE_HEADER_LEN;
    let mut steps = Vec::new();
    let mut adaptive_shift_events = Vec::new();
    let mut summary = None;

    while offset < bytes.len() {
        let tag = bytes[offset];
        match tag {
            MINI_TRANSFORMER_BINARY_TAG_STEP_SAMPLE => {
                let end = checked_record_end(
                    offset,
                    MINI_TRANSFORMER_BINARY_STEP_SAMPLE_RECORD_LEN,
                    bytes.len(),
                    "step sample",
                )?;
                steps.push(decode_step_record(&bytes[offset..end])?);
                offset = end;
            }
            MINI_TRANSFORMER_BINARY_TAG_ADAPTIVE_SHIFT => {
                let end = checked_record_end(
                    offset,
                    MINI_TRANSFORMER_BINARY_ADAPTIVE_SHIFT_RECORD_LEN,
                    bytes.len(),
                    "adaptive shift event",
                )?;
                adaptive_shift_events.push(decode_adaptive_shift_record(&bytes[offset..end])?);
                offset = end;
            }
            MINI_TRANSFORMER_BINARY_TAG_FINAL_SUMMARY => {
                let end = checked_record_end(
                    offset,
                    MINI_TRANSFORMER_BINARY_FINAL_SUMMARY_RECORD_LEN,
                    bytes.len(),
                    "final summary",
                )?;
                if end != bytes.len() {
                    return Err("final summary record must be the last trace record".into());
                }
                summary = Some(decode_final_summary(&bytes[offset..end])?);
                offset = end;
            }
            _ => {
                return Err(
                    format!("unknown binary trace tag 0x{tag:02x} at byte {offset}").into(),
                );
            }
        }
    }

    let summary = summary.ok_or("trace is missing final summary record")?;
    if header.initial_model_hash != summary.initial_model_hash {
        return Err("header initial model hash does not match final summary".into());
    }

    Ok(BinaryTrace {
        header,
        steps,
        adaptive_shift_events,
        summary,
    })
}

fn checked_record_end(
    offset: usize,
    len: usize,
    trace_len: usize,
    label: &str,
) -> Result<usize, Box<dyn std::error::Error>> {
    let end = offset
        .checked_add(len)
        .ok_or_else(|| format!("{label} record length overflow"))?;
    if end > trace_len {
        return Err(format!("{label} record is truncated at byte {offset}").into());
    }
    Ok(end)
}

fn decode_step_record(bytes: &[u8]) -> Result<StepRecord, Box<dyn std::error::Error>> {
    Ok(StepRecord {
        update_index: read_u32_at(bytes, 1)?,
        window_start: read_u32_at(bytes, 5)?,
        first_token: bytes[9],
        last_token: bytes[10],
        target_token: bytes[11],
        predicted_token_before: bytes[12],
        predicted_token_after: bytes[13],
        target_probability_before_q15: read_i16_at(bytes, 14)?,
        target_probability_after_q15: read_i16_at(bytes, 16)?,
        residual_saturation_count: read_u16_at(bytes, 18)?,
        component_saturation_count: read_u16_at(bytes, 20)?,
        zero_delta_count: read_u16_at(bytes, 22)?,
        attention_delta_l1: read_u32_at(bytes, 24)?,
        total_delta_l1: read_u32_at(bytes, 28)?,
    })
}

fn decode_adaptive_shift_record(
    bytes: &[u8],
) -> Result<AdaptiveShiftRecord, Box<dyn std::error::Error>> {
    Ok(AdaptiveShiftRecord {
        batch_index: read_u32_at(bytes, 1)?,
        component_code: bytes[5],
        reason_code: bytes[6],
        previous_shift: bytes[7],
        next_shift: bytes[8],
        delta: bytes[9] as i8,
        observation_batches: read_u16_at(bytes, 10)?,
        rejected_batches: read_u16_at(bytes, 12)?,
        saturation_count: read_u16_at(bytes, 14)?,
        zero_delta_count: read_u16_at(bytes, 16)?,
        weight_delta_l1: read_u32_at(bytes, 18)?,
    })
}

fn decode_final_summary(bytes: &[u8]) -> Result<FinalSummary, Box<dyn std::error::Error>> {
    let mut cursor = TraceCursor::new(bytes);
    cursor.expect_tag(MINI_TRANSFORMER_BINARY_TAG_FINAL_SUMMARY)?;
    let summary = FinalSummary {
        trace_detail_code: cursor.read_u8()?,
        tokenizer_code: cursor.read_u8()?,
        attention_code: cursor.read_u8()?,
        position_code: cursor.read_u8()?,
        config_flags: cursor.read_u16()?,
        epochs: cursor.read_u32()?,
        seq_len: cursor.read_u32()?,
        stride: cursor.read_u32()?,
        window_offset: cursor.read_u32()?,
        max_windows: decode_optional_u32(cursor.read_u32()?),
        batch_windows: cursor.read_u32()?,
        learning_rate: cursor.read_i32()?,
        output_learning_rate_shift: cursor.read_u8()?,
        mlp_learning_rate_shift: cursor.read_u8()?,
        embedding_learning_rate_shift: cursor.read_u8()?,
        attention_learning_rate_shift: cursor.read_u8()?,
        attention_q_learning_rate_shift: cursor.read_u8()?,
        attention_qk_learning_rate_shift: cursor.read_u8()?,
        token_count: cursor.read_u64()?,
        token_hash: cursor.read_u64()?,
        window_hash: cursor.read_u64()?,
        windows: cursor.read_u64()?,
        examined_windows: cursor.read_u64()?,
        updates: cursor.read_u64()?,
        accepted_batch_count: cursor.read_u64()?,
        rejected_batch_count: cursor.read_u64()?,
        rollback_count: cursor.read_u64()?,
        rejected_window_count: cursor.read_u64()?,
        loss_regression_rejected_batch_count: cursor.read_u64()?,
        final_invalid_forward_count: cursor.read_u64()?,
        initial_total_error: cursor.read_u64()?,
        final_total_error: cursor.read_u64()?,
        initial_probability_error_q15: cursor.read_u64()?,
        final_probability_error_q15: cursor.read_u64()?,
        probability_error_delta_q15: cursor.read_i64()?,
        initial_mistakes: cursor.read_u64()?,
        final_mistakes: cursor.read_u64()?,
        final_accuracy_per_mille: cursor.read_u16()?,
        output_head_saturation_count: cursor.read_u64()?,
        mlp_saturation_count: cursor.read_u64()?,
        embedding_saturation_count: cursor.read_u64()?,
        attention_saturation_count: cursor.read_u64()?,
        residual_saturation_count: cursor.read_u64()?,
        output_head_zero_delta_count: cursor.read_u64()?,
        mlp_zero_delta_count: cursor.read_u64()?,
        embedding_zero_delta_count: cursor.read_u64()?,
        attention_zero_delta_count: cursor.read_u64()?,
        output_head_delta_l1: cursor.read_u64()?,
        mlp_delta_l1: cursor.read_u64()?,
        embedding_delta_l1: cursor.read_u64()?,
        attention_delta_l1: cursor.read_u64()?,
        attention_q_delta_l1: cursor.read_u64()?,
        attention_k_delta_l1: cursor.read_u64()?,
        attention_v_delta_l1: cursor.read_u64()?,
        attention_o_delta_l1: cursor.read_u64()?,
        adaptive_rule_shift_adjustment_count: cursor.read_u64()?,
        adaptive_rule_update_count: cursor.read_u64()?,
        adaptive_rule_event_count: cursor.read_u64()?,
        adaptive_holographic_shift_adjustment_count: cursor.read_u64()?,
        adaptive_holographic_update_count: cursor.read_u64()?,
        adaptive_holographic_hash: cursor.read_u64()?,
        adaptive_attention_shift_adjustment_count: cursor.read_u64()?,
        adaptive_attention_holographic_update_count: cursor.read_u64()?,
        adaptive_attention_holographic_hash: cursor.read_u64()?,
        final_output_learning_rate_shift: cursor.read_u8()?,
        final_mlp_learning_rate_shift: cursor.read_u8()?,
        final_embedding_learning_rate_shift: cursor.read_u8()?,
        final_attention_learning_rate_shift: cursor.read_u8()?,
        final_attention_q_learning_rate_shift: cursor.read_u8()?,
        final_attention_qk_learning_rate_shift: cursor.read_u8()?,
        initial_model_hash: cursor.read_u64()?,
        final_model_hash: cursor.read_u64()?,
        initial_embedding_hash: cursor.read_u64()?,
        final_embedding_hash: cursor.read_u64()?,
        initial_output_head_hash: cursor.read_u64()?,
        final_output_head_hash: cursor.read_u64()?,
        initial_mlp_hash: cursor.read_u64()?,
        final_mlp_hash: cursor.read_u64()?,
        initial_attention_hash: cursor.read_u64()?,
        final_attention_hash: cursor.read_u64()?,
        initial_attention_q_hash: cursor.read_u64()?,
        final_attention_q_hash: cursor.read_u64()?,
        initial_attention_k_hash: cursor.read_u64()?,
        final_attention_k_hash: cursor.read_u64()?,
        initial_attention_v_hash: cursor.read_u64()?,
        final_attention_v_hash: cursor.read_u64()?,
        initial_attention_o_hash: cursor.read_u64()?,
        final_attention_o_hash: cursor.read_u64()?,
        final_logits_hash: cursor.read_u64()?,
    };
    cursor.expect_end()?;
    Ok(summary)
}

fn decode_optional_u32(value: u32) -> Option<u32> {
    if value == u32::MAX { None } else { Some(value) }
}

#[derive(Debug, Clone, Copy)]
struct TraceCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> TraceCursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn expect_tag(&mut self, expected: u8) -> Result<(), Box<dyn std::error::Error>> {
        let tag = self.read_u8()?;
        if tag != expected {
            return Err(format!("expected tag 0x{expected:02x}, got 0x{tag:02x}").into());
        }
        Ok(())
    }

    fn read_u8(&mut self) -> Result<u8, Box<dyn std::error::Error>> {
        let value = *self
            .bytes
            .get(self.offset)
            .ok_or("unexpected end of trace record")?;
        self.offset += 1;
        Ok(value)
    }

    fn read_u16(&mut self) -> Result<u16, Box<dyn std::error::Error>> {
        let value = read_u16_at(self.bytes, self.offset)?;
        self.offset += 2;
        Ok(value)
    }

    fn read_u32(&mut self) -> Result<u32, Box<dyn std::error::Error>> {
        let value = read_u32_at(self.bytes, self.offset)?;
        self.offset += 4;
        Ok(value)
    }

    fn read_i32(&mut self) -> Result<i32, Box<dyn std::error::Error>> {
        let value = read_i32_at(self.bytes, self.offset)?;
        self.offset += 4;
        Ok(value)
    }

    fn read_u64(&mut self) -> Result<u64, Box<dyn std::error::Error>> {
        let value = read_u64_at(self.bytes, self.offset)?;
        self.offset += 8;
        Ok(value)
    }

    fn read_i64(&mut self) -> Result<i64, Box<dyn std::error::Error>> {
        let value = read_i64_at(self.bytes, self.offset)?;
        self.offset += 8;
        Ok(value)
    }

    fn expect_end(&self) -> Result<(), Box<dyn std::error::Error>> {
        if self.offset != self.bytes.len() {
            return Err(format!(
                "final summary decoder consumed {} bytes but record is {} bytes",
                self.offset,
                self.bytes.len()
            )
            .into());
        }
        Ok(())
    }
}

fn read_i16_at(bytes: &[u8], offset: usize) -> Result<i16, Box<dyn std::error::Error>> {
    Ok(i16::from_le_bytes(read_array_at(bytes, offset)?))
}

fn read_u16_at(bytes: &[u8], offset: usize) -> Result<u16, Box<dyn std::error::Error>> {
    Ok(u16::from_le_bytes(read_array_at(bytes, offset)?))
}

fn read_u32_at(bytes: &[u8], offset: usize) -> Result<u32, Box<dyn std::error::Error>> {
    Ok(u32::from_le_bytes(read_array_at(bytes, offset)?))
}

fn read_i32_at(bytes: &[u8], offset: usize) -> Result<i32, Box<dyn std::error::Error>> {
    Ok(i32::from_le_bytes(read_array_at(bytes, offset)?))
}

fn read_u64_at(bytes: &[u8], offset: usize) -> Result<u64, Box<dyn std::error::Error>> {
    Ok(u64::from_le_bytes(read_array_at(bytes, offset)?))
}

fn read_i64_at(bytes: &[u8], offset: usize) -> Result<i64, Box<dyn std::error::Error>> {
    Ok(i64::from_le_bytes(read_array_at(bytes, offset)?))
}

fn read_array_at<const N: usize>(
    bytes: &[u8],
    offset: usize,
) -> Result<[u8; N], Box<dyn std::error::Error>> {
    let end = offset
        .checked_add(N)
        .ok_or("trace offset overflow while reading integer")?;
    let slice = bytes
        .get(offset..end)
        .ok_or("unexpected end of trace while reading integer")?;
    Ok(slice.try_into()?)
}

fn write_table<W: Write>(
    writer: &mut W,
    trace: &BinaryTrace,
    step_limit: StepLimit,
) -> Result<(), Box<dyn std::error::Error>> {
    let summary = &trace.summary;
    writeln!(writer, "schema: {MINI_TRANSFORMER_BINARY_TRACE_SCHEMA}")?;
    writeln!(
        writer,
        "header: version={} schema_id={} reserved={} initial_model_hash={}",
        trace.header.version,
        trace.header.schema_id,
        trace.header.reserved,
        format_hash(trace.header.initial_model_hash)
    )?;
    writeln!(
        writer,
        "records: step_samples={} adaptive_shift_events={} final_summary=1",
        trace.steps.len(),
        trace.adaptive_shift_events.len()
    )?;
    writeln!(
        writer,
        "training: trace_detail={} tokenizer={} attention={} position={} epochs={} seq_len={} stride={} window_offset={} max_windows={} batch_windows={}",
        trace_detail_label(summary.trace_detail_code),
        tokenizer_label(summary.tokenizer_code),
        attention_label(summary.attention_code),
        position_label(summary.position_code),
        summary.epochs,
        summary.seq_len,
        summary.stride,
        summary.window_offset,
        optional_u32_label(summary.max_windows),
        summary.batch_windows
    )?;
    writeln!(
        writer,
        "learning: rate={} shifts output={} mlp={} emb={} attn={} q={} qk={} final_output={} final_mlp={} final_emb={} final_attn={} final_q={} final_qk={}",
        summary.learning_rate,
        summary.output_learning_rate_shift,
        summary.mlp_learning_rate_shift,
        summary.embedding_learning_rate_shift,
        summary.attention_learning_rate_shift,
        summary.attention_q_learning_rate_shift,
        summary.attention_qk_learning_rate_shift,
        summary.final_output_learning_rate_shift,
        summary.final_mlp_learning_rate_shift,
        summary.final_embedding_learning_rate_shift,
        summary.final_attention_learning_rate_shift,
        summary.final_attention_q_learning_rate_shift,
        summary.final_attention_qk_learning_rate_shift
    )?;
    writeln!(
        writer,
        "metrics: tokens={} windows={} examined={} updates={} accepted={} rejected={} rollbacks={} rejected_windows={} accuracy_per_mille={} probability_error_delta_q15={}",
        summary.token_count,
        summary.windows,
        summary.examined_windows,
        summary.updates,
        summary.accepted_batch_count,
        summary.rejected_batch_count,
        summary.rollback_count,
        summary.rejected_window_count,
        summary.final_accuracy_per_mille,
        summary.probability_error_delta_q15
    )?;
    writeln!(
        writer,
        "deltas: output={} mlp={} embedding={} attention={} q={} k={} v={} o={}",
        summary.output_head_delta_l1,
        summary.mlp_delta_l1,
        summary.embedding_delta_l1,
        summary.attention_delta_l1,
        summary.attention_q_delta_l1,
        summary.attention_k_delta_l1,
        summary.attention_v_delta_l1,
        summary.attention_o_delta_l1
    )?;
    writeln!(
        writer,
        "adaptive: rule_adjustments={} rule_updates={} rule_events={} holographic_adjustments={} attention_adjustments={}",
        summary.adaptive_rule_shift_adjustment_count,
        summary.adaptive_rule_update_count,
        summary.adaptive_rule_event_count,
        summary.adaptive_holographic_shift_adjustment_count,
        summary.adaptive_attention_shift_adjustment_count
    )?;
    writeln!(
        writer,
        "hashes: token={} window={} initial_model={} final_model={} logits={}",
        format_hash(summary.token_hash),
        format_hash(summary.window_hash),
        format_hash(summary.initial_model_hash),
        format_hash(summary.final_model_hash),
        format_hash(summary.final_logits_hash)
    )?;

    if !trace.adaptive_shift_events.is_empty() {
        writeln!(writer)?;
        writeln!(writer, "adaptive shift events:")?;
        writeln!(
            writer,
            "  batch component      reason        prev next delta obs rejected saturation zero_delta weight_delta"
        )?;
        for event in &trace.adaptive_shift_events {
            writeln!(
                writer,
                "  {:>5} {:<14} {:<12} {:>4} {:>4} {:>5} {:>3} {:>8} {:>10} {:>10} {:>12}",
                event.batch_index,
                component_label(event.component_code),
                reason_label(event.reason_code),
                event.previous_shift,
                event.next_shift,
                event.delta,
                event.observation_batches,
                event.rejected_batches,
                event.saturation_count,
                event.zero_delta_count,
                event.weight_delta_l1
            )?;
        }
    }

    let rendered_step_count = limited_len(trace.steps.len(), step_limit);
    if rendered_step_count > 0 {
        writeln!(writer)?;
        writeln!(writer, "step samples:")?;
        writeln!(
            writer,
            "  update window target before after prob_before prob_after residual_sat component_sat zero_delta attn_delta total_delta"
        )?;
        for step in trace.steps.iter().take(rendered_step_count) {
            writeln!(
                writer,
                "  {:>6} {:>6} {:>6} {:>6} {:>5} {:>11} {:>10} {:>12} {:>13} {:>10} {:>10} {:>11}",
                step.update_index,
                step.window_start,
                step.target_token,
                step.predicted_token_before,
                step.predicted_token_after,
                step.target_probability_before_q15,
                step.target_probability_after_q15,
                step.residual_saturation_count,
                step.component_saturation_count,
                step.zero_delta_count,
                step.attention_delta_l1,
                step.total_delta_l1
            )?;
        }
        if rendered_step_count < trace.steps.len() {
            writeln!(
                writer,
                "  ... {} more step sample records omitted",
                trace.steps.len() - rendered_step_count
            )?;
        }
    }

    Ok(())
}

fn write_json<W: Write>(
    writer: &mut W,
    trace: &BinaryTrace,
    step_limit: StepLimit,
) -> Result<(), Box<dyn std::error::Error>> {
    let summary = &trace.summary;
    write!(
        writer,
        "{{\"schema\":\"{}\",\"header\":{{\"version\":{},\"schema_id\":{},\"reserved\":{},\"initial_model_hash\":{}}},",
        MINI_TRANSFORMER_BINARY_TRACE_SCHEMA,
        trace.header.version,
        trace.header.schema_id,
        trace.header.reserved,
        trace.header.initial_model_hash
    )?;
    write!(
        writer,
        "\"record_counts\":{{\"step_samples\":{},\"adaptive_shift_events\":{},\"final_summary\":1}},",
        trace.steps.len(),
        trace.adaptive_shift_events.len()
    )?;
    write_final_summary_json(writer, summary)?;
    write!(writer, ",\"adaptive_shift_events\":[")?;
    for (index, event) in trace.adaptive_shift_events.iter().enumerate() {
        if index > 0 {
            write!(writer, ",")?;
        }
        write_adaptive_shift_json(writer, event)?;
    }
    write!(writer, "],\"steps\":[")?;
    let rendered_step_count = limited_len(trace.steps.len(), step_limit);
    for (index, step) in trace.steps.iter().take(rendered_step_count).enumerate() {
        if index > 0 {
            write!(writer, ",")?;
        }
        write_step_json(writer, step)?;
    }
    writeln!(
        writer,
        "],\"omitted_step_records\":{}}}",
        trace.steps.len() - rendered_step_count
    )?;
    Ok(())
}

fn write_final_summary_json<W: Write>(
    writer: &mut W,
    summary: &FinalSummary,
) -> Result<(), Box<dyn std::error::Error>> {
    write!(
        writer,
        "\"summary\":{{\"trace_detail\":\"{}\",\"trace_detail_code\":{},\"tokenizer\":\"{}\",\"tokenizer_code\":{},\"attention\":\"{}\",\"attention_code\":{},\"position\":\"{}\",\"position_code\":{},\"config_flags\":{},",
        trace_detail_label(summary.trace_detail_code),
        summary.trace_detail_code,
        tokenizer_label(summary.tokenizer_code),
        summary.tokenizer_code,
        attention_label(summary.attention_code),
        summary.attention_code,
        position_label(summary.position_code),
        summary.position_code,
        summary.config_flags
    )?;
    write!(
        writer,
        "\"training\":{{\"epochs\":{},\"seq_len\":{},\"stride\":{},\"window_offset\":{},\"max_windows\":{},\"batch_windows\":{},\"learning_rate\":{},\"output_learning_rate_shift\":{},\"mlp_learning_rate_shift\":{},\"embedding_learning_rate_shift\":{},\"attention_learning_rate_shift\":{},\"attention_q_learning_rate_shift\":{},\"attention_qk_learning_rate_shift\":{},\"final_output_learning_rate_shift\":{},\"final_mlp_learning_rate_shift\":{},\"final_embedding_learning_rate_shift\":{},\"final_attention_learning_rate_shift\":{},\"final_attention_q_learning_rate_shift\":{},\"final_attention_qk_learning_rate_shift\":{}}},",
        summary.epochs,
        summary.seq_len,
        summary.stride,
        summary.window_offset,
        optional_u32_json(summary.max_windows),
        summary.batch_windows,
        summary.learning_rate,
        summary.output_learning_rate_shift,
        summary.mlp_learning_rate_shift,
        summary.embedding_learning_rate_shift,
        summary.attention_learning_rate_shift,
        summary.attention_q_learning_rate_shift,
        summary.attention_qk_learning_rate_shift,
        summary.final_output_learning_rate_shift,
        summary.final_mlp_learning_rate_shift,
        summary.final_embedding_learning_rate_shift,
        summary.final_attention_learning_rate_shift,
        summary.final_attention_q_learning_rate_shift,
        summary.final_attention_qk_learning_rate_shift
    )?;
    write!(
        writer,
        "\"data\":{{\"token_count\":{},\"token_hash\":{},\"window_hash\":{},\"windows\":{},\"examined_windows\":{},\"updates\":{}}},",
        summary.token_count,
        summary.token_hash,
        summary.window_hash,
        summary.windows,
        summary.examined_windows,
        summary.updates
    )?;
    write!(
        writer,
        "\"metrics\":{{\"accepted_batch_count\":{},\"rejected_batch_count\":{},\"rollback_count\":{},\"rejected_window_count\":{},\"loss_regression_rejected_batch_count\":{},\"final_invalid_forward_count\":{},\"initial_total_error\":{},\"final_total_error\":{},\"initial_probability_error_q15\":{},\"final_probability_error_q15\":{},\"probability_error_delta_q15\":{},\"initial_mistakes\":{},\"final_mistakes\":{},\"final_accuracy_per_mille\":{}}},",
        summary.accepted_batch_count,
        summary.rejected_batch_count,
        summary.rollback_count,
        summary.rejected_window_count,
        summary.loss_regression_rejected_batch_count,
        summary.final_invalid_forward_count,
        summary.initial_total_error,
        summary.final_total_error,
        summary.initial_probability_error_q15,
        summary.final_probability_error_q15,
        summary.probability_error_delta_q15,
        summary.initial_mistakes,
        summary.final_mistakes,
        summary.final_accuracy_per_mille
    )?;
    write!(
        writer,
        "\"saturation\":{{\"output_head\":{},\"mlp\":{},\"embedding\":{},\"attention\":{},\"residual\":{}}},",
        summary.output_head_saturation_count,
        summary.mlp_saturation_count,
        summary.embedding_saturation_count,
        summary.attention_saturation_count,
        summary.residual_saturation_count
    )?;
    write!(
        writer,
        "\"zero_delta\":{{\"output_head\":{},\"mlp\":{},\"embedding\":{},\"attention\":{}}},",
        summary.output_head_zero_delta_count,
        summary.mlp_zero_delta_count,
        summary.embedding_zero_delta_count,
        summary.attention_zero_delta_count
    )?;
    write!(
        writer,
        "\"delta_l1\":{{\"output_head\":{},\"mlp\":{},\"embedding\":{},\"attention\":{},\"attention_q\":{},\"attention_k\":{},\"attention_v\":{},\"attention_o\":{}}},",
        summary.output_head_delta_l1,
        summary.mlp_delta_l1,
        summary.embedding_delta_l1,
        summary.attention_delta_l1,
        summary.attention_q_delta_l1,
        summary.attention_k_delta_l1,
        summary.attention_v_delta_l1,
        summary.attention_o_delta_l1
    )?;
    write!(
        writer,
        "\"adaptive\":{{\"rule_shift_adjustment_count\":{},\"rule_update_count\":{},\"rule_event_count\":{},\"holographic_shift_adjustment_count\":{},\"holographic_update_count\":{},\"holographic_hash\":{},\"attention_shift_adjustment_count\":{},\"attention_holographic_update_count\":{},\"attention_holographic_hash\":{}}},",
        summary.adaptive_rule_shift_adjustment_count,
        summary.adaptive_rule_update_count,
        summary.adaptive_rule_event_count,
        summary.adaptive_holographic_shift_adjustment_count,
        summary.adaptive_holographic_update_count,
        summary.adaptive_holographic_hash,
        summary.adaptive_attention_shift_adjustment_count,
        summary.adaptive_attention_holographic_update_count,
        summary.adaptive_attention_holographic_hash
    )?;
    write!(
        writer,
        "\"hashes\":{{\"initial_model\":{},\"final_model\":{},\"initial_embedding\":{},\"final_embedding\":{},\"initial_output_head\":{},\"final_output_head\":{},\"initial_mlp\":{},\"final_mlp\":{},\"initial_attention\":{},\"final_attention\":{},\"initial_attention_q\":{},\"final_attention_q\":{},\"initial_attention_k\":{},\"final_attention_k\":{},\"initial_attention_v\":{},\"final_attention_v\":{},\"initial_attention_o\":{},\"final_attention_o\":{},\"final_logits\":{}}}}}",
        summary.initial_model_hash,
        summary.final_model_hash,
        summary.initial_embedding_hash,
        summary.final_embedding_hash,
        summary.initial_output_head_hash,
        summary.final_output_head_hash,
        summary.initial_mlp_hash,
        summary.final_mlp_hash,
        summary.initial_attention_hash,
        summary.final_attention_hash,
        summary.initial_attention_q_hash,
        summary.final_attention_q_hash,
        summary.initial_attention_k_hash,
        summary.final_attention_k_hash,
        summary.initial_attention_v_hash,
        summary.final_attention_v_hash,
        summary.initial_attention_o_hash,
        summary.final_attention_o_hash,
        summary.final_logits_hash
    )?;
    Ok(())
}

fn write_step_json<W: Write>(
    writer: &mut W,
    step: &StepRecord,
) -> Result<(), Box<dyn std::error::Error>> {
    write!(
        writer,
        "{{\"update_index\":{},\"window_start\":{},\"first_token\":{},\"last_token\":{},\"target_token\":{},\"predicted_token_before\":{},\"predicted_token_after\":{},\"target_probability_before_q15\":{},\"target_probability_after_q15\":{},\"residual_saturation_count\":{},\"component_saturation_count\":{},\"zero_delta_count\":{},\"attention_delta_l1\":{},\"total_delta_l1\":{}}}",
        step.update_index,
        step.window_start,
        step.first_token,
        step.last_token,
        step.target_token,
        step.predicted_token_before,
        step.predicted_token_after,
        step.target_probability_before_q15,
        step.target_probability_after_q15,
        step.residual_saturation_count,
        step.component_saturation_count,
        step.zero_delta_count,
        step.attention_delta_l1,
        step.total_delta_l1
    )?;
    Ok(())
}

fn write_adaptive_shift_json<W: Write>(
    writer: &mut W,
    event: &AdaptiveShiftRecord,
) -> Result<(), Box<dyn std::error::Error>> {
    write!(
        writer,
        "{{\"batch_index\":{},\"component\":\"{}\",\"component_code\":{},\"reason\":\"{}\",\"reason_code\":{},\"previous_shift\":{},\"next_shift\":{},\"delta\":{},\"observation_batches\":{},\"rejected_batches\":{},\"saturation_count\":{},\"zero_delta_count\":{},\"weight_delta_l1\":{}}}",
        event.batch_index,
        component_label(event.component_code),
        event.component_code,
        reason_label(event.reason_code),
        event.reason_code,
        event.previous_shift,
        event.next_shift,
        event.delta,
        event.observation_batches,
        event.rejected_batches,
        event.saturation_count,
        event.zero_delta_count,
        event.weight_delta_l1
    )?;
    Ok(())
}

fn limited_len(len: usize, limit: StepLimit) -> usize {
    match limit {
        StepLimit::All => len,
        StepLimit::Count(limit) => len.min(limit),
    }
}

fn optional_u32_label(value: Option<u32>) -> String {
    match value {
        Some(value) => value.to_string(),
        None => String::from("none"),
    }
}

fn optional_u32_json(value: Option<u32>) -> String {
    match value {
        Some(value) => value.to_string(),
        None => String::from("null"),
    }
}

fn format_hash(value: u64) -> String {
    format!("0x{value:016x}")
}

fn trace_detail_label(code: u8) -> &'static str {
    match code {
        0 => "full",
        1 => "summary",
        2 => "none",
        _ => "unknown",
    }
}

fn tokenizer_label(code: u8) -> &'static str {
    match code {
        0 => "identity",
        1 => "ascii_lower_text",
        _ => "unknown",
    }
}

fn attention_label(code: u8) -> &'static str {
    match code {
        0 => "base2_softmax",
        1 => "linear",
        2 => "linear_streaming_nope",
        3 => "linear_streaming_ttt_nope",
        _ => "unknown",
    }
}

fn position_label(code: u8) -> &'static str {
    match code {
        0 => "learned_absolute_i16",
        1 => "nope",
        _ => "unknown",
    }
}

fn component_label(code: u8) -> &'static str {
    match code {
        0 => "output_head",
        1 => "mlp",
        2 => "embedding",
        3 => "attention",
        4 => "attention_q",
        5 => "attention_qk",
        _ => "unknown",
    }
}

fn reason_label(code: u8) -> &'static str {
    match code {
        0 => "rollback",
        1 => "saturation",
        2 => "zero_delta",
        3 => "movement",
        4 => "holographic",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nsrl_train::{
        MiniTransformerAttentionKind, MiniTransformerMlpModel, MiniTransformerMlpTrainConfig,
        MiniTransformerPositionPolicy, MiniTransformerTraceDetail,
        run_mini_transformer_mlp_training_from_model_with_progress_and_trace_detail,
    };

    #[test]
    fn decodes_generated_binary_trace() {
        let tokens: Vec<u8> = (0..64).map(|value| value as u8).collect();
        let config = MiniTransformerMlpTrainConfig {
            seq_len: 4,
            max_windows: Some(8),
            batch_windows: 2,
            attention_kind: MiniTransformerAttentionKind::Linear,
            position_policy: MiniTransformerPositionPolicy::Nope,
            ..MiniTransformerMlpTrainConfig::default()
        };
        let model = MiniTransformerMlpModel::new_initial_with_seq_len(config.seq_len);
        let run = run_mini_transformer_mlp_training_from_model_with_progress_and_trace_detail(
            &tokens,
            config,
            model,
            0,
            MiniTransformerTraceDetail::Summary,
            |_| Ok(()),
        )
        .expect("training run");

        let decoded = decode_trace(&run.trace.to_binary_trace_v1()).expect("decode trace");

        assert_eq!(
            decoded.header.version,
            MINI_TRANSFORMER_BINARY_TRACE_VERSION
        );
        assert_eq!(
            decoded.header.schema_id,
            MINI_TRANSFORMER_BINARY_TRACE_SCHEMA_ID
        );
        assert_eq!(decoded.steps.len(), run.trace.steps.len());
        assert_eq!(decoded.summary.updates, run.trace.updates as u64);
        assert_eq!(decoded.summary.final_model_hash, run.trace.final_model_hash);
    }

    #[test]
    fn rejects_truncated_step_record() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MINI_TRANSFORMER_BINARY_TRACE_MAGIC);
        bytes.push(MINI_TRANSFORMER_BINARY_TRACE_VERSION);
        bytes.push(MINI_TRANSFORMER_BINARY_TRACE_SCHEMA_ID);
        bytes.extend_from_slice(&0_u16.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.push(MINI_TRANSFORMER_BINARY_TAG_STEP_SAMPLE);

        let error = decode_trace(&bytes).expect_err("truncated trace should fail");

        assert!(
            error
                .to_string()
                .contains("step sample record is truncated")
        );
    }
}
