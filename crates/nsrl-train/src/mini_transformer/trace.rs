//! Stable JSON and binary trace serialization.

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MiniTransformerBinaryTraceRecord<'a> {
    Header { initial_model_hash: u64 },
    StepSample(&'a MiniTransformerMlpTrainingStepTrace),
    AdaptiveShift(&'a MiniTransformerAdaptiveShiftEventTrace),
    FinalSummary(&'a MiniTransformerMlpTrainingTrace),
}

pub struct MiniTransformerBinaryTraceWriter<W: std::io::Write> {
    writer: W,
}

impl<W: std::io::Write> MiniTransformerBinaryTraceWriter<W> {
    pub fn new(writer: W) -> Self {
        Self { writer }
    }

    pub fn write_record(
        &mut self,
        record: MiniTransformerBinaryTraceRecord<'_>,
    ) -> std::io::Result<()> {
        match record {
            MiniTransformerBinaryTraceRecord::Header { initial_model_hash } => self
                .writer
                .write_all(&mini_transformer_binary_trace_header_v1(initial_model_hash)),
            MiniTransformerBinaryTraceRecord::StepSample(step) => self
                .writer
                .write_all(&mini_transformer_binary_step_sample_record_v1(step)),
            MiniTransformerBinaryTraceRecord::AdaptiveShift(event) => self
                .writer
                .write_all(&mini_transformer_binary_adaptive_shift_record_v1(event)),
            MiniTransformerBinaryTraceRecord::FinalSummary(trace) => self
                .writer
                .write_all(&mini_transformer_binary_final_summary_record_v1(trace)),
        }
    }

    pub fn write_trace_tail(
        &mut self,
        trace: &MiniTransformerMlpTrainingTrace,
    ) -> std::io::Result<()> {
        for event in &trace.adaptive_shift_events {
            self.write_record(MiniTransformerBinaryTraceRecord::AdaptiveShift(event))?;
        }
        self.write_record(MiniTransformerBinaryTraceRecord::FinalSummary(trace))?;
        self.writer.flush()
    }

    pub fn into_inner(self) -> W {
        self.writer
    }
}

pub fn mini_transformer_binary_trace_header_v1(initial_model_hash: u64) -> [u8; 16] {
    let mut out = Vec::with_capacity(MINI_TRANSFORMER_BINARY_TRACE_HEADER_LEN);
    out.extend_from_slice(MINI_TRANSFORMER_BINARY_TRACE_MAGIC);
    out.push(MINI_TRANSFORMER_BINARY_TRACE_VERSION);
    out.push(MINI_TRANSFORMER_BINARY_TRACE_SCHEMA_ID);
    push_binary_u16(&mut out, 0);
    push_binary_u64(&mut out, initial_model_hash);
    debug_assert_eq!(out.len(), MINI_TRANSFORMER_BINARY_TRACE_HEADER_LEN);
    let mut record = [0_u8; MINI_TRANSFORMER_BINARY_TRACE_HEADER_LEN];
    record.copy_from_slice(&out);
    record
}

pub fn mini_transformer_binary_step_sample_record_v1(
    step: &MiniTransformerMlpTrainingStepTrace,
) -> [u8; 32] {
    let mut out = Vec::with_capacity(MINI_TRANSFORMER_BINARY_STEP_SAMPLE_RECORD_LEN);
    push_mini_transformer_binary_step_sample(&mut out, step);
    debug_assert_eq!(out.len(), MINI_TRANSFORMER_BINARY_STEP_SAMPLE_RECORD_LEN);
    let mut record = [0_u8; MINI_TRANSFORMER_BINARY_STEP_SAMPLE_RECORD_LEN];
    record.copy_from_slice(&out);
    record
}

pub fn mini_transformer_binary_adaptive_shift_record_v1(
    event: &MiniTransformerAdaptiveShiftEventTrace,
) -> [u8; 22] {
    let mut out = Vec::with_capacity(MINI_TRANSFORMER_BINARY_ADAPTIVE_SHIFT_RECORD_LEN);
    push_mini_transformer_binary_adaptive_shift(&mut out, event);
    debug_assert_eq!(out.len(), MINI_TRANSFORMER_BINARY_ADAPTIVE_SHIFT_RECORD_LEN);
    let mut record = [0_u8; MINI_TRANSFORMER_BINARY_ADAPTIVE_SHIFT_RECORD_LEN];
    record.copy_from_slice(&out);
    record
}

pub fn mini_transformer_binary_final_summary_record_v1(
    trace: &MiniTransformerMlpTrainingTrace,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(MINI_TRANSFORMER_BINARY_FINAL_SUMMARY_RECORD_LEN);
    push_mini_transformer_binary_final_summary(&mut out, trace);
    debug_assert_eq!(out.len(), MINI_TRANSFORMER_BINARY_FINAL_SUMMARY_RECORD_LEN);
    out
}

impl MiniTransformerMlpSwarmTrainingTrace {
    pub fn to_json_line(&self) -> String {
        let mut out = String::new();
        out.push('{');
        push_string_field(&mut out, "schema", MINI_TRANSFORMER_SWARM_SCHEMA);
        comma(&mut out);
        push_string_field(&mut out, "authority", AUTHORITY);
        comma(&mut out);
        push_string_field(&mut out, "task", "wiki_bard_mini_transformer_mlp_swarm");
        comma(&mut out);
        out.push_str("\"data\":{");
        push_string_field(&mut out, "tokenizer", self.config.tokenizer_id.as_str());
        comma(&mut out);
        push_usize_field(&mut out, "token_count", self.token_count);
        comma(&mut out);
        push_hash_field(&mut out, "token_hash", self.token_hash);
        out.push('}');
        comma(&mut out);
        out.push_str("\"swarm\":{");
        push_usize_field(&mut out, "worker_count", self.worker_count);
        comma(&mut out);
        push_usize_field(&mut out, "best_worker_index", self.best_worker_index);
        comma(&mut out);
        push_usize_field(&mut out, "base_window_offset", self.base_window_offset);
        comma(&mut out);
        push_usize_field(&mut out, "base_stride", self.base_stride);
        comma(&mut out);
        push_string_field(
            &mut out,
            "trace_detail",
            self.swarm_config.trace_detail.as_str(),
        );
        comma(&mut out);
        push_hash_field(&mut out, "final_model_hash", self.final_model_hash);
        out.push('}');
        comma(&mut out);
        out.push_str("\"model\":{");
        push_usize_field(&mut out, "vocab", BYTE_VOCAB);
        comma(&mut out);
        push_usize_field(&mut out, "seq_len", self.config.seq_len);
        comma(&mut out);
        push_usize_field(&mut out, "d_model", MINI_TRANSFORMER_D_MODEL);
        comma(&mut out);
        push_usize_field(&mut out, "heads", MINI_TRANSFORMER_HEADS);
        comma(&mut out);
        push_usize_field(&mut out, "hidden_dim", MINI_TRANSFORMER_HIDDEN_DIM);
        comma(&mut out);
        push_string_field(
            &mut out,
            "attention_kind",
            self.config.attention_kind.as_str(),
        );
        comma(&mut out);
        push_string_field(&mut out, "position", self.config.position_policy.as_str());
        out.push('}');
        comma(&mut out);
        out.push_str("\"training\":{");
        push_usize_field(&mut out, "epochs", self.config.epochs);
        comma(&mut out);
        push_usize_field(&mut out, "seq_len", self.config.seq_len);
        comma(&mut out);
        push_usize_field(&mut out, "stride", self.config.stride);
        comma(&mut out);
        push_usize_field(&mut out, "window_offset", self.config.window_offset);
        comma(&mut out);
        push_optional_usize_field(&mut out, "max_windows", self.config.max_windows);
        comma(&mut out);
        push_usize_field(&mut out, "batch_windows", self.config.batch_windows);
        comma(&mut out);
        push_i32_field(&mut out, "learning_rate", self.config.learning_rate);
        comma(&mut out);
        push_usize_field(
            &mut out,
            "output_learning_rate_shift",
            usize::from(self.config.output_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "mlp_learning_rate_shift",
            usize::from(self.config.mlp_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "embedding_learning_rate_shift",
            usize::from(self.config.embedding_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "attention_learning_rate_shift",
            usize::from(self.config.attention_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "attention_q_learning_rate_shift",
            usize::from(self.config.attention_q_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "attention_qk_learning_rate_shift",
            usize::from(self.config.attention_qk_learning_rate_shift),
        );
        out.push('}');
        comma(&mut out);
        push_mini_transformer_swarm_workers_field(&mut out, "workers", &self.workers);
        comma(&mut out);
        push_string_array_field(
            &mut out,
            "known_non_claims",
            &MINI_TRANSFORMER_MLP_KNOWN_NON_CLAIMS,
        );
        out.push('}');
        out.push('\n');
        out
    }
}

impl MiniTransformerMlpSwarmTrainingProgressTrace {
    pub fn to_json_line(&self) -> String {
        let mut out = String::new();
        out.push('{');
        push_string_field(&mut out, "schema", MINI_TRANSFORMER_SWARM_PROGRESS_SCHEMA);
        comma(&mut out);
        push_string_field(&mut out, "authority", AUTHORITY);
        comma(&mut out);
        push_string_field(
            &mut out,
            "task",
            "wiki_bard_mini_transformer_mlp_swarm_progress",
        );
        comma(&mut out);
        out.push_str("\"data\":{");
        push_string_field(&mut out, "tokenizer", self.config.tokenizer_id.as_str());
        comma(&mut out);
        push_usize_field(&mut out, "token_count", self.token_count);
        comma(&mut out);
        push_hash_field(&mut out, "token_hash", self.token_hash);
        out.push('}');
        comma(&mut out);
        out.push_str("\"swarm\":{");
        push_usize_field(&mut out, "worker_count", self.worker_count);
        comma(&mut out);
        push_usize_field(&mut out, "base_window_offset", self.base_window_offset);
        comma(&mut out);
        push_usize_field(&mut out, "base_stride", self.base_stride);
        comma(&mut out);
        push_string_field(
            &mut out,
            "trace_detail",
            self.swarm_config.trace_detail.as_str(),
        );
        out.push('}');
        comma(&mut out);
        out.push_str("\"model\":{");
        push_usize_field(&mut out, "vocab", BYTE_VOCAB);
        comma(&mut out);
        push_usize_field(&mut out, "seq_len", self.config.seq_len);
        comma(&mut out);
        push_usize_field(&mut out, "d_model", MINI_TRANSFORMER_D_MODEL);
        comma(&mut out);
        push_usize_field(&mut out, "heads", MINI_TRANSFORMER_HEADS);
        comma(&mut out);
        push_usize_field(&mut out, "hidden_dim", MINI_TRANSFORMER_HIDDEN_DIM);
        comma(&mut out);
        push_string_field(
            &mut out,
            "attention_kind",
            self.config.attention_kind.as_str(),
        );
        comma(&mut out);
        push_string_field(&mut out, "position", self.config.position_policy.as_str());
        out.push('}');
        comma(&mut out);
        out.push_str("\"training\":{");
        push_usize_field(&mut out, "epochs", self.config.epochs);
        comma(&mut out);
        push_usize_field(&mut out, "seq_len", self.config.seq_len);
        comma(&mut out);
        push_usize_field(&mut out, "stride", self.config.stride);
        comma(&mut out);
        push_usize_field(&mut out, "window_offset", self.config.window_offset);
        comma(&mut out);
        push_optional_usize_field(&mut out, "max_windows", self.config.max_windows);
        comma(&mut out);
        push_usize_field(&mut out, "batch_windows", self.config.batch_windows);
        comma(&mut out);
        push_i32_field(&mut out, "learning_rate", self.config.learning_rate);
        comma(&mut out);
        push_usize_field(
            &mut out,
            "output_learning_rate_shift",
            usize::from(self.config.output_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "mlp_learning_rate_shift",
            usize::from(self.config.mlp_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "embedding_learning_rate_shift",
            usize::from(self.config.embedding_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "attention_learning_rate_shift",
            usize::from(self.config.attention_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "attention_q_learning_rate_shift",
            usize::from(self.config.attention_q_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "attention_qk_learning_rate_shift",
            usize::from(self.config.attention_qk_learning_rate_shift),
        );
        out.push('}');
        comma(&mut out);
        push_mini_transformer_swarm_progress_workers_field(&mut out, "workers", &self.workers);
        out.push('}');
        out.push('\n');
        out
    }
}

impl MiniTransformerMlpSwarmScalingTrace {
    pub fn to_json_line(&self) -> String {
        let mut out = String::new();
        out.push('{');
        push_string_field(&mut out, "schema", MINI_TRANSFORMER_SWARM_SCALING_SCHEMA);
        comma(&mut out);
        push_string_field(&mut out, "authority", AUTHORITY);
        comma(&mut out);
        push_string_field(
            &mut out,
            "task",
            "wiki_bard_mini_transformer_mlp_swarm_scaling",
        );
        comma(&mut out);
        out.push_str("\"data\":{");
        push_string_field(&mut out, "tokenizer", self.config.tokenizer_id.as_str());
        comma(&mut out);
        push_usize_field(&mut out, "token_count", self.token_count);
        comma(&mut out);
        push_hash_field(&mut out, "token_hash", self.token_hash);
        out.push('}');
        comma(&mut out);
        out.push_str("\"host\":{");
        push_usize_field(
            &mut out,
            "available_parallelism",
            self.available_parallelism,
        );
        out.push('}');
        comma(&mut out);
        out.push_str("\"benchmark\":{");
        push_usize_field(
            &mut out,
            "requested_max_workers",
            self.requested_max_workers,
        );
        comma(&mut out);
        push_usize_field(&mut out, "run_count", self.runs.len());
        comma(&mut out);
        push_usize_array_field(&mut out, "worker_counts", &self.worker_counts);
        out.push('}');
        comma(&mut out);
        out.push_str("\"model\":{");
        push_usize_field(&mut out, "vocab", BYTE_VOCAB);
        comma(&mut out);
        push_usize_field(&mut out, "seq_len", self.config.seq_len);
        comma(&mut out);
        push_usize_field(&mut out, "d_model", MINI_TRANSFORMER_D_MODEL);
        comma(&mut out);
        push_usize_field(&mut out, "heads", MINI_TRANSFORMER_HEADS);
        comma(&mut out);
        push_usize_field(&mut out, "hidden_dim", MINI_TRANSFORMER_HIDDEN_DIM);
        comma(&mut out);
        push_string_field(
            &mut out,
            "attention_kind",
            self.config.attention_kind.as_str(),
        );
        comma(&mut out);
        push_string_field(&mut out, "position", self.config.position_policy.as_str());
        out.push('}');
        comma(&mut out);
        out.push_str("\"training\":{");
        push_usize_field(&mut out, "epochs", self.config.epochs);
        comma(&mut out);
        push_usize_field(&mut out, "seq_len", self.config.seq_len);
        comma(&mut out);
        push_usize_field(&mut out, "stride", self.config.stride);
        comma(&mut out);
        push_usize_field(&mut out, "window_offset", self.config.window_offset);
        comma(&mut out);
        push_optional_usize_field(&mut out, "max_windows", self.config.max_windows);
        comma(&mut out);
        push_usize_field(&mut out, "batch_windows", self.config.batch_windows);
        out.push('}');
        comma(&mut out);
        push_mini_transformer_swarm_scaling_runs_field(&mut out, "runs", &self.runs);
        comma(&mut out);
        push_string_array_field(
            &mut out,
            "known_non_claims",
            &MINI_TRANSFORMER_SWARM_SCALING_KNOWN_NON_CLAIMS,
        );
        out.push('}');
        out.push('\n');
        out
    }
}

pub(super) fn push_usize_array_field(out: &mut String, field: &str, values: &[usize]) {
    out.push('"');
    out.push_str(field);
    out.push_str("\":");
    out.push('[');
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            comma(out);
        }
        out.push_str(&value.to_string());
    }
    out.push(']');
}

fn push_mini_transformer_swarm_scaling_runs_field(
    out: &mut String,
    field: &str,
    runs: &[MiniTransformerMlpSwarmScalingRunTrace],
) {
    out.push('"');
    out.push_str(field);
    out.push_str("\":");
    out.push('[');
    for (index, run) in runs.iter().enumerate() {
        if index > 0 {
            comma(out);
        }
        push_mini_transformer_swarm_scaling_run(out, run);
    }
    out.push(']');
}

fn push_mini_transformer_swarm_scaling_run(
    out: &mut String,
    run: &MiniTransformerMlpSwarmScalingRunTrace,
) {
    out.push('{');
    push_usize_field(out, "requested_worker_count", run.requested_worker_count);
    comma(out);
    push_usize_field(out, "effective_worker_count", run.effective_worker_count);
    comma(out);
    push_u64_field(out, "elapsed_ns", run.elapsed_ns);
    comma(out);
    push_milli_decimal_field(out, "elapsed_ms", run.elapsed_ns / 1_000);
    comma(out);
    push_u64_field(out, "speedup_per_mille", run.speedup_per_mille);
    comma(out);
    push_milli_decimal_field(out, "speedup", run.speedup_per_mille);
    comma(out);
    push_u64_field(
        out,
        "parallel_efficiency_per_mille",
        run.parallel_efficiency_per_mille,
    );
    comma(out);
    push_milli_decimal_field(
        out,
        "parallel_efficiency",
        run.parallel_efficiency_per_mille,
    );
    comma(out);
    push_u64_field(
        out,
        "windows_per_second_milli",
        run.windows_per_second_milli,
    );
    comma(out);
    push_milli_decimal_field(out, "windows_per_second", run.windows_per_second_milli);
    comma(out);
    push_u64_field(
        out,
        "updates_per_second_milli",
        run.updates_per_second_milli,
    );
    comma(out);
    push_milli_decimal_field(out, "updates_per_second", run.updates_per_second_milli);
    comma(out);
    push_usize_field(out, "examined_windows", run.examined_windows);
    comma(out);
    push_usize_field(out, "updates", run.updates);
    comma(out);
    push_usize_field(out, "accepted_batch_count", run.accepted_batch_count);
    comma(out);
    push_usize_field(out, "rejected_batch_count", run.rejected_batch_count);
    comma(out);
    push_usize_field(out, "rollback_count", run.rollback_count);
    comma(out);
    push_usize_field(out, "best_worker_index", run.best_worker_index);
    comma(out);
    push_usize_field(out, "best_final_total_error", run.best_final_total_error);
    comma(out);
    push_usize_field(
        out,
        "best_final_probability_error_q15",
        run.best_final_probability_error_q15,
    );
    comma(out);
    push_usize_field(
        out,
        "best_final_accuracy_per_mille",
        run.best_final_accuracy_per_mille,
    );
    comma(out);
    push_hash_field(out, "final_model_hash", run.final_model_hash);
    out.push('}');
}

pub(super) fn push_mini_transformer_swarm_route_config_field(
    out: &mut String,
    field: &str,
    config: &MiniTransformerSwarmRouteConfig,
) {
    out.push('"');
    out.push_str(field);
    out.push_str("\":");
    out.push('{');
    match config.required_capabilities.first().map(String::as_str) {
        Some(capability) => push_string_field(out, "required_capability", capability),
        None => {
            push_quoted(out, "required_capability");
            out.push_str(":null");
        }
    }
    comma(out);
    push_string_vec_field(out, "required_capabilities", &config.required_capabilities);
    comma(out);
    push_optional_usize_field(out, "max_artifact_bytes", config.max_artifact_bytes);
    comma(out);
    push_optional_usize_field(out, "max_parameter_bytes", config.max_parameter_bytes);
    comma(out);
    push_usize_field(out, "active_expert_limit", config.active_expert_limit);
    comma(out);
    push_bool_field(out, "prompt_affinity", config.prompt_affinity);
    comma(out);
    push_usize_field(
        out,
        "prompt_affinity_max_windows",
        config.prompt_affinity_max_windows,
    );
    out.push('}');
}

pub(super) fn push_mini_transformer_swarm_route_candidates_field(
    out: &mut String,
    field: &str,
    candidates: &[MiniTransformerSwarmRouteCandidateTrace],
) {
    out.push('"');
    out.push_str(field);
    out.push_str("\":");
    out.push('[');
    for (index, candidate) in candidates.iter().enumerate() {
        if index > 0 {
            comma(out);
        }
        push_mini_transformer_swarm_route_candidate(out, candidate);
    }
    out.push(']');
}

fn push_mini_transformer_swarm_route_candidate(
    out: &mut String,
    candidate: &MiniTransformerSwarmRouteCandidateTrace,
) {
    out.push('{');
    push_usize_field(out, "expert_index", candidate.expert_index);
    comma(out);
    push_string_field(out, "expert_id", &candidate.expert_id);
    comma(out);
    push_bool_field(out, "accepted", candidate.accepted);
    comma(out);
    push_string_field(out, "reject_reason", candidate.reject_reason);
    comma(out);
    push_i64_field(out, "score", candidate.score);
    comma(out);
    push_i64_field(out, "manifest_score", candidate.manifest_score);
    comma(out);
    push_i64_field(
        out,
        "prompt_affinity_score",
        candidate.prompt_affinity_score,
    );
    comma(out);
    push_usize_field(out, "prompt_eval_windows", candidate.prompt_eval_windows);
    comma(out);
    push_optional_usize_field(
        out,
        "prompt_probability_error_q15",
        candidate.prompt_probability_error_q15,
    );
    comma(out);
    push_bool_field(out, "capability_match", candidate.capability_match);
    comma(out);
    push_string_vec_field(out, "matched_capabilities", &candidate.matched_capabilities);
    comma(out);
    push_string_vec_field(out, "missing_capabilities", &candidate.missing_capabilities);
    comma(out);
    push_hash_field(out, "model_hash", candidate.model_hash);
    comma(out);
    push_usize_field(out, "artifact_bytes", candidate.artifact_bytes);
    comma(out);
    push_usize_field(out, "parameter_bytes", candidate.parameter_bytes);
    comma(out);
    push_usize_field(out, "worker_count", candidate.worker_count);
    comma(out);
    push_usize_field(out, "context_seq_len", candidate.context_seq_len);
    comma(out);
    push_string_field(out, "default_composition", candidate.default_composition);
    out.push('}');
}

fn push_mini_transformer_swarm_workers_field(
    out: &mut String,
    field: &str,
    workers: &[MiniTransformerMlpSwarmWorkerTrace],
) {
    out.push('"');
    out.push_str(field);
    out.push_str("\":");
    out.push('[');
    for (index, worker) in workers.iter().enumerate() {
        if index > 0 {
            comma(out);
        }
        push_mini_transformer_swarm_worker(out, worker);
    }
    out.push(']');
}

pub(super) fn push_mini_transformer_swarm_worker(
    out: &mut String,
    worker: &MiniTransformerMlpSwarmWorkerTrace,
) {
    out.push('{');
    push_usize_field(out, "worker_index", worker.worker_index);
    comma(out);
    push_usize_field(out, "window_offset", worker.window_offset);
    comma(out);
    push_usize_field(out, "stride", worker.stride);
    comma(out);
    push_optional_usize_field(out, "max_windows", worker.max_windows);
    comma(out);
    push_hash_field(out, "token_hash", worker.token_hash);
    comma(out);
    push_hash_field(out, "window_hash", worker.window_hash);
    comma(out);
    push_usize_field(out, "windows", worker.windows);
    comma(out);
    push_usize_field(out, "examined_windows", worker.examined_windows);
    comma(out);
    push_usize_field(out, "updates", worker.updates);
    comma(out);
    push_usize_field(out, "accepted_batch_count", worker.accepted_batch_count);
    comma(out);
    push_usize_field(out, "rejected_batch_count", worker.rejected_batch_count);
    comma(out);
    push_usize_field(out, "rollback_count", worker.rollback_count);
    comma(out);
    push_usize_field(out, "rejected_window_count", worker.rejected_window_count);
    comma(out);
    push_usize_field(
        out,
        "final_invalid_forward_count",
        worker.final_invalid_forward_count,
    );
    comma(out);
    push_usize_field(out, "initial_total_error", worker.initial_total_error);
    comma(out);
    push_usize_field(out, "final_total_error", worker.final_total_error);
    comma(out);
    push_usize_field(
        out,
        "initial_probability_error_q15",
        worker.initial_probability_error_q15,
    );
    comma(out);
    push_usize_field(
        out,
        "final_probability_error_q15",
        worker.final_probability_error_q15,
    );
    comma(out);
    push_usize_field(
        out,
        "final_accuracy_per_mille",
        worker.final_accuracy_per_mille,
    );
    comma(out);
    push_hash_field(out, "final_model_hash", worker.final_model_hash);
    comma(out);
    push_hash_field(out, "final_logits_hash", worker.final_logits_hash);
    out.push('}');
}

fn push_mini_transformer_swarm_progress_workers_field(
    out: &mut String,
    field: &str,
    workers: &[MiniTransformerMlpSwarmWorkerProgressTrace],
) {
    out.push('"');
    out.push_str(field);
    out.push_str("\":");
    out.push('[');
    for (index, worker) in workers.iter().enumerate() {
        if index > 0 {
            comma(out);
        }
        push_mini_transformer_swarm_progress_worker(out, worker);
    }
    out.push(']');
}

fn push_mini_transformer_swarm_progress_worker(
    out: &mut String,
    worker: &MiniTransformerMlpSwarmWorkerProgressTrace,
) {
    let progress = &worker.progress;
    out.push('{');
    push_usize_field(out, "worker_index", worker.worker_index);
    comma(out);
    push_usize_field(out, "window_offset", progress.config.window_offset);
    comma(out);
    push_usize_field(out, "stride", progress.config.stride);
    comma(out);
    push_optional_usize_field(out, "max_windows", progress.config.max_windows);
    comma(out);
    push_hash_field(out, "token_hash", progress.token_hash);
    comma(out);
    push_hash_field(out, "window_hash", progress.window_hash);
    comma(out);
    push_usize_field(out, "windows", progress.windows);
    comma(out);
    push_usize_field(out, "examined_windows", progress.examined_windows);
    comma(out);
    push_usize_field(out, "updates", progress.updates);
    comma(out);
    push_usize_field(out, "accepted_batch_count", progress.accepted_batch_count);
    comma(out);
    push_usize_field(out, "rejected_batch_count", progress.rejected_batch_count);
    comma(out);
    push_usize_field(out, "rollback_count", progress.rollback_count);
    comma(out);
    push_usize_field(out, "rejected_window_count", progress.rejected_window_count);
    comma(out);
    push_u64_field(out, "output_head_delta_l1", progress.output_head_delta_l1);
    comma(out);
    push_u64_field(out, "mlp_delta_l1", progress.mlp_delta_l1);
    comma(out);
    push_u64_field(out, "embedding_delta_l1", progress.embedding_delta_l1);
    comma(out);
    push_u64_field(out, "attention_delta_l1", progress.attention_delta_l1);
    comma(out);
    push_u64_field(out, "attention_q_delta_l1", progress.attention_q_delta_l1);
    comma(out);
    push_u64_field(out, "attention_k_delta_l1", progress.attention_k_delta_l1);
    comma(out);
    push_u64_field(out, "attention_v_delta_l1", progress.attention_v_delta_l1);
    comma(out);
    push_u64_field(out, "attention_o_delta_l1", progress.attention_o_delta_l1);
    comma(out);
    push_u64_field(out, "output_head_carry_l1", progress.output_head_carry_l1);
    comma(out);
    push_u64_field(out, "mlp_carry_l1", progress.mlp_carry_l1);
    comma(out);
    push_u64_field(out, "embedding_carry_l1", progress.embedding_carry_l1);
    comma(out);
    push_u64_field(out, "attention_carry_l1", progress.attention_carry_l1);
    comma(out);
    push_u64_field(out, "attention_q_carry_l1", progress.attention_q_carry_l1);
    comma(out);
    push_u64_field(out, "attention_k_carry_l1", progress.attention_k_carry_l1);
    comma(out);
    push_u64_field(out, "attention_v_carry_l1", progress.attention_v_carry_l1);
    comma(out);
    push_u64_field(out, "attention_o_carry_l1", progress.attention_o_carry_l1);
    comma(out);
    push_usize_field(
        out,
        "adaptive_rule_shift_adjustment_count",
        progress.adaptive_rule_shift_adjustment_count,
    );
    comma(out);
    push_usize_field(
        out,
        "adaptive_holographic_shift_adjustment_count",
        progress.adaptive_holographic_shift_adjustment_count,
    );
    comma(out);
    push_hash_field(out, "current_model_hash", progress.model_hash);
    out.push('}');
}

impl MiniTransformerMlpTrainingTrace {
    pub fn to_json_line(&self) -> String {
        let mut out = String::new();
        out.push('{');
        push_string_field(&mut out, "schema", MINI_TRANSFORMER_MLP_SCHEMA);
        comma(&mut out);
        push_string_field(&mut out, "authority", AUTHORITY);
        comma(&mut out);
        push_string_field(&mut out, "task", MINI_TRANSFORMER_MLP_TASK);
        comma(&mut out);
        out.push_str("\"data\":{");
        push_string_field(&mut out, "tokenizer", self.config.tokenizer_id.as_str());
        comma(&mut out);
        push_usize_field(&mut out, "token_count", self.token_count);
        comma(&mut out);
        push_hash_field(&mut out, "token_hash", self.token_hash);
        comma(&mut out);
        push_hash_field(&mut out, "window_hash", self.window_hash);
        comma(&mut out);
        push_usize_field(&mut out, "windows", self.windows);
        out.push('}');
        comma(&mut out);
        out.push_str("\"model\":{");
        push_usize_field(&mut out, "vocab", BYTE_VOCAB);
        comma(&mut out);
        push_usize_field(&mut out, "seq_len", self.config.seq_len);
        comma(&mut out);
        push_usize_field(&mut out, "d_model", MINI_TRANSFORMER_D_MODEL);
        comma(&mut out);
        push_usize_field(&mut out, "heads", MINI_TRANSFORMER_HEADS);
        comma(&mut out);
        push_usize_field(&mut out, "hidden_dim", MINI_TRANSFORMER_HIDDEN_DIM);
        comma(&mut out);
        push_string_field(
            &mut out,
            "trained_component",
            "embedding_i16_plus_output_head_i8_plus_gated_mlp_i8_plus_attention_qkvo_i8",
        );
        comma(&mut out);
        push_string_field(&mut out, "attention", "updates_q_k_v_o_i8");
        comma(&mut out);
        push_string_field(
            &mut out,
            "attention_kind",
            self.config.attention_kind.as_str(),
        );
        comma(&mut out);
        let attention_backward = match self.config.attention_kind {
            MiniTransformerAttentionKind::Base2Softmax => "base2_softmax_jacobian",
            MiniTransformerAttentionKind::Linear => {
                "linear_numerator_straight_through_denominator_constant"
            }
            MiniTransformerAttentionKind::LinearStreamingNope => "unsupported_for_training",
            MiniTransformerAttentionKind::LinearStreamingTttNope => "unsupported_for_training",
        };
        push_string_field(&mut out, "attention_backward", attention_backward);
        comma(&mut out);
        push_string_field(&mut out, "position", self.config.position_policy.as_str());
        out.push('}');
        comma(&mut out);
        out.push_str("\"optimizer\":{");
        push_string_field(&mut out, "kind", "base2_softmax_cross_entropy_sgd");
        comma(&mut out);
        push_string_field(&mut out, "feature_scale", "q15");
        comma(&mut out);
        push_string_field(&mut out, "activation", "hard_silu_shift2_q15");
        comma(&mut out);
        push_string_field(&mut out, "weight_dtype", "i8");
        comma(&mut out);
        push_i32_field(&mut out, "learning_rate", self.config.learning_rate);
        comma(&mut out);
        push_usize_field(
            &mut out,
            "output_learning_rate_shift",
            usize::from(self.config.output_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "mlp_learning_rate_shift",
            usize::from(self.config.mlp_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "embedding_learning_rate_shift",
            usize::from(self.config.embedding_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "attention_learning_rate_shift",
            usize::from(self.config.attention_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "attention_q_learning_rate_shift",
            usize::from(self.config.attention_q_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "attention_qk_learning_rate_shift",
            usize::from(self.config.attention_qk_learning_rate_shift),
        );
        comma(&mut out);
        push_bool_field(
            &mut out,
            "adaptive_rule_shifts",
            self.config.adaptive_rule_shifts,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "adaptive_rule_interval_batches",
            self.config.adaptive_rule_interval_batches,
        );
        comma(&mut out);
        push_bool_field(
            &mut out,
            "adaptive_attention_shifts",
            self.config.adaptive_attention_shifts,
        );
        comma(&mut out);
        push_bool_field(
            &mut out,
            "adaptive_holographic_shifts",
            self.config.adaptive_holographic_shifts,
        );
        comma(&mut out);
        push_bool_field(
            &mut out,
            "attention_vo_error_feedback",
            self.config.attention_vo_error_feedback,
        );
        comma(&mut out);
        push_bool_field(
            &mut out,
            "attention_vo_oracle",
            self.config.attention_vo_oracle,
        );
        comma(&mut out);
        push_bool_field(
            &mut out,
            "reject_loss_regression",
            self.config.reject_loss_regression,
        );
        comma(&mut out);
        push_string_field(&mut out, "batch_mode", self.config.batch_mode.as_str());
        comma(&mut out);
        push_usize_field(
            &mut out,
            "map_reduce_workers",
            self.config.map_reduce_workers,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "effective_map_reduce_workers",
            mini_transformer_effective_map_reduce_workers(self.config),
        );
        comma(&mut out);
        push_string_field(&mut out, "trace_detail", self.trace_detail.as_str());
        out.push('}');
        comma(&mut out);
        out.push_str("\"training\":{");
        push_usize_field(&mut out, "epochs", self.config.epochs);
        comma(&mut out);
        push_usize_field(&mut out, "seq_len", self.config.seq_len);
        comma(&mut out);
        push_usize_field(&mut out, "stride", self.config.stride);
        comma(&mut out);
        push_usize_field(&mut out, "window_offset", self.config.window_offset);
        comma(&mut out);
        push_optional_usize_field(&mut out, "max_windows", self.config.max_windows);
        comma(&mut out);
        push_usize_field(&mut out, "batch_windows", self.config.batch_windows);
        comma(&mut out);
        push_usize_field(
            &mut out,
            "target_token_min",
            usize::from(self.config.target_token_min),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "target_token_max",
            usize::from(self.config.target_token_max),
        );
        comma(&mut out);
        push_string_field(
            &mut out,
            "target_segment",
            self.config.target_segment.as_str(),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "target_frequency_cap",
            self.config.target_frequency_cap as usize,
        );
        comma(&mut out);
        push_i16_field(
            &mut out,
            "target_frequency_min_weight_q15",
            self.config.target_frequency_min_weight_q15,
        );
        comma(&mut out);
        push_i16_field(
            &mut out,
            "argmax_margin_weight_q15",
            self.config.argmax_margin_weight_q15,
        );
        comma(&mut out);
        push_string_field(&mut out, "batch_mode", self.config.batch_mode.as_str());
        comma(&mut out);
        push_usize_field(
            &mut out,
            "map_reduce_workers",
            self.config.map_reduce_workers,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "effective_map_reduce_workers",
            mini_transformer_effective_map_reduce_workers(self.config),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "batch_average_shift",
            usize::from(
                mini_transformer_batch_learning_rate_shift(self.config.batch_windows).unwrap_or(0),
            ),
        );
        comma(&mut out);
        push_usize_field(&mut out, "examined_windows", self.examined_windows);
        comma(&mut out);
        push_usize_field(&mut out, "updates", self.updates);
        comma(&mut out);
        push_string_field(&mut out, "trace_detail", self.trace_detail.as_str());
        comma(&mut out);
        push_usize_field(
            &mut out,
            "rollback_history_limit",
            MINI_TRANSFORMER_ROLLBACK_HISTORY_LIMIT,
        );
        out.push('}');
        comma(&mut out);
        out.push_str("\"metrics\":{");
        push_usize_field(&mut out, "initial_total_error", self.initial_total_error);
        comma(&mut out);
        push_usize_field(&mut out, "final_total_error", self.final_total_error);
        comma(&mut out);
        push_usize_field(
            &mut out,
            "initial_probability_error_q15",
            self.initial_probability_error_q15,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "final_probability_error_q15",
            self.final_probability_error_q15,
        );
        comma(&mut out);
        push_i32_field(
            &mut out,
            "probability_error_delta_i32",
            self.final_probability_error_q15 as i32 - self.initial_probability_error_q15 as i32,
        );
        comma(&mut out);
        push_usize_field(&mut out, "initial_mistakes", self.initial_mistakes);
        comma(&mut out);
        push_usize_field(&mut out, "final_mistakes", self.final_mistakes);
        comma(&mut out);
        push_usize_field(
            &mut out,
            "final_accuracy_per_mille",
            self.final_accuracy_per_mille,
        );
        comma(&mut out);
        push_usize_field(&mut out, "accepted_batch_count", self.accepted_batch_count);
        comma(&mut out);
        push_usize_field(&mut out, "rejected_batch_count", self.rejected_batch_count);
        comma(&mut out);
        push_usize_field(
            &mut out,
            "output_head_accumulator_batch_count",
            self.output_head_accumulator_batch_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "output_head_accumulator_window_count",
            self.output_head_accumulator_window_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "mlp_accumulator_batch_count",
            self.mlp_accumulator_batch_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "mlp_accumulator_window_count",
            self.mlp_accumulator_window_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "attention_accumulator_batch_count",
            self.attention_accumulator_batch_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "attention_accumulator_window_count",
            self.attention_accumulator_window_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "embedding_accumulator_batch_count",
            self.embedding_accumulator_batch_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "embedding_accumulator_window_count",
            self.embedding_accumulator_window_count,
        );
        comma(&mut out);
        push_usize_field(&mut out, "rollback_count", self.rollback_count);
        comma(&mut out);
        push_usize_field(
            &mut out,
            "rejected_window_count",
            self.rejected_window_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "loss_regression_rejected_batch_count",
            self.loss_regression_rejected_batch_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "final_invalid_forward_count",
            self.final_invalid_forward_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "output_head_saturation_count",
            self.output_head_saturation_count,
        );
        comma(&mut out);
        push_usize_field(&mut out, "mlp_saturation_count", self.mlp_saturation_count);
        comma(&mut out);
        push_usize_field(
            &mut out,
            "embedding_saturation_count",
            self.embedding_saturation_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "attention_saturation_count",
            self.attention_saturation_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "residual_saturation_count",
            self.residual_saturation_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "output_head_zero_delta_count",
            self.output_head_zero_delta_count,
        );
        comma(&mut out);
        push_usize_field(&mut out, "mlp_zero_delta_count", self.mlp_zero_delta_count);
        comma(&mut out);
        push_usize_field(
            &mut out,
            "embedding_zero_delta_count",
            self.embedding_zero_delta_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "attention_zero_delta_count",
            self.attention_zero_delta_count,
        );
        comma(&mut out);
        push_u64_field(&mut out, "output_head_delta_l1", self.output_head_delta_l1);
        comma(&mut out);
        push_u64_field(&mut out, "mlp_delta_l1", self.mlp_delta_l1);
        comma(&mut out);
        push_u64_field(&mut out, "embedding_delta_l1", self.embedding_delta_l1);
        comma(&mut out);
        push_u64_field(&mut out, "attention_delta_l1", self.attention_delta_l1);
        comma(&mut out);
        push_u64_field(&mut out, "attention_q_delta_l1", self.attention_q_delta_l1);
        comma(&mut out);
        push_u64_field(&mut out, "attention_k_delta_l1", self.attention_k_delta_l1);
        comma(&mut out);
        push_u64_field(&mut out, "attention_v_delta_l1", self.attention_v_delta_l1);
        comma(&mut out);
        push_u64_field(&mut out, "attention_o_delta_l1", self.attention_o_delta_l1);
        comma(&mut out);
        push_u64_field(&mut out, "output_head_carry_l1", self.output_head_carry_l1);
        comma(&mut out);
        push_u64_field(&mut out, "mlp_carry_l1", self.mlp_carry_l1);
        comma(&mut out);
        push_u64_field(&mut out, "embedding_carry_l1", self.embedding_carry_l1);
        comma(&mut out);
        push_u64_field(&mut out, "attention_carry_l1", self.attention_carry_l1);
        comma(&mut out);
        push_u64_field(&mut out, "attention_q_carry_l1", self.attention_q_carry_l1);
        comma(&mut out);
        push_u64_field(&mut out, "attention_k_carry_l1", self.attention_k_carry_l1);
        comma(&mut out);
        push_u64_field(&mut out, "attention_v_carry_l1", self.attention_v_carry_l1);
        comma(&mut out);
        push_u64_field(&mut out, "attention_o_carry_l1", self.attention_o_carry_l1);
        comma(&mut out);
        push_usize_field(
            &mut out,
            "adaptive_rule_shift_adjustment_count",
            self.adaptive_rule_shift_adjustment_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "adaptive_rule_update_count",
            self.adaptive_rule_update_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "adaptive_rule_event_count",
            self.adaptive_rule_event_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "adaptive_rule_trace_event_limit",
            MINI_TRANSFORMER_ADAPTIVE_RULE_TRACE_EVENT_LIMIT,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "adaptive_holographic_shift_adjustment_count",
            self.adaptive_holographic_shift_adjustment_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "adaptive_holographic_update_count",
            self.adaptive_holographic_update_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "adaptive_holographic_meta_dim",
            MINI_TRANSFORMER_HOLO_META_DIM,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "adaptive_holographic_action_count",
            MINI_TRANSFORMER_HOLO_ACTION_COUNT,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "adaptive_holographic_memory_update_shift",
            MINI_TRANSFORMER_HOLO_MEMORY_UPDATE_SHIFT as usize,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "adaptive_holographic_query_shift",
            MINI_TRANSFORMER_HOLO_QUERY_SHIFT as usize,
        );
        comma(&mut out);
        push_hash_field(
            &mut out,
            "adaptive_holographic_hash",
            self.adaptive_holographic_hash,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "adaptive_attention_shift_adjustment_count",
            self.adaptive_attention_shift_adjustment_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "adaptive_attention_holographic_update_count",
            self.adaptive_attention_holographic_update_count,
        );
        comma(&mut out);
        push_hash_field(
            &mut out,
            "adaptive_attention_holographic_hash",
            self.adaptive_attention_holographic_hash,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "final_output_learning_rate_shift",
            usize::from(self.final_output_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "final_mlp_learning_rate_shift",
            usize::from(self.final_mlp_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "final_embedding_learning_rate_shift",
            usize::from(self.final_embedding_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "final_attention_learning_rate_shift",
            usize::from(self.final_attention_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "final_attention_q_learning_rate_shift",
            usize::from(self.final_attention_q_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "final_attention_qk_learning_rate_shift",
            usize::from(self.final_attention_qk_learning_rate_shift),
        );
        out.push('}');
        comma(&mut out);
        push_hash_field(&mut out, "initial_model_hash", self.initial_model_hash);
        comma(&mut out);
        push_hash_field(&mut out, "final_model_hash", self.final_model_hash);
        comma(&mut out);
        push_hash_field(
            &mut out,
            "initial_embedding_hash",
            self.initial_embedding_hash,
        );
        comma(&mut out);
        push_hash_field(&mut out, "final_embedding_hash", self.final_embedding_hash);
        comma(&mut out);
        push_hash_field(
            &mut out,
            "initial_output_head_hash",
            self.initial_output_head_hash,
        );
        comma(&mut out);
        push_hash_field(
            &mut out,
            "final_output_head_hash",
            self.final_output_head_hash,
        );
        comma(&mut out);
        push_hash_field(&mut out, "initial_mlp_hash", self.initial_mlp_hash);
        comma(&mut out);
        push_hash_field(&mut out, "final_mlp_hash", self.final_mlp_hash);
        comma(&mut out);
        push_hash_field(
            &mut out,
            "initial_attention_hash",
            self.initial_attention_hash,
        );
        comma(&mut out);
        push_hash_field(&mut out, "final_attention_hash", self.final_attention_hash);
        comma(&mut out);
        push_hash_field(
            &mut out,
            "initial_attention_q_hash",
            self.initial_attention_q_hash,
        );
        comma(&mut out);
        push_hash_field(
            &mut out,
            "final_attention_q_hash",
            self.final_attention_q_hash,
        );
        comma(&mut out);
        push_hash_field(
            &mut out,
            "initial_attention_k_hash",
            self.initial_attention_k_hash,
        );
        comma(&mut out);
        push_hash_field(
            &mut out,
            "final_attention_k_hash",
            self.final_attention_k_hash,
        );
        comma(&mut out);
        push_hash_field(
            &mut out,
            "initial_attention_v_hash",
            self.initial_attention_v_hash,
        );
        comma(&mut out);
        push_hash_field(
            &mut out,
            "final_attention_v_hash",
            self.final_attention_v_hash,
        );
        comma(&mut out);
        push_hash_field(
            &mut out,
            "initial_attention_o_hash",
            self.initial_attention_o_hash,
        );
        comma(&mut out);
        push_hash_field(
            &mut out,
            "final_attention_o_hash",
            self.final_attention_o_hash,
        );
        comma(&mut out);
        push_hash_field(&mut out, "final_logits_hash", self.final_logits_hash);
        comma(&mut out);
        push_mini_transformer_adaptive_shift_events_field(
            &mut out,
            "adaptive_shift_events",
            &self.adaptive_shift_events,
        );
        comma(&mut out);
        push_mini_transformer_mlp_steps_field(&mut out, "steps", &self.steps);
        comma(&mut out);
        push_string_array_field(
            &mut out,
            "known_non_claims",
            &MINI_TRANSFORMER_MLP_KNOWN_NON_CLAIMS,
        );
        out.push('}');
        out.push('\n');
        out
    }

    pub fn to_binary_trace_v1(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(
            MINI_TRANSFORMER_BINARY_TRACE_HEADER_LEN
                + self
                    .steps
                    .len()
                    .saturating_mul(MINI_TRANSFORMER_BINARY_STEP_SAMPLE_RECORD_LEN)
                + self
                    .adaptive_shift_events
                    .len()
                    .saturating_mul(MINI_TRANSFORMER_BINARY_ADAPTIVE_SHIFT_RECORD_LEN)
                + MINI_TRANSFORMER_BINARY_FINAL_SUMMARY_RECORD_LEN,
        );
        out.extend_from_slice(&mini_transformer_binary_trace_header_v1(
            self.initial_model_hash,
        ));

        for step in &self.steps {
            out.extend_from_slice(&mini_transformer_binary_step_sample_record_v1(step));
        }
        for event in &self.adaptive_shift_events {
            out.extend_from_slice(&mini_transformer_binary_adaptive_shift_record_v1(event));
        }
        out.extend_from_slice(&mini_transformer_binary_final_summary_record_v1(self));
        out
    }
}

fn push_mini_transformer_binary_step_sample(
    out: &mut Vec<u8>,
    step: &MiniTransformerMlpTrainingStepTrace,
) {
    let start = out.len();
    out.push(MINI_TRANSFORMER_BINARY_TAG_STEP_SAMPLE);
    push_binary_u32_clamped(out, step.update_index);
    push_binary_u32_clamped(out, step.window_start);
    out.push(step.first_token);
    out.push(step.last_token);
    out.push(step.target_token);
    out.push(step.predicted_token_before);
    out.push(step.predicted_token_after);
    push_binary_i16(out, step.target_probability_before_q15);
    push_binary_i16(out, step.target_probability_after_q15);
    push_binary_u16_clamped(out, step.residual_saturation_count);
    push_binary_u16_clamped(
        out,
        step.output_head_saturation_count
            .saturating_add(step.mlp_saturation_count)
            .saturating_add(step.embedding_saturation_count)
            .saturating_add(step.attention_saturation_count),
    );
    push_binary_u16_clamped(
        out,
        step.output_head_zero_delta_count
            .saturating_add(step.mlp_zero_delta_count)
            .saturating_add(step.embedding_zero_delta_count)
            .saturating_add(step.attention_zero_delta_count),
    );
    push_binary_u32_saturating(out, step.attention_delta_l1);
    push_binary_u32_saturating(
        out,
        step.output_head_delta_l1
            .saturating_add(step.mlp_delta_l1)
            .saturating_add(step.embedding_delta_l1)
            .saturating_add(step.attention_delta_l1),
    );
    debug_assert_eq!(out.len() - start, 32);
}

fn push_mini_transformer_binary_adaptive_shift(
    out: &mut Vec<u8>,
    event: &MiniTransformerAdaptiveShiftEventTrace,
) {
    out.push(MINI_TRANSFORMER_BINARY_TAG_ADAPTIVE_SHIFT);
    push_binary_u32_clamped(out, event.batch_index);
    out.push(mini_transformer_binary_component_code(event.component));
    out.push(mini_transformer_binary_reason_code(event.reason));
    out.push(event.previous_shift);
    out.push(event.next_shift);
    out.push(event.delta as u8);
    push_binary_u16_clamped(out, event.observation_batches);
    push_binary_u16_clamped(out, event.rejected_batches);
    push_binary_u16_clamped(out, event.saturation_count);
    push_binary_u16_clamped(out, event.zero_delta_count);
    push_binary_u32_saturating(out, event.weight_delta_l1);
}

fn push_mini_transformer_binary_final_summary(
    out: &mut Vec<u8>,
    trace: &MiniTransformerMlpTrainingTrace,
) {
    out.push(MINI_TRANSFORMER_BINARY_TAG_FINAL_SUMMARY);
    out.push(mini_transformer_binary_trace_detail_code(
        trace.trace_detail,
    ));
    out.push(mini_transformer_binary_tokenizer_code(
        trace.config.tokenizer_id,
    ));
    out.push(mini_transformer_binary_attention_code(
        trace.config.attention_kind,
    ));
    out.push(mini_transformer_binary_position_code(
        trace.config.position_policy,
    ));
    push_binary_u16(out, mini_transformer_binary_config_flags(trace.config));
    push_binary_u32_clamped(out, trace.config.epochs);
    push_binary_u32_clamped(out, trace.config.seq_len);
    push_binary_u32_clamped(out, trace.config.stride);
    push_binary_u32_clamped(out, trace.config.window_offset);
    push_binary_optional_u32(out, trace.config.max_windows);
    push_binary_u32_clamped(out, trace.config.batch_windows);
    push_binary_i32(out, trace.config.learning_rate);
    out.push(trace.config.output_learning_rate_shift);
    out.push(trace.config.mlp_learning_rate_shift);
    out.push(trace.config.embedding_learning_rate_shift);
    out.push(trace.config.attention_learning_rate_shift);
    out.push(trace.config.attention_q_learning_rate_shift);
    out.push(trace.config.attention_qk_learning_rate_shift);

    push_binary_u64(out, trace.token_count as u64);
    push_binary_u64(out, trace.token_hash);
    push_binary_u64(out, trace.window_hash);
    push_binary_u64(out, trace.windows as u64);
    push_binary_u64(out, trace.examined_windows as u64);
    push_binary_u64(out, trace.updates as u64);
    push_binary_u64(out, trace.accepted_batch_count as u64);
    push_binary_u64(out, trace.rejected_batch_count as u64);
    push_binary_u64(out, trace.rollback_count as u64);
    push_binary_u64(out, trace.rejected_window_count as u64);
    push_binary_u64(out, trace.loss_regression_rejected_batch_count as u64);
    push_binary_u64(out, trace.final_invalid_forward_count as u64);

    push_binary_u64(out, trace.initial_total_error as u64);
    push_binary_u64(out, trace.final_total_error as u64);
    push_binary_u64(out, trace.initial_probability_error_q15 as u64);
    push_binary_u64(out, trace.final_probability_error_q15 as u64);
    push_binary_i64(
        out,
        trace.final_probability_error_q15 as i64 - trace.initial_probability_error_q15 as i64,
    );
    push_binary_u64(out, trace.initial_mistakes as u64);
    push_binary_u64(out, trace.final_mistakes as u64);
    push_binary_u16_clamped(out, trace.final_accuracy_per_mille);

    push_binary_u64(out, trace.output_head_saturation_count as u64);
    push_binary_u64(out, trace.mlp_saturation_count as u64);
    push_binary_u64(out, trace.embedding_saturation_count as u64);
    push_binary_u64(out, trace.attention_saturation_count as u64);
    push_binary_u64(out, trace.residual_saturation_count as u64);
    push_binary_u64(out, trace.output_head_zero_delta_count as u64);
    push_binary_u64(out, trace.mlp_zero_delta_count as u64);
    push_binary_u64(out, trace.embedding_zero_delta_count as u64);
    push_binary_u64(out, trace.attention_zero_delta_count as u64);
    push_binary_u64(out, trace.output_head_delta_l1);
    push_binary_u64(out, trace.mlp_delta_l1);
    push_binary_u64(out, trace.embedding_delta_l1);
    push_binary_u64(out, trace.attention_delta_l1);
    push_binary_u64(out, trace.attention_q_delta_l1);
    push_binary_u64(out, trace.attention_k_delta_l1);
    push_binary_u64(out, trace.attention_v_delta_l1);
    push_binary_u64(out, trace.attention_o_delta_l1);

    push_binary_u64(out, trace.adaptive_rule_shift_adjustment_count as u64);
    push_binary_u64(out, trace.adaptive_rule_update_count as u64);
    push_binary_u64(out, trace.adaptive_rule_event_count as u64);
    push_binary_u64(
        out,
        trace.adaptive_holographic_shift_adjustment_count as u64,
    );
    push_binary_u64(out, trace.adaptive_holographic_update_count as u64);
    push_binary_u64(out, trace.adaptive_holographic_hash);
    push_binary_u64(out, trace.adaptive_attention_shift_adjustment_count as u64);
    push_binary_u64(
        out,
        trace.adaptive_attention_holographic_update_count as u64,
    );
    push_binary_u64(out, trace.adaptive_attention_holographic_hash);
    out.push(trace.final_output_learning_rate_shift);
    out.push(trace.final_mlp_learning_rate_shift);
    out.push(trace.final_embedding_learning_rate_shift);
    out.push(trace.final_attention_learning_rate_shift);
    out.push(trace.final_attention_q_learning_rate_shift);
    out.push(trace.final_attention_qk_learning_rate_shift);

    push_binary_u64(out, trace.initial_model_hash);
    push_binary_u64(out, trace.final_model_hash);
    push_binary_u64(out, trace.initial_embedding_hash);
    push_binary_u64(out, trace.final_embedding_hash);
    push_binary_u64(out, trace.initial_output_head_hash);
    push_binary_u64(out, trace.final_output_head_hash);
    push_binary_u64(out, trace.initial_mlp_hash);
    push_binary_u64(out, trace.final_mlp_hash);
    push_binary_u64(out, trace.initial_attention_hash);
    push_binary_u64(out, trace.final_attention_hash);
    push_binary_u64(out, trace.initial_attention_q_hash);
    push_binary_u64(out, trace.final_attention_q_hash);
    push_binary_u64(out, trace.initial_attention_k_hash);
    push_binary_u64(out, trace.final_attention_k_hash);
    push_binary_u64(out, trace.initial_attention_v_hash);
    push_binary_u64(out, trace.final_attention_v_hash);
    push_binary_u64(out, trace.initial_attention_o_hash);
    push_binary_u64(out, trace.final_attention_o_hash);
    push_binary_u64(out, trace.final_logits_hash);
}

fn mini_transformer_binary_trace_detail_code(trace_detail: MiniTransformerTraceDetail) -> u8 {
    match trace_detail {
        MiniTransformerTraceDetail::Full => 0,
        MiniTransformerTraceDetail::Summary => 1,
        MiniTransformerTraceDetail::None => 2,
    }
}

fn mini_transformer_binary_tokenizer_code(tokenizer: ByteTokenizerId) -> u8 {
    match tokenizer {
        ByteTokenizerId::Identity => 0,
        ByteTokenizerId::AsciiLowerText => 1,
    }
}

fn mini_transformer_binary_attention_code(attention: MiniTransformerAttentionKind) -> u8 {
    match attention {
        MiniTransformerAttentionKind::Base2Softmax => 0,
        MiniTransformerAttentionKind::Linear => 1,
        MiniTransformerAttentionKind::LinearStreamingNope => 2,
        MiniTransformerAttentionKind::LinearStreamingTttNope => 3,
    }
}

fn mini_transformer_binary_position_code(position: MiniTransformerPositionPolicy) -> u8 {
    match position {
        MiniTransformerPositionPolicy::LearnedAbsolute => 0,
        MiniTransformerPositionPolicy::Nope => 1,
    }
}

fn mini_transformer_binary_config_flags(config: MiniTransformerMlpTrainConfig) -> u16 {
    let mut flags = 0_u16;
    if config.adaptive_rule_shifts {
        flags |= 1 << 0;
    }
    if config.adaptive_attention_shifts {
        flags |= 1 << 1;
    }
    if config.adaptive_holographic_shifts {
        flags |= 1 << 2;
    }
    if config.attention_vo_error_feedback {
        flags |= 1 << 3;
    }
    if config.attention_vo_oracle {
        flags |= 1 << 4;
    }
    if config.reject_loss_regression {
        flags |= 1 << 5;
    }
    if config.batch_mode == MiniTransformerBatchMode::MapReduce {
        flags |= 1 << 6;
    }
    flags
}

fn mini_transformer_binary_component_code(component: &str) -> u8 {
    match component {
        "output" | "output_head" => 0,
        "mlp" => 1,
        "embedding" => 2,
        "attention" => 3,
        "attention_q" => 4,
        "attention_qk" | "attention_k" => 5,
        _ => u8::MAX,
    }
}

fn mini_transformer_binary_reason_code(reason: &str) -> u8 {
    match reason {
        "rollback" | "rejected" => 0,
        "saturation" => 1,
        "zero_delta" | "dead_component" => 2,
        "movement" | "active_delta" => 3,
        "holographic" | "holographic_advisory" => 4,
        _ => u8::MAX,
    }
}

fn push_binary_optional_u32(out: &mut Vec<u8>, value: Option<usize>) {
    match value {
        Some(value) => push_binary_u32_clamped(out, value),
        None => push_binary_u32(out, u32::MAX),
    }
}

fn push_binary_u16_clamped(out: &mut Vec<u8>, value: usize) {
    push_binary_u16(out, value.min(usize::from(u16::MAX)) as u16);
}

fn push_binary_u32_clamped(out: &mut Vec<u8>, value: usize) {
    push_binary_u32(out, value.min(u32::MAX as usize) as u32);
}

fn push_binary_u32_saturating(out: &mut Vec<u8>, value: u64) {
    push_binary_u32(out, value.min(u64::from(u32::MAX)) as u32);
}

fn push_binary_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_binary_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_binary_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_binary_i16(out: &mut Vec<u8>, value: i16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_binary_i32(out: &mut Vec<u8>, value: i32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_binary_i64(out: &mut Vec<u8>, value: i64) {
    out.extend_from_slice(&value.to_le_bytes());
}

impl MiniTransformerMlpTrainingProgressTrace {
    pub fn to_json_line(&self) -> String {
        let mut out = String::new();
        out.push('{');
        push_string_field(
            &mut out,
            "schema",
            "nsrl.training_mini_transformer_progress.v1",
        );
        comma(&mut out);
        push_string_field(&mut out, "authority", AUTHORITY);
        comma(&mut out);
        push_string_field(&mut out, "task", MINI_TRANSFORMER_MLP_TASK);
        comma(&mut out);
        out.push_str("\"data\":{");
        push_string_field(&mut out, "tokenizer", self.config.tokenizer_id.as_str());
        comma(&mut out);
        push_usize_field(&mut out, "token_count", self.token_count);
        comma(&mut out);
        push_hash_field(&mut out, "token_hash", self.token_hash);
        comma(&mut out);
        push_hash_field(&mut out, "window_hash", self.window_hash);
        comma(&mut out);
        push_usize_field(&mut out, "windows", self.windows);
        out.push('}');
        comma(&mut out);
        out.push_str("\"model\":{");
        push_usize_field(&mut out, "seq_len", self.config.seq_len);
        comma(&mut out);
        push_usize_field(&mut out, "d_model", MINI_TRANSFORMER_D_MODEL);
        comma(&mut out);
        push_usize_field(&mut out, "heads", MINI_TRANSFORMER_HEADS);
        comma(&mut out);
        push_string_field(
            &mut out,
            "attention_kind",
            self.config.attention_kind.as_str(),
        );
        comma(&mut out);
        push_string_field(&mut out, "position", self.config.position_policy.as_str());
        out.push('}');
        comma(&mut out);
        out.push_str("\"training\":{");
        push_usize_field(&mut out, "epochs", self.config.epochs);
        comma(&mut out);
        push_usize_field(&mut out, "stride", self.config.stride);
        comma(&mut out);
        push_usize_field(&mut out, "window_offset", self.config.window_offset);
        comma(&mut out);
        push_optional_usize_field(&mut out, "max_windows", self.config.max_windows);
        comma(&mut out);
        push_usize_field(&mut out, "batch_windows", self.config.batch_windows);
        comma(&mut out);
        push_usize_field(&mut out, "examined_windows", self.examined_windows);
        comma(&mut out);
        push_usize_field(&mut out, "updates", self.updates);
        out.push('}');
        comma(&mut out);
        out.push_str("\"metrics\":{");
        push_usize_field(&mut out, "accepted_batch_count", self.accepted_batch_count);
        comma(&mut out);
        push_usize_field(&mut out, "rejected_batch_count", self.rejected_batch_count);
        comma(&mut out);
        push_usize_field(&mut out, "rollback_count", self.rollback_count);
        comma(&mut out);
        push_usize_field(
            &mut out,
            "rejected_window_count",
            self.rejected_window_count,
        );
        comma(&mut out);
        push_u64_field(&mut out, "output_head_delta_l1", self.output_head_delta_l1);
        comma(&mut out);
        push_u64_field(&mut out, "mlp_delta_l1", self.mlp_delta_l1);
        comma(&mut out);
        push_u64_field(&mut out, "embedding_delta_l1", self.embedding_delta_l1);
        comma(&mut out);
        push_u64_field(&mut out, "attention_delta_l1", self.attention_delta_l1);
        comma(&mut out);
        push_u64_field(&mut out, "attention_q_delta_l1", self.attention_q_delta_l1);
        comma(&mut out);
        push_u64_field(&mut out, "attention_k_delta_l1", self.attention_k_delta_l1);
        comma(&mut out);
        push_u64_field(&mut out, "attention_v_delta_l1", self.attention_v_delta_l1);
        comma(&mut out);
        push_u64_field(&mut out, "attention_o_delta_l1", self.attention_o_delta_l1);
        comma(&mut out);
        push_u64_field(&mut out, "output_head_carry_l1", self.output_head_carry_l1);
        comma(&mut out);
        push_u64_field(&mut out, "mlp_carry_l1", self.mlp_carry_l1);
        comma(&mut out);
        push_u64_field(&mut out, "embedding_carry_l1", self.embedding_carry_l1);
        comma(&mut out);
        push_u64_field(&mut out, "attention_carry_l1", self.attention_carry_l1);
        comma(&mut out);
        push_u64_field(&mut out, "attention_q_carry_l1", self.attention_q_carry_l1);
        comma(&mut out);
        push_u64_field(&mut out, "attention_k_carry_l1", self.attention_k_carry_l1);
        comma(&mut out);
        push_u64_field(&mut out, "attention_v_carry_l1", self.attention_v_carry_l1);
        comma(&mut out);
        push_u64_field(&mut out, "attention_o_carry_l1", self.attention_o_carry_l1);
        comma(&mut out);
        push_usize_field(
            &mut out,
            "adaptive_rule_shift_adjustment_count",
            self.adaptive_rule_shift_adjustment_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "adaptive_holographic_shift_adjustment_count",
            self.adaptive_holographic_shift_adjustment_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "current_output_learning_rate_shift",
            usize::from(self.current_output_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "current_mlp_learning_rate_shift",
            usize::from(self.current_mlp_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "current_embedding_learning_rate_shift",
            usize::from(self.current_embedding_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "current_attention_learning_rate_shift",
            usize::from(self.current_attention_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "current_attention_q_learning_rate_shift",
            usize::from(self.current_attention_q_learning_rate_shift),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "current_attention_qk_learning_rate_shift",
            usize::from(self.current_attention_qk_learning_rate_shift),
        );
        out.push('}');
        comma(&mut out);
        push_hash_field(&mut out, "model_hash", self.model_hash);
        comma(&mut out);
        push_hash_field(&mut out, "embedding_hash", self.embedding_hash);
        comma(&mut out);
        push_hash_field(&mut out, "attention_hash", self.attention_hash);
        comma(&mut out);
        push_hash_field(&mut out, "mlp_hash", self.mlp_hash);
        comma(&mut out);
        push_hash_field(&mut out, "output_head_hash", self.output_head_hash);
        out.push('}');
        out.push('\n');
        out
    }
}
