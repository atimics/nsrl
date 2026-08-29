//! Frozen-trunk target-margin training for the production output matrix.

use std::{fmt::Write, thread};

use nsrl_core::{DEFAULT_ZERO_PROBABILITY_NLL_MILLIBITS, base2_softmax_nll_millibits};

use super::{
    FNV_OFFSET, FNV_PRIME, ProductionModelV1, TrainError, argmax_except, document_windows, fnv1a,
    output_logits, spread_document_windows, update_i16,
};

const MARGIN_OPTIMIZER_MAGIC: &[u8; 8] = b"NSRLMT1\n";
const MARGIN_OPTIMIZER_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductionMarginTrainConfig {
    pub context_tokens: usize,
    pub max_windows: usize,
    pub window_schedule_windows: usize,
    pub spread_windows: bool,
    pub targets_per_window: usize,
    pub training_workers: usize,
    pub epochs: usize,
    pub feature_shift: u8,
    pub margin_q8: i32,
    pub batch_windows: usize,
    pub max_optimizer_steps: usize,
    pub evaluation_windows: usize,
    pub descent_guard_windows: usize,
}

impl Default for ProductionMarginTrainConfig {
    fn default() -> Self {
        Self {
            context_tokens: 4,
            max_windows: 8,
            window_schedule_windows: 0,
            spread_windows: false,
            targets_per_window: 1,
            training_workers: 1,
            epochs: 1,
            feature_shift: 13,
            margin_q8: 8,
            batch_windows: 4,
            max_optimizer_steps: usize::MAX,
            evaluation_windows: usize::MAX,
            descent_guard_windows: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionMarginOptimizerStateV1 {
    pub tokenizer_hash: u64,
    pub token_stream_hash: u64,
    pub bound_model_hash: u64,
    pub step: u64,
    pub schedule_hash: u64,
    pub next_epoch: u64,
    pub next_window: u64,
}

impl ProductionMarginOptimizerStateV1 {
    pub fn new(
        model: &ProductionModelV1,
        token_stream_hash: u64,
        config: ProductionMarginTrainConfig,
    ) -> Self {
        Self {
            tokenizer_hash: model.tokenizer_hash,
            token_stream_hash,
            bound_model_hash: model.model_hash(),
            step: 0,
            schedule_hash: margin_schedule_hash(config),
            next_epoch: 0,
            next_window: 0,
        }
    }

    pub fn validate_binding(
        &self,
        model: &ProductionModelV1,
        token_stream_hash: u64,
        config: ProductionMarginTrainConfig,
    ) -> Result<(), TrainError> {
        if self.tokenizer_hash != model.tokenizer_hash
            || self.token_stream_hash != token_stream_hash
            || self.bound_model_hash != model.model_hash()
            || self.schedule_hash != margin_schedule_hash(config)
        {
            return Err(TrainError::InvalidModel(
                "production margin optimizer binding mismatch",
            ));
        }
        Ok(())
    }

    pub fn state_hash(&self) -> u64 {
        fnv1a(&self.bytes_without_checksum())
    }

    pub fn try_to_bytes(&self) -> Result<Vec<u8>, TrainError> {
        if self.tokenizer_hash == 0 || self.token_stream_hash == 0 || self.bound_model_hash == 0 {
            return Err(TrainError::InvalidModel(
                "invalid production margin optimizer state",
            ));
        }
        let mut bytes = self.bytes_without_checksum();
        bytes.extend_from_slice(&fnv1a(&bytes).to_le_bytes());
        Ok(bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TrainError> {
        if bytes.len() != 76 || &bytes[..8] != MARGIN_OPTIMIZER_MAGIC {
            return Err(TrainError::InvalidModel(
                "bad production margin optimizer artifact",
            ));
        }
        let checksum_offset = bytes.len() - 8;
        let expected = u64::from_le_bytes(bytes[checksum_offset..].try_into().unwrap());
        if fnv1a(&bytes[..checksum_offset]) != expected {
            return Err(TrainError::InvalidModel(
                "bad production margin optimizer checksum",
            ));
        }
        if u32::from_le_bytes(bytes[8..12].try_into().unwrap()) != MARGIN_OPTIMIZER_VERSION {
            return Err(TrainError::InvalidModel(
                "unsupported production margin optimizer version",
            ));
        }
        Ok(Self {
            tokenizer_hash: read_u64(bytes, 12),
            token_stream_hash: read_u64(bytes, 20),
            bound_model_hash: read_u64(bytes, 28),
            step: read_u64(bytes, 36),
            schedule_hash: read_u64(bytes, 44),
            next_epoch: read_u64(bytes, 52),
            next_window: read_u64(bytes, 60),
        })
    }

    fn bytes_without_checksum(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(76);
        bytes.extend_from_slice(MARGIN_OPTIMIZER_MAGIC);
        bytes.extend_from_slice(&MARGIN_OPTIMIZER_VERSION.to_le_bytes());
        for value in [
            self.tokenizer_hash,
            self.token_stream_hash,
            self.bound_model_hash,
            self.step,
            self.schedule_hash,
            self.next_epoch,
            self.next_window,
        ] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductionMarginEvaluation {
    pub supervised_targets: usize,
    pub mistakes: usize,
    pub top5_hits: usize,
    pub top10_hits: usize,
    pub margin_satisfied: usize,
    pub mean_target_rank_x1000: u64,
    pub mean_target_margin_q8: i64,
    pub minimum_target_margin_q8: i64,
}

impl ProductionMarginEvaluation {
    fn to_json(self) -> String {
        format!(
            concat!(
                "{{\"supervised_targets\":{},\"mistakes\":{},",
                "\"top1_hits\":{},\"top5_hits\":{},\"top10_hits\":{},",
                "\"margin_satisfied\":{},\"mean_target_rank_x1000\":{},",
                "\"mean_target_margin_q8\":{},\"minimum_target_margin_q8\":{}}}"
            ),
            self.supervised_targets,
            self.mistakes,
            self.supervised_targets.saturating_sub(self.mistakes),
            self.top5_hits,
            self.top10_hits,
            self.margin_satisfied,
            self.mean_target_rank_x1000,
            self.mean_target_margin_q8,
            self.minimum_target_margin_q8,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductionMarginTrainTrace {
    pub profile: &'static str,
    pub parameter_count: usize,
    pub tokenizer_hash: u64,
    pub token_stream_hash: u64,
    pub context_tokens: usize,
    pub windows: usize,
    pub window_schedule_windows: usize,
    pub window_schedule_rank_hash: u64,
    pub training_window_rank_hash: u64,
    pub evaluation_windows: usize,
    pub targets_per_window: usize,
    pub training_workers: usize,
    pub epochs: usize,
    pub feature_shift: u8,
    pub margin_q8: i32,
    pub batch_windows: usize,
    pub optimizer_steps: usize,
    pub total_optimizer_step: u64,
    pub updates: usize,
    pub movement_l1: u64,
    pub weight_saturation_count: usize,
    pub rejected_batch_start_window: Option<u64>,
    pub descent_guard_windows: usize,
    pub descent_guard_window_rank_hash: u64,
    pub descent_guard_batches_evaluated: usize,
    pub descent_guard_batches_accepted: usize,
    pub descent_guard_batches_rejected: usize,
    pub descent_guard_initial_nll_millibits: u64,
    pub descent_guard_final_nll_millibits: u64,
    pub descent_guard_initial_evaluation: ProductionMarginEvaluation,
    pub descent_guard_final_evaluation: ProductionMarginEvaluation,
    pub start_epoch: u64,
    pub start_window: u64,
    pub next_epoch: u64,
    pub next_window: u64,
    pub schedule_complete: bool,
    pub initial_evaluation: ProductionMarginEvaluation,
    pub final_evaluation: ProductionMarginEvaluation,
    pub initial_model_hash: u64,
    pub final_model_hash: u64,
    pub optimizer_state_hash: u64,
    pub initial_frozen_parameter_hash: u64,
    pub final_frozen_parameter_hash: u64,
    pub initial_output_bias_hash: u64,
    pub final_output_bias_hash: u64,
    pub spread_windows: bool,
}

impl ProductionMarginTrainTrace {
    pub fn to_json_line(self) -> String {
        let mut output = String::new();
        write!(
            output,
            concat!(
                "{{\"schema\":\"nsrl.production_target_margin_train.v1\",",
                "\"profile\":\"{}\",\"parameter_count\":{},",
                "\"bindings\":{{\"tokenizer_hash\":\"0x{:016x}\",\"token_stream_hash\":\"0x{:016x}\"}},",
                "\"training\":{{\"objective\":\"hard_negative_hinge_q8\",",
                "\"parameter_scope\":\"output_matrix_only\",\"output_bias_frozen\":true,",
                "\"context_tokens\":{},\"windows\":{},\"evaluation_windows\":{},",
                "\"window_schedule_windows\":{},\"window_schedule_rank_hash\":\"0x{:016x}\",",
                "\"training_window_rank_hash\":\"0x{:016x}\",",
                "\"targets_per_window\":{},\"epochs\":{},\"feature_shift\":{},",
                "\"margin_q8\":{},\"batch_windows\":{},\"optimizer_steps\":{},",
                "\"total_optimizer_step\":{},\"updates\":{},\"movement_l1\":{}}},",
                "\"evaluation\":{{\"initial\":{},\"final\":{}}},",
                "\"cursor\":{{\"start_epoch\":{},\"start_window\":{},",
                "\"next_epoch\":{},\"next_window\":{},\"schedule_complete\":{}}},",
                "\"transaction\":{{\"saturation_policy\":\"reject_batch_stop\",",
                "\"rejected_batch_start_window\":{}}},",
                "\"descent_guard\":{{\"policy\":\"reject_worsening_batch_consume_cursor\",",
                "\"windows\":{},\"window_rank_hash\":\"0x{:016x}\",",
                "\"batches_evaluated\":{},\"batches_accepted\":{},\"batches_rejected\":{},",
                "\"initial_nll_millibits\":{},\"final_nll_millibits\":{},",
                "\"initial_evaluation\":{},\"final_evaluation\":{}}},",
                "\"health\":{{\"weight_saturation_count\":{}}},",
                "\"hashes\":{{\"initial_model\":\"0x{:016x}\",\"final_model\":\"0x{:016x}\",",
                "\"optimizer_state\":\"0x{:016x}\",",
                "\"initial_frozen_parameters\":\"0x{:016x}\",\"final_frozen_parameters\":\"0x{:016x}\",",
                "\"initial_output_bias\":\"0x{:016x}\",\"final_output_bias\":\"0x{:016x}\"}},",
                "\"gates\":{{\"frozen_parameters_unchanged\":{},\"output_bias_unchanged\":{},",
                "\"zero_weight_saturation\":{},\"descent_guard_nonworsening\":{},",
                "\"descent_guard_disjoint_from_window_schedule\":true,",
                "\"resumable_optimizer_state\":true}},",
                "\"known_non_claims\":[\"bounded_output_matrix_pilot_not_scaling_run\",",
                "\"training_surface_metrics_not_held_out_quality\",\"not_open_generation_quality\"]}}\n"
            ),
            self.profile,
            self.parameter_count,
            self.tokenizer_hash,
            self.token_stream_hash,
            self.context_tokens,
            self.windows,
            self.evaluation_windows,
            self.window_schedule_windows,
            self.window_schedule_rank_hash,
            self.training_window_rank_hash,
            self.targets_per_window,
            self.epochs,
            self.feature_shift,
            self.margin_q8,
            self.batch_windows,
            self.optimizer_steps,
            self.total_optimizer_step,
            self.updates,
            self.movement_l1,
            self.initial_evaluation.to_json(),
            self.final_evaluation.to_json(),
            self.start_epoch,
            self.start_window,
            self.next_epoch,
            self.next_window,
            self.schedule_complete,
            self.rejected_batch_start_window
                .map_or_else(|| "null".to_string(), |value| value.to_string()),
            self.descent_guard_windows,
            self.descent_guard_window_rank_hash,
            self.descent_guard_batches_evaluated,
            self.descent_guard_batches_accepted,
            self.descent_guard_batches_rejected,
            self.descent_guard_initial_nll_millibits,
            self.descent_guard_final_nll_millibits,
            self.descent_guard_initial_evaluation.to_json(),
            self.descent_guard_final_evaluation.to_json(),
            self.weight_saturation_count,
            self.initial_model_hash,
            self.final_model_hash,
            self.optimizer_state_hash,
            self.initial_frozen_parameter_hash,
            self.final_frozen_parameter_hash,
            self.initial_output_bias_hash,
            self.final_output_bias_hash,
            self.initial_frozen_parameter_hash == self.final_frozen_parameter_hash,
            self.initial_output_bias_hash == self.final_output_bias_hash,
            self.weight_saturation_count == 0,
            self.descent_guard_final_nll_millibits
                <= self.descent_guard_initial_nll_millibits,
        )
        .expect("writing JSON to String cannot fail");
        if self.spread_windows {
            output = output.replace(
                ",\"evaluation_windows\"",
                ",\"window_selection\":\"deterministic_uniform_target_rank_over_all_documents\",\"evaluation_windows\"",
            );
        }
        if self.training_workers > 1 {
            output = output.replace(
                ",\"epochs\"",
                &format!(",\"training_workers\":{},\"epochs\"", self.training_workers),
            );
        }
        output
    }
}

#[derive(Clone)]
struct FrozenTarget {
    features: Vec<i16>,
    target: usize,
}

#[derive(Debug, Clone, Copy)]
struct TargetDecision {
    competitor: usize,
    rank: usize,
    margin_q8: i64,
}

pub fn train_production_target_margin(
    model: &mut ProductionModelV1,
    tokens: &[u32],
    token_stream_hash: u64,
    config: ProductionMarginTrainConfig,
    state: Option<ProductionMarginOptimizerStateV1>,
) -> Result<(ProductionMarginTrainTrace, ProductionMarginOptimizerStateV1), TrainError> {
    model.validate()?;
    validate_config(model, tokens, config)?;
    let requested_schedule_windows = if config.window_schedule_windows == 0 {
        config.max_windows
    } else {
        config.window_schedule_windows
    };
    let scheduled_windows = if config.spread_windows {
        spread_document_windows(tokens, config.context_tokens, requested_schedule_windows)
    } else {
        document_windows(tokens, config.context_tokens, requested_schedule_windows)
    };
    let window_schedule_windows = scheduled_windows.len();
    let windows = scheduled_windows[..config.max_windows.min(scheduled_windows.len())].to_vec();
    if windows.is_empty() {
        return Err(TrainError::InvalidConfig);
    }
    let total_windows = super::training::document_window_count(tokens, config.context_tokens);
    let window_schedule_ranks = super::training::update_window_ranks(
        total_windows,
        window_schedule_windows,
        config.spread_windows,
    );
    let training_window_ranks = window_schedule_ranks[..windows.len()].to_vec();
    let descent_guard_ranks = super::training::descent_guard_window_ranks(
        total_windows,
        &window_schedule_ranks,
        config.descent_guard_windows,
    )?;
    let descent_guard_windows = super::training::document_windows_at_ranks(
        tokens,
        config.context_tokens,
        &descent_guard_ranks,
    );
    if descent_guard_windows.len() != config.descent_guard_windows {
        return Err(TrainError::InvalidConfig);
    }

    let initial_model_hash = model.model_hash();
    let initial_frozen_parameter_hash = frozen_parameter_hash(model);
    let initial_output_bias_hash = i32_slice_hash(&model.output_bias_q8);
    let (cached, source_decisions) = cache_targets(model, &windows, config)?;
    let evaluation_windows = config.evaluation_windows.min(windows.len());
    let evaluation_targets = evaluation_windows * config.targets_per_window;
    let initial_evaluation =
        summarize_decisions(&source_decisions[..evaluation_targets], config.margin_q8);
    let descent_guard_targets =
        cache_final_targets(model, &descent_guard_windows, config.training_workers)?;
    let (mut descent_guard_current_nll_millibits, descent_guard_source_decisions) =
        evaluate_guard(model, &descent_guard_targets)?;
    let descent_guard_initial_nll_millibits = descent_guard_current_nll_millibits;
    let descent_guard_initial_evaluation =
        summarize_decisions(&descent_guard_source_decisions, config.margin_q8);

    let mut state = state
        .unwrap_or_else(|| ProductionMarginOptimizerStateV1::new(model, token_stream_hash, config));
    state.validate_binding(model, token_stream_hash, config)?;
    if state.next_epoch > config.epochs as u64
        || state.next_window > windows.len() as u64
        || (state.next_epoch == config.epochs as u64 && state.next_window != 0)
    {
        return Err(TrainError::InvalidModel(
            "production margin optimizer cursor mismatch",
        ));
    }

    let start_epoch = state.next_epoch;
    let start_window = state.next_window;
    let mut optimizer_steps = 0_usize;
    let mut updates = 0_usize;
    let mut movement_l1 = 0_u64;
    let mut weight_saturation_count = 0_usize;
    let mut rejected_batch_start_window = None;
    let mut descent_guard_batches_evaluated = 0_usize;
    let mut descent_guard_batches_accepted = 0_usize;
    let mut descent_guard_batches_rejected = 0_usize;

    while state.next_epoch < config.epochs as u64 && optimizer_steps < config.max_optimizer_steps {
        let batch_start = state.next_window as usize;
        let batch_end = batch_start
            .saturating_add(config.batch_windows)
            .min(windows.len());
        let row_start = batch_start * config.targets_per_window;
        let row_end = batch_end * config.targets_per_window;
        let decisions =
            evaluate_targets(model, &cached[row_start..row_end], config.training_workers)?;
        let output_before = model.output_weights.clone();
        let mut batch_saturation = 0_usize;
        let mut batch_updates = 0_usize;
        let mut batch_movement = 0_u64;
        for (row, decision) in cached[row_start..row_end].iter().zip(&decisions) {
            if decision.margin_q8 >= i64::from(config.margin_q8) {
                continue;
            }
            batch_updates = batch_updates.saturating_add(1);
            for (dimension, &feature) in row.features.iter().enumerate() {
                let mut delta = i32::from(feature) >> config.feature_shift;
                if delta == 0 && feature != 0 {
                    delta = i32::from(feature.signum());
                }
                let target_offset = row.target * model.config.d_model + dimension;
                let competitor_offset = decision.competitor * model.config.d_model + dimension;
                let target_before = model.output_weights[target_offset];
                let competitor_before = model.output_weights[competitor_offset];
                batch_saturation = batch_saturation
                    .saturating_add(update_i16(&mut model.output_weights[target_offset], delta));
                batch_saturation = batch_saturation.saturating_add(update_i16(
                    &mut model.output_weights[competitor_offset],
                    -delta,
                ));
                batch_movement = batch_movement
                    .saturating_add(
                        (i32::from(model.output_weights[target_offset]) - i32::from(target_before))
                            .unsigned_abs() as u64,
                    )
                    .saturating_add(
                        (i32::from(model.output_weights[competitor_offset])
                            - i32::from(competitor_before))
                        .unsigned_abs() as u64,
                    );
            }
        }
        if batch_saturation != 0 {
            model.output_weights = output_before;
            weight_saturation_count = weight_saturation_count.saturating_add(batch_saturation);
            rejected_batch_start_window = Some(state.next_window);
            break;
        }

        let guard_accepted = if descent_guard_targets.is_empty() {
            true
        } else {
            descent_guard_batches_evaluated = descent_guard_batches_evaluated.saturating_add(1);
            let (candidate_nll_millibits, _) = evaluate_guard(model, &descent_guard_targets)?;
            if candidate_nll_millibits <= descent_guard_current_nll_millibits {
                descent_guard_current_nll_millibits = candidate_nll_millibits;
                descent_guard_batches_accepted = descent_guard_batches_accepted.saturating_add(1);
                true
            } else {
                model.output_weights = output_before;
                descent_guard_batches_rejected = descent_guard_batches_rejected.saturating_add(1);
                false
            }
        };

        if guard_accepted {
            updates = updates.saturating_add(batch_updates);
            movement_l1 = movement_l1.saturating_add(batch_movement);
        }
        optimizer_steps = optimizer_steps.saturating_add(1);
        state.step = state.step.checked_add(1).ok_or(TrainError::InvalidConfig)?;
        state.next_window = batch_end as u64;
        if batch_end == windows.len() {
            state.next_epoch = state
                .next_epoch
                .checked_add(1)
                .ok_or(TrainError::InvalidConfig)?;
            state.next_window = 0;
        }
        state.bound_model_hash = model.model_hash();
    }

    let final_decisions = evaluate_targets(
        model,
        &cached[..evaluation_targets],
        config.training_workers,
    )?;
    let final_evaluation = summarize_decisions(&final_decisions, config.margin_q8);
    let (descent_guard_final_nll_millibits, descent_guard_final_decisions) =
        evaluate_guard(model, &descent_guard_targets)?;
    let descent_guard_final_evaluation =
        summarize_decisions(&descent_guard_final_decisions, config.margin_q8);
    let final_model_hash = model.model_hash();
    let final_frozen_parameter_hash = frozen_parameter_hash(model);
    let final_output_bias_hash = i32_slice_hash(&model.output_bias_q8);
    state.bound_model_hash = final_model_hash;
    let trace = ProductionMarginTrainTrace {
        profile: model.config.profile_id().unwrap_or("custom"),
        parameter_count: model.parameter_count(),
        tokenizer_hash: model.tokenizer_hash,
        token_stream_hash,
        context_tokens: config.context_tokens,
        windows: windows.len(),
        window_schedule_windows,
        window_schedule_rank_hash: super::training::window_rank_hash(&window_schedule_ranks),
        training_window_rank_hash: super::training::window_rank_hash(&training_window_ranks),
        evaluation_windows,
        targets_per_window: config.targets_per_window,
        training_workers: config.training_workers,
        epochs: config.epochs,
        feature_shift: config.feature_shift,
        margin_q8: config.margin_q8,
        batch_windows: config.batch_windows,
        optimizer_steps,
        total_optimizer_step: state.step,
        updates,
        movement_l1,
        weight_saturation_count,
        rejected_batch_start_window,
        descent_guard_windows: descent_guard_windows.len(),
        descent_guard_window_rank_hash: super::training::window_rank_hash(&descent_guard_ranks),
        descent_guard_batches_evaluated,
        descent_guard_batches_accepted,
        descent_guard_batches_rejected,
        descent_guard_initial_nll_millibits,
        descent_guard_final_nll_millibits,
        descent_guard_initial_evaluation,
        descent_guard_final_evaluation,
        start_epoch,
        start_window,
        next_epoch: state.next_epoch,
        next_window: state.next_window,
        schedule_complete: state.next_epoch == config.epochs as u64,
        initial_evaluation,
        final_evaluation,
        initial_model_hash,
        final_model_hash,
        optimizer_state_hash: state.state_hash(),
        initial_frozen_parameter_hash,
        final_frozen_parameter_hash,
        initial_output_bias_hash,
        final_output_bias_hash,
        spread_windows: config.spread_windows,
    };
    Ok((trace, state))
}

fn validate_config(
    model: &ProductionModelV1,
    tokens: &[u32],
    config: ProductionMarginTrainConfig,
) -> Result<(), TrainError> {
    if config.context_tokens == 0
        || config.context_tokens > model.config.context_tokens
        || config.max_windows == 0
        || (config.window_schedule_windows != 0
            && config.window_schedule_windows < config.max_windows)
        || config.targets_per_window == 0
        || config.targets_per_window > config.context_tokens
        || config.training_workers == 0
        || config.training_workers > 256
        || config.epochs == 0
        || config.feature_shift > 15
        || config.margin_q8 < 0
        || config.batch_windows == 0
        || config.max_optimizer_steps == 0
        || config.evaluation_windows == 0
        || tokens
            .iter()
            .any(|&token| token as usize >= model.config.vocab_size)
    {
        return Err(TrainError::InvalidConfig);
    }
    Ok(())
}

fn cache_targets(
    model: &ProductionModelV1,
    windows: &[(Vec<u32>, u32)],
    config: ProductionMarginTrainConfig,
) -> Result<(Vec<FrozenTarget>, Vec<TargetDecision>), TrainError> {
    let mut cached = Vec::with_capacity(windows.len() * config.targets_per_window);
    let mut decisions = Vec::with_capacity(cached.capacity());
    for (context, final_target) in windows {
        let first_context_target = context.len() - config.targets_per_window + 1;
        let mut targets = context[first_context_target..]
            .iter()
            .map(|&token| token as usize)
            .collect::<Vec<_>>();
        targets.push(*final_target as usize);
        let (features, logits) = super::training::frozen_target_rows(
            model,
            context,
            config.targets_per_window,
            config.training_workers,
        )?;
        for (row, &target) in targets.iter().enumerate() {
            let feature_start = row * model.config.d_model;
            let logit_start = row * model.config.vocab_size;
            cached.push(FrozenTarget {
                features: features[feature_start..feature_start + model.config.d_model].to_vec(),
                target,
            });
            decisions.push(decision_from_logits(
                &logits[logit_start..logit_start + model.config.vocab_size],
                target,
            ));
        }
    }
    Ok((cached, decisions))
}

fn cache_final_targets(
    model: &ProductionModelV1,
    windows: &[(Vec<u32>, u32)],
    workers: usize,
) -> Result<Vec<FrozenTarget>, TrainError> {
    windows
        .iter()
        .map(|(context, target)| {
            let (features, _) = super::training::frozen_target_rows(model, context, 1, workers)?;
            Ok(FrozenTarget {
                features,
                target: *target as usize,
            })
        })
        .collect()
}

fn evaluate_guard(
    model: &ProductionModelV1,
    targets: &[FrozenTarget],
) -> Result<(u64, Vec<TargetDecision>), TrainError> {
    let mut total_nll_millibits = 0_u64;
    let mut decisions = Vec::with_capacity(targets.len());
    for row in targets {
        let logits = output_logits(model, &row.features)?;
        let nll_millibits = base2_softmax_nll_millibits(
            &logits,
            row.target,
            DEFAULT_ZERO_PROBABILITY_NLL_MILLIBITS,
        )
        .ok_or(TrainError::CoreRejected(
            "production_margin_descent_guard_nll",
        ))?;
        total_nll_millibits =
            total_nll_millibits
                .checked_add(nll_millibits)
                .ok_or(TrainError::CoreRejected(
                    "production_margin_descent_guard_nll_overflow",
                ))?;
        decisions.push(decision_from_logits(&logits, row.target));
    }
    Ok((total_nll_millibits, decisions))
}

fn evaluate_targets(
    model: &ProductionModelV1,
    targets: &[FrozenTarget],
    workers: usize,
) -> Result<Vec<TargetDecision>, TrainError> {
    let workers = workers.min(targets.len()).max(1);
    if workers == 1 {
        return targets
            .iter()
            .map(|row| {
                output_logits(model, &row.features)
                    .map(|logits| decision_from_logits(&logits, row.target))
            })
            .collect();
    }
    let rows_per_worker = targets.len().div_ceil(workers);
    thread::scope(|scope| {
        let handles = targets
            .chunks(rows_per_worker)
            .map(|chunk| {
                scope.spawn(move || {
                    chunk
                        .iter()
                        .map(|row| {
                            output_logits(model, &row.features)
                                .map(|logits| decision_from_logits(&logits, row.target))
                        })
                        .collect::<Result<Vec<_>, _>>()
                })
            })
            .collect::<Vec<_>>();
        let mut output = Vec::with_capacity(targets.len());
        for handle in handles {
            output.extend(
                handle
                    .join()
                    .map_err(|_| TrainError::CoreRejected("production_margin_worker_panic"))??,
            );
        }
        Ok(output)
    })
}

fn decision_from_logits(logits: &[i32], target: usize) -> TargetDecision {
    let competitor = argmax_except(logits, target);
    let target_logit = logits[target];
    let rank = 1 + logits
        .iter()
        .enumerate()
        .filter(|&(index, &logit)| {
            index != target && (logit > target_logit || (logit == target_logit && index < target))
        })
        .count();
    TargetDecision {
        competitor,
        rank,
        margin_q8: i64::from(target_logit) - i64::from(logits[competitor]),
    }
}

fn summarize_decisions(
    decisions: &[TargetDecision],
    required_margin_q8: i32,
) -> ProductionMarginEvaluation {
    let count = decisions.len().max(1);
    let rank_sum = decisions
        .iter()
        .map(|decision| decision.rank as u64)
        .sum::<u64>();
    let margin_sum = decisions
        .iter()
        .map(|decision| i128::from(decision.margin_q8))
        .sum::<i128>();
    ProductionMarginEvaluation {
        supervised_targets: decisions.len(),
        mistakes: decisions
            .iter()
            .filter(|decision| decision.rank != 1)
            .count(),
        top5_hits: decisions
            .iter()
            .filter(|decision| decision.rank <= 5)
            .count(),
        top10_hits: decisions
            .iter()
            .filter(|decision| decision.rank <= 10)
            .count(),
        margin_satisfied: decisions
            .iter()
            .filter(|decision| decision.margin_q8 >= i64::from(required_margin_q8))
            .count(),
        mean_target_rank_x1000: rank_sum.saturating_mul(1_000) / count as u64,
        mean_target_margin_q8: (margin_sum / count as i128)
            .clamp(i128::from(i64::MIN), i128::from(i64::MAX))
            as i64,
        minimum_target_margin_q8: decisions
            .iter()
            .map(|decision| decision.margin_q8)
            .min()
            .unwrap_or(0),
    }
}

fn margin_schedule_hash(config: ProductionMarginTrainConfig) -> u64 {
    let mut bytes = Vec::new();
    for value in [
        config.context_tokens,
        config.max_windows,
        config.targets_per_window,
        config.epochs,
        config.batch_windows,
    ] {
        bytes.extend_from_slice(&(value as u64).to_le_bytes());
    }
    bytes.extend_from_slice(&[u8::from(config.spread_windows), config.feature_shift]);
    bytes.extend_from_slice(&config.margin_q8.to_le_bytes());
    if config.window_schedule_windows != 0 || config.descent_guard_windows != 0 {
        bytes.extend_from_slice(b"margin-trust-region-v1");
        bytes.extend_from_slice(&(config.window_schedule_windows as u64).to_le_bytes());
        bytes.extend_from_slice(&(config.descent_guard_windows as u64).to_le_bytes());
    }
    fnv1a(&bytes)
}

fn frozen_parameter_hash(model: &ProductionModelV1) -> u64 {
    let mut hash = FNV_OFFSET;
    let mut update = |bytes: &[u8]| {
        for &byte in bytes {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    };
    update(&model.tokenizer_hash.to_le_bytes());
    update(&model.initialization_seed.to_le_bytes());
    for value in [
        model.config.vocab_size,
        model.config.d_model,
        model.config.heads,
        model.config.layers,
        model.config.hidden_dim,
        model.config.context_tokens,
    ] {
        update(&(value as u64).to_le_bytes());
    }
    update(&[
        model.scales.qkv_shift,
        model.scales.o_shift,
        model.scales.up_shift,
        model.scales.gate_shift,
        model.scales.down_shift,
        model.scales.output_shift,
    ]);
    for values in [
        &model.embeddings,
        &model.attention_rms_weights,
        &model.mlp_rms_weights,
        &model.final_rms_weights,
    ] {
        for value in values {
            update(&value.to_le_bytes());
        }
    }
    for values in [
        &model.q_weights,
        &model.k_weights,
        &model.v_weights,
        &model.o_weights,
        &model.up_weights,
        &model.gate_weights,
        &model.down_weights,
    ] {
        for &value in values {
            update(&[value as u8]);
        }
    }
    hash
}

fn i32_slice_hash(values: &[i32]) -> u64 {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fnv1a(&bytes)
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::production::ProductionModelConfig;
    use nsrl_corpus::subword::{BOS_TOKEN_ID, EOS_TOKEN_ID};

    fn tiny_model() -> ProductionModelV1 {
        ProductionModelV1::new_initial(
            ProductionModelConfig {
                vocab_size: 320,
                d_model: 16,
                heads: 4,
                layers: 2,
                hidden_dim: 48,
                context_tokens: 16,
            },
            0x1234,
            11,
        )
        .expect("model")
    }

    fn tokens() -> Vec<u32> {
        vec![
            BOS_TOKEN_ID,
            300,
            301,
            302,
            303,
            304,
            305,
            306,
            307,
            EOS_TOKEN_ID,
        ]
    }

    fn long_tokens() -> Vec<u32> {
        let mut tokens = vec![BOS_TOKEN_ID];
        tokens.extend((0..48).map(|index| 300 + index % 20));
        tokens.push(EOS_TOKEN_ID);
        tokens
    }

    fn config() -> ProductionMarginTrainConfig {
        ProductionMarginTrainConfig {
            context_tokens: 4,
            max_windows: 4,
            targets_per_window: 2,
            training_workers: 2,
            epochs: 2,
            feature_shift: 13,
            margin_q8: 8,
            batch_windows: 1,
            max_optimizer_steps: usize::MAX,
            evaluation_windows: 4,
            ..ProductionMarginTrainConfig::default()
        }
    }

    #[test]
    fn margin_optimizer_round_trips_and_rejects_corruption() {
        let model = tiny_model();
        let state = ProductionMarginOptimizerStateV1::new(&model, 0x5678, config());
        let bytes = state.try_to_bytes().expect("serialize");
        assert_eq!(
            ProductionMarginOptimizerStateV1::from_bytes(&bytes).expect("decode"),
            state
        );
        let mut corrupt = bytes;
        corrupt[20] ^= 1;
        assert!(ProductionMarginOptimizerStateV1::from_bytes(&corrupt).is_err());
    }

    #[test]
    fn target_margin_training_only_changes_output_matrix() {
        let mut model = tiny_model();
        let source = model.clone();
        let (trace, _) =
            train_production_target_margin(&mut model, &tokens(), 0x5678, config(), None)
                .expect("margin train");
        assert_ne!(model.output_weights, source.output_weights);
        assert_eq!(model.output_bias_q8, source.output_bias_q8);
        assert_eq!(
            frozen_parameter_hash(&model),
            frozen_parameter_hash(&source)
        );
        assert_eq!(trace.weight_saturation_count, 0);
        assert!(
            trace.final_evaluation.mean_target_rank_x1000
                <= trace.initial_evaluation.mean_target_rank_x1000
        );
        assert!(
            trace
                .to_json_line()
                .contains("\"output_bias_unchanged\":true")
        );
    }

    #[test]
    fn midpoint_restart_is_byte_exact() {
        let mut uninterrupted = tiny_model();
        let source = uninterrupted.clone();
        let (_, uninterrupted_state) =
            train_production_target_margin(&mut uninterrupted, &tokens(), 0x5678, config(), None)
                .expect("uninterrupted");

        let mut midpoint = source;
        let partial = ProductionMarginTrainConfig {
            max_optimizer_steps: 4,
            ..config()
        };
        let (_, midpoint_state) =
            train_production_target_margin(&mut midpoint, &tokens(), 0x5678, partial, None)
                .expect("midpoint");
        let (_, resumed_state) = train_production_target_margin(
            &mut midpoint,
            &tokens(),
            0x5678,
            config(),
            Some(midpoint_state),
        )
        .expect("resume");
        assert_eq!(midpoint, uninterrupted);
        assert_eq!(resumed_state, uninterrupted_state);
    }

    #[test]
    fn short_and_full_runs_share_one_bound_schedule_and_disjoint_guard() {
        let tokens = long_tokens();
        let total = super::super::training::document_window_count(&tokens, 4);
        let schedule = super::super::training::update_window_ranks(total, 24, true);
        let short_schedule = super::super::training::update_window_ranks(total, 24, true);
        let guard = super::super::training::descent_guard_window_ranks(total, &schedule, 8)
            .expect("guard ranks");
        assert_eq!(short_schedule, schedule);
        assert_eq!(&short_schedule[..6], &schedule[..6]);
        assert!(
            guard
                .iter()
                .all(|rank| schedule.binary_search(rank).is_err())
        );

        let mut model = tiny_model();
        let guarded = ProductionMarginTrainConfig {
            context_tokens: 4,
            max_windows: 6,
            window_schedule_windows: 24,
            spread_windows: true,
            targets_per_window: 1,
            training_workers: 2,
            epochs: 1,
            feature_shift: 13,
            margin_q8: 8,
            batch_windows: 1,
            max_optimizer_steps: usize::MAX,
            evaluation_windows: 6,
            descent_guard_windows: 8,
        };
        let (trace, _) = train_production_target_margin(&mut model, &tokens, 0x9876, guarded, None)
            .expect("guarded train");
        assert_eq!(trace.windows, 6);
        assert_eq!(trace.window_schedule_windows, 24);
        assert_eq!(trace.descent_guard_windows, 8);
        assert_eq!(
            trace.descent_guard_batches_evaluated,
            trace.descent_guard_batches_accepted + trace.descent_guard_batches_rejected
        );
        assert!(trace.descent_guard_batches_rejected > 0);
        assert!(
            trace.descent_guard_final_nll_millibits <= trace.descent_guard_initial_nll_millibits
        );
    }

    #[test]
    fn guarded_midpoint_restart_is_byte_exact() {
        let tokens = long_tokens();
        let guarded = ProductionMarginTrainConfig {
            context_tokens: 4,
            max_windows: 12,
            window_schedule_windows: 24,
            spread_windows: true,
            targets_per_window: 2,
            training_workers: 2,
            epochs: 2,
            feature_shift: 13,
            margin_q8: 8,
            batch_windows: 2,
            max_optimizer_steps: usize::MAX,
            evaluation_windows: 12,
            descent_guard_windows: 8,
        };
        let mut uninterrupted = tiny_model();
        let source = uninterrupted.clone();
        let (_, uninterrupted_state) =
            train_production_target_margin(&mut uninterrupted, &tokens, 0x9876, guarded, None)
                .expect("uninterrupted guarded train");

        let mut midpoint = source;
        let partial = ProductionMarginTrainConfig {
            max_optimizer_steps: 5,
            ..guarded
        };
        let (_, midpoint_state) =
            train_production_target_margin(&mut midpoint, &tokens, 0x9876, partial, None)
                .expect("guarded midpoint");
        let (_, resumed_state) = train_production_target_margin(
            &mut midpoint,
            &tokens,
            0x9876,
            guarded,
            Some(midpoint_state),
        )
        .expect("guarded resume");
        assert_eq!(midpoint, uninterrupted);
        assert_eq!(resumed_state, uninterrupted_state);
    }
}
