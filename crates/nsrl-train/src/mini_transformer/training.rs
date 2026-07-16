//! Mini-transformer training, evaluation, adaptive control, and map-reduce orchestration.

use super::*;

pub(super) const MINI_TRANSFORMER_HOLO_META_DIM: usize = 8;
pub(super) const MINI_TRANSFORMER_HOLO_ACTION_COUNT: usize = 5;
pub(super) const MINI_TRANSFORMER_HOLO_MEMORY_UPDATE_SHIFT: u32 = 8;
pub(super) const MINI_TRANSFORMER_HOLO_QUERY_SHIFT: u32 = 15;
pub(super) const MINI_TRANSFORMER_HOLO_MEMORY_MIN_UPDATES: usize = 8;
pub(super) const MINI_TRANSFORMER_HOLO_ADJUSTMENT_COOLDOWN_BATCHES: usize = 32;
pub(super) const MINI_TRANSFORMER_STACKED_BLOCK_LEARNING_RATE_EXTRA_SHIFT: u8 = 2;
pub(super) const MINI_TRANSFORMER_STACKED_LOWER_LAYER_LEARNING_RATE_EXTRA_SHIFT: u8 = 14;
pub(super) const MINI_TRANSFORMER_STACKED_EMBEDDING_LEARNING_RATE_EXTRA_SHIFT: u8 = 0;
pub(super) const MINI_TRANSFORMER_HOLO_ACTION_ATOMS: [[i16; MINI_TRANSFORMER_HOLO_META_DIM];
    MINI_TRANSFORMER_HOLO_ACTION_COUNT] = [
    [16384, -16384, 16384, -16384, 8192, -8192, 4096, -4096],
    [16384, 16384, -16384, -16384, 8192, 8192, -4096, -4096],
    [16384, 0, -16384, 0, 16384, 0, -16384, 0],
    [-16384, 16384, 16384, -16384, -8192, 8192, 4096, -4096],
    [-16384, -16384, -16384, -16384, 8192, 8192, 4096, 4096],
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct IntegerHolographicShiftMemory {
    pub(super) memory: [[i64; MINI_TRANSFORMER_HOLO_META_DIM]; MINI_TRANSFORMER_HOLO_META_DIM],
    pub(super) normalizer: [i64; MINI_TRANSFORMER_HOLO_META_DIM],
    pub(super) update_count: usize,
}

impl IntegerHolographicShiftMemory {
    pub(super) fn new() -> Self {
        Self {
            memory: [[0_i64; MINI_TRANSFORMER_HOLO_META_DIM]; MINI_TRANSFORMER_HOLO_META_DIM],
            normalizer: [0_i64; MINI_TRANSFORMER_HOLO_META_DIM],
            update_count: 0,
        }
    }

    pub(super) fn remember(
        &mut self,
        state_q15: &[i16; MINI_TRANSFORMER_HOLO_META_DIM],
        delta: i8,
    ) {
        let atom = mini_transformer_holo_action_atom(delta);
        for (row, &atom_value) in atom.iter().enumerate() {
            for (col, &state_value) in state_q15.iter().enumerate() {
                let wide = i64::from(atom_value) * i64::from(state_value);
                self.memory[row][col] = self.memory[row][col]
                    .saturating_add(wide >> MINI_TRANSFORMER_HOLO_MEMORY_UPDATE_SHIFT);
            }
        }
        for (slot, &value) in self.normalizer.iter_mut().zip(state_q15.iter()) {
            *slot = slot.saturating_add(i64::from(value).abs());
        }
        self.update_count = self.update_count.saturating_add(1);
    }

    pub(super) fn retrieve_delta(
        &self,
        state_q15: &[i16; MINI_TRANSFORMER_HOLO_META_DIM],
    ) -> Option<i8> {
        if self.update_count == 0 {
            return None;
        }
        let mut denominator = 0_i128;
        for (&norm, &state) in self.normalizer.iter().zip(state_q15.iter()) {
            denominator += i128::from(norm) * i128::from(state).abs();
        }
        if denominator == 0 {
            return None;
        }

        let mut recalled = [0_i64; MINI_TRANSFORMER_HOLO_META_DIM];
        for (row, out) in recalled.iter_mut().enumerate() {
            let mut acc = 0_i128;
            for (col, &state) in state_q15.iter().enumerate() {
                acc += i128::from(self.memory[row][col]) * i128::from(state);
            }
            *out = (acc >> MINI_TRANSFORMER_HOLO_QUERY_SHIFT)
                .clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64;
        }

        let mut best_delta = 0_i8;
        let mut best_score = i128::MIN;
        for delta in -2_i8..=2 {
            let atom = mini_transformer_holo_action_atom(delta);
            let mut score = 0_i128;
            for (&value, &basis) in recalled.iter().zip(atom.iter()) {
                score += i128::from(value) * i128::from(basis);
            }
            if score > best_score {
                best_score = score;
                best_delta = delta;
            }
        }

        if best_score <= 0 {
            None
        } else {
            Some(best_delta)
        }
    }

    pub(super) fn hash_into(&self, hasher: &mut StableHasher) {
        hasher.update_usize(self.update_count);
        for row in self.memory {
            for value in row {
                hasher.update_bytes(&value.to_le_bytes());
            }
        }
        for value in self.normalizer {
            hasher.update_bytes(&value.to_le_bytes());
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct MiniTransformerRuleShiftWindow {
    pub(super) observation_batches: usize,
    pub(super) rejected_batches: usize,
    pub(super) stats: LinearWeightUpdateStats,
}

impl MiniTransformerRuleShiftWindow {
    pub(super) fn new() -> Self {
        Self {
            observation_batches: 0,
            rejected_batches: 0,
            stats: empty_linear_weight_update_stats(),
        }
    }

    pub(super) fn observe_accepted(&mut self, stats: LinearWeightUpdateStats) {
        self.observation_batches = self.observation_batches.saturating_add(1);
        self.stats.gradient_saturation_count = self
            .stats
            .gradient_saturation_count
            .saturating_add(stats.gradient_saturation_count);
        self.stats.zero_delta_count = self
            .stats
            .zero_delta_count
            .saturating_add(stats.zero_delta_count);
        self.stats.weight_delta_l1 = self
            .stats
            .weight_delta_l1
            .saturating_add(stats.weight_delta_l1);
    }

    pub(super) fn observe_rejected(&mut self) {
        self.rejected_batches = self.rejected_batches.saturating_add(1);
        self.observation_batches = self.observation_batches.saturating_add(1);
    }

    pub(super) fn reset(&mut self) {
        *self = Self::new();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct MiniTransformerAdaptiveShiftState {
    pub(super) output_memory: IntegerHolographicShiftMemory,
    pub(super) mlp_memory: IntegerHolographicShiftMemory,
    pub(super) embedding_memory: IntegerHolographicShiftMemory,
    pub(super) q_memory: IntegerHolographicShiftMemory,
    pub(super) k_memory: IntegerHolographicShiftMemory,
    pub(super) v_memory: IntegerHolographicShiftMemory,
    pub(super) o_memory: IntegerHolographicShiftMemory,
    pub(super) output_previous_state: Option<[i16; MINI_TRANSFORMER_HOLO_META_DIM]>,
    pub(super) mlp_previous_state: Option<[i16; MINI_TRANSFORMER_HOLO_META_DIM]>,
    pub(super) embedding_previous_state: Option<[i16; MINI_TRANSFORMER_HOLO_META_DIM]>,
    pub(super) q_previous_state: Option<[i16; MINI_TRANSFORMER_HOLO_META_DIM]>,
    pub(super) k_previous_state: Option<[i16; MINI_TRANSFORMER_HOLO_META_DIM]>,
    pub(super) v_previous_state: Option<[i16; MINI_TRANSFORMER_HOLO_META_DIM]>,
    pub(super) o_previous_state: Option<[i16; MINI_TRANSFORMER_HOLO_META_DIM]>,
    pub(super) output_holo_last_adjust_batch: Option<usize>,
    pub(super) mlp_holo_last_adjust_batch: Option<usize>,
    pub(super) embedding_holo_last_adjust_batch: Option<usize>,
    pub(super) q_holo_last_adjust_batch: Option<usize>,
    pub(super) k_holo_last_adjust_batch: Option<usize>,
    pub(super) vo_holo_last_adjust_batch: Option<usize>,
    pub(super) output_rule: MiniTransformerRuleShiftWindow,
    pub(super) mlp_rule: MiniTransformerRuleShiftWindow,
    pub(super) embedding_rule: MiniTransformerRuleShiftWindow,
    pub(super) q_rule: MiniTransformerRuleShiftWindow,
    pub(super) k_rule: MiniTransformerRuleShiftWindow,
    pub(super) v_rule: MiniTransformerRuleShiftWindow,
    pub(super) o_rule: MiniTransformerRuleShiftWindow,
    pub(super) output_learning_rate_shift: u8,
    pub(super) mlp_learning_rate_shift: u8,
    pub(super) embedding_learning_rate_shift: u8,
    pub(super) attention_learning_rate_shift: u8,
    pub(super) attention_q_learning_rate_shift: u8,
    pub(super) attention_qk_learning_rate_shift: u8,
    pub(super) adjustment_count: usize,
    pub(super) rule_adjustment_count: usize,
    pub(super) rule_update_count: usize,
    pub(super) rule_event_count: usize,
    pub(super) holographic_adjustment_count: usize,
}

impl MiniTransformerAdaptiveShiftState {
    pub(super) fn new(config: MiniTransformerMlpTrainConfig) -> Self {
        Self {
            output_memory: IntegerHolographicShiftMemory::new(),
            mlp_memory: IntegerHolographicShiftMemory::new(),
            embedding_memory: IntegerHolographicShiftMemory::new(),
            q_memory: IntegerHolographicShiftMemory::new(),
            k_memory: IntegerHolographicShiftMemory::new(),
            v_memory: IntegerHolographicShiftMemory::new(),
            o_memory: IntegerHolographicShiftMemory::new(),
            output_previous_state: None,
            mlp_previous_state: None,
            embedding_previous_state: None,
            q_previous_state: None,
            k_previous_state: None,
            v_previous_state: None,
            o_previous_state: None,
            output_holo_last_adjust_batch: None,
            mlp_holo_last_adjust_batch: None,
            embedding_holo_last_adjust_batch: None,
            q_holo_last_adjust_batch: None,
            k_holo_last_adjust_batch: None,
            vo_holo_last_adjust_batch: None,
            output_rule: MiniTransformerRuleShiftWindow::new(),
            mlp_rule: MiniTransformerRuleShiftWindow::new(),
            embedding_rule: MiniTransformerRuleShiftWindow::new(),
            q_rule: MiniTransformerRuleShiftWindow::new(),
            k_rule: MiniTransformerRuleShiftWindow::new(),
            v_rule: MiniTransformerRuleShiftWindow::new(),
            o_rule: MiniTransformerRuleShiftWindow::new(),
            output_learning_rate_shift: config.output_learning_rate_shift,
            mlp_learning_rate_shift: config.mlp_learning_rate_shift,
            embedding_learning_rate_shift: config.embedding_learning_rate_shift,
            attention_learning_rate_shift: config.attention_learning_rate_shift,
            attention_q_learning_rate_shift: config.attention_q_learning_rate_shift,
            attention_qk_learning_rate_shift: config.attention_qk_learning_rate_shift,
            adjustment_count: 0,
            rule_adjustment_count: 0,
            rule_update_count: 0,
            rule_event_count: 0,
            holographic_adjustment_count: 0,
        }
    }

    pub(super) fn runtime_config(
        &self,
        mut config: MiniTransformerMlpTrainConfig,
    ) -> MiniTransformerMlpTrainConfig {
        if config.adaptive_shift_controller_enabled() {
            config.output_learning_rate_shift = self.output_learning_rate_shift;
            config.mlp_learning_rate_shift = self.mlp_learning_rate_shift;
            config.embedding_learning_rate_shift = self.embedding_learning_rate_shift;
            config.attention_learning_rate_shift = self.attention_learning_rate_shift;
            config.attention_q_learning_rate_shift = self.attention_q_learning_rate_shift;
            config.attention_qk_learning_rate_shift = self.attention_qk_learning_rate_shift;
        }
        config
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn observe_accepted(
        &mut self,
        output: LinearWeightUpdateStats,
        mlp: GatedMlpWeightUpdateStats,
        embedding: SoftmaxUpdateStats,
        attention: &MiniTransformerAttentionWeightUpdateStats,
        accepted_batches: usize,
        enabled: bool,
        config: MiniTransformerMlpTrainConfig,
        adaptive_shift_events: &mut Vec<MiniTransformerAdaptiveShiftEventTrace>,
    ) {
        let rule_enabled = enabled && config.adaptive_rule_shift_controller_enabled();
        let holographic_enabled = enabled && config.adaptive_holographic_shift_controller_enabled();
        if !rule_enabled && !holographic_enabled {
            return;
        }

        let mlp_stats = mini_transformer_gated_mlp_update_stats_as_linear(mlp);
        let embedding_stats = mini_transformer_softmax_update_stats_as_linear(embedding);
        if rule_enabled {
            self.observe_rule_accepted(
                output,
                mlp_stats,
                embedding_stats,
                attention,
                accepted_batches,
                config,
                adaptive_shift_events,
            );
        }
        if !holographic_enabled {
            return;
        }

        let output_state = mini_transformer_holo_shift_state(
            &output,
            mini_transformer_peer_delta_l1(&[mlp_stats, embedding_stats, attention.o]),
            mini_transformer_output_weight_count(),
            false,
            accepted_batches,
        );
        let mlp_state = mini_transformer_holo_shift_state(
            &mlp_stats,
            output.weight_delta_l1,
            mini_transformer_mlp_weight_count(),
            false,
            accepted_batches,
        );
        let embedding_state = mini_transformer_holo_shift_state(
            &embedding_stats,
            output
                .weight_delta_l1
                .saturating_add(mlp_stats.weight_delta_l1),
            mini_transformer_embedding_weight_count(config),
            false,
            accepted_batches,
        );
        let q_state = mini_transformer_holo_shift_state(
            &attention.q,
            attention.k.weight_delta_l1,
            mini_transformer_attention_projection_weight_count(),
            false,
            accepted_batches,
        );
        let k_state = mini_transformer_holo_shift_state(
            &attention.k,
            attention.q.weight_delta_l1,
            mini_transformer_attention_projection_weight_count(),
            false,
            accepted_batches,
        );
        let v_state = mini_transformer_holo_shift_state(
            &attention.v,
            attention.o.weight_delta_l1,
            mini_transformer_attention_projection_weight_count(),
            false,
            accepted_batches,
        );
        let o_state = mini_transformer_holo_shift_state(
            &attention.o,
            attention.v.weight_delta_l1,
            mini_transformer_attention_projection_weight_count(),
            false,
            accepted_batches,
        );

        let output_teacher = mini_transformer_generic_shift_teacher_delta(
            &output,
            mini_transformer_output_weight_count(),
        );
        let mlp_teacher = mini_transformer_generic_shift_teacher_delta(
            &mlp_stats,
            mini_transformer_mlp_weight_count(),
        );
        let embedding_teacher = mini_transformer_generic_shift_teacher_delta(
            &embedding_stats,
            mini_transformer_embedding_weight_count(config),
        );
        let q_teacher = mini_transformer_attention_q_teacher_delta(attention);
        let k_teacher = mini_transformer_attention_k_teacher_delta(attention);
        let v_teacher = mini_transformer_generic_shift_teacher_delta(
            &attention.v,
            mini_transformer_attention_projection_weight_count(),
        );
        let o_teacher = mini_transformer_generic_shift_teacher_delta(
            &attention.o,
            mini_transformer_attention_projection_weight_count(),
        );

        mini_transformer_holo_remember_lagged(
            &mut self.output_memory,
            &mut self.output_previous_state,
            output_state,
            output_teacher,
        );
        mini_transformer_holo_remember_lagged(
            &mut self.mlp_memory,
            &mut self.mlp_previous_state,
            mlp_state,
            mlp_teacher,
        );
        mini_transformer_holo_remember_lagged(
            &mut self.embedding_memory,
            &mut self.embedding_previous_state,
            embedding_state,
            embedding_teacher,
        );
        mini_transformer_holo_remember_lagged(
            &mut self.q_memory,
            &mut self.q_previous_state,
            q_state,
            q_teacher,
        );
        mini_transformer_holo_remember_lagged(
            &mut self.k_memory,
            &mut self.k_previous_state,
            k_state,
            k_teacher,
        );
        mini_transformer_holo_remember_lagged(
            &mut self.v_memory,
            &mut self.v_previous_state,
            v_state,
            v_teacher,
        );
        mini_transformer_holo_remember_lagged(
            &mut self.o_memory,
            &mut self.o_previous_state,
            o_state,
            o_teacher,
        );

        let output_delta = mini_transformer_holo_authorized_delta(
            mini_transformer_holo_safety_delta(
                output_teacher,
                self.output_memory
                    .retrieve_delta(&output_state)
                    .unwrap_or(0),
                !rule_enabled,
            ),
            output_teacher,
            self.output_memory.update_count,
            accepted_batches,
            &mut self.output_holo_last_adjust_batch,
        );
        let mlp_delta = mini_transformer_holo_authorized_delta(
            mini_transformer_holo_safety_delta(
                mlp_teacher,
                self.mlp_memory.retrieve_delta(&mlp_state).unwrap_or(0),
                !rule_enabled,
            ),
            mlp_teacher,
            self.mlp_memory.update_count,
            accepted_batches,
            &mut self.mlp_holo_last_adjust_batch,
        );
        let embedding_delta = mini_transformer_holo_authorized_delta(
            mini_transformer_holo_safety_delta(
                embedding_teacher,
                self.embedding_memory
                    .retrieve_delta(&embedding_state)
                    .unwrap_or(0),
                !rule_enabled,
            ),
            embedding_teacher,
            self.embedding_memory.update_count,
            accepted_batches,
            &mut self.embedding_holo_last_adjust_batch,
        );
        let q_delta = mini_transformer_holo_authorized_delta(
            mini_transformer_holo_safety_delta(
                q_teacher,
                self.q_memory.retrieve_delta(&q_state).unwrap_or(0),
                !rule_enabled,
            ),
            q_teacher,
            self.q_memory.update_count,
            accepted_batches,
            &mut self.q_holo_last_adjust_batch,
        );
        let k_delta = mini_transformer_holo_authorized_delta(
            mini_transformer_holo_safety_delta(
                k_teacher,
                self.k_memory.retrieve_delta(&k_state).unwrap_or(0),
                !rule_enabled,
            ),
            k_teacher,
            self.k_memory.update_count,
            accepted_batches,
            &mut self.k_holo_last_adjust_batch,
        );
        let v_delta = mini_transformer_holo_safety_delta(
            v_teacher,
            self.v_memory.retrieve_delta(&v_state).unwrap_or(0),
            !rule_enabled,
        );
        let o_delta = mini_transformer_holo_safety_delta(
            o_teacher,
            self.o_memory.retrieve_delta(&o_state).unwrap_or(0),
            !rule_enabled,
        );
        let vo_teacher = mini_transformer_join_shift_deltas(v_teacher, o_teacher);
        let vo_delta = mini_transformer_holo_authorized_delta(
            mini_transformer_join_shift_deltas(v_delta, o_delta),
            vo_teacher,
            self.v_memory.update_count.min(self.o_memory.update_count),
            accepted_batches,
            &mut self.vo_holo_last_adjust_batch,
        );

        let adjustment_count_before = self.adjustment_count;
        self.adjust_output(output_delta);
        self.adjust_mlp(mlp_delta);
        self.adjust_embedding(embedding_delta);
        self.adjust_q(q_delta);
        self.adjust_k(k_delta);
        self.adjust_vo(vo_delta);
        self.holographic_adjustment_count = self.holographic_adjustment_count.saturating_add(
            self.adjustment_count
                .saturating_sub(adjustment_count_before),
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn observe_rule_accepted(
        &mut self,
        output: LinearWeightUpdateStats,
        mlp: LinearWeightUpdateStats,
        embedding: LinearWeightUpdateStats,
        attention: &MiniTransformerAttentionWeightUpdateStats,
        accepted_batches: usize,
        config: MiniTransformerMlpTrainConfig,
        adaptive_shift_events: &mut Vec<MiniTransformerAdaptiveShiftEventTrace>,
    ) {
        self.output_rule.observe_accepted(output);
        self.mlp_rule.observe_accepted(mlp);
        self.embedding_rule.observe_accepted(embedding);
        self.q_rule.observe_accepted(attention.q);
        self.k_rule.observe_accepted(attention.k);
        self.v_rule.observe_accepted(attention.v);
        self.o_rule.observe_accepted(attention.o);
        self.apply_rule_controls(accepted_batches, config, adaptive_shift_events);
    }

    pub(super) fn observe_rule_rejected(
        &mut self,
        rejected_batches: usize,
        config: MiniTransformerMlpTrainConfig,
        adaptive_shift_events: &mut Vec<MiniTransformerAdaptiveShiftEventTrace>,
    ) {
        self.output_rule.observe_rejected();
        self.mlp_rule.observe_rejected();
        self.embedding_rule.observe_rejected();
        self.q_rule.observe_rejected();
        self.k_rule.observe_rejected();
        self.v_rule.observe_rejected();
        self.o_rule.observe_rejected();
        self.apply_rule_controls(rejected_batches, config, adaptive_shift_events);
    }

    pub(super) fn apply_rule_controls(
        &mut self,
        batch_index: usize,
        config: MiniTransformerMlpTrainConfig,
        adaptive_shift_events: &mut Vec<MiniTransformerAdaptiveShiftEventTrace>,
    ) {
        let interval = config.adaptive_rule_interval_batches.max(1);
        self.rule_update_count = self.rule_update_count.saturating_add(1);

        let output_rule = self.output_rule;
        self.apply_rule_output(output_rule, batch_index, interval, adaptive_shift_events);
        if mini_transformer_rule_should_reset(output_rule, interval) {
            self.output_rule.reset();
        }

        let mlp_rule = self.mlp_rule;
        self.apply_rule_mlp(mlp_rule, batch_index, interval, adaptive_shift_events);
        if mini_transformer_rule_should_reset(mlp_rule, interval) {
            self.mlp_rule.reset();
        }

        let embedding_rule = self.embedding_rule;
        self.apply_rule_embedding(
            embedding_rule,
            batch_index,
            interval,
            mini_transformer_embedding_weight_count(config),
            adaptive_shift_events,
        );
        if mini_transformer_rule_should_reset(embedding_rule, interval) {
            self.embedding_rule.reset();
        }

        let q_rule = self.q_rule;
        let k_rule = self.k_rule;
        self.apply_rule_q(q_rule, k_rule, batch_index, interval, adaptive_shift_events);
        if mini_transformer_rule_should_reset(q_rule, interval) {
            self.q_rule.reset();
        }

        self.apply_rule_k(k_rule, q_rule, batch_index, interval, adaptive_shift_events);
        if mini_transformer_rule_should_reset(k_rule, interval) {
            self.k_rule.reset();
        }

        let v_rule = self.v_rule;
        let o_rule = self.o_rule;
        self.apply_rule_vo(v_rule, o_rule, batch_index, interval, adaptive_shift_events);
        if mini_transformer_rule_should_reset(v_rule, interval) {
            self.v_rule.reset();
        }
        if mini_transformer_rule_should_reset(o_rule, interval) {
            self.o_rule.reset();
        }
    }

    pub(super) fn apply_rule_output(
        &mut self,
        window: MiniTransformerRuleShiftWindow,
        batch_index: usize,
        interval: usize,
        adaptive_shift_events: &mut Vec<MiniTransformerAdaptiveShiftEventTrace>,
    ) {
        if let Some((delta, reason)) = mini_transformer_rule_generic_delta(
            window,
            mini_transformer_output_weight_count(),
            interval,
        ) {
            let previous = self.output_learning_rate_shift;
            let next = mini_transformer_adjust_shift(previous, delta);
            self.record_rule_event(
                adaptive_shift_events,
                mini_transformer_rule_event(
                    batch_index,
                    "output",
                    reason,
                    previous,
                    next,
                    delta,
                    window,
                ),
            );
            self.output_learning_rate_shift = next;
        }
    }

    pub(super) fn apply_rule_mlp(
        &mut self,
        window: MiniTransformerRuleShiftWindow,
        batch_index: usize,
        interval: usize,
        adaptive_shift_events: &mut Vec<MiniTransformerAdaptiveShiftEventTrace>,
    ) {
        if let Some((delta, reason)) = mini_transformer_rule_generic_delta(
            window,
            mini_transformer_mlp_weight_count(),
            interval,
        ) {
            let previous = self.mlp_learning_rate_shift;
            let next = mini_transformer_adjust_shift(previous, delta);
            self.record_rule_event(
                adaptive_shift_events,
                mini_transformer_rule_event(
                    batch_index,
                    "mlp",
                    reason,
                    previous,
                    next,
                    delta,
                    window,
                ),
            );
            self.mlp_learning_rate_shift = next;
        }
    }

    pub(super) fn apply_rule_embedding(
        &mut self,
        window: MiniTransformerRuleShiftWindow,
        batch_index: usize,
        interval: usize,
        weight_count: usize,
        adaptive_shift_events: &mut Vec<MiniTransformerAdaptiveShiftEventTrace>,
    ) {
        if let Some((delta, reason)) =
            mini_transformer_rule_generic_delta(window, weight_count, interval)
        {
            let previous = self.embedding_learning_rate_shift;
            let next = mini_transformer_adjust_shift(previous, delta);
            self.record_rule_event(
                adaptive_shift_events,
                mini_transformer_rule_event(
                    batch_index,
                    "embedding",
                    reason,
                    previous,
                    next,
                    delta,
                    window,
                ),
            );
            self.embedding_learning_rate_shift = next;
        }
    }

    pub(super) fn apply_rule_q(
        &mut self,
        q_window: MiniTransformerRuleShiftWindow,
        k_window: MiniTransformerRuleShiftWindow,
        batch_index: usize,
        interval: usize,
        adaptive_shift_events: &mut Vec<MiniTransformerAdaptiveShiftEventTrace>,
    ) {
        if let Some((delta, reason)) = mini_transformer_rule_q_delta(q_window, k_window, interval) {
            let previous = self.attention_q_learning_rate_shift;
            let next = mini_transformer_adjust_shift(previous, delta);
            self.record_rule_event(
                adaptive_shift_events,
                mini_transformer_rule_event(
                    batch_index,
                    "attention_q",
                    reason,
                    previous,
                    next,
                    delta,
                    q_window,
                ),
            );
            self.attention_q_learning_rate_shift = next;
        }
    }

    pub(super) fn apply_rule_k(
        &mut self,
        k_window: MiniTransformerRuleShiftWindow,
        q_window: MiniTransformerRuleShiftWindow,
        batch_index: usize,
        interval: usize,
        adaptive_shift_events: &mut Vec<MiniTransformerAdaptiveShiftEventTrace>,
    ) {
        if let Some((delta, reason)) = mini_transformer_rule_k_delta(k_window, q_window, interval) {
            let previous = self.attention_qk_learning_rate_shift;
            let next = mini_transformer_adjust_shift(previous, delta);
            self.record_rule_event(
                adaptive_shift_events,
                mini_transformer_rule_event(
                    batch_index,
                    "attention_k",
                    reason,
                    previous,
                    next,
                    delta,
                    k_window,
                ),
            );
            self.attention_qk_learning_rate_shift = next;
        }
    }

    pub(super) fn apply_rule_vo(
        &mut self,
        v_window: MiniTransformerRuleShiftWindow,
        o_window: MiniTransformerRuleShiftWindow,
        batch_index: usize,
        interval: usize,
        adaptive_shift_events: &mut Vec<MiniTransformerAdaptiveShiftEventTrace>,
    ) {
        let v_decision = mini_transformer_rule_generic_delta(
            v_window,
            mini_transformer_attention_projection_weight_count(),
            interval,
        );
        let o_decision = mini_transformer_rule_generic_delta(
            o_window,
            mini_transformer_attention_projection_weight_count(),
            interval,
        );
        let Some((delta, reason, window)) =
            mini_transformer_rule_join_vo_decisions(v_decision, o_decision, v_window, o_window)
        else {
            return;
        };
        let previous = self.attention_learning_rate_shift;
        let next = mini_transformer_adjust_shift(previous, delta);
        self.record_rule_event(
            adaptive_shift_events,
            mini_transformer_rule_event(
                batch_index,
                "attention_vo",
                reason,
                previous,
                next,
                delta,
                window,
            ),
        );
        self.attention_learning_rate_shift = next;
    }

    pub(super) fn record_rule_event(
        &mut self,
        adaptive_shift_events: &mut Vec<MiniTransformerAdaptiveShiftEventTrace>,
        event: MiniTransformerAdaptiveShiftEventTrace,
    ) {
        if event.previous_shift == event.next_shift {
            return;
        }
        self.rule_adjustment_count = self.rule_adjustment_count.saturating_add(1);
        self.adjustment_count = self.adjustment_count.saturating_add(1);
        self.rule_event_count = self.rule_event_count.saturating_add(1);
        if adaptive_shift_events.len() < MINI_TRANSFORMER_ADAPTIVE_RULE_TRACE_EVENT_LIMIT {
            adaptive_shift_events.push(event);
        }
    }

    pub(super) fn observe_rejected(
        &mut self,
        rejected_batches: usize,
        enabled: bool,
        config: MiniTransformerMlpTrainConfig,
        adaptive_shift_events: &mut Vec<MiniTransformerAdaptiveShiftEventTrace>,
    ) {
        let rule_enabled = enabled && config.adaptive_rule_shift_controller_enabled();
        let holographic_enabled = enabled && config.adaptive_holographic_shift_controller_enabled();
        if !rule_enabled && !holographic_enabled {
            return;
        }
        if rule_enabled {
            self.observe_rule_rejected(rejected_batches, config, adaptive_shift_events);
        }
        if !holographic_enabled {
            return;
        }
        let output_rejected =
            mini_transformer_rejected_shift_stats(mini_transformer_output_weight_count());
        let mlp_rejected =
            mini_transformer_rejected_shift_stats(mini_transformer_mlp_weight_count());
        let embedding_rejected =
            mini_transformer_rejected_shift_stats(mini_transformer_embedding_weight_count(config));
        let attention_rejected = mini_transformer_rejected_shift_stats(
            mini_transformer_attention_projection_weight_count(),
        );
        let output_state = mini_transformer_holo_shift_state(
            &output_rejected,
            0,
            mini_transformer_output_weight_count(),
            true,
            rejected_batches,
        );
        let mlp_state = mini_transformer_holo_shift_state(
            &mlp_rejected,
            0,
            mini_transformer_mlp_weight_count(),
            true,
            rejected_batches,
        );
        let embedding_state = mini_transformer_holo_shift_state(
            &embedding_rejected,
            0,
            mini_transformer_embedding_weight_count(config),
            true,
            rejected_batches,
        );
        let attention_state = mini_transformer_holo_shift_state(
            &attention_rejected,
            0,
            mini_transformer_attention_projection_weight_count(),
            true,
            rejected_batches,
        );

        mini_transformer_holo_remember_lagged(
            &mut self.output_memory,
            &mut self.output_previous_state,
            output_state,
            1,
        );
        mini_transformer_holo_remember_lagged(
            &mut self.mlp_memory,
            &mut self.mlp_previous_state,
            mlp_state,
            1,
        );
        mini_transformer_holo_remember_lagged(
            &mut self.embedding_memory,
            &mut self.embedding_previous_state,
            embedding_state,
            1,
        );
        mini_transformer_holo_remember_lagged(
            &mut self.q_memory,
            &mut self.q_previous_state,
            attention_state,
            1,
        );
        mini_transformer_holo_remember_lagged(
            &mut self.k_memory,
            &mut self.k_previous_state,
            attention_state,
            1,
        );
        mini_transformer_holo_remember_lagged(
            &mut self.v_memory,
            &mut self.v_previous_state,
            attention_state,
            1,
        );
        mini_transformer_holo_remember_lagged(
            &mut self.o_memory,
            &mut self.o_previous_state,
            attention_state,
            1,
        );
        let adjustment_count_before = self.adjustment_count;
        self.adjust_output(1);
        self.adjust_mlp(1);
        self.adjust_embedding(1);
        self.adjust_q(1);
        self.adjust_k(1);
        self.adjust_vo(1);
        self.holographic_adjustment_count = self.holographic_adjustment_count.saturating_add(
            self.adjustment_count
                .saturating_sub(adjustment_count_before),
        );
    }

    pub(super) fn total_memory_updates(&self) -> usize {
        self.output_memory
            .update_count
            .saturating_add(self.mlp_memory.update_count)
            .saturating_add(self.embedding_memory.update_count)
            .saturating_add(self.q_memory.update_count)
            .saturating_add(self.k_memory.update_count)
            .saturating_add(self.v_memory.update_count)
            .saturating_add(self.o_memory.update_count)
    }

    pub(super) fn attention_memory_updates(&self) -> usize {
        self.q_memory
            .update_count
            .saturating_add(self.k_memory.update_count)
            .saturating_add(self.v_memory.update_count)
            .saturating_add(self.o_memory.update_count)
    }

    pub(super) fn memory_hash(&self) -> u64 {
        let mut hasher = StableHasher::new();
        hasher.update_usize(usize::from(self.output_learning_rate_shift));
        hasher.update_usize(usize::from(self.mlp_learning_rate_shift));
        hasher.update_usize(usize::from(self.embedding_learning_rate_shift));
        hasher.update_usize(usize::from(self.attention_learning_rate_shift));
        hasher.update_usize(usize::from(self.attention_q_learning_rate_shift));
        hasher.update_usize(usize::from(self.attention_qk_learning_rate_shift));
        self.output_memory.hash_into(&mut hasher);
        self.mlp_memory.hash_into(&mut hasher);
        self.embedding_memory.hash_into(&mut hasher);
        self.q_memory.hash_into(&mut hasher);
        self.k_memory.hash_into(&mut hasher);
        self.v_memory.hash_into(&mut hasher);
        self.o_memory.hash_into(&mut hasher);
        mini_transformer_hash_holo_previous_state(&mut hasher, self.output_previous_state);
        mini_transformer_hash_holo_previous_state(&mut hasher, self.mlp_previous_state);
        mini_transformer_hash_holo_previous_state(&mut hasher, self.embedding_previous_state);
        mini_transformer_hash_holo_previous_state(&mut hasher, self.q_previous_state);
        mini_transformer_hash_holo_previous_state(&mut hasher, self.k_previous_state);
        mini_transformer_hash_holo_previous_state(&mut hasher, self.v_previous_state);
        mini_transformer_hash_holo_previous_state(&mut hasher, self.o_previous_state);
        mini_transformer_hash_optional_usize(&mut hasher, self.output_holo_last_adjust_batch);
        mini_transformer_hash_optional_usize(&mut hasher, self.mlp_holo_last_adjust_batch);
        mini_transformer_hash_optional_usize(&mut hasher, self.embedding_holo_last_adjust_batch);
        mini_transformer_hash_optional_usize(&mut hasher, self.q_holo_last_adjust_batch);
        mini_transformer_hash_optional_usize(&mut hasher, self.k_holo_last_adjust_batch);
        mini_transformer_hash_optional_usize(&mut hasher, self.vo_holo_last_adjust_batch);
        hasher.finish()
    }

    pub(super) fn attention_memory_hash(&self) -> u64 {
        let mut hasher = StableHasher::new();
        hasher.update_usize(usize::from(self.attention_learning_rate_shift));
        hasher.update_usize(usize::from(self.attention_q_learning_rate_shift));
        hasher.update_usize(usize::from(self.attention_qk_learning_rate_shift));
        self.q_memory.hash_into(&mut hasher);
        self.k_memory.hash_into(&mut hasher);
        self.v_memory.hash_into(&mut hasher);
        self.o_memory.hash_into(&mut hasher);
        mini_transformer_hash_holo_previous_state(&mut hasher, self.q_previous_state);
        mini_transformer_hash_holo_previous_state(&mut hasher, self.k_previous_state);
        mini_transformer_hash_holo_previous_state(&mut hasher, self.v_previous_state);
        mini_transformer_hash_holo_previous_state(&mut hasher, self.o_previous_state);
        mini_transformer_hash_optional_usize(&mut hasher, self.q_holo_last_adjust_batch);
        mini_transformer_hash_optional_usize(&mut hasher, self.k_holo_last_adjust_batch);
        mini_transformer_hash_optional_usize(&mut hasher, self.vo_holo_last_adjust_batch);
        hasher.finish()
    }

    pub(super) fn adjust_output(&mut self, delta: i8) {
        let next = mini_transformer_adjust_shift(self.output_learning_rate_shift, delta);
        if next != self.output_learning_rate_shift {
            self.output_learning_rate_shift = next;
            self.adjustment_count = self.adjustment_count.saturating_add(1);
        }
    }

    pub(super) fn adjust_mlp(&mut self, delta: i8) {
        let next = mini_transformer_adjust_shift(self.mlp_learning_rate_shift, delta);
        if next != self.mlp_learning_rate_shift {
            self.mlp_learning_rate_shift = next;
            self.adjustment_count = self.adjustment_count.saturating_add(1);
        }
    }

    pub(super) fn adjust_embedding(&mut self, delta: i8) {
        let next = mini_transformer_adjust_shift(self.embedding_learning_rate_shift, delta);
        if next != self.embedding_learning_rate_shift {
            self.embedding_learning_rate_shift = next;
            self.adjustment_count = self.adjustment_count.saturating_add(1);
        }
    }

    pub(super) fn adjust_q(&mut self, delta: i8) {
        let next = mini_transformer_adjust_shift(self.attention_q_learning_rate_shift, delta);
        if next != self.attention_q_learning_rate_shift {
            self.attention_q_learning_rate_shift = next;
            self.adjustment_count = self.adjustment_count.saturating_add(1);
        }
    }

    pub(super) fn adjust_k(&mut self, delta: i8) {
        let next = mini_transformer_adjust_shift(self.attention_qk_learning_rate_shift, delta);
        if next != self.attention_qk_learning_rate_shift {
            self.attention_qk_learning_rate_shift = next;
            self.adjustment_count = self.adjustment_count.saturating_add(1);
        }
    }

    pub(super) fn adjust_vo(&mut self, delta: i8) {
        let next = mini_transformer_adjust_shift(self.attention_learning_rate_shift, delta);
        if next != self.attention_learning_rate_shift {
            self.attention_learning_rate_shift = next;
            self.adjustment_count = self.adjustment_count.saturating_add(1);
        }
    }
}

pub(super) fn mini_transformer_holo_action_atom(
    delta: i8,
) -> &'static [i16; MINI_TRANSFORMER_HOLO_META_DIM] {
    match delta.clamp(-2, 2) {
        -2 => &MINI_TRANSFORMER_HOLO_ACTION_ATOMS[0],
        -1 => &MINI_TRANSFORMER_HOLO_ACTION_ATOMS[1],
        0 => &MINI_TRANSFORMER_HOLO_ACTION_ATOMS[2],
        1 => &MINI_TRANSFORMER_HOLO_ACTION_ATOMS[3],
        _ => &MINI_TRANSFORMER_HOLO_ACTION_ATOMS[4],
    }
}

pub(super) fn mini_transformer_adjust_shift(current: u8, delta: i8) -> u8 {
    let next = i16::from(current) + i16::from(delta);
    next.clamp(0, i16::from(MAX_RIGHT_SHIFT)) as u8
}

pub(super) fn mini_transformer_rule_generic_delta(
    window: MiniTransformerRuleShiftWindow,
    weight_count: usize,
    interval: usize,
) -> Option<(i8, &'static str)> {
    if window.rejected_batches > 0 {
        return Some((1, "rollback"));
    }
    if window.observation_batches < interval.max(1) {
        return None;
    }
    if mini_transformer_rule_saturation_pressure(window, weight_count) {
        return Some((1, "saturation"));
    }
    if mini_transformer_rule_zero_pressure(window, weight_count) {
        return Some((-1, "zero_delta"));
    }
    None
}

pub(super) fn mini_transformer_rule_q_delta(
    q_window: MiniTransformerRuleShiftWindow,
    k_window: MiniTransformerRuleShiftWindow,
    interval: usize,
) -> Option<(i8, &'static str)> {
    if q_window.rejected_batches > 0 {
        return Some((1, "rollback"));
    }
    if q_window.observation_batches < interval.max(1) {
        return None;
    }
    if q_window.stats.weight_delta_l1 == 0
        || mini_transformer_rule_zero_pressure(
            q_window,
            mini_transformer_attention_projection_weight_count(),
        )
    {
        return Some((-1, "zero_delta"));
    }
    if k_window.stats.weight_delta_l1 > 0
        && q_window.stats.weight_delta_l1.saturating_mul(8) < k_window.stats.weight_delta_l1
    {
        return Some((-1, "lagging_k"));
    }
    if mini_transformer_rule_saturation_pressure(
        q_window,
        mini_transformer_attention_projection_weight_count(),
    ) {
        return Some((1, "saturation"));
    }
    if k_window.stats.weight_delta_l1 > 0
        && q_window.stats.weight_delta_l1 > k_window.stats.weight_delta_l1.saturating_mul(4)
    {
        return Some((1, "overpowering_k"));
    }
    None
}

pub(super) fn mini_transformer_rule_k_delta(
    k_window: MiniTransformerRuleShiftWindow,
    q_window: MiniTransformerRuleShiftWindow,
    interval: usize,
) -> Option<(i8, &'static str)> {
    if k_window.rejected_batches > 0 {
        return Some((1, "rollback"));
    }
    if k_window.observation_batches < interval.max(1) {
        return None;
    }
    if mini_transformer_rule_zero_pressure(
        k_window,
        mini_transformer_attention_projection_weight_count(),
    ) {
        return Some((-1, "zero_delta"));
    }
    if mini_transformer_rule_saturation_pressure(
        k_window,
        mini_transformer_attention_projection_weight_count(),
    ) {
        return Some((1, "saturation"));
    }
    if q_window.stats.weight_delta_l1 > 0
        && k_window.stats.weight_delta_l1 > q_window.stats.weight_delta_l1.saturating_mul(64)
    {
        return Some((1, "overpowering_q"));
    }
    None
}

pub(super) fn mini_transformer_rule_saturation_pressure(
    window: MiniTransformerRuleShiftWindow,
    weight_count: usize,
) -> bool {
    if window.stats.gradient_saturation_count == 0 {
        return false;
    }
    let total_slots = weight_count
        .max(1)
        .saturating_mul(window.observation_batches.max(1));
    let threshold = (total_slots / MINI_TRANSFORMER_RULE_SATURATION_PRESSURE_DIVISOR)
        .max(window.observation_batches.max(1));
    window.stats.gradient_saturation_count >= threshold
}

pub(super) fn mini_transformer_rule_join_vo_decisions(
    v_decision: Option<(i8, &'static str)>,
    o_decision: Option<(i8, &'static str)>,
    v_window: MiniTransformerRuleShiftWindow,
    o_window: MiniTransformerRuleShiftWindow,
) -> Option<(i8, &'static str, MiniTransformerRuleShiftWindow)> {
    let (delta, reason) = match (v_decision, o_decision) {
        (Some((v_delta, v_reason)), Some((o_delta, o_reason))) => {
            let delta = mini_transformer_join_shift_deltas(v_delta, o_delta);
            let reason = if v_reason == o_reason || delta == v_delta {
                v_reason
            } else {
                o_reason
            };
            (delta, reason)
        }
        (Some(decision), None) | (None, Some(decision)) => decision,
        (None, None) => return None,
    };
    Some((
        delta,
        reason,
        mini_transformer_rule_join_windows(v_window, o_window),
    ))
}

pub(super) fn mini_transformer_rule_join_windows(
    left: MiniTransformerRuleShiftWindow,
    right: MiniTransformerRuleShiftWindow,
) -> MiniTransformerRuleShiftWindow {
    MiniTransformerRuleShiftWindow {
        observation_batches: left.observation_batches.max(right.observation_batches),
        rejected_batches: left.rejected_batches.saturating_add(right.rejected_batches),
        stats: LinearWeightUpdateStats {
            gradient_saturation_count: left
                .stats
                .gradient_saturation_count
                .saturating_add(right.stats.gradient_saturation_count),
            zero_delta_count: left
                .stats
                .zero_delta_count
                .saturating_add(right.stats.zero_delta_count),
            weight_delta_l1: left
                .stats
                .weight_delta_l1
                .saturating_add(right.stats.weight_delta_l1),
        },
    }
}

pub(super) fn mini_transformer_rule_zero_pressure(
    window: MiniTransformerRuleShiftWindow,
    weight_count: usize,
) -> bool {
    if window.stats.weight_delta_l1 == 0 {
        return true;
    }
    let total_slots = weight_count
        .max(1)
        .saturating_mul(window.observation_batches.max(1));
    let zero_pressure_threshold =
        ((total_slots as u128) * (MINI_TRANSFORMER_RULE_ZERO_PRESSURE_NUMERATOR as u128)
            / (MINI_TRANSFORMER_RULE_ZERO_PRESSURE_DENOMINATOR as u128)) as usize;
    window.stats.zero_delta_count > zero_pressure_threshold
}

pub(super) fn mini_transformer_rule_should_reset(
    window: MiniTransformerRuleShiftWindow,
    interval: usize,
) -> bool {
    window.rejected_batches > 0 || window.observation_batches >= interval.max(1)
}

pub(super) fn mini_transformer_rule_event(
    batch_index: usize,
    component: &'static str,
    reason: &'static str,
    previous_shift: u8,
    next_shift: u8,
    delta: i8,
    window: MiniTransformerRuleShiftWindow,
) -> MiniTransformerAdaptiveShiftEventTrace {
    MiniTransformerAdaptiveShiftEventTrace {
        batch_index,
        component,
        reason,
        previous_shift,
        next_shift,
        delta,
        observation_batches: window.observation_batches,
        rejected_batches: window.rejected_batches,
        saturation_count: window.stats.gradient_saturation_count,
        zero_delta_count: window.stats.zero_delta_count,
        weight_delta_l1: window.stats.weight_delta_l1,
    }
}

pub(super) fn mini_transformer_output_weight_count() -> usize {
    BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL
}

pub(super) fn mini_transformer_mlp_weight_count() -> usize {
    MINI_TRANSFORMER_D_MODEL
        .saturating_mul(MINI_TRANSFORMER_HIDDEN_DIM)
        .saturating_mul(2)
        .saturating_add(MINI_TRANSFORMER_HIDDEN_DIM.saturating_mul(MINI_TRANSFORMER_D_MODEL))
}

pub(super) fn mini_transformer_attention_projection_weight_count() -> usize {
    MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_D_MODEL
}

pub(super) fn mini_transformer_embedding_weight_count(
    config: MiniTransformerMlpTrainConfig,
) -> usize {
    let token_embeddings = BYTE_VOCAB.saturating_mul(MINI_TRANSFORMER_D_MODEL);
    if config.position_policy.uses_position_embeddings() {
        token_embeddings.saturating_add(config.seq_len.saturating_mul(MINI_TRANSFORMER_D_MODEL))
    } else {
        token_embeddings
    }
}

pub(super) fn mini_transformer_peer_delta_l1(stats: &[LinearWeightUpdateStats]) -> u64 {
    stats.iter().fold(0_u64, |total, stats| {
        total.saturating_add(stats.weight_delta_l1)
    })
}

pub(super) fn mini_transformer_softmax_update_stats_as_linear(
    stats: SoftmaxUpdateStats,
) -> LinearWeightUpdateStats {
    LinearWeightUpdateStats {
        gradient_saturation_count: stats.gradient_saturation_count,
        zero_delta_count: stats.zero_delta_count,
        weight_delta_l1: stats.weight_delta_l1,
    }
}

pub(super) fn mini_transformer_gated_mlp_update_stats_as_linear(
    stats: GatedMlpWeightUpdateStats,
) -> LinearWeightUpdateStats {
    LinearWeightUpdateStats {
        gradient_saturation_count: stats.gradient_saturation_count().unwrap_or(usize::MAX),
        zero_delta_count: stats.zero_delta_count().unwrap_or(usize::MAX),
        weight_delta_l1: stats.weight_delta_l1().unwrap_or(u64::MAX),
    }
}

pub(super) fn mini_transformer_generic_shift_teacher_delta(
    stats: &LinearWeightUpdateStats,
    weight_count: usize,
) -> i8 {
    if stats.gradient_saturation_count > 0 {
        return 1;
    }
    if stats.weight_delta_l1 == 0 || stats.zero_delta_count > weight_count.max(1) / 2 {
        return -1;
    }
    0
}

pub(super) fn mini_transformer_holo_remember_lagged(
    memory: &mut IntegerHolographicShiftMemory,
    previous_state: &mut Option<[i16; MINI_TRANSFORMER_HOLO_META_DIM]>,
    current_state: [i16; MINI_TRANSFORMER_HOLO_META_DIM],
    teacher: i8,
) {
    if let Some(state) = previous_state {
        memory.remember(state, teacher);
    }
    *previous_state = Some(current_state);
}

pub(super) fn mini_transformer_hash_holo_previous_state(
    hasher: &mut StableHasher,
    previous_state: Option<[i16; MINI_TRANSFORMER_HOLO_META_DIM]>,
) {
    match previous_state {
        Some(state) => {
            hasher.update_usize(1);
            for value in state {
                hasher.update_bytes(&value.to_le_bytes());
            }
        }
        None => hasher.update_usize(0),
    }
}

pub(super) fn mini_transformer_hash_optional_usize(
    hasher: &mut StableHasher,
    value: Option<usize>,
) {
    match value {
        Some(value) => {
            hasher.update_usize(1);
            hasher.update_usize(value);
        }
        None => hasher.update_usize(0),
    }
}

pub(super) fn mini_transformer_rejected_shift_stats(
    weight_count: usize,
) -> LinearWeightUpdateStats {
    LinearWeightUpdateStats {
        gradient_saturation_count: 1,
        zero_delta_count: weight_count,
        weight_delta_l1: 0,
    }
}

pub(super) fn mini_transformer_join_shift_deltas(left: i8, right: i8) -> i8 {
    if left > 0 || right > 0 {
        left.max(right)
    } else {
        left.min(right)
    }
}

pub(super) fn mini_transformer_holo_safety_delta(
    teacher: i8,
    recalled: i8,
    teacher_can_act: bool,
) -> i8 {
    if teacher_can_act && teacher != 0 {
        teacher
    } else if teacher == 0 {
        recalled.clamp(-1, 1)
    } else {
        0
    }
}

pub(super) fn mini_transformer_holo_authorized_delta(
    candidate: i8,
    teacher: i8,
    memory_update_count: usize,
    batch_index: usize,
    last_adjust_batch: &mut Option<usize>,
) -> i8 {
    if candidate == 0 {
        return 0;
    }
    if teacher == 0 && memory_update_count < MINI_TRANSFORMER_HOLO_MEMORY_MIN_UPDATES {
        return 0;
    }
    if let Some(last_batch) = *last_adjust_batch
        && batch_index.saturating_sub(last_batch)
            < MINI_TRANSFORMER_HOLO_ADJUSTMENT_COOLDOWN_BATCHES
    {
        return 0;
    }
    *last_adjust_batch = Some(batch_index);
    candidate
}

pub(super) fn mini_transformer_holo_shift_state(
    stats: &LinearWeightUpdateStats,
    peer_delta_l1: u64,
    weight_count: usize,
    rejected: bool,
    phase: usize,
) -> [i16; MINI_TRANSFORMER_HOLO_META_DIM] {
    let movement = mini_transformer_log_u64_q15(stats.weight_delta_l1);
    let peer_movement = mini_transformer_log_u64_q15(peer_delta_l1);
    let zero_ratio = mini_transformer_ratio_q15(stats.zero_delta_count, weight_count);
    let saturation_ratio =
        mini_transformer_ratio_q15(stats.gradient_saturation_count, weight_count.max(1));
    let phase_q15 = mini_transformer_ratio_q15(phase.min(1024), 1024);
    let rejected_q15 = if rejected { i16::MAX } else { 0 };
    let signed_pressure = if stats.gradient_saturation_count > 0 || rejected {
        i16::MAX
    } else if stats.weight_delta_l1 == 0 || stats.zero_delta_count > weight_count / 2 {
        -i16::MAX
    } else {
        0
    };

    [
        i16::MAX,
        movement,
        peer_movement,
        zero_ratio,
        saturation_ratio,
        rejected_q15,
        phase_q15,
        signed_pressure,
    ]
}

pub(super) fn mini_transformer_log_u64_q15(value: u64) -> i16 {
    if value == 0 {
        return 0;
    }
    let bits = 64_u32.saturating_sub(value.leading_zeros());
    let scaled = bits.saturating_mul(512).min(i16::MAX as u32);
    scaled as i16
}

pub(super) fn mini_transformer_ratio_q15(numerator: usize, denominator: usize) -> i16 {
    if denominator == 0 {
        return 0;
    }
    let wide = (numerator as u128).saturating_mul(i16::MAX as u128) / (denominator as u128);
    wide.min(i16::MAX as u128) as i16
}

pub(super) const MINI_TRANSFORMER_TRACE_SUMMARY_INITIAL_STEPS: usize = 16;
pub(super) const MINI_TRANSFORMER_TRACE_SUMMARY_DEFAULT_INTERVAL_STEPS: usize = 1024;

pub(super) fn mini_transformer_trace_sample_interval(
    progress_interval_batches: usize,
    batch_windows: usize,
) -> usize {
    let progress_windows = progress_interval_batches.saturating_mul(batch_windows);
    if progress_windows == 0 {
        MINI_TRANSFORMER_TRACE_SUMMARY_DEFAULT_INTERVAL_STEPS
    } else {
        progress_windows
    }
}

pub(super) fn mini_transformer_should_record_step(
    trace_detail: MiniTransformerTraceDetail,
    update_index: usize,
    sample_interval: usize,
) -> bool {
    match trace_detail {
        MiniTransformerTraceDetail::Full => true,
        MiniTransformerTraceDetail::Summary => {
            update_index <= MINI_TRANSFORMER_TRACE_SUMMARY_INITIAL_STEPS
                || update_index.is_multiple_of(sample_interval.max(1))
        }
        MiniTransformerTraceDetail::None => false,
    }
}

pub(super) fn emit_mini_transformer_committed_binary_steps<G>(
    steps: &[MiniTransformerMlpTrainingStepTrace],
    start_index: usize,
    binary_trace: &mut G,
) -> Result<(), TrainError>
where
    G: FnMut(MiniTransformerBinaryTraceRecord<'_>) -> Result<(), TrainError>,
{
    for step in &steps[start_index.min(steps.len())..] {
        binary_trace(MiniTransformerBinaryTraceRecord::StepSample(step))?;
    }
    Ok(())
}

pub(super) fn mini_transformer_attention_q_teacher_delta(
    stats: &MiniTransformerAttentionWeightUpdateStats,
) -> i8 {
    if stats.q.gradient_saturation_count > 0 {
        return 1;
    }
    if stats.q.weight_delta_l1 == 0 {
        return -1;
    }
    if stats.k.weight_delta_l1 > 0
        && stats.q.weight_delta_l1.saturating_mul(8) < stats.k.weight_delta_l1
    {
        return -1;
    }
    if stats.k.weight_delta_l1 > 0
        && stats.q.weight_delta_l1 > stats.k.weight_delta_l1.saturating_mul(4)
    {
        return 1;
    }
    0
}

pub(super) fn mini_transformer_attention_k_teacher_delta(
    stats: &MiniTransformerAttentionWeightUpdateStats,
) -> i8 {
    if stats.k.gradient_saturation_count > 0 {
        return 1;
    }
    if stats.k.weight_delta_l1 == 0 {
        return -1;
    }
    if stats.q.weight_delta_l1 > 0
        && stats.k.weight_delta_l1 > stats.q.weight_delta_l1.saturating_mul(64)
    {
        return 1;
    }
    0
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mini_transformer_training_progress_trace(
    config: MiniTransformerMlpTrainConfig,
    token_count: usize,
    token_hash: u64,
    window_hash: u64,
    windows: usize,
    examined_windows: usize,
    updates: usize,
    accepted_batch_count: usize,
    rejected_batch_count: usize,
    rollback_count: usize,
    rejected_window_count: usize,
    output_head_delta_l1: u64,
    mlp_delta_l1: u64,
    embedding_delta_l1: u64,
    attention_delta_l1: u64,
    attention_q_delta_l1: u64,
    attention_k_delta_l1: u64,
    attention_v_delta_l1: u64,
    attention_o_delta_l1: u64,
    output_head_carry_l1: u64,
    mlp_carry_l1: u64,
    embedding_carry_l1: u64,
    attention_carry_l1: u64,
    attention_q_carry_l1: u64,
    attention_k_carry_l1: u64,
    attention_v_carry_l1: u64,
    attention_o_carry_l1: u64,
    adaptive_attention_shifts: &MiniTransformerAdaptiveShiftState,
    model: &MiniTransformerMlpModel,
) -> MiniTransformerMlpTrainingProgressTrace {
    let runtime_config = adaptive_attention_shifts.runtime_config(config);
    MiniTransformerMlpTrainingProgressTrace {
        config,
        token_count,
        token_hash,
        window_hash,
        windows,
        examined_windows,
        updates,
        accepted_batch_count,
        rejected_batch_count,
        rollback_count,
        rejected_window_count,
        output_head_delta_l1,
        mlp_delta_l1,
        embedding_delta_l1,
        attention_delta_l1,
        attention_q_delta_l1,
        attention_k_delta_l1,
        attention_v_delta_l1,
        attention_o_delta_l1,
        output_head_carry_l1,
        mlp_carry_l1,
        embedding_carry_l1,
        attention_carry_l1,
        attention_q_carry_l1,
        attention_k_carry_l1,
        attention_v_carry_l1,
        attention_o_carry_l1,
        adaptive_rule_shift_adjustment_count: adaptive_attention_shifts.rule_adjustment_count,
        adaptive_holographic_shift_adjustment_count: adaptive_attention_shifts
            .holographic_adjustment_count,
        current_output_learning_rate_shift: runtime_config.output_learning_rate_shift,
        current_mlp_learning_rate_shift: runtime_config.mlp_learning_rate_shift,
        current_embedding_learning_rate_shift: runtime_config.embedding_learning_rate_shift,
        current_attention_learning_rate_shift: runtime_config.attention_learning_rate_shift,
        current_attention_q_learning_rate_shift: runtime_config.attention_q_learning_rate_shift,
        current_attention_qk_learning_rate_shift: runtime_config.attention_qk_learning_rate_shift,
        model_hash: model.model_hash(),
        embedding_hash: model.embedding_hash(),
        attention_hash: model.attention_hash(),
        mlp_hash: model.mlp_hash(),
        output_head_hash: model.output_head_hash(),
    }
}

pub(super) struct MiniTransformerHostTrainCoreWorkspaceBuffers {
    pub(super) embedding_output: Vec<i16>,
    pub(super) attention_norm: Vec<i16>,
    pub(super) attention_q: Vec<i16>,
    pub(super) attention_k: Vec<i16>,
    pub(super) attention_v: Vec<i16>,
    pub(super) attention_context: Vec<i16>,
    pub(super) attention_output: Vec<i16>,
    pub(super) attention_residual: Vec<i16>,
    pub(super) attention_state_kv: Vec<i64>,
    pub(super) attention_key_sums: Vec<i64>,
    pub(super) mlp_norm: Vec<i16>,
    pub(super) mlp_up: Vec<i16>,
    pub(super) mlp_gate: Vec<i16>,
    pub(super) mlp_gated: Vec<i16>,
    pub(super) mlp_output: Vec<i16>,
    pub(super) block_output: Vec<i16>,
    pub(super) logits_q8: Vec<i32>,
    pub(super) probabilities_q15: Vec<i16>,
    pub(super) grad_output_q15: Vec<i16>,
    pub(super) output_scaled_grad: Vec<i32>,
    pub(super) grad_last_features: Vec<i16>,
    pub(super) grad_mlp_output: Vec<i16>,
    pub(super) grad_mlp_input: Vec<i16>,
    pub(super) mlp_scaled_grad: Vec<i32>,
    pub(super) mlp_input_grad_gated: Vec<i16>,
    pub(super) mlp_input_grad_up: Vec<i16>,
    pub(super) mlp_input_grad_gate: Vec<i16>,
    pub(super) mlp_input_grad_up_input: Vec<i16>,
    pub(super) mlp_input_grad_gate_input: Vec<i16>,
    pub(super) mlp_update_grad_gated: Vec<i16>,
    pub(super) mlp_update_grad_up: Vec<i16>,
    pub(super) mlp_update_grad_gate: Vec<i16>,
    pub(super) grad_attention_output: Vec<i16>,
    pub(super) grad_attention_context: Vec<i16>,
    pub(super) attention_scaled_grad: Vec<i32>,
    pub(super) linear_prefix_states: Vec<i64>,
    pub(super) linear_denominators: Vec<i64>,
    pub(super) linear_grad_state_q15: Vec<i64>,
    pub(super) linear_grad_q_acc: Vec<i64>,
    pub(super) linear_grad_k_acc: Vec<i64>,
    pub(super) linear_grad_v_acc: Vec<i64>,
    pub(super) grad_attention_q: Vec<i16>,
    pub(super) grad_attention_k: Vec<i16>,
    pub(super) grad_attention_v: Vec<i16>,
    pub(super) grad_attention_norm_input: Vec<i16>,
    pub(super) grad_embedding_output: Vec<i16>,
}

impl MiniTransformerHostTrainCoreWorkspaceBuffers {
    pub(super) fn new(seq_len: usize) -> Result<Self, TrainError> {
        let total = seq_len
            .checked_mul(MINI_TRANSFORMER_D_MODEL)
            .ok_or(TrainError::InvalidConfig)?;
        let hidden_total = seq_len
            .checked_mul(MINI_TRANSFORMER_HIDDEN_DIM)
            .ok_or(TrainError::InvalidConfig)?;
        let head_dim = mini_transformer_head_dim()?;
        let head_state_len = head_dim
            .checked_mul(head_dim)
            .ok_or(TrainError::InvalidConfig)?;
        let state_len = MINI_TRANSFORMER_HEADS
            .checked_mul(head_state_len)
            .ok_or(TrainError::InvalidConfig)?;
        let key_sum_len = MINI_TRANSFORMER_HEADS
            .checked_mul(head_dim)
            .ok_or(TrainError::InvalidConfig)?;
        let prefix_len = seq_len
            .checked_mul(state_len)
            .ok_or(TrainError::InvalidConfig)?;
        let denom_len = seq_len
            .checked_mul(MINI_TRANSFORMER_HEADS)
            .ok_or(TrainError::InvalidConfig)?;
        let scaled_len = MINI_TRANSFORMER_D_MODEL.max(MINI_TRANSFORMER_HIDDEN_DIM);

        Ok(Self {
            embedding_output: vec![0_i16; total],
            attention_norm: vec![0_i16; total],
            attention_q: vec![0_i16; total],
            attention_k: vec![0_i16; total],
            attention_v: vec![0_i16; total],
            attention_context: vec![0_i16; total],
            attention_output: vec![0_i16; total],
            attention_residual: vec![0_i16; total],
            attention_state_kv: vec![0_i64; state_len],
            attention_key_sums: vec![0_i64; key_sum_len],
            mlp_norm: vec![0_i16; total],
            mlp_up: vec![0_i16; hidden_total],
            mlp_gate: vec![0_i16; hidden_total],
            mlp_gated: vec![0_i16; hidden_total],
            mlp_output: vec![0_i16; total],
            block_output: vec![0_i16; total],
            logits_q8: vec![0_i32; BYTE_VOCAB],
            probabilities_q15: vec![0_i16; BYTE_VOCAB],
            grad_output_q15: vec![0_i16; BYTE_VOCAB],
            output_scaled_grad: vec![0_i32; BYTE_VOCAB],
            grad_last_features: vec![0_i16; MINI_TRANSFORMER_D_MODEL],
            grad_mlp_output: vec![0_i16; total],
            grad_mlp_input: vec![0_i16; total],
            mlp_scaled_grad: vec![0_i32; scaled_len],
            mlp_input_grad_gated: vec![0_i16; hidden_total],
            mlp_input_grad_up: vec![0_i16; hidden_total],
            mlp_input_grad_gate: vec![0_i16; hidden_total],
            mlp_input_grad_up_input: vec![0_i16; total],
            mlp_input_grad_gate_input: vec![0_i16; total],
            mlp_update_grad_gated: vec![0_i16; hidden_total],
            mlp_update_grad_up: vec![0_i16; hidden_total],
            mlp_update_grad_gate: vec![0_i16; hidden_total],
            grad_attention_output: vec![0_i16; total],
            grad_attention_context: vec![0_i16; total],
            attention_scaled_grad: vec![0_i32; MINI_TRANSFORMER_D_MODEL],
            linear_prefix_states: vec![0_i64; prefix_len],
            linear_denominators: vec![0_i64; denom_len],
            linear_grad_state_q15: vec![0_i64; head_state_len],
            linear_grad_q_acc: vec![0_i64; total],
            linear_grad_k_acc: vec![0_i64; total],
            linear_grad_v_acc: vec![0_i64; total],
            grad_attention_q: vec![0_i16; total],
            grad_attention_k: vec![0_i16; total],
            grad_attention_v: vec![0_i16; total],
            grad_attention_norm_input: vec![0_i16; total],
            grad_embedding_output: vec![0_i16; total],
        })
    }

    pub(super) fn as_workspace(&mut self) -> nsrl_train_core::MiniTransformerStepWorkspace<'_> {
        nsrl_train_core::MiniTransformerStepWorkspace {
            embedding_output: &mut self.embedding_output,
            attention_norm: &mut self.attention_norm,
            attention_q: &mut self.attention_q,
            attention_k: &mut self.attention_k,
            attention_v: &mut self.attention_v,
            attention_context: &mut self.attention_context,
            attention_output: &mut self.attention_output,
            attention_residual: &mut self.attention_residual,
            attention_state_kv: &mut self.attention_state_kv,
            attention_key_sums: &mut self.attention_key_sums,
            mlp_norm: &mut self.mlp_norm,
            mlp_up: &mut self.mlp_up,
            mlp_gate: &mut self.mlp_gate,
            mlp_gated: &mut self.mlp_gated,
            mlp_output: &mut self.mlp_output,
            block_output: &mut self.block_output,
            logits_q8: &mut self.logits_q8,
            probabilities_q15: &mut self.probabilities_q15,
            grad_output_q15: &mut self.grad_output_q15,
            output_scaled_grad: &mut self.output_scaled_grad,
            grad_last_features: &mut self.grad_last_features,
            grad_mlp_output: &mut self.grad_mlp_output,
            grad_mlp_input: &mut self.grad_mlp_input,
            mlp_scaled_grad: &mut self.mlp_scaled_grad,
            mlp_input_grad_gated: &mut self.mlp_input_grad_gated,
            mlp_input_grad_up: &mut self.mlp_input_grad_up,
            mlp_input_grad_gate: &mut self.mlp_input_grad_gate,
            mlp_input_grad_up_input: &mut self.mlp_input_grad_up_input,
            mlp_input_grad_gate_input: &mut self.mlp_input_grad_gate_input,
            mlp_update_grad_gated: &mut self.mlp_update_grad_gated,
            mlp_update_grad_up: &mut self.mlp_update_grad_up,
            mlp_update_grad_gate: &mut self.mlp_update_grad_gate,
            grad_attention_output: &mut self.grad_attention_output,
            grad_attention_context: &mut self.grad_attention_context,
            attention_scaled_grad: &mut self.attention_scaled_grad,
            linear_prefix_states: &mut self.linear_prefix_states,
            linear_denominators: &mut self.linear_denominators,
            linear_grad_state_q15: &mut self.linear_grad_state_q15,
            linear_grad_q_acc: &mut self.linear_grad_q_acc,
            linear_grad_k_acc: &mut self.linear_grad_k_acc,
            linear_grad_v_acc: &mut self.linear_grad_v_acc,
            grad_attention_q: &mut self.grad_attention_q,
            grad_attention_k: &mut self.grad_attention_k,
            grad_attention_v: &mut self.grad_attention_v,
            grad_attention_norm_input: &mut self.grad_attention_norm_input,
            grad_embedding_output: &mut self.grad_embedding_output,
        }
    }

    pub(super) fn reset_host_training_step(&mut self) {
        self.grad_mlp_output.fill(0);
    }

    pub(super) fn validate_host_training_step_shape(
        &self,
        seq_len: usize,
    ) -> Result<(), TrainError> {
        let total = seq_len
            .checked_mul(MINI_TRANSFORMER_D_MODEL)
            .ok_or(TrainError::InvalidConfig)?;
        let hidden_total = seq_len
            .checked_mul(MINI_TRANSFORMER_HIDDEN_DIM)
            .ok_or(TrainError::InvalidConfig)?;
        let head_dim = mini_transformer_head_dim()?;
        let head_state_len = head_dim
            .checked_mul(head_dim)
            .ok_or(TrainError::InvalidConfig)?;
        let state_len = MINI_TRANSFORMER_HEADS
            .checked_mul(head_state_len)
            .ok_or(TrainError::InvalidConfig)?;
        let key_sum_len = MINI_TRANSFORMER_HEADS
            .checked_mul(head_dim)
            .ok_or(TrainError::InvalidConfig)?;
        let prefix_len = seq_len
            .checked_mul(state_len)
            .ok_or(TrainError::InvalidConfig)?;
        let denom_len = seq_len
            .checked_mul(MINI_TRANSFORMER_HEADS)
            .ok_or(TrainError::InvalidConfig)?;
        let scaled_len = MINI_TRANSFORMER_D_MODEL.max(MINI_TRANSFORMER_HIDDEN_DIM);

        if self.output_scaled_grad.len() != BYTE_VOCAB
            || self.grad_last_features.len() != MINI_TRANSFORMER_D_MODEL
            || self.grad_mlp_output.len() != total
            || self.grad_mlp_input.len() != total
            || self.mlp_scaled_grad.len() != scaled_len
            || self.mlp_input_grad_gated.len() != hidden_total
            || self.mlp_input_grad_up.len() != hidden_total
            || self.mlp_input_grad_gate.len() != hidden_total
            || self.mlp_input_grad_up_input.len() != total
            || self.mlp_input_grad_gate_input.len() != total
            || self.mlp_update_grad_gated.len() != hidden_total
            || self.mlp_update_grad_up.len() != hidden_total
            || self.mlp_update_grad_gate.len() != hidden_total
            || self.grad_attention_output.len() != total
            || self.grad_attention_context.len() != total
            || self.attention_scaled_grad.len() < MINI_TRANSFORMER_D_MODEL
            || self.attention_state_kv.len() != state_len
            || self.attention_key_sums.len() != key_sum_len
            || self.linear_prefix_states.len() != prefix_len
            || self.linear_denominators.len() != denom_len
            || self.linear_grad_state_q15.len() != head_state_len
            || self.linear_grad_q_acc.len() != total
            || self.linear_grad_k_acc.len() != total
            || self.linear_grad_v_acc.len() != total
            || self.grad_attention_q.len() != total
            || self.grad_attention_k.len() != total
            || self.grad_attention_v.len() != total
            || self.grad_attention_norm_input.len() != total
            || self.grad_embedding_output.len() != total
        {
            return Err(TrainError::InvalidConfig);
        }
        Ok(())
    }
}

pub(super) fn mini_transformer_uses_train_core_step(config: MiniTransformerMlpTrainConfig) -> bool {
    config.batch_windows == 1
        && config.tokenizer_id == ByteTokenizerId::Identity
        && config.attention_kind == MiniTransformerAttentionKind::Linear
        && config.position_policy == MiniTransformerPositionPolicy::Nope
        && !config.adaptive_shift_controller_enabled()
        && !config.attention_vo_error_feedback
        && !config.attention_vo_oracle
        && !config.reject_loss_regression
}

pub(super) fn mini_transformer_uses_train_core_step_for_model(
    config: MiniTransformerMlpTrainConfig,
    model: &MiniTransformerMlpModel,
) -> bool {
    model.transformer_layers() == 1 && mini_transformer_uses_train_core_step(config)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrainError {
    InvalidConfig,
    InvalidModel(&'static str),
    CoreRejected(&'static str),
    TraceWrite,
}

impl core::fmt::Display for TrainError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidConfig => write!(f, "invalid training config"),
            Self::InvalidModel(message) => write!(f, "invalid model artifact: {message}"),
            Self::CoreRejected(stage) => write!(f, "nsrl-core rejected training stage: {stage}"),
            Self::TraceWrite => write!(f, "failed to write training trace"),
        }
    }
}

impl std::error::Error for TrainError {}

pub(super) fn train_core_error_to_train_error(
    error: nsrl_train_core::TrainCoreError,
    stage: &'static str,
) -> TrainError {
    match error {
        nsrl_train_core::TrainCoreError::InvalidConfig
        | nsrl_train_core::TrainCoreError::InvalidShape => TrainError::InvalidConfig,
        nsrl_train_core::TrainCoreError::CoreRejected => TrainError::CoreRejected(stage),
    }
}

pub fn run_mini_transformer_mlp_swarm_training(
    tokens: &[u8],
    config: MiniTransformerMlpTrainConfig,
    swarm_config: MiniTransformerMlpSwarmTrainConfig,
) -> Result<MiniTransformerMlpSwarmTrainingRun, TrainError> {
    let model = MiniTransformerMlpModel::new_initial_with_seq_len(config.seq_len);
    run_mini_transformer_mlp_swarm_training_from_model(tokens, config, swarm_config, model)
}

pub fn run_mini_transformer_mlp_swarm_training_from_model(
    tokens: &[u8],
    config: MiniTransformerMlpTrainConfig,
    swarm_config: MiniTransformerMlpSwarmTrainConfig,
    base_model: MiniTransformerMlpModel,
) -> Result<MiniTransformerMlpSwarmTrainingRun, TrainError> {
    run_mini_transformer_mlp_swarm_training_from_model_with_progress(
        tokens,
        config,
        swarm_config,
        base_model,
        0,
        |_| Ok(()),
    )
}

pub fn run_mini_transformer_mlp_swarm_training_from_model_with_progress<F>(
    tokens: &[u8],
    config: MiniTransformerMlpTrainConfig,
    swarm_config: MiniTransformerMlpSwarmTrainConfig,
    base_model: MiniTransformerMlpModel,
    progress_interval_batches: usize,
    mut progress: F,
) -> Result<MiniTransformerMlpSwarmTrainingRun, TrainError>
where
    F: FnMut(&MiniTransformerMlpSwarmTrainingProgressTrace) -> Result<(), TrainError>,
{
    if swarm_config.workers == 0 || config.stride == 0 || config.window_offset >= tokens.len() {
        return Err(TrainError::InvalidConfig);
    }
    if base_model.context_seq_len != config.seq_len {
        return Err(TrainError::InvalidConfig);
    }

    let available_windows =
        mini_transformer_filtered_window_starts(tokens.len(), tokens, config).len();
    if available_windows == 0 {
        return Err(TrainError::InvalidConfig);
    }
    let worker_count = mini_transformer_swarm_effective_worker_count(
        config,
        swarm_config.workers,
        available_windows,
    );
    let mut worker_runs = mini_transformer_train_swarm_workers(
        tokens,
        config,
        swarm_config,
        &base_model,
        worker_count,
        progress_interval_batches,
        &mut progress,
    )?;
    if worker_runs.is_empty() {
        return Err(TrainError::InvalidConfig);
    }
    worker_runs.sort_by_key(|run| run.worker_index);

    let best_position = worker_runs
        .iter()
        .enumerate()
        .min_by_key(|(_, run)| {
            (
                run.run.trace.final_total_error,
                run.run.trace.final_probability_error_q15,
                run.run.trace.final_invalid_forward_count,
                run.worker_index,
            )
        })
        .map(|(position, _)| position)
        .ok_or(TrainError::InvalidConfig)?;
    let best_worker_index = worker_runs[best_position].worker_index;
    let model = worker_runs[best_position].run.model.clone();
    let swarm_model = MiniTransformerMlpSwarmModel::new(
        best_worker_index,
        worker_runs
            .iter()
            .map(|run| run.run.model.clone())
            .collect::<Vec<_>>(),
    )?;
    let final_model_hash = model.model_hash();
    let workers = worker_runs
        .iter()
        .map(|run| mini_transformer_swarm_worker_trace(run.worker_index, &run.run.trace))
        .collect::<Vec<_>>();

    let trace = MiniTransformerMlpSwarmTrainingTrace {
        config,
        swarm_config: MiniTransformerMlpSwarmTrainConfig {
            workers: worker_count,
            trace_detail: swarm_config.trace_detail,
        },
        token_count: tokens.len(),
        token_hash: hash_u8_slice(tokens),
        worker_count,
        base_window_offset: config.window_offset,
        base_stride: config.stride,
        best_worker_index,
        final_model_hash,
        workers,
    };

    Ok(MiniTransformerMlpSwarmTrainingRun {
        trace,
        model,
        swarm_model,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn run_mini_transformer_mlp_swarm_worker_from_model_with_progress<F>(
    tokens: &[u8],
    config: MiniTransformerMlpTrainConfig,
    worker_index: usize,
    worker_count: usize,
    base_model: MiniTransformerMlpModel,
    progress_interval_batches: usize,
    trace_detail: MiniTransformerTraceDetail,
    mut progress: F,
) -> Result<MiniTransformerMlpSwarmWorkerTrainingRun, TrainError>
where
    F: FnMut(&MiniTransformerMlpSwarmTrainingProgressTrace) -> Result<(), TrainError>,
{
    if worker_count == 0
        || worker_index >= worker_count
        || config.stride == 0
        || config.window_offset >= tokens.len()
    {
        return Err(TrainError::InvalidConfig);
    }
    if base_model.context_seq_len != config.seq_len {
        return Err(TrainError::InvalidConfig);
    }

    let available_windows =
        mini_transformer_filtered_window_starts(tokens.len(), tokens, config).len();
    if available_windows == 0
        || mini_transformer_swarm_effective_worker_count(config, worker_count, available_windows)
            != worker_count
    {
        return Err(TrainError::InvalidConfig);
    }

    let worker_config = mini_transformer_swarm_worker_config(config, worker_index, worker_count);
    let mut latest_progress = vec![None; worker_count];
    let mut worker_progress = |worker_progress: &MiniTransformerMlpTrainingProgressTrace| {
        latest_progress[worker_index] = Some(MiniTransformerMlpSwarmWorkerProgressTrace {
            worker_index,
            progress: worker_progress.clone(),
        });
        progress(&mini_transformer_swarm_training_progress_trace(
            tokens,
            config,
            MiniTransformerMlpSwarmTrainConfig {
                workers: worker_count,
                trace_detail,
            },
            worker_count,
            &latest_progress,
        ))
    };
    let base_model_hash = base_model.model_hash();
    let run = run_mini_transformer_mlp_training_from_model_with_progress_and_trace_detail(
        tokens,
        worker_config,
        base_model,
        progress_interval_batches,
        trace_detail,
        &mut worker_progress,
    )?;
    let worker = mini_transformer_swarm_worker_trace(worker_index, &run.trace);
    let artifact = MiniTransformerMlpSwarmWorkerArtifact {
        worker_count,
        token_count: tokens.len(),
        token_hash: hash_u8_slice(tokens),
        base_window_offset: config.window_offset,
        base_stride: config.stride,
        base_max_windows: config.max_windows,
        base_model_hash,
        worker,
        model: run.model,
    };

    Ok(MiniTransformerMlpSwarmWorkerTrainingRun { artifact })
}

pub fn assemble_mini_transformer_mlp_swarm_worker_artifacts(
    tokens: &[u8],
    config: MiniTransformerMlpTrainConfig,
    base_model: &MiniTransformerMlpModel,
    artifacts: Vec<MiniTransformerMlpSwarmWorkerArtifact>,
) -> Result<MiniTransformerMlpSwarmTrainingRun, TrainError> {
    let first = artifacts.first().ok_or(TrainError::InvalidConfig)?;
    let worker_count = first.worker_count;
    if worker_count == 0
        || artifacts.len() != worker_count
        || config.stride == 0
        || config.window_offset >= tokens.len()
        || base_model.context_seq_len != config.seq_len
    {
        return Err(TrainError::InvalidConfig);
    }

    let available_windows =
        mini_transformer_filtered_window_starts(tokens.len(), tokens, config).len();
    if available_windows == 0
        || mini_transformer_swarm_effective_worker_count(config, worker_count, available_windows)
            != worker_count
    {
        return Err(TrainError::InvalidConfig);
    }

    let token_hash = hash_u8_slice(tokens);
    let base_model_hash = base_model.model_hash();
    let mut slots = vec![None; worker_count];
    for artifact in artifacts {
        validate_mini_transformer_swarm_worker_artifact(
            tokens,
            config,
            base_model,
            worker_count,
            token_hash,
            base_model_hash,
            &artifact,
        )?;
        let worker_index = artifact.worker.worker_index;
        if slots[worker_index].is_some() {
            return Err(TrainError::InvalidConfig);
        }
        slots[worker_index] = Some(artifact);
    }

    let artifacts = slots
        .into_iter()
        .map(|slot| slot.ok_or(TrainError::InvalidConfig))
        .collect::<Result<Vec<_>, _>>()?;
    let best_position = artifacts
        .iter()
        .enumerate()
        .min_by_key(|(_, artifact)| {
            (
                artifact.worker.final_total_error,
                artifact.worker.final_probability_error_q15,
                artifact.worker.final_invalid_forward_count,
                artifact.worker.worker_index,
            )
        })
        .map(|(position, _)| position)
        .ok_or(TrainError::InvalidConfig)?;
    let best_worker_index = artifacts[best_position].worker.worker_index;
    let model = artifacts[best_position].model.clone();
    let swarm_model = MiniTransformerMlpSwarmModel::new(
        best_worker_index,
        artifacts
            .iter()
            .map(|artifact| artifact.model.clone())
            .collect(),
    )?;
    let workers = artifacts
        .iter()
        .map(|artifact| artifact.worker.clone())
        .collect::<Vec<_>>();
    let trace = MiniTransformerMlpSwarmTrainingTrace {
        config,
        swarm_config: MiniTransformerMlpSwarmTrainConfig {
            workers: worker_count,
            trace_detail: MiniTransformerTraceDetail::None,
        },
        token_count: tokens.len(),
        token_hash,
        worker_count,
        base_window_offset: config.window_offset,
        base_stride: config.stride,
        best_worker_index,
        final_model_hash: model.model_hash(),
        workers,
    };

    Ok(MiniTransformerMlpSwarmTrainingRun {
        trace,
        model,
        swarm_model,
    })
}

pub fn run_mini_transformer_mlp_swarm_scaling_benchmark(
    tokens: &[u8],
    config: MiniTransformerMlpTrainConfig,
    max_workers: usize,
    trace_detail: MiniTransformerTraceDetail,
) -> Result<MiniTransformerMlpSwarmScalingTrace, TrainError> {
    let model = MiniTransformerMlpModel::new_initial_with_seq_len(config.seq_len);
    run_mini_transformer_mlp_swarm_scaling_benchmark_from_model(
        tokens,
        config,
        max_workers,
        trace_detail,
        model,
    )
}

pub fn run_mini_transformer_mlp_swarm_scaling_benchmark_from_model(
    tokens: &[u8],
    config: MiniTransformerMlpTrainConfig,
    max_workers: usize,
    trace_detail: MiniTransformerTraceDetail,
    base_model: MiniTransformerMlpModel,
) -> Result<MiniTransformerMlpSwarmScalingTrace, TrainError> {
    if max_workers == 0 {
        return Err(TrainError::InvalidConfig);
    }
    if base_model.context_seq_len != config.seq_len {
        return Err(TrainError::InvalidConfig);
    }

    let worker_counts = mini_transformer_swarm_scaling_worker_counts(max_workers);
    let available_parallelism = std::thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(1);
    let mut runs = Vec::with_capacity(worker_counts.len());
    let mut baseline_elapsed_ns = 0_u64;

    for &requested_worker_count in &worker_counts {
        let start = std::time::Instant::now();
        let run = run_mini_transformer_mlp_swarm_training_from_model(
            tokens,
            config,
            MiniTransformerMlpSwarmTrainConfig {
                workers: requested_worker_count,
                trace_detail,
            },
            base_model.clone(),
        )?;
        let elapsed_ns = mini_transformer_elapsed_ns_u64(start.elapsed());
        if baseline_elapsed_ns == 0 {
            baseline_elapsed_ns = elapsed_ns.max(1);
        }

        let examined_windows = run
            .trace
            .workers
            .iter()
            .map(|worker| worker.examined_windows)
            .sum::<usize>();
        let updates = run
            .trace
            .workers
            .iter()
            .map(|worker| worker.updates)
            .sum::<usize>();
        let accepted_batch_count = run
            .trace
            .workers
            .iter()
            .map(|worker| worker.accepted_batch_count)
            .sum::<usize>();
        let rejected_batch_count = run
            .trace
            .workers
            .iter()
            .map(|worker| worker.rejected_batch_count)
            .sum::<usize>();
        let rollback_count = run
            .trace
            .workers
            .iter()
            .map(|worker| worker.rollback_count)
            .sum::<usize>();
        let best_worker = run
            .trace
            .workers
            .iter()
            .find(|worker| worker.worker_index == run.trace.best_worker_index)
            .ok_or(TrainError::InvalidConfig)?;
        let speedup_per_mille =
            mini_transformer_ratio_per_mille_u64(baseline_elapsed_ns, elapsed_ns.max(1));
        let parallel_efficiency_per_mille =
            speedup_per_mille / u64::try_from(run.trace.worker_count.max(1)).unwrap_or(u64::MAX);

        runs.push(MiniTransformerMlpSwarmScalingRunTrace {
            requested_worker_count,
            effective_worker_count: run.trace.worker_count,
            elapsed_ns,
            speedup_per_mille,
            parallel_efficiency_per_mille,
            windows_per_second_milli: mini_transformer_rate_per_second_milli(
                examined_windows,
                elapsed_ns,
            ),
            updates_per_second_milli: mini_transformer_rate_per_second_milli(updates, elapsed_ns),
            examined_windows,
            updates,
            accepted_batch_count,
            rejected_batch_count,
            rollback_count,
            best_worker_index: run.trace.best_worker_index,
            best_final_total_error: best_worker.final_total_error,
            best_final_probability_error_q15: best_worker.final_probability_error_q15,
            best_final_accuracy_per_mille: best_worker.final_accuracy_per_mille,
            final_model_hash: run.trace.final_model_hash,
        });
    }

    Ok(MiniTransformerMlpSwarmScalingTrace {
        config,
        token_count: tokens.len(),
        token_hash: hash_u8_slice(tokens),
        available_parallelism,
        requested_max_workers: max_workers,
        worker_counts,
        runs,
    })
}

pub(super) fn mini_transformer_swarm_scaling_worker_counts(max_workers: usize) -> Vec<usize> {
    let max_workers = max_workers.max(1);
    let mut counts = Vec::new();
    let mut worker_count = 1_usize;
    while worker_count < max_workers {
        counts.push(worker_count);
        worker_count = worker_count.saturating_mul(2);
        if worker_count == 0 {
            break;
        }
    }
    if counts.last().copied() != Some(max_workers) {
        counts.push(max_workers);
    }
    counts
}

pub(super) fn mini_transformer_elapsed_ns_u64(elapsed: std::time::Duration) -> u64 {
    elapsed.as_nanos().min(u128::from(u64::MAX)) as u64
}

pub(super) fn mini_transformer_rate_per_second_milli(events: usize, elapsed_ns: u64) -> u64 {
    if elapsed_ns == 0 {
        return 0;
    }
    ((events as u128).saturating_mul(1_000_000_000_000_u128) / u128::from(elapsed_ns))
        .min(u128::from(u64::MAX)) as u64
}

pub(super) fn mini_transformer_ratio_per_mille_u64(numerator: u64, denominator: u64) -> u64 {
    if denominator == 0 {
        return 0;
    }
    (u128::from(numerator).saturating_mul(1000) / u128::from(denominator)).min(u128::from(u64::MAX))
        as u64
}

pub(super) struct MiniTransformerMlpSwarmWorkerRun {
    pub(super) worker_index: usize,
    pub(super) run: MiniTransformerMlpTrainingRun,
}

pub(super) fn mini_transformer_train_swarm_workers(
    tokens: &[u8],
    config: MiniTransformerMlpTrainConfig,
    swarm_config: MiniTransformerMlpSwarmTrainConfig,
    base_model: &MiniTransformerMlpModel,
    worker_count: usize,
    progress_interval_batches: usize,
    progress: &mut impl FnMut(&MiniTransformerMlpSwarmTrainingProgressTrace) -> Result<(), TrainError>,
) -> Result<Vec<MiniTransformerMlpSwarmWorkerRun>, TrainError> {
    if worker_count <= 1 {
        let worker_config = mini_transformer_swarm_worker_config(config, 0, 1);
        let mut latest_progress = vec![None; worker_count];
        let mut worker_progress = |worker_progress: &MiniTransformerMlpTrainingProgressTrace| {
            latest_progress[0] = Some(MiniTransformerMlpSwarmWorkerProgressTrace {
                worker_index: 0,
                progress: worker_progress.clone(),
            });
            progress(&mini_transformer_swarm_training_progress_trace(
                tokens,
                config,
                swarm_config,
                worker_count,
                &latest_progress,
            ))
        };
        let run = run_mini_transformer_mlp_training_from_model_with_progress_and_trace_detail(
            tokens,
            worker_config,
            base_model.clone(),
            progress_interval_batches,
            swarm_config.trace_detail,
            &mut worker_progress,
        )?;
        return Ok(vec![MiniTransformerMlpSwarmWorkerRun {
            worker_index: 0,
            run,
        }]);
    }

    std::thread::scope(|scope| {
        let (progress_tx, progress_rx) =
            std::sync::mpsc::channel::<MiniTransformerMlpSwarmWorkerProgressTrace>();
        let mut handles = Vec::with_capacity(worker_count);
        for worker_index in 0..worker_count {
            let worker_config =
                mini_transformer_swarm_worker_config(config, worker_index, worker_count);
            let worker_model = base_model.clone();
            let progress_tx = progress_tx.clone();
            handles.push(scope.spawn(move || {
                let mut worker_progress = |progress: &MiniTransformerMlpTrainingProgressTrace| {
                    progress_tx
                        .send(MiniTransformerMlpSwarmWorkerProgressTrace {
                            worker_index,
                            progress: progress.clone(),
                        })
                        .map_err(|_| TrainError::TraceWrite)
                };
                let run =
                    run_mini_transformer_mlp_training_from_model_with_progress_and_trace_detail(
                        tokens,
                        worker_config,
                        worker_model,
                        progress_interval_batches,
                        swarm_config.trace_detail,
                        &mut worker_progress,
                    )?;
                Ok(MiniTransformerMlpSwarmWorkerRun { worker_index, run })
            }));
        }
        drop(progress_tx);

        let mut latest_progress = vec![None; worker_count];
        for worker_progress in progress_rx {
            let worker_index = worker_progress.worker_index;
            if worker_index < latest_progress.len() {
                latest_progress[worker_index] = Some(worker_progress);
                progress(&mini_transformer_swarm_training_progress_trace(
                    tokens,
                    config,
                    swarm_config,
                    worker_count,
                    &latest_progress,
                ))?;
            }
        }

        let mut runs = Vec::with_capacity(worker_count);
        for handle in handles {
            match handle.join() {
                Ok(result) => runs.push(result?),
                Err(payload) => std::panic::resume_unwind(payload),
            }
        }
        Ok(runs)
    })
}

pub(super) fn mini_transformer_swarm_training_progress_trace(
    tokens: &[u8],
    config: MiniTransformerMlpTrainConfig,
    swarm_config: MiniTransformerMlpSwarmTrainConfig,
    worker_count: usize,
    latest_progress: &[Option<MiniTransformerMlpSwarmWorkerProgressTrace>],
) -> MiniTransformerMlpSwarmTrainingProgressTrace {
    MiniTransformerMlpSwarmTrainingProgressTrace {
        config,
        swarm_config: MiniTransformerMlpSwarmTrainConfig {
            workers: worker_count,
            trace_detail: swarm_config.trace_detail,
        },
        token_count: tokens.len(),
        token_hash: hash_u8_slice(tokens),
        worker_count,
        base_window_offset: config.window_offset,
        base_stride: config.stride,
        workers: latest_progress
            .iter()
            .filter_map(|progress| progress.clone())
            .collect(),
    }
}

pub(super) fn mini_transformer_swarm_effective_worker_count(
    config: MiniTransformerMlpTrainConfig,
    requested_workers: usize,
    available_windows: usize,
) -> usize {
    let by_requested = match config.max_windows {
        Some(max_windows) => requested_workers.min(max_windows.max(1)).max(1),
        None => requested_workers.max(1),
    };
    by_requested.min(available_windows.max(1)).max(1)
}

pub(super) fn mini_transformer_swarm_worker_config(
    mut config: MiniTransformerMlpTrainConfig,
    worker_index: usize,
    worker_count: usize,
) -> MiniTransformerMlpTrainConfig {
    let base_stride = config.stride.max(1);
    config.window_offset = config
        .window_offset
        .saturating_add(worker_index.saturating_mul(base_stride));
    config.stride = base_stride.saturating_mul(worker_count.max(1));
    config.max_windows = config.max_windows.map(|max_windows| {
        mini_transformer_swarm_worker_window_limit(max_windows, worker_index, worker_count)
    });
    config
}

pub(super) fn mini_transformer_swarm_worker_window_limit(
    max_windows: usize,
    worker_index: usize,
    worker_count: usize,
) -> usize {
    let base = max_windows / worker_count.max(1);
    let remainder = max_windows % worker_count.max(1);
    base + usize::from(worker_index < remainder)
}

pub(super) fn mini_transformer_swarm_worker_trace(
    worker_index: usize,
    trace: &MiniTransformerMlpTrainingTrace,
) -> MiniTransformerMlpSwarmWorkerTrace {
    MiniTransformerMlpSwarmWorkerTrace {
        worker_index,
        window_offset: trace.config.window_offset,
        stride: trace.config.stride,
        max_windows: trace.config.max_windows,
        token_hash: trace.token_hash,
        window_hash: trace.window_hash,
        windows: trace.windows,
        examined_windows: trace.examined_windows,
        updates: trace.updates,
        accepted_batch_count: trace.accepted_batch_count,
        rejected_batch_count: trace.rejected_batch_count,
        rollback_count: trace.rollback_count,
        rejected_window_count: trace.rejected_window_count,
        final_invalid_forward_count: trace.final_invalid_forward_count,
        initial_total_error: trace.initial_total_error,
        final_total_error: trace.final_total_error,
        initial_probability_error_q15: trace.initial_probability_error_q15,
        final_probability_error_q15: trace.final_probability_error_q15,
        final_accuracy_per_mille: trace.final_accuracy_per_mille,
        final_model_hash: trace.final_model_hash,
        final_logits_hash: trace.final_logits_hash,
    }
}

pub(super) fn validate_mini_transformer_swarm_worker_artifact(
    tokens: &[u8],
    config: MiniTransformerMlpTrainConfig,
    base_model: &MiniTransformerMlpModel,
    worker_count: usize,
    token_hash: u64,
    base_model_hash: u64,
    artifact: &MiniTransformerMlpSwarmWorkerArtifact,
) -> Result<(), TrainError> {
    let worker_index = artifact.worker.worker_index;
    if artifact.worker_count != worker_count
        || artifact.token_count != tokens.len()
        || artifact.token_hash != token_hash
        || artifact.base_window_offset != config.window_offset
        || artifact.base_stride != config.stride
        || artifact.base_max_windows != config.max_windows
        || artifact.base_model_hash != base_model_hash
        || worker_index >= worker_count
        || artifact.model.context_seq_len != config.seq_len
        || artifact.model.model_hash() != artifact.worker.final_model_hash
    {
        return Err(TrainError::InvalidConfig);
    }

    let worker_config = mini_transformer_swarm_worker_config(config, worker_index, worker_count);
    let starts = mini_transformer_filtered_window_starts(tokens.len(), tokens, worker_config);
    if starts.is_empty()
        || artifact.worker.window_offset != worker_config.window_offset
        || artifact.worker.stride != worker_config.stride
        || artifact.worker.max_windows != worker_config.max_windows
        || artifact.worker.token_hash != token_hash
        || artifact.worker.window_hash
            != hash_mini_transformer_windows(tokens, worker_config, &starts)
        || artifact.worker.windows != starts.len()
    {
        return Err(TrainError::InvalidConfig);
    }

    let initial_eval = mini_transformer_eval_summary_with_attention_and_position_policy(
        tokens,
        &starts,
        base_model,
        worker_config.seq_len,
        worker_config.attention_kind,
        worker_config.position_policy,
    )?;
    let final_eval = mini_transformer_eval_summary_with_attention_and_position_policy(
        tokens,
        &starts,
        &artifact.model,
        worker_config.seq_len,
        worker_config.attention_kind,
        worker_config.position_policy,
    )?;
    let final_accuracy_per_mille =
        starts.len().saturating_sub(final_eval.mistakes) * 1000 / starts.len();
    if artifact.worker.initial_total_error != initial_eval.mistakes
        || artifact.worker.initial_probability_error_q15 != initial_eval.probability_error_q15
        || artifact.worker.final_total_error != final_eval.mistakes
        || artifact.worker.final_probability_error_q15 != final_eval.probability_error_q15
        || artifact.worker.final_invalid_forward_count != final_eval.invalid_forward_count
        || artifact.worker.final_logits_hash != final_eval.logits_hash
        || artifact.worker.final_accuracy_per_mille != final_accuracy_per_mille
    {
        return Err(TrainError::InvalidConfig);
    }

    Ok(())
}

#[derive(Debug, Clone)]
pub(super) struct MiniTransformerAdamStateRanges {
    pub(super) embeddings: Range<usize>,
    pub(super) position_embeddings: Range<usize>,
    pub(super) attention_rms: Range<usize>,
    pub(super) mlp_rms: Range<usize>,
    pub(super) q: Range<usize>,
    pub(super) k: Range<usize>,
    pub(super) v: Range<usize>,
    pub(super) o: Range<usize>,
    pub(super) up: Range<usize>,
    pub(super) gate: Range<usize>,
    pub(super) down: Range<usize>,
    pub(super) output: Range<usize>,
}

#[derive(Debug, Clone)]
pub(super) struct MiniTransformerAdamBatchUpdateStats {
    pub(super) output_head: LinearWeightUpdateStats,
    pub(super) mlp: GatedMlpWeightUpdateStats,
    pub(super) embedding: SoftmaxUpdateStats,
    pub(super) rms_norm: SoftmaxUpdateStats,
    pub(super) attention: MiniTransformerAttentionWeightUpdateStats,
}

pub(super) fn mini_transformer_adam_state_ranges(
    model: &MiniTransformerMlpModel,
) -> Result<MiniTransformerAdamStateRanges, TrainError> {
    fn take(cursor: &mut usize, len: usize) -> Result<Range<usize>, TrainError> {
        let start = *cursor;
        let end = start
            .checked_add(len)
            .ok_or(TrainError::InvalidModel("optimizer range overflow"))?;
        *cursor = end;
        Ok(start..end)
    }

    let mut cursor = 0_usize;
    let ranges = MiniTransformerAdamStateRanges {
        embeddings: take(&mut cursor, model.embeddings.len())?,
        position_embeddings: take(&mut cursor, model.position_embeddings.len())?,
        attention_rms: take(&mut cursor, model.attention_rms_weights.len())?,
        mlp_rms: take(&mut cursor, model.mlp_rms_weights.len())?,
        q: take(&mut cursor, model.q_weights.len())?,
        k: take(&mut cursor, model.k_weights.len())?,
        v: take(&mut cursor, model.v_weights.len())?,
        o: take(&mut cursor, model.o_weights.len())?,
        up: take(&mut cursor, model.up_weights.len())?,
        gate: take(&mut cursor, model.gate_weights.len())?,
        down: take(&mut cursor, model.down_weights.len())?,
        output: take(&mut cursor, model.output_weights.len())?,
    };
    if cursor != model.optimizer_parameter_count()? {
        return Err(TrainError::InvalidModel("optimizer range mismatch"));
    }
    Ok(ranges)
}

pub(super) fn offset_optimizer_range(
    tensor: &Range<usize>,
    local: &Range<usize>,
) -> Result<Range<usize>, TrainError> {
    if local.start > local.end || local.end > tensor.len() {
        return Err(TrainError::InvalidConfig);
    }
    let start = tensor
        .start
        .checked_add(local.start)
        .ok_or(TrainError::InvalidConfig)?;
    let end = tensor
        .start
        .checked_add(local.end)
        .ok_or(TrainError::InvalidConfig)?;
    Ok(start..end)
}

pub(super) fn apply_integer_adam_state_slice_i8(
    accumulators: &[i64],
    sample_count: usize,
    weights: &mut [i8],
    state: &mut MiniTransformerAdamOptimizerState,
    state_range: Range<usize>,
) -> Result<LinearWeightUpdateStats, TrainError> {
    if accumulators.len() != weights.len() || state_range.len() != weights.len() {
        return Err(TrainError::InvalidConfig);
    }
    let config = state.config;
    let mut workspace = nsrl_train_core::IntegerAdamStateWorkspace {
        step: state.step,
        first_moments: &mut state.first_moments[state_range.clone()],
        second_moments: &mut state.second_moments[state_range.clone()],
        update_residuals: &mut state.update_residuals[state_range],
    };
    nsrl_train_core::apply_integer_adam_accumulators_i64_to_i8(
        accumulators,
        sample_count,
        weights,
        config,
        &mut workspace,
    )
    .map_err(|error| train_core_error_to_train_error(error, "integer_adam_i8_apply"))
}

pub(super) fn apply_integer_adam_state_slice_i16(
    accumulators: &[i64],
    sample_count: usize,
    weights: &mut [i16],
    state: &mut MiniTransformerAdamOptimizerState,
    state_range: Range<usize>,
) -> Result<LinearWeightUpdateStats, TrainError> {
    if accumulators.len() != weights.len() || state_range.len() != weights.len() {
        return Err(TrainError::InvalidConfig);
    }
    let config = state.config;
    let mut workspace = nsrl_train_core::IntegerAdamStateWorkspace {
        step: state.step,
        first_moments: &mut state.first_moments[state_range.clone()],
        second_moments: &mut state.second_moments[state_range.clone()],
        update_residuals: &mut state.update_residuals[state_range],
    };
    nsrl_train_core::apply_integer_adam_accumulators_i64_to_i16(
        accumulators,
        sample_count,
        weights,
        config,
        &mut workspace,
    )
    .map_err(|error| train_core_error_to_train_error(error, "integer_adam_i16_apply"))
}

pub(super) fn mini_transformer_apply_integer_adam_batch(
    batch: &MiniTransformerMapReduceBatchResult,
    model: &mut MiniTransformerMlpModel,
    state: &mut MiniTransformerAdamOptimizerState,
    position_policy: MiniTransformerPositionPolicy,
    train_scope: MiniTransformerAdamTrainScope,
) -> Result<MiniTransformerAdamBatchUpdateStats, TrainError> {
    state.validate_for_model(model)?;
    let ranges = mini_transformer_adam_state_ranges(model)?;
    let output_head = if matches!(
        train_scope,
        MiniTransformerAdamTrainScope::All
            | MiniTransformerAdamTrainScope::Output
            | MiniTransformerAdamTrainScope::FinalMlpAndOutput
    ) {
        apply_integer_adam_state_slice_i8(
            &batch.output_head_gradient.accumulators,
            batch.output_head_gradient.sample_count,
            &mut model.output_weights,
            state,
            ranges.output.clone(),
        )?
    } else {
        empty_linear_weight_update_stats()
    };
    let mut mlp = empty_gated_mlp_weight_update_stats();
    let mut attention = empty_mini_transformer_attention_weight_update_stats();
    let mut rms_linear = empty_linear_weight_update_stats();
    let layers = model.checked_transformer_layers()?;
    if batch.mlp_weight_gradients.len() != layers
        || batch.attention_weight_gradients.len() != layers
        || batch.rms_weight_gradients.len() != layers
    {
        return Err(TrainError::InvalidConfig);
    }
    for layer_index in 0..layers {
        let attention_local = model.attention_weight_range(layer_index)?;
        let q_state_range = offset_optimizer_range(&ranges.q, &attention_local)?;
        let k_state_range = offset_optimizer_range(&ranges.k, &attention_local)?;
        let v_state_range = offset_optimizer_range(&ranges.v, &attention_local)?;
        let o_state_range = offset_optimizer_range(&ranges.o, &attention_local)?;
        let up_local = model.mlp_up_or_gate_weight_range(layer_index)?;
        let down_local = model.mlp_down_weight_range(layer_index)?;
        if train_scope == MiniTransformerAdamTrainScope::All {
            let attention_gradient = &batch.attention_weight_gradients[layer_index];
            let q = apply_integer_adam_state_slice_i8(
                &attention_gradient.q.accumulators,
                attention_gradient.q.sample_count,
                &mut model.q_weights[attention_local.clone()],
                state,
                q_state_range,
            )?;
            let k = apply_integer_adam_state_slice_i8(
                &attention_gradient.k.accumulators,
                attention_gradient.k.sample_count,
                &mut model.k_weights[attention_local.clone()],
                state,
                k_state_range,
            )?;
            let v = apply_integer_adam_state_slice_i8(
                &attention_gradient.v.accumulators,
                attention_gradient.v.sample_count,
                &mut model.v_weights[attention_local.clone()],
                state,
                v_state_range,
            )?;
            let o = apply_integer_adam_state_slice_i8(
                &attention_gradient.o.accumulators,
                attention_gradient.o.sample_count,
                &mut model.o_weights[attention_local.clone()],
                state,
                o_state_range,
            )?;
            let mut projection_total = empty_linear_weight_update_stats();
            for stats in [q, k, v, o] {
                add_linear_weight_update_stats_checked(&mut projection_total, stats)?;
            }
            add_mini_transformer_attention_weight_update_stats_checked(
                &mut attention,
                MiniTransformerAttentionWeightUpdateStats {
                    q,
                    k,
                    v,
                    o,
                    gradient_saturation_count: projection_total.gradient_saturation_count,
                    zero_delta_count: projection_total.zero_delta_count,
                    weight_delta_l1: projection_total.weight_delta_l1,
                    grad_embedding_output: Vec::new(),
                },
            )?;
        }

        if train_scope == MiniTransformerAdamTrainScope::All
            || (matches!(
                train_scope,
                MiniTransformerAdamTrainScope::FinalMlp
                    | MiniTransformerAdamTrainScope::FinalMlpAndOutput
            ) && layer_index + 1 == layers)
        {
            let mlp_gradient = &batch.mlp_weight_gradients[layer_index];
            let layer_mlp = GatedMlpWeightUpdateStats {
                down: apply_integer_adam_state_slice_i8(
                    &mlp_gradient.down.accumulators,
                    mlp_gradient.down.sample_count,
                    &mut model.down_weights[down_local.clone()],
                    state,
                    offset_optimizer_range(&ranges.down, &down_local)?,
                )?,
                up: apply_integer_adam_state_slice_i8(
                    &mlp_gradient.up.accumulators,
                    mlp_gradient.up.sample_count,
                    &mut model.up_weights[up_local.clone()],
                    state,
                    offset_optimizer_range(&ranges.up, &up_local)?,
                )?,
                gate: apply_integer_adam_state_slice_i8(
                    &mlp_gradient.gate.accumulators,
                    mlp_gradient.gate.sample_count,
                    &mut model.gate_weights[up_local.clone()],
                    state,
                    offset_optimizer_range(&ranges.gate, &up_local)?,
                )?,
            };
            add_gated_mlp_weight_update_stats_checked(&mut mlp, layer_mlp)?;
        }
        if matches!(
            train_scope,
            MiniTransformerAdamTrainScope::All | MiniTransformerAdamTrainScope::RmsNorm
        ) && model.rms_norm_enabled()
        {
            let rms_local = model.rms_weight_range(layer_index)?;
            let rms_gradient = &batch.rms_weight_gradients[layer_index];
            let attention_rms = apply_integer_adam_state_slice_i16(
                &rms_gradient.attention.accumulators,
                rms_gradient.attention.sample_count,
                &mut model.attention_rms_weights[rms_local.clone()],
                state,
                offset_optimizer_range(&ranges.attention_rms, &rms_local)?,
            )?;
            let mlp_rms = apply_integer_adam_state_slice_i16(
                &rms_gradient.mlp.accumulators,
                rms_gradient.mlp.sample_count,
                &mut model.mlp_rms_weights[rms_local.clone()],
                state,
                offset_optimizer_range(&ranges.mlp_rms, &rms_local)?,
            )?;
            add_linear_weight_update_stats_checked(&mut rms_linear, attention_rms)?;
            add_linear_weight_update_stats_checked(&mut rms_linear, mlp_rms)?;
        }
    }

    let mut embedding_linear = empty_linear_weight_update_stats();
    if train_scope == MiniTransformerAdamTrainScope::All {
        embedding_linear = apply_integer_adam_state_slice_i16(
            &batch.embedding_gradient.token_accumulators,
            batch.embedding_gradient.sample_count,
            &mut model.embeddings,
            state,
            ranges.embeddings,
        )?;
        if position_policy.uses_position_embeddings() {
            let position = apply_integer_adam_state_slice_i16(
                &batch.embedding_gradient.position_accumulators,
                batch.embedding_gradient.sample_count,
                &mut model.position_embeddings,
                state,
                ranges.position_embeddings,
            )?;
            add_linear_weight_update_stats_checked(&mut embedding_linear, position)?;
        }
    }
    let embedding = SoftmaxUpdateStats {
        gradient_saturation_count: embedding_linear.gradient_saturation_count,
        zero_delta_count: embedding_linear.zero_delta_count,
        weight_delta_l1: embedding_linear.weight_delta_l1,
    };
    let rms_norm = SoftmaxUpdateStats {
        gradient_saturation_count: rms_linear.gradient_saturation_count,
        zero_delta_count: rms_linear.zero_delta_count,
        weight_delta_l1: rms_linear.weight_delta_l1,
    };
    // All tensor slices represent one optimizer batch; slice-local workspace
    // steps are intentionally ignored and the global step advances exactly once.
    state.step = state
        .step
        .checked_add(1)
        .ok_or(TrainError::CoreRejected("integer_adam_step_overflow"))?;
    state.bind_to_model(model)?;
    Ok(MiniTransformerAdamBatchUpdateStats {
        output_head,
        mlp,
        embedding,
        rms_norm,
        attention,
    })
}

pub fn run_mini_transformer_mlp_integer_adam_training(
    tokens: &[u8],
    config: MiniTransformerMlpTrainConfig,
    optimizer_config: IntegerAdamConfig,
) -> Result<MiniTransformerAdamTrainingRun, TrainError> {
    let model = MiniTransformerMlpModel::new_initial_with_seq_len(config.seq_len);
    run_mini_transformer_mlp_integer_adam_training_from_model(
        tokens,
        config,
        optimizer_config,
        model,
        None,
    )
}

pub fn run_mini_transformer_mlp_integer_adam_training_from_model(
    tokens: &[u8],
    config: MiniTransformerMlpTrainConfig,
    optimizer_config: IntegerAdamConfig,
    model: MiniTransformerMlpModel,
    optimizer_state: Option<MiniTransformerAdamOptimizerState>,
) -> Result<MiniTransformerAdamTrainingRun, TrainError> {
    run_mini_transformer_mlp_integer_adam_training_from_model_with_scope(
        tokens,
        config,
        optimizer_config,
        model,
        optimizer_state,
        MiniTransformerAdamTrainScope::All,
    )
}

pub fn run_mini_transformer_mlp_integer_adam_training_from_model_with_scope(
    tokens: &[u8],
    config: MiniTransformerMlpTrainConfig,
    optimizer_config: IntegerAdamConfig,
    mut model: MiniTransformerMlpModel,
    optimizer_state: Option<MiniTransformerAdamOptimizerState>,
    train_scope: MiniTransformerAdamTrainScope,
) -> Result<MiniTransformerAdamTrainingRun, TrainError> {
    if config.epochs == 0
        || config.seq_len == 0
        || config.stride == 0
        || config.batch_windows == 0
        || config.target_token_min > config.target_token_max
        || !valid_mini_transformer_target_segment(config.target_segment)
        || !valid_q15_weight_floor(config.target_frequency_min_weight_q15)
        || config.argmax_margin_weight_q15 < 0
        || !optimizer_config.is_valid()
        || config.output_learning_rate_shift > MAX_RIGHT_SHIFT
        || config.mlp_learning_rate_shift > MAX_RIGHT_SHIFT
        || config.embedding_learning_rate_shift > MAX_RIGHT_SHIFT
        || config.attention_learning_rate_shift > MAX_RIGHT_SHIFT
        || config.attention_q_learning_rate_shift > MAX_RIGHT_SHIFT
        || config.attention_qk_learning_rate_shift > MAX_RIGHT_SHIFT
        || config.attention_kind.uses_incremental_state()
        || config.attention_vo_error_feedback
        || config.attention_vo_oracle
        || config.adaptive_shift_controller_enabled()
        || model.context_seq_len != config.seq_len
    {
        return Err(TrainError::InvalidConfig);
    }
    model.checked_transformer_layers()?;
    let starts = mini_transformer_filtered_window_starts(tokens.len(), tokens, config);
    if starts.is_empty() {
        return Err(TrainError::InvalidConfig);
    }
    let target_frequency_weights_q15 = byte_target_frequency_weights_q15(
        tokens,
        &starts,
        config.seq_len,
        config.target_frequency_cap,
        config.target_frequency_min_weight_q15,
    )?;
    let token_hash = hash_u8_slice(tokens);
    let window_hash = hash_mini_transformer_windows(tokens, config, &starts);
    let initial_model_hash = model.model_hash();
    let initial_mistakes = mini_transformer_total_error_with_attention_and_position_policy(
        tokens,
        &starts,
        &model,
        config.seq_len,
        config.attention_kind,
        config.position_policy,
    )?;
    let initial_probability_error_q15 =
        mini_transformer_total_probability_error_q15_with_attention_and_position_policy(
            tokens,
            &starts,
            &model,
            config.seq_len,
            config.attention_kind,
            config.position_policy,
        )?;
    let mut optimizer_state = match optimizer_state {
        Some(state) => {
            state.validate_for_model(&model)?;
            if state.config != optimizer_config {
                return Err(TrainError::InvalidConfig);
            }
            state
        }
        None => MiniTransformerAdamOptimizerState::new_for_model(&model, optimizer_config)?,
    };
    #[cfg(feature = "mini-calibrated")]
    if config.position_policy == MiniTransformerPositionPolicy::Nope
        && mini_transformer_suffix_memory_is_installed(&model.position_embeddings)
    {
        for weight in model
            .position_embeddings
            .iter_mut()
            .take(MINI_TRANSFORMER_SUFFIX_MEMORY_MAGIC.len() / 2)
        {
            *weight = 0;
        }
        optimizer_state.bind_to_model(&model)?;
    }
    let mut examined_windows = 0_usize;
    let mut updates = 0_usize;
    let mut accepted_batch_count = 0_usize;
    let mut rejected_batch_count = 0_usize;
    let mut output_head_delta_l1 = 0_u64;
    let mut mlp_delta_l1 = 0_u64;
    let mut embedding_delta_l1 = 0_u64;
    let mut rms_norm_delta_l1 = 0_u64;
    let mut attention_delta_l1 = 0_u64;
    let mut attention_q_delta_l1 = 0_u64;
    let mut attention_k_delta_l1 = 0_u64;
    let mut attention_v_delta_l1 = 0_u64;
    let mut attention_o_delta_l1 = 0_u64;
    let mut mlp_saturation_count = 0_usize;
    let mut attention_saturation_count = 0_usize;
    let mut residual_saturation_count = 0_usize;

    for epoch in 0..config.epochs {
        let mut batch_start = 0_usize;
        while batch_start < starts.len() {
            let batch_end = batch_start
                .saturating_add(config.batch_windows)
                .min(starts.len());
            examined_windows = examined_windows.saturating_add(batch_end - batch_start);
            let batch_result = if config.batch_mode == MiniTransformerBatchMode::MapReduce {
                mini_transformer_map_reduce_batch(
                    tokens,
                    &starts,
                    &target_frequency_weights_q15,
                    batch_start,
                    batch_end,
                    epoch,
                    &model,
                    config,
                    updates,
                    MiniTransformerTraceDetail::None,
                    usize::MAX,
                )
            } else {
                mini_transformer_map_reduce_worker_batch(
                    tokens,
                    &starts,
                    &target_frequency_weights_q15,
                    batch_start,
                    batch_end,
                    batch_start,
                    epoch,
                    &model,
                    config,
                    updates,
                    MiniTransformerTraceDetail::None,
                    usize::MAX,
                )
            };
            let batch_result = match batch_result {
                Ok(result) if result.accepted_window_count > 0 => result,
                Ok(_) | Err(TrainError::CoreRejected(_)) => {
                    rejected_batch_count = rejected_batch_count.saturating_add(1);
                    batch_start = batch_end;
                    continue;
                }
                Err(error) => return Err(error),
            };
            mlp_saturation_count =
                mlp_saturation_count.saturating_add(batch_result.mlp_saturation_count);
            attention_saturation_count =
                attention_saturation_count.saturating_add(batch_result.attention_saturation_count);
            residual_saturation_count =
                residual_saturation_count.saturating_add(batch_result.residual_saturation_count);

            let mut candidate_model = model.clone();
            let mut candidate_state = optimizer_state.clone();
            let update = mini_transformer_apply_integer_adam_batch(
                &batch_result,
                &mut candidate_model,
                &mut candidate_state,
                config.position_policy,
                train_scope,
            )?;
            let batch_starts = &starts[batch_start..batch_end];
            let batch_valid = mini_transformer_validate_batch_windows(
                &candidate_model,
                tokens,
                batch_starts,
                config.seq_len,
                config.attention_kind,
                config.position_policy,
            )
            .and_then(|_| {
                mini_transformer_validate_guard_windows(
                    &candidate_model,
                    tokens,
                    &starts,
                    config.seq_len,
                    config.attention_kind,
                    config.position_policy,
                    epoch,
                    batch_end - 1,
                    config.epochs,
                )
            })
            .is_ok();
            let loss_regressed = if batch_valid && config.reject_loss_regression {
                let guard = mini_transformer_loss_guard_starts(&starts, batch_start, batch_end);
                let before = mini_transformer_total_probability_error_q15_with_attention_and_position_policy(
                    tokens,
                    &guard,
                    &model,
                    config.seq_len,
                    config.attention_kind,
                    config.position_policy,
                )?;
                match mini_transformer_total_probability_error_q15_with_attention_and_position_policy(
                    tokens,
                    &guard,
                    &candidate_model,
                    config.seq_len,
                    config.attention_kind,
                    config.position_policy,
                ) {
                    Ok(after) => mini_transformer_loss_guard_regressed(before, after, guard.len()),
                    Err(TrainError::CoreRejected(_)) => true,
                    Err(error) => return Err(error),
                }
            } else {
                false
            };
            if batch_valid && !loss_regressed {
                model = candidate_model;
                optimizer_state = candidate_state;
                updates = updates.saturating_add(batch_result.accepted_window_count);
                accepted_batch_count = accepted_batch_count.saturating_add(1);
                output_head_delta_l1 =
                    output_head_delta_l1.saturating_add(update.output_head.weight_delta_l1);
                mlp_delta_l1 =
                    mlp_delta_l1.saturating_add(update.mlp.weight_delta_l1().unwrap_or(0));
                embedding_delta_l1 =
                    embedding_delta_l1.saturating_add(update.embedding.weight_delta_l1);
                rms_norm_delta_l1 =
                    rms_norm_delta_l1.saturating_add(update.rms_norm.weight_delta_l1);
                attention_delta_l1 =
                    attention_delta_l1.saturating_add(update.attention.weight_delta_l1);
                attention_q_delta_l1 =
                    attention_q_delta_l1.saturating_add(update.attention.q.weight_delta_l1);
                attention_k_delta_l1 =
                    attention_k_delta_l1.saturating_add(update.attention.k.weight_delta_l1);
                attention_v_delta_l1 =
                    attention_v_delta_l1.saturating_add(update.attention.v.weight_delta_l1);
                attention_o_delta_l1 =
                    attention_o_delta_l1.saturating_add(update.attention.o.weight_delta_l1);
            } else {
                rejected_batch_count = rejected_batch_count.saturating_add(1);
            }
            batch_start = batch_end;
        }
    }

    #[cfg(feature = "mini-calibrated")]
    if config.position_policy == MiniTransformerPositionPolicy::Nope
        && train_scope == MiniTransformerAdamTrainScope::All
    {
        mini_transformer_install_ngram_cache(&mut model, tokens)?;
    }
    optimizer_state.bind_to_model(&model)?;
    let final_mistakes = mini_transformer_total_error_with_attention_and_position_policy(
        tokens,
        &starts,
        &model,
        config.seq_len,
        config.attention_kind,
        config.position_policy,
    )?;
    let final_probability_error_q15 =
        mini_transformer_total_probability_error_q15_with_attention_and_position_policy(
            tokens,
            &starts,
            &model,
            config.seq_len,
            config.attention_kind,
            config.position_policy,
        )?;
    let optimizer_state_hash = optimizer_state.state_hash()?;
    Ok(MiniTransformerAdamTrainingRun {
        trace: MiniTransformerAdamTrainingTrace {
            schema: MINI_TRANSFORMER_ADAM_SCHEMA,
            config,
            optimizer_config,
            train_scope,
            token_count: tokens.len(),
            token_hash,
            window_hash,
            windows: starts.len(),
            examined_windows,
            updates,
            accepted_batch_count,
            rejected_batch_count,
            initial_mistakes,
            final_mistakes,
            initial_probability_error_q15,
            final_probability_error_q15,
            transformer_layers: model.transformer_layers(),
            rms_norm_enabled: model.rms_norm_enabled(),
            output_head_delta_l1,
            mlp_delta_l1,
            embedding_delta_l1,
            rms_norm_delta_l1,
            attention_delta_l1,
            attention_q_delta_l1,
            attention_k_delta_l1,
            attention_v_delta_l1,
            attention_o_delta_l1,
            mlp_saturation_count,
            attention_saturation_count,
            residual_saturation_count,
            initial_model_hash,
            final_model_hash: model.model_hash(),
            optimizer_step: optimizer_state.step,
            optimizer_state_hash,
        },
        model,
        optimizer_state,
    })
}

pub fn run_mini_transformer_mlp_training(
    tokens: &[u8],
    config: MiniTransformerMlpTrainConfig,
) -> Result<MiniTransformerMlpTrainingTrace, TrainError> {
    Ok(run_mini_transformer_mlp_training_with_model(tokens, config)?.trace)
}

pub const MINI_TRANSFORMER_EVAL_SCHEMA: &str = "nsrl.mini_transformer_eval.v1";
pub const MINI_TRANSFORMER_ROUTER_HIDDEN_FEATURES: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MiniTransformerMlpEvalConfig {
    pub seq_len: usize,
    pub stride: usize,
    pub max_windows: Option<usize>,
    pub attention_kind: MiniTransformerAttentionKind,
    pub position_policy: MiniTransformerPositionPolicy,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniTransformerMlpEvalTrace {
    pub token_count: usize,
    pub token_hash: u64,
    pub window_hash: u64,
    pub windows: usize,
    pub config: MiniTransformerMlpEvalConfig,
    pub model_hash: u64,
    pub mistakes: usize,
    pub accuracy_per_mille: usize,
    pub probability_error_q15: usize,
    pub mean_probability_error_q15: usize,
    pub invalid_forward_count: usize,
    pub unique_predicted_tokens: usize,
    pub most_predicted_token: Option<u8>,
    pub most_predicted_token_count: usize,
    pub most_predicted_token_share_per_mille: usize,
    pub logits_hash: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniTransformerMlpWindowEvalRecord {
    pub start: usize,
    pub end: usize,
    pub mistakes: usize,
    pub probability_error_q15: usize,
    pub invalid_forward_count: usize,
    pub predicted_token: Option<u8>,
    pub last_hidden_q15: [i16; MINI_TRANSFORMER_D_MODEL],
    pub router_hidden_features_q15: [i16; MINI_TRANSFORMER_ROUTER_HIDDEN_FEATURES],
    pub logits_q8: Option<[i32; BYTE_VOCAB]>,
}

impl MiniTransformerMlpEvalTrace {
    pub fn to_json_line(&self) -> String {
        let mut out = String::new();
        out.push('{');
        push_string_field(&mut out, "schema", MINI_TRANSFORMER_EVAL_SCHEMA);
        comma(&mut out);
        out.push_str("\"data\":{");
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
        push_hash_field(&mut out, "hash", self.model_hash);
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
        out.push_str("\"evaluation\":{");
        push_usize_field(&mut out, "stride", self.config.stride);
        comma(&mut out);
        push_optional_usize_field(&mut out, "max_windows", self.config.max_windows);
        comma(&mut out);
        push_usize_field(&mut out, "mistakes", self.mistakes);
        comma(&mut out);
        push_usize_field(&mut out, "accuracy_per_mille", self.accuracy_per_mille);
        comma(&mut out);
        push_usize_field(
            &mut out,
            "probability_error_q15",
            self.probability_error_q15,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "mean_probability_error_q15",
            self.mean_probability_error_q15,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "invalid_forward_count",
            self.invalid_forward_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "unique_predicted_tokens",
            self.unique_predicted_tokens,
        );
        comma(&mut out);
        push_optional_usize_field(
            &mut out,
            "most_predicted_token",
            self.most_predicted_token.map(usize::from),
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "most_predicted_token_count",
            self.most_predicted_token_count,
        );
        comma(&mut out);
        push_usize_field(
            &mut out,
            "most_predicted_token_share_per_mille",
            self.most_predicted_token_share_per_mille,
        );
        comma(&mut out);
        push_hash_field(&mut out, "logits_hash", self.logits_hash);
        out.push_str("}}\n");
        out
    }
}

pub fn evaluate_mini_transformer_mlp_model(
    tokens: &[u8],
    model: &MiniTransformerMlpModel,
    config: MiniTransformerMlpEvalConfig,
) -> Result<MiniTransformerMlpEvalTrace, TrainError> {
    if config.seq_len == 0
        || config.stride == 0
        || model.context_seq_len != config.seq_len
        || matches!(
            config.attention_kind,
            MiniTransformerAttentionKind::LinearStreamingNope
                | MiniTransformerAttentionKind::LinearStreamingTttNope
        )
    {
        return Err(TrainError::InvalidConfig);
    }
    let starts_config = MiniTransformerMlpTrainConfig {
        seq_len: config.seq_len,
        stride: config.stride,
        max_windows: config.max_windows,
        attention_kind: config.attention_kind,
        position_policy: config.position_policy,
        ..MiniTransformerMlpTrainConfig::default()
    };
    let starts = mini_transformer_filtered_window_starts(tokens.len(), tokens, starts_config);
    if starts.is_empty() {
        return Err(TrainError::InvalidConfig);
    }
    let summary = mini_transformer_eval_summary_with_attention_and_position_policy(
        tokens,
        &starts,
        model,
        config.seq_len,
        config.attention_kind,
        config.position_policy,
    )?;
    let windows = starts.len();
    Ok(MiniTransformerMlpEvalTrace {
        token_count: tokens.len(),
        token_hash: hash_u8_slice(tokens),
        window_hash: hash_mini_transformer_windows(tokens, starts_config, &starts),
        windows,
        config,
        model_hash: model.model_hash(),
        mistakes: summary.mistakes,
        accuracy_per_mille: windows.saturating_sub(summary.mistakes) * 1000 / windows,
        probability_error_q15: summary.probability_error_q15,
        mean_probability_error_q15: summary.probability_error_q15 / windows,
        invalid_forward_count: summary.invalid_forward_count,
        unique_predicted_tokens: summary.unique_predicted_tokens,
        most_predicted_token: summary.most_predicted_token,
        most_predicted_token_count: summary.most_predicted_token_count,
        most_predicted_token_share_per_mille: summary
            .most_predicted_token_count
            .saturating_mul(1000)
            / windows,
        logits_hash: summary.logits_hash,
    })
}

pub fn evaluate_mini_transformer_mlp_windows(
    tokens: &[u8],
    model: &MiniTransformerMlpModel,
    config: MiniTransformerMlpEvalConfig,
) -> Result<Vec<MiniTransformerMlpWindowEvalRecord>, TrainError> {
    if config.seq_len == 0
        || config.stride == 0
        || model.context_seq_len != config.seq_len
        || matches!(
            config.attention_kind,
            MiniTransformerAttentionKind::LinearStreamingNope
                | MiniTransformerAttentionKind::LinearStreamingTttNope
        )
    {
        return Err(TrainError::InvalidConfig);
    }
    let starts_config = MiniTransformerMlpTrainConfig {
        seq_len: config.seq_len,
        stride: config.stride,
        max_windows: config.max_windows,
        attention_kind: config.attention_kind,
        position_policy: config.position_policy,
        ..MiniTransformerMlpTrainConfig::default()
    };
    let starts = mini_transformer_filtered_window_starts(tokens.len(), tokens, starts_config);
    if starts.is_empty() {
        return Err(TrainError::InvalidConfig);
    }
    mini_transformer_window_eval_records_with_attention_and_position_policy(
        tokens,
        &starts,
        model,
        config.seq_len,
        config.attention_kind,
        config.position_policy,
    )
}

pub fn run_mini_transformer_mlp_training_with_model(
    tokens: &[u8],
    config: MiniTransformerMlpTrainConfig,
) -> Result<MiniTransformerMlpTrainingRun, TrainError> {
    let model = MiniTransformerMlpModel::new_initial_with_seq_len(config.seq_len);
    run_mini_transformer_mlp_training_from_model(tokens, config, model)
}

pub fn run_mini_transformer_mlp_training_from_model(
    tokens: &[u8],
    config: MiniTransformerMlpTrainConfig,
    model: MiniTransformerMlpModel,
) -> Result<MiniTransformerMlpTrainingRun, TrainError> {
    run_mini_transformer_mlp_training_from_model_with_progress(tokens, config, model, 0, |_| Ok(()))
}

pub fn run_mini_transformer_mlp_training_from_model_with_progress<F>(
    tokens: &[u8],
    config: MiniTransformerMlpTrainConfig,
    model: MiniTransformerMlpModel,
    progress_interval_batches: usize,
    progress: F,
) -> Result<MiniTransformerMlpTrainingRun, TrainError>
where
    F: FnMut(&MiniTransformerMlpTrainingProgressTrace) -> Result<(), TrainError>,
{
    run_mini_transformer_mlp_training_from_model_with_progress_and_trace_detail(
        tokens,
        config,
        model,
        progress_interval_batches,
        MiniTransformerTraceDetail::Full,
        progress,
    )
}

pub fn run_mini_transformer_mlp_training_from_model_with_progress_and_trace_detail<F>(
    tokens: &[u8],
    config: MiniTransformerMlpTrainConfig,
    model: MiniTransformerMlpModel,
    progress_interval_batches: usize,
    trace_detail: MiniTransformerTraceDetail,
    progress: F,
) -> Result<MiniTransformerMlpTrainingRun, TrainError>
where
    F: FnMut(&MiniTransformerMlpTrainingProgressTrace) -> Result<(), TrainError>,
{
    run_mini_transformer_mlp_training_from_model_with_progress_trace_detail_and_binary_trace(
        tokens,
        config,
        model,
        progress_interval_batches,
        trace_detail,
        progress,
        |_| Ok(()),
    )
}

pub fn run_mini_transformer_mlp_training_from_model_with_progress_trace_detail_and_binary_trace<
    F,
    G,
>(
    tokens: &[u8],
    config: MiniTransformerMlpTrainConfig,
    mut model: MiniTransformerMlpModel,
    progress_interval_batches: usize,
    trace_detail: MiniTransformerTraceDetail,
    mut progress: F,
    mut binary_trace: G,
) -> Result<MiniTransformerMlpTrainingRun, TrainError>
where
    F: FnMut(&MiniTransformerMlpTrainingProgressTrace) -> Result<(), TrainError>,
    G: FnMut(MiniTransformerBinaryTraceRecord<'_>) -> Result<(), TrainError>,
{
    if config.epochs == 0
        || config.seq_len == 0
        || config.stride == 0
        || config.batch_windows == 0
        || config.learning_rate <= 0
        || config.target_token_min > config.target_token_max
        || !valid_mini_transformer_target_segment(config.target_segment)
        || !valid_q15_weight_floor(config.target_frequency_min_weight_q15)
        || config.argmax_margin_weight_q15 < 0
        || config.output_learning_rate_shift > MAX_RIGHT_SHIFT
        || config.mlp_learning_rate_shift > MAX_RIGHT_SHIFT
        || config.embedding_learning_rate_shift > MAX_RIGHT_SHIFT
        || config.attention_learning_rate_shift > MAX_RIGHT_SHIFT
        || config.attention_q_learning_rate_shift > MAX_RIGHT_SHIFT
        || config.attention_qk_learning_rate_shift > MAX_RIGHT_SHIFT
        || config.attention_kind == MiniTransformerAttentionKind::LinearStreamingNope
        || config.attention_kind == MiniTransformerAttentionKind::LinearStreamingTttNope
        || (config.attention_vo_oracle && config.batch_windows <= 1)
        || (config.attention_vo_oracle
            && MINI_TRANSFORMER_D_MODEL > MINI_TRANSFORMER_ATTENTION_VO_ORACLE_MAX_D_MODEL)
    {
        return Err(TrainError::InvalidConfig);
    }
    validate_mini_transformer_batch_mode(config)?;
    validate_mini_transformer_effective_learning_rate_shifts(config)?;
    if model.context_seq_len != config.seq_len {
        return Err(TrainError::InvalidConfig);
    }
    if model.rms_norm_enabled() {
        return Err(TrainError::InvalidConfig);
    }
    let transformer_layers = model.checked_transformer_layers()?;
    let use_stacked_serial_backprop =
        transformer_layers > 1 && config.batch_mode == MiniTransformerBatchMode::Serial;

    let starts = mini_transformer_filtered_window_starts(tokens.len(), tokens, config);
    if starts.is_empty() {
        return Err(TrainError::InvalidConfig);
    }

    let token_hash = hash_u8_slice(tokens);
    let window_hash = hash_mini_transformer_windows(tokens, config, &starts);
    let target_frequency_weights_q15 = byte_target_frequency_weights_q15(
        tokens,
        &starts,
        config.seq_len,
        config.target_frequency_cap,
        config.target_frequency_min_weight_q15,
    )?;
    let initial_model_hash = model.model_hash();
    let initial_embedding_hash = model.embedding_hash();
    let initial_output_head_hash = model.output_head_hash();
    let initial_mlp_hash = model.mlp_hash();
    let initial_attention_hash = model.attention_hash();
    let initial_attention_q_hash = model.attention_q_hash();
    let initial_attention_k_hash = model.attention_k_hash();
    let initial_attention_v_hash = model.attention_v_hash();
    let initial_attention_o_hash = model.attention_o_hash();
    let initial_total_error = mini_transformer_total_error_with_attention_and_position_policy(
        tokens,
        &starts,
        &model,
        config.seq_len,
        config.attention_kind,
        config.position_policy,
    )?;
    let initial_probability_error_q15 =
        mini_transformer_total_probability_error_q15_with_attention_and_position_policy(
            tokens,
            &starts,
            &model,
            config.seq_len,
            config.attention_kind,
            config.position_policy,
        )?;
    let initial_mistakes = initial_total_error;
    binary_trace(MiniTransformerBinaryTraceRecord::Header { initial_model_hash })?;
    let mut updates = 0_usize;
    let mut examined_windows = 0_usize;
    let mut accepted_batch_count = 0_usize;
    let mut rejected_batch_count = 0_usize;
    let mut output_head_accumulator_batch_count = 0_usize;
    let mut output_head_accumulator_window_count = 0_usize;
    let mut mlp_accumulator_batch_count = 0_usize;
    let mut mlp_accumulator_window_count = 0_usize;
    let mut attention_accumulator_batch_count = 0_usize;
    let mut attention_accumulator_window_count = 0_usize;
    let mut embedding_accumulator_batch_count = 0_usize;
    let mut embedding_accumulator_window_count = 0_usize;
    let mut rollback_count = 0_usize;
    let mut rejected_window_count = 0_usize;
    let mut loss_regression_rejected_batch_count = 0_usize;
    let mut output_head_saturation_count = 0_usize;
    let mut mlp_saturation_count = 0_usize;
    let mut embedding_saturation_count = 0_usize;
    let mut attention_saturation_count = 0_usize;
    let mut residual_saturation_count = 0_usize;
    let mut output_head_zero_delta_count = 0_usize;
    let mut mlp_zero_delta_count = 0_usize;
    let mut embedding_zero_delta_count = 0_usize;
    let mut attention_zero_delta_count = 0_usize;
    let mut output_head_delta_l1 = 0_u64;
    let mut mlp_delta_l1 = 0_u64;
    let mut embedding_delta_l1 = 0_u64;
    let mut attention_delta_l1 = 0_u64;
    let mut attention_q_delta_l1 = 0_u64;
    let mut attention_k_delta_l1 = 0_u64;
    let mut attention_v_delta_l1 = 0_u64;
    let mut attention_o_delta_l1 = 0_u64;
    let mut output_head_carry_l1 = 0_u64;
    let mut mlp_carry_l1 = 0_u64;
    let mut embedding_carry_l1 = 0_u64;
    let mut attention_carry_l1 = 0_u64;
    let mut attention_q_carry_l1 = 0_u64;
    let mut attention_k_carry_l1 = 0_u64;
    let mut attention_v_carry_l1 = 0_u64;
    let mut attention_o_carry_l1 = 0_u64;
    let mut steps = Vec::new();
    let trace_sample_interval =
        mini_transformer_trace_sample_interval(progress_interval_batches, config.batch_windows);
    let mut rollback_history = vec![model.clone()];
    let mut output_head_gradient =
        LinearWeightGradientI64::new(MINI_TRANSFORMER_D_MODEL, BYTE_VOCAB)
            .ok_or(TrainError::InvalidConfig)?;
    let mut mlp_weight_gradients =
        mini_transformer_new_gated_mlp_weight_gradients(transformer_layers)?;
    let mut attention_weight_gradients =
        mini_transformer_new_attention_weight_gradients(transformer_layers)?;
    let mut embedding_gradient = MiniTransformerEmbeddingGradientI64::new(config.seq_len)
        .ok_or(TrainError::InvalidConfig)?;
    let mut adaptive_attention_shifts = MiniTransformerAdaptiveShiftState::new(config);
    let mut adaptive_shift_events = Vec::new();
    let adaptive_shift_controller_enabled = config.adaptive_shift_controller_enabled();
    let use_output_head_accumulator = config.batch_windows > 1 && !use_stacked_serial_backprop;
    let use_mlp_accumulator = config.batch_windows > 1 && !use_stacked_serial_backprop;
    let use_attention_accumulator = config.batch_windows > 1 && !use_stacked_serial_backprop;
    let use_embedding_accumulator = config.batch_windows > 1 && !use_stacked_serial_backprop;
    let use_train_core_step = mini_transformer_uses_train_core_step_for_model(config, &model);
    let mut train_core_workspace = if use_train_core_step {
        Some(MiniTransformerHostTrainCoreWorkspaceBuffers::new(
            config.seq_len,
        )?)
    } else {
        None
    };
    let mut host_training_workspace = if use_train_core_step {
        None
    } else {
        Some(MiniTransformerHostTrainCoreWorkspaceBuffers::new(
            config.seq_len,
        )?)
    };
    if progress_interval_batches > 0 {
        progress(&mini_transformer_training_progress_trace(
            config,
            tokens.len(),
            token_hash,
            window_hash,
            starts.len(),
            examined_windows,
            updates,
            accepted_batch_count,
            rejected_batch_count,
            rollback_count,
            rejected_window_count,
            output_head_delta_l1,
            mlp_delta_l1,
            embedding_delta_l1,
            attention_delta_l1,
            attention_q_delta_l1,
            attention_k_delta_l1,
            attention_v_delta_l1,
            attention_o_delta_l1,
            output_head_carry_l1,
            mlp_carry_l1,
            embedding_carry_l1,
            attention_carry_l1,
            attention_q_carry_l1,
            attention_k_carry_l1,
            attention_v_carry_l1,
            attention_o_carry_l1,
            &adaptive_attention_shifts,
            &model,
        ))?;
    }

    for epoch in 0..config.epochs {
        let mut batch_start_index = 0_usize;
        while batch_start_index < starts.len() {
            let batch_end_index = batch_start_index
                .saturating_add(config.batch_windows)
                .min(starts.len());
            let batch_model_checkpoint = model.clone();
            let updates_before_batch = updates;
            let steps_before_batch = steps.len();
            let rollbacks_before_batch = rollback_count;

            if config.batch_mode == MiniTransformerBatchMode::MapReduce {
                let batch_window_count = batch_end_index.saturating_sub(batch_start_index);
                examined_windows = examined_windows.saturating_add(batch_window_count);
                output_head_gradient.clear();
                mini_transformer_clear_gated_mlp_weight_gradient_i64_layers(
                    &mut mlp_weight_gradients,
                );
                mini_transformer_clear_attention_weight_gradient_i64_layers(
                    &mut attention_weight_gradients,
                );
                embedding_gradient.clear();

                match mini_transformer_map_reduce_batch(
                    tokens,
                    &starts,
                    &target_frequency_weights_q15,
                    batch_start_index,
                    batch_end_index,
                    epoch,
                    &model,
                    config,
                    updates_before_batch,
                    trace_detail,
                    trace_sample_interval,
                ) {
                    Ok(batch_result) => {
                        mini_transformer_merge_linear_weight_gradient_i64(
                            &mut output_head_gradient,
                            &batch_result.output_head_gradient,
                        )?;
                        mini_transformer_merge_gated_mlp_weight_gradient_i64_layers(
                            &mut mlp_weight_gradients,
                            &batch_result.mlp_weight_gradients,
                        )?;
                        mini_transformer_merge_attention_weight_gradient_i64_layers(
                            &mut attention_weight_gradients,
                            &batch_result.attention_weight_gradients,
                        )?;
                        mini_transformer_merge_embedding_gradient_i64(
                            &mut embedding_gradient,
                            &batch_result.embedding_gradient,
                        )?;
                        updates = updates.saturating_add(batch_result.accepted_window_count);
                        mlp_saturation_count =
                            mlp_saturation_count.saturating_add(batch_result.mlp_saturation_count);
                        attention_saturation_count = attention_saturation_count
                            .saturating_add(batch_result.attention_saturation_count);
                        residual_saturation_count = residual_saturation_count
                            .saturating_add(batch_result.residual_saturation_count);
                        steps.extend(batch_result.steps);
                    }
                    Err(TrainError::CoreRejected(_)) => {
                        rejected_window_count =
                            rejected_window_count.saturating_add(batch_window_count);
                    }
                    Err(error) => return Err(error),
                }
            } else {
                for (relative_window_index, &window_start) in starts
                    [batch_start_index..batch_end_index]
                    .iter()
                    .enumerate()
                {
                    let window_index = batch_start_index + relative_window_index;
                    examined_windows += 1;
                    let target_token = tokens[window_start + config.seq_len];
                    let cache_before = match mini_transformer_forward_for_attention_and_position(
                        &model,
                        &tokens[window_start..window_start + config.seq_len],
                        config.attention_kind,
                        config.position_policy,
                    ) {
                        Ok(cache) => cache,
                        Err(_) => {
                            let mut recovered = None;
                            for checkpoint in rollback_history.iter().rev() {
                                if let Ok(cache) =
                                    mini_transformer_forward_for_attention_and_position(
                                        checkpoint,
                                        &tokens[window_start..window_start + config.seq_len],
                                        config.attention_kind,
                                        config.position_policy,
                                    )
                                {
                                    recovered = Some((checkpoint.clone(), cache));
                                    break;
                                }
                            }

                            match recovered {
                                Some((checkpoint, cache)) => {
                                    model = checkpoint;
                                    rollback_count = rollback_count.saturating_add(1);
                                    rejected_window_count = rejected_window_count.saturating_add(1);
                                    cache
                                }
                                None => {
                                    rejected_window_count = rejected_window_count.saturating_add(1);
                                    adaptive_attention_shifts.observe_rejected(
                                        rejected_batch_count.saturating_add(rejected_window_count),
                                        adaptive_shift_controller_enabled,
                                        config,
                                        &mut adaptive_shift_events,
                                    );
                                    continue;
                                }
                            }
                        }
                    };
                    let should_record_step = mini_transformer_should_record_step(
                        trace_detail,
                        updates.saturating_add(1),
                        trace_sample_interval,
                    );
                    let predicted_token_before = if should_record_step {
                        byte_argmax_i32(&cache_before.logits_q8)
                    } else {
                        0
                    };
                    let mut gradient_q15 = byte_vocab_softmax_gradient_q15(
                        &cache_before.probabilities_q15,
                        target_token,
                    );
                    apply_byte_argmax_margin_gradient_q15(
                        &mut gradient_q15,
                        &cache_before.logits_q8,
                        target_token,
                        config.argmax_margin_weight_q15,
                    );
                    let target_frequency_weight_q15 =
                        target_frequency_weights_q15[usize::from(target_token)];
                    let weighted_gradient_q15 =
                        byte_scale_gradient_q15(&gradient_q15, target_frequency_weight_q15);
                    let grad_output_q15 = byte_gradient_i32_to_i16(&weighted_gradient_q15);
                    let output_head_hash_before = if should_record_step {
                        model.output_head_hash()
                    } else {
                        0
                    };
                    let mlp_hash_before = if should_record_step {
                        model.mlp_hash()
                    } else {
                        0
                    };
                    let attention_hash_before = if should_record_step {
                        model.attention_hash()
                    } else {
                        0
                    };
                    let embedding_hash_before = if should_record_step {
                        model.embedding_hash()
                    } else {
                        0
                    };
                    let model_checkpoint = model.clone();
                    rollback_history.push(model_checkpoint.clone());
                    if rollback_history.len() > MINI_TRANSFORMER_ROLLBACK_HISTORY_LIMIT {
                        rollback_history.remove(0);
                    }

                    if use_train_core_step {
                        let core_stats = {
                            let mut model_slices = nsrl_train_core::MiniTransformerModelSlicesMut {
                                embeddings: &mut model.embeddings,
                                q_weights: &mut model.q_weights,
                                k_weights: &mut model.k_weights,
                                v_weights: &mut model.v_weights,
                                o_weights: &mut model.o_weights,
                                up_weights: &mut model.up_weights,
                                gate_weights: &mut model.gate_weights,
                                down_weights: &mut model.down_weights,
                                output_weights: &mut model.output_weights,
                            };
                            let workspace_buffers = train_core_workspace
                                .as_mut()
                                .ok_or(TrainError::InvalidConfig)?;
                            let mut workspace = workspace_buffers.as_workspace();
                            nsrl_train_core::mini_transformer_linear_nope_train_step(
                                &mut model_slices,
                                &tokens[window_start..window_start + config.seq_len],
                                target_token,
                                nsrl_train_core::MiniTransformerStepConfig {
                                    seq_len: config.seq_len,
                                    learning_rate: config.learning_rate,
                                    output_learning_rate_shift: config.output_learning_rate_shift,
                                    mlp_learning_rate_shift: config.mlp_learning_rate_shift,
                                    embedding_learning_rate_shift: config
                                        .embedding_learning_rate_shift,
                                    attention_learning_rate_shift: config
                                        .attention_learning_rate_shift,
                                    attention_q_learning_rate_shift: config
                                        .attention_q_learning_rate_shift,
                                    attention_qk_learning_rate_shift: config
                                        .attention_qk_learning_rate_shift,
                                },
                                &mut workspace,
                            )
                        };
                        let core_stats = match core_stats {
                            Ok(stats) => stats,
                            Err(_) => {
                                model = model_checkpoint;
                                rollback_count = rollback_count.saturating_add(1);
                                rejected_window_count = rejected_window_count.saturating_add(1);
                                adaptive_attention_shifts.observe_rejected(
                                    rejected_batch_count.saturating_add(rejected_window_count),
                                    adaptive_shift_controller_enabled,
                                    config,
                                    &mut adaptive_shift_events,
                                );
                                continue;
                            }
                        };

                        let cache_after = match mini_transformer_forward_for_attention_and_position(
                            &model,
                            &tokens[window_start..window_start + config.seq_len],
                            config.attention_kind,
                            config.position_policy,
                        ) {
                            Ok(cache) => cache,
                            Err(_) => {
                                model = model_checkpoint;
                                rollback_count = rollback_count.saturating_add(1);
                                rejected_window_count = rejected_window_count.saturating_add(1);
                                adaptive_attention_shifts.observe_rejected(
                                    rejected_batch_count.saturating_add(rejected_window_count),
                                    adaptive_shift_controller_enabled,
                                    config,
                                    &mut adaptive_shift_events,
                                );
                                continue;
                            }
                        };

                        if mini_transformer_validate_guard_windows(
                            &model,
                            tokens,
                            &starts,
                            config.seq_len,
                            config.attention_kind,
                            config.position_policy,
                            epoch,
                            window_index,
                            config.epochs,
                        )
                        .is_err()
                        {
                            model = model_checkpoint;
                            rollback_count = rollback_count.saturating_add(1);
                            rejected_window_count = rejected_window_count.saturating_add(1);
                            adaptive_attention_shifts.observe_rejected(
                                rejected_batch_count.saturating_add(rejected_window_count),
                                adaptive_shift_controller_enabled,
                                config,
                                &mut adaptive_shift_events,
                            );
                            continue;
                        }

                        let predicted_token_after = if should_record_step {
                            byte_argmax_i32(&cache_after.logits_q8)
                        } else {
                            0
                        };
                        let output_head_hash_after = if should_record_step {
                            model.output_head_hash()
                        } else {
                            0
                        };
                        let mlp_hash_after = if should_record_step {
                            model.mlp_hash()
                        } else {
                            0
                        };
                        let attention_hash_after = if should_record_step {
                            model.attention_hash()
                        } else {
                            0
                        };
                        let embedding_hash_after = if should_record_step {
                            model.embedding_hash()
                        } else {
                            0
                        };

                        updates += 1;
                        output_head_saturation_count +=
                            core_stats.output_head.gradient_saturation_count;
                        output_head_zero_delta_count += core_stats.output_head.zero_delta_count;
                        output_head_delta_l1 = output_head_delta_l1
                            .saturating_add(core_stats.output_head.weight_delta_l1);
                        mlp_saturation_count += core_stats.mlp.gradient_saturation_count();
                        mlp_zero_delta_count += core_stats.mlp.zero_delta_count();
                        mlp_delta_l1 =
                            mlp_delta_l1.saturating_add(core_stats.mlp.weight_delta_l1());
                        embedding_saturation_count +=
                            core_stats.embedding.gradient_saturation_count;
                        embedding_zero_delta_count += core_stats.embedding.zero_delta_count;
                        embedding_delta_l1 =
                            embedding_delta_l1.saturating_add(core_stats.embedding.weight_delta_l1);
                        attention_saturation_count +=
                            core_stats.attention.gradient_saturation_count();
                        attention_zero_delta_count += core_stats.attention.zero_delta_count();
                        attention_delta_l1 = attention_delta_l1
                            .saturating_add(core_stats.attention.weight_delta_l1());
                        attention_q_delta_l1 = attention_q_delta_l1
                            .saturating_add(core_stats.attention.q.weight_delta_l1);
                        attention_k_delta_l1 = attention_k_delta_l1
                            .saturating_add(core_stats.attention.k.weight_delta_l1);
                        attention_v_delta_l1 = attention_v_delta_l1
                            .saturating_add(core_stats.attention.v.weight_delta_l1);
                        attention_o_delta_l1 = attention_o_delta_l1
                            .saturating_add(core_stats.attention.o.weight_delta_l1);
                        residual_saturation_count = residual_saturation_count
                            .saturating_add(core_stats.residual_saturation_count);

                        if should_record_step {
                            steps.push(MiniTransformerMlpTrainingStepTrace {
                                update_index: updates,
                                epoch,
                                window_index,
                                window_start,
                                first_token: tokens[window_start],
                                last_token: tokens[window_start + config.seq_len - 1],
                                target_token,
                                predicted_token_before,
                                predicted_token_after,
                                target_probability_before_q15: cache_before.probabilities_q15
                                    [usize::from(target_token)],
                                target_probability_after_q15: cache_after.probabilities_q15
                                    [usize::from(target_token)],
                                embedding_cache_hash: hash_i16_slice(
                                    &cache_before.embedding_output,
                                ),
                                attention_cache_hash: hash_i16_slice(
                                    &cache_before.attention_output,
                                ),
                                mlp_cache_hash: hash_i16_slice(&cache_before.mlp_gated),
                                block_output_hash_before: hash_i16_slice(
                                    &cache_before.block_output,
                                ),
                                block_output_hash_after: hash_i16_slice(&cache_after.block_output),
                                output_head_hash_before,
                                output_head_hash_after,
                                mlp_hash_before,
                                mlp_hash_after,
                                attention_hash_before,
                                attention_hash_after,
                                embedding_hash_before,
                                embedding_hash_after,
                                output_head_saturation_count: core_stats
                                    .output_head
                                    .gradient_saturation_count,
                                mlp_saturation_count: core_stats.mlp.gradient_saturation_count(),
                                embedding_saturation_count: core_stats
                                    .embedding
                                    .gradient_saturation_count,
                                attention_saturation_count: core_stats
                                    .attention
                                    .gradient_saturation_count(),
                                residual_saturation_count: core_stats.residual_saturation_count,
                                output_head_zero_delta_count: core_stats
                                    .output_head
                                    .zero_delta_count,
                                mlp_zero_delta_count: core_stats.mlp.zero_delta_count(),
                                embedding_zero_delta_count: core_stats.embedding.zero_delta_count,
                                attention_zero_delta_count: core_stats.attention.zero_delta_count(),
                                output_head_delta_l1: core_stats.output_head.weight_delta_l1,
                                mlp_delta_l1: core_stats.mlp.weight_delta_l1(),
                                embedding_delta_l1: core_stats.embedding.weight_delta_l1,
                                attention_delta_l1: core_stats.attention.weight_delta_l1(),
                                attention_q_delta_l1: core_stats.attention.q.weight_delta_l1,
                                attention_k_delta_l1: core_stats.attention.k.weight_delta_l1,
                                attention_v_delta_l1: core_stats.attention.v.weight_delta_l1,
                                attention_o_delta_l1: core_stats.attention.o.weight_delta_l1,
                            });
                        }
                        continue;
                    }

                    let workspace = host_training_workspace
                        .as_mut()
                        .ok_or(TrainError::InvalidConfig)?;
                    workspace.reset_host_training_step();
                    linear_backward_input_i16_i8_i16_per_channel_checked(
                        &grad_output_q15,
                        LinearBackwardInputI16I8Params {
                            weights: &model.output_weights,
                            forward_scales: &MINI_TRANSFORMER_OUTPUT_SCALES,
                            grad_input_scales: &MINI_TRANSFORMER_OUTPUT_GRAD_INPUT_SCALES,
                            input_dim: MINI_TRANSFORMER_D_MODEL,
                            output_dim: BYTE_VOCAB,
                        },
                        LinearBackwardInputWorkspace {
                            scaled_grad_output: &mut workspace.output_scaled_grad,
                        },
                        &mut workspace.grad_last_features,
                    )
                    .ok_or(TrainError::CoreRejected(
                        "mini_transformer_output_head_backward_input",
                    ))?;

                    let last_start = (config.seq_len - 1) * MINI_TRANSFORMER_D_MODEL;
                    let last_end = last_start + MINI_TRANSFORMER_D_MODEL;
                    let runtime_config = adaptive_attention_shifts.runtime_config(config);
                    let output_update = if use_output_head_accumulator {
                        empty_linear_weight_update_stats()
                    } else {
                        linear_backward_weight_update_i8_checked(
                            &cache_before.output_features,
                            &grad_output_q15,
                            &mut model.output_weights,
                            LinearBackwardWeightUpdateI8Params {
                                forward_scales: &MINI_TRANSFORMER_OUTPUT_SCALES,
                                input_dim: MINI_TRANSFORMER_D_MODEL,
                                output_dim: BYTE_VOCAB,
                                learning_rate: config.learning_rate,
                                learning_rate_shift: runtime_config.output_learning_rate_shift,
                            },
                            LinearBackwardWeightUpdateWorkspace {
                                scaled_grad_output: &mut workspace.output_scaled_grad,
                            },
                        )
                        .ok_or(TrainError::CoreRejected(
                            "mini_transformer_output_head_update",
                        ))?
                    };

                    let total = config
                        .seq_len
                        .checked_mul(MINI_TRANSFORMER_D_MODEL)
                        .ok_or(TrainError::InvalidConfig)?;
                    let (
                        mlp_input_saturation_count,
                        gradient_residual_saturation_count,
                        mlp_update,
                        attention_update,
                        embedding_gradient_saturation_count,
                        embedding_update,
                    ) = if use_stacked_serial_backprop {
                        let mut grad_block_output = vec![0_i16; total];
                        grad_block_output[last_start..last_end]
                            .copy_from_slice(&workspace.grad_last_features);
                        let mut stacked_mlp_update = empty_gated_mlp_weight_update_stats();
                        let mut stacked_attention_update =
                            empty_mini_transformer_attention_weight_update_stats();
                        let mut stacked_mlp_input_saturation_count = 0_usize;
                        let mut stacked_gradient_residual_saturation_count = 0_usize;
                        let mut stacked_input_gradient_saturation_count = 0_usize;
                        let mut stacked_rejected = false;

                        for layer_index in (0..cache_before.layers.len()).rev() {
                            let layer_runtime_config =
                                mini_transformer_stacked_layer_runtime_config(
                                    runtime_config,
                                    layer_index,
                                    cache_before.layers.len(),
                                );
                            let block_update =
                                match mini_transformer_block_backward_update_i8_checked(
                                    &cache_before.layers[layer_index],
                                    &grad_block_output,
                                    &mut model,
                                    layer_index,
                                    layer_runtime_config,
                                    workspace,
                                ) {
                                    Ok(update) => update,
                                    Err(TrainError::CoreRejected(_)) => {
                                        model = model_checkpoint.clone();
                                        rollback_count = rollback_count.saturating_add(1);
                                        rejected_window_count =
                                            rejected_window_count.saturating_add(1);
                                        adaptive_attention_shifts.observe_rejected(
                                            rejected_batch_count
                                                .saturating_add(rejected_window_count),
                                            adaptive_shift_controller_enabled,
                                            config,
                                            &mut adaptive_shift_events,
                                        );
                                        stacked_rejected = true;
                                        break;
                                    }
                                    Err(error) => return Err(error),
                                };
                            add_gated_mlp_weight_update_stats_checked(
                                &mut stacked_mlp_update,
                                block_update.mlp_update,
                            )?;
                            add_mini_transformer_attention_weight_update_stats_checked(
                                &mut stacked_attention_update,
                                block_update.attention_update,
                            )?;
                            stacked_mlp_input_saturation_count = stacked_mlp_input_saturation_count
                                .saturating_add(block_update.mlp_input_saturation_count);
                            stacked_gradient_residual_saturation_count =
                                stacked_gradient_residual_saturation_count.saturating_add(
                                    block_update.gradient_residual_saturation_count,
                                );
                            stacked_input_gradient_saturation_count =
                                stacked_input_gradient_saturation_count
                                    .saturating_add(block_update.input_gradient_saturation_count);
                            grad_block_output = block_update.grad_input;
                        }
                        if stacked_rejected {
                            continue;
                        }

                        workspace.grad_embedding_output[..total]
                            .copy_from_slice(&grad_block_output);
                        let stacked_embedding_learning_rate_shift = runtime_config
                            .embedding_learning_rate_shift
                            .saturating_add(
                                MINI_TRANSFORMER_STACKED_EMBEDDING_LEARNING_RATE_EXTRA_SHIFT,
                            )
                            .min(MAX_RIGHT_SHIFT);
                        let embedding_update =
                            apply_mini_transformer_embedding_update_with_position_policy(
                                &mut model.embeddings,
                                &mut model.position_embeddings,
                                &tokens[window_start..window_start + config.seq_len],
                                &workspace.grad_embedding_output,
                                config.position_policy,
                                config.learning_rate,
                                stacked_embedding_learning_rate_shift,
                            )?;
                        (
                            stacked_mlp_input_saturation_count,
                            stacked_gradient_residual_saturation_count,
                            stacked_mlp_update,
                            stacked_attention_update,
                            stacked_input_gradient_saturation_count,
                            embedding_update,
                        )
                    } else {
                        workspace.grad_mlp_output[last_start..last_end]
                            .copy_from_slice(&workspace.grad_last_features);
                        let mlp_input_saturation_count = gated_mlp_backward_input_i16_q15_checked(
                            &workspace.grad_mlp_output,
                            mini_transformer_final_mlp_params(&model, config.seq_len)?,
                            &cache_before.mlp_up,
                            &cache_before.mlp_gate,
                            GatedMlpBackwardScales {
                                down_to_hidden: &MINI_TRANSFORMER_HIDDEN_GRAD_INPUT_SCALES,
                                up_to_input: &MINI_TRANSFORMER_D_MODEL_GRAD_INPUT_SCALES,
                                gate_to_input: &MINI_TRANSFORMER_D_MODEL_GRAD_INPUT_SCALES,
                            },
                            GatedMlpBackwardWorkspace {
                                scaled_grad_output: &mut workspace.mlp_scaled_grad,
                                grad_gated: &mut workspace.mlp_input_grad_gated,
                                grad_up: &mut workspace.mlp_input_grad_up,
                                grad_gate: &mut workspace.mlp_input_grad_gate,
                                grad_up_input: &mut workspace.mlp_input_grad_up_input,
                                grad_gate_input: &mut workspace.mlp_input_grad_gate_input,
                            },
                            &mut workspace.grad_mlp_input,
                        )
                        .ok_or(TrainError::CoreRejected(
                            "mini_transformer_mlp_backward_input",
                        ))?;

                        let gradient_residual_saturation_count = add_i16_residual_rows_checked(
                            &workspace.grad_mlp_output,
                            &workspace.grad_mlp_input,
                            &mut workspace.grad_attention_output,
                        )?;

                        let mlp_update = if use_mlp_accumulator {
                            empty_gated_mlp_weight_update_stats()
                        } else {
                            let up_or_gate_range = model.final_mlp_up_or_gate_weight_range()?;
                            let down_range = model.final_mlp_down_weight_range()?;
                            gated_mlp_backward_weight_update_i8_checked(
                                &cache_before.mlp_norm,
                                &workspace.grad_mlp_output,
                                &cache_before.mlp_up,
                                &cache_before.mlp_gate,
                                &cache_before.mlp_gated,
                                &mut model.up_weights[up_or_gate_range.clone()],
                                &mut model.gate_weights[up_or_gate_range],
                                &mut model.down_weights[down_range],
                                GatedMlpWeightUpdateParams {
                                    up_scales: &MINI_TRANSFORMER_HIDDEN_SCALES,
                                    gate_scales: &MINI_TRANSFORMER_HIDDEN_SCALES,
                                    down_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                                    down_to_hidden_scales:
                                        &MINI_TRANSFORMER_HIDDEN_GRAD_INPUT_SCALES,
                                    seq_len: config.seq_len,
                                    d_model: MINI_TRANSFORMER_D_MODEL,
                                    hidden_dim: MINI_TRANSFORMER_HIDDEN_DIM,
                                    learning_rate: config.learning_rate,
                                    learning_rate_shift: runtime_config.mlp_learning_rate_shift,
                                },
                                GatedMlpWeightUpdateWorkspace {
                                    scaled_grad_output: &mut workspace.mlp_scaled_grad,
                                    grad_gated: &mut workspace.mlp_update_grad_gated,
                                    grad_up: &mut workspace.mlp_update_grad_up,
                                    grad_gate: &mut workspace.mlp_update_grad_gate,
                                },
                            )
                            .ok_or(TrainError::CoreRejected("mini_transformer_mlp_update"))?
                        };

                        let attention_update = match mini_transformer_attention_update_i8_checked(
                            cache_before
                                .layers
                                .last()
                                .ok_or(TrainError::InvalidConfig)?,
                            &mut model,
                            transformer_layers - 1,
                            runtime_config,
                            workspace,
                            if use_attention_accumulator {
                                Some(&mut attention_weight_gradients[transformer_layers - 1])
                            } else {
                                None
                            },
                        ) {
                            Ok(update) => update,
                            Err(TrainError::CoreRejected(_)) => {
                                model = model_checkpoint;
                                rollback_count = rollback_count.saturating_add(1);
                                rejected_window_count = rejected_window_count.saturating_add(1);
                                adaptive_attention_shifts.observe_rejected(
                                    rejected_batch_count.saturating_add(rejected_window_count),
                                    adaptive_shift_controller_enabled,
                                    config,
                                    &mut adaptive_shift_events,
                                );
                                continue;
                            }
                            Err(error) => return Err(error),
                        };

                        let embedding_gradient_saturation_count = add_i16_residual_rows_checked(
                            &workspace.grad_attention_output,
                            &workspace.grad_attention_norm_input,
                            &mut workspace.grad_embedding_output,
                        )?;
                        let embedding_update = if use_embedding_accumulator {
                            empty_softmax_update_stats()
                        } else {
                            apply_mini_transformer_embedding_update_with_position_policy(
                                &mut model.embeddings,
                                &mut model.position_embeddings,
                                &tokens[window_start..window_start + config.seq_len],
                                &workspace.grad_embedding_output,
                                config.position_policy,
                                config.learning_rate,
                                runtime_config.embedding_learning_rate_shift,
                            )?
                        };

                        (
                            mlp_input_saturation_count,
                            gradient_residual_saturation_count,
                            mlp_update,
                            attention_update,
                            embedding_gradient_saturation_count,
                            embedding_update,
                        )
                    };
                    let mlp_rms_backward_saturation_count = 0_usize;
                    let attention_rms_backward_saturation_count = 0_usize;

                    let cache_after = match mini_transformer_forward_for_attention_and_position(
                        &model,
                        &tokens[window_start..window_start + config.seq_len],
                        config.attention_kind,
                        config.position_policy,
                    ) {
                        Ok(cache) => cache,
                        Err(error) => {
                            let _ = error;
                            model = model_checkpoint;
                            rollback_count = rollback_count.saturating_add(1);
                            rejected_window_count = rejected_window_count.saturating_add(1);
                            adaptive_attention_shifts.observe_rejected(
                                rejected_batch_count.saturating_add(rejected_window_count),
                                adaptive_shift_controller_enabled,
                                config,
                                &mut adaptive_shift_events,
                            );
                            continue;
                        }
                    };

                    if mini_transformer_validate_guard_windows(
                        &model,
                        tokens,
                        &starts,
                        config.seq_len,
                        config.attention_kind,
                        config.position_policy,
                        epoch,
                        window_index,
                        config.epochs,
                    )
                    .is_err()
                    {
                        model = model_checkpoint;
                        rollback_count = rollback_count.saturating_add(1);
                        rejected_window_count = rejected_window_count.saturating_add(1);
                        adaptive_attention_shifts.observe_rejected(
                            rejected_batch_count.saturating_add(rejected_window_count),
                            adaptive_shift_controller_enabled,
                            config,
                            &mut adaptive_shift_events,
                        );
                        continue;
                    }
                    if use_output_head_accumulator {
                        accumulate_linear_weight_gradient_i64_prescaled(
                            &cache_before.output_features,
                            &workspace.output_scaled_grad,
                            &mut output_head_gradient,
                        )?;
                    }
                    if use_mlp_accumulator {
                        accumulate_gated_mlp_weight_gradient_i64(
                            &cache_before.mlp_norm,
                            &workspace.grad_mlp_output,
                            &cache_before.mlp_gated,
                            &workspace.mlp_input_grad_up,
                            &workspace.mlp_input_grad_gate,
                            GatedMlpWeightUpdateParams {
                                up_scales: &MINI_TRANSFORMER_HIDDEN_SCALES,
                                gate_scales: &MINI_TRANSFORMER_HIDDEN_SCALES,
                                down_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                                down_to_hidden_scales: &MINI_TRANSFORMER_HIDDEN_GRAD_INPUT_SCALES,
                                seq_len: config.seq_len,
                                d_model: MINI_TRANSFORMER_D_MODEL,
                                hidden_dim: MINI_TRANSFORMER_HIDDEN_DIM,
                                learning_rate: config.learning_rate,
                                learning_rate_shift: config.mlp_learning_rate_shift,
                            },
                            &mut mlp_weight_gradients[transformer_layers - 1],
                            &mut workspace.mlp_scaled_grad,
                        )?;
                    }
                    if use_embedding_accumulator {
                        accumulate_mini_transformer_embedding_gradient_i64_with_position_policy(
                            &tokens[window_start..window_start + config.seq_len],
                            &workspace.grad_embedding_output,
                            config.position_policy,
                            &mut embedding_gradient,
                        )?;
                    }
                    let predicted_token_after = if should_record_step {
                        byte_argmax_i32(&cache_after.logits_q8)
                    } else {
                        0
                    };
                    let output_head_hash_after = if should_record_step {
                        model.output_head_hash()
                    } else {
                        0
                    };
                    let mlp_hash_after = if should_record_step {
                        model.mlp_hash()
                    } else {
                        0
                    };
                    let attention_hash_after = if should_record_step {
                        model.attention_hash()
                    } else {
                        0
                    };
                    let embedding_hash_after = if should_record_step {
                        model.embedding_hash()
                    } else {
                        0
                    };

                    updates += 1;
                    output_head_saturation_count += output_update.gradient_saturation_count;
                    output_head_zero_delta_count += output_update.zero_delta_count;
                    output_head_delta_l1 =
                        output_head_delta_l1.saturating_add(output_update.weight_delta_l1);
                    mlp_saturation_count += mlp_input_saturation_count;
                    mlp_saturation_count += mlp_rms_backward_saturation_count;
                    mlp_saturation_count +=
                        mlp_update.gradient_saturation_count().unwrap_or(usize::MAX);
                    mlp_zero_delta_count += mlp_update.zero_delta_count().unwrap_or(usize::MAX);
                    mlp_delta_l1 =
                        mlp_delta_l1.saturating_add(mlp_update.weight_delta_l1().unwrap_or(0));
                    embedding_saturation_count += embedding_update.gradient_saturation_count;
                    embedding_zero_delta_count += embedding_update.zero_delta_count;
                    embedding_delta_l1 =
                        embedding_delta_l1.saturating_add(embedding_update.weight_delta_l1);
                    attention_saturation_count += attention_update.gradient_saturation_count;
                    attention_saturation_count += attention_rms_backward_saturation_count;
                    attention_zero_delta_count += attention_update.zero_delta_count;
                    attention_delta_l1 =
                        attention_delta_l1.saturating_add(attention_update.weight_delta_l1);
                    attention_q_delta_l1 =
                        attention_q_delta_l1.saturating_add(attention_update.q.weight_delta_l1);
                    attention_k_delta_l1 =
                        attention_k_delta_l1.saturating_add(attention_update.k.weight_delta_l1);
                    attention_v_delta_l1 =
                        attention_v_delta_l1.saturating_add(attention_update.v.weight_delta_l1);
                    attention_o_delta_l1 =
                        attention_o_delta_l1.saturating_add(attention_update.o.weight_delta_l1);
                    residual_saturation_count += gradient_residual_saturation_count;
                    residual_saturation_count += embedding_gradient_saturation_count;
                    residual_saturation_count += cache_before.residual_saturation_count;
                    residual_saturation_count += cache_after.residual_saturation_count;
                    if !use_attention_accumulator {
                        adaptive_attention_shifts.observe_accepted(
                            output_update,
                            mlp_update,
                            embedding_update,
                            &attention_update,
                            accepted_batch_count.saturating_add(updates),
                            adaptive_shift_controller_enabled,
                            config,
                            &mut adaptive_shift_events,
                        );
                    }

                    if should_record_step {
                        steps.push(MiniTransformerMlpTrainingStepTrace {
                            update_index: updates,
                            epoch,
                            window_index,
                            window_start,
                            first_token: tokens[window_start],
                            last_token: tokens[window_start + config.seq_len - 1],
                            target_token,
                            predicted_token_before,
                            predicted_token_after,
                            target_probability_before_q15: cache_before.probabilities_q15
                                [usize::from(target_token)],
                            target_probability_after_q15: cache_after.probabilities_q15
                                [usize::from(target_token)],
                            embedding_cache_hash: hash_i16_slice(&cache_before.embedding_output),
                            attention_cache_hash: hash_i16_slice(&cache_before.attention_output),
                            mlp_cache_hash: hash_i16_slice(&cache_before.mlp_gated),
                            block_output_hash_before: hash_i16_slice(&cache_before.block_output),
                            block_output_hash_after: hash_i16_slice(&cache_after.block_output),
                            output_head_hash_before,
                            output_head_hash_after,
                            mlp_hash_before,
                            mlp_hash_after,
                            attention_hash_before,
                            attention_hash_after,
                            embedding_hash_before,
                            embedding_hash_after,
                            output_head_saturation_count: output_update.gradient_saturation_count,
                            mlp_saturation_count: mlp_input_saturation_count
                                + mlp_rms_backward_saturation_count
                                + mlp_update.gradient_saturation_count().unwrap_or(usize::MAX),
                            embedding_saturation_count: embedding_update.gradient_saturation_count,
                            attention_saturation_count: attention_update.gradient_saturation_count
                                + attention_rms_backward_saturation_count,
                            residual_saturation_count: cache_before.residual_saturation_count
                                + cache_after.residual_saturation_count
                                + gradient_residual_saturation_count
                                + embedding_gradient_saturation_count,
                            output_head_zero_delta_count: output_update.zero_delta_count,
                            mlp_zero_delta_count: mlp_update
                                .zero_delta_count()
                                .unwrap_or(usize::MAX),
                            embedding_zero_delta_count: embedding_update.zero_delta_count,
                            attention_zero_delta_count: attention_update.zero_delta_count,
                            output_head_delta_l1: output_update.weight_delta_l1,
                            mlp_delta_l1: mlp_update.weight_delta_l1().unwrap_or(0),
                            embedding_delta_l1: embedding_update.weight_delta_l1,
                            attention_delta_l1: attention_update.weight_delta_l1,
                            attention_q_delta_l1: attention_update.q.weight_delta_l1,
                            attention_k_delta_l1: attention_update.k.weight_delta_l1,
                            attention_v_delta_l1: attention_update.v.weight_delta_l1,
                            attention_o_delta_l1: attention_update.o.weight_delta_l1,
                        });
                    }
                }
            }

            let accepted_windows_in_batch = updates.saturating_sub(updates_before_batch);
            if accepted_windows_in_batch > 0 {
                if rollback_count == rollbacks_before_batch {
                    let mut candidate_model = model.clone();
                    average_mini_transformer_batch_movement(
                        &batch_model_checkpoint,
                        &mut candidate_model,
                        accepted_windows_in_batch,
                        !use_output_head_accumulator,
                        !use_mlp_accumulator,
                        !use_attention_accumulator,
                        !use_embedding_accumulator,
                    )?;

                    let mut batch_output_head_saturation_count = 0_usize;
                    let mut batch_output_head_zero_delta_count = 0_usize;
                    let mut batch_output_head_delta_l1 = 0_u64;
                    let mut batch_mlp_saturation_count = 0_usize;
                    let mut batch_mlp_zero_delta_count = 0_usize;
                    let mut batch_mlp_delta_l1 = 0_u64;
                    let mut batch_attention_saturation_count = 0_usize;
                    let mut batch_attention_zero_delta_count = 0_usize;
                    let mut batch_attention_delta_l1 = 0_u64;
                    let mut batch_attention_q_delta_l1 = 0_u64;
                    let mut batch_attention_k_delta_l1 = 0_u64;
                    let mut batch_attention_v_delta_l1 = 0_u64;
                    let mut batch_attention_o_delta_l1 = 0_u64;
                    let mut batch_attention_update_for_controller = None;
                    let mut batch_embedding_saturation_count = 0_usize;
                    let mut batch_embedding_zero_delta_count = 0_usize;
                    let mut batch_embedding_delta_l1 = 0_u64;

                    let mut batch_output_head_accumulator_batch_count = 0_usize;
                    let mut batch_output_head_accumulator_window_count = 0_usize;
                    let mut batch_mlp_accumulator_batch_count = 0_usize;
                    let mut batch_mlp_accumulator_window_count = 0_usize;
                    let mut batch_attention_accumulator_batch_count = 0_usize;
                    let mut batch_attention_accumulator_window_count = 0_usize;
                    let mut batch_embedding_accumulator_batch_count = 0_usize;
                    let mut batch_embedding_accumulator_window_count = 0_usize;

                    let output_head_gradient_checkpoint = output_head_gradient.clone();
                    let mlp_weight_gradients_checkpoint = mlp_weight_gradients.clone();
                    let attention_weight_gradients_checkpoint = attention_weight_gradients.clone();
                    let embedding_gradient_checkpoint = embedding_gradient.clone();
                    let batch_windows = &starts[batch_start_index..batch_end_index];
                    let batch_runtime_config = adaptive_attention_shifts.runtime_config(config);
                    let batch_apply_config = mini_transformer_batch_component_shift_config(
                        batch_runtime_config,
                        accepted_windows_in_batch,
                    )?;

                    if use_output_head_accumulator {
                        let output_batch_update = apply_linear_weight_gradient_i64_to_i8(
                            &mut output_head_gradient,
                            &mut candidate_model.output_weights,
                            batch_apply_config.learning_rate,
                            batch_apply_config.output_learning_rate_shift,
                            true,
                        )?;
                        batch_output_head_saturation_count =
                            output_batch_update.gradient_saturation_count;
                        batch_output_head_zero_delta_count = output_batch_update.zero_delta_count;
                        batch_output_head_delta_l1 = output_batch_update.weight_delta_l1;
                        batch_output_head_accumulator_batch_count = 1;
                        batch_output_head_accumulator_window_count = accepted_windows_in_batch;
                    }
                    if use_mlp_accumulator {
                        let mut mlp_batch_update = empty_gated_mlp_weight_update_stats();
                        for (layer_index, gradient) in mlp_weight_gradients
                            .iter_mut()
                            .enumerate()
                            .take(transformer_layers)
                        {
                            let layer_apply_config = if transformer_layers > 1 {
                                mini_transformer_stacked_layer_runtime_config(
                                    batch_apply_config,
                                    layer_index,
                                    transformer_layers,
                                )
                            } else {
                                batch_apply_config
                            };
                            let up_or_gate_range =
                                candidate_model.mlp_up_or_gate_weight_range(layer_index)?;
                            let down_range = candidate_model.mlp_down_weight_range(layer_index)?;
                            let layer_update = apply_gated_mlp_weight_gradient_i64_to_i8(
                                gradient,
                                &mut candidate_model.up_weights[up_or_gate_range.clone()],
                                &mut candidate_model.gate_weights[up_or_gate_range],
                                &mut candidate_model.down_weights[down_range],
                                layer_apply_config.learning_rate,
                                layer_apply_config.mlp_learning_rate_shift,
                                true,
                            )?;
                            add_gated_mlp_weight_update_stats_checked(
                                &mut mlp_batch_update,
                                layer_update,
                            )?;
                        }
                        batch_mlp_saturation_count = mlp_batch_update
                            .gradient_saturation_count()
                            .unwrap_or(usize::MAX);
                        batch_mlp_zero_delta_count =
                            mlp_batch_update.zero_delta_count().unwrap_or(usize::MAX);
                        batch_mlp_delta_l1 = mlp_batch_update.weight_delta_l1().unwrap_or(0);
                        batch_mlp_accumulator_batch_count = 1;
                        batch_mlp_accumulator_window_count = accepted_windows_in_batch;
                    }
                    if use_attention_accumulator {
                        let mut attention_batch_update =
                            empty_mini_transformer_attention_weight_update_stats();
                        for (layer_index, gradient) in attention_weight_gradients
                            .iter_mut()
                            .enumerate()
                            .take(transformer_layers)
                        {
                            let layer_apply_config = if transformer_layers > 1 {
                                mini_transformer_stacked_layer_runtime_config(
                                    batch_apply_config,
                                    layer_index,
                                    transformer_layers,
                                )
                            } else {
                                batch_apply_config
                            };
                            let layer_update = if transformer_layers == 1 {
                                apply_mini_transformer_attention_weight_gradient_i64_to_i8(
                                    gradient,
                                    &mut candidate_model,
                                    layer_apply_config,
                                )?
                            } else {
                                apply_mini_transformer_attention_weight_gradient_i64_to_i8_for_layer(
                                    gradient,
                                    &mut candidate_model,
                                    layer_index,
                                    layer_apply_config,
                                )?
                            };
                            add_mini_transformer_attention_weight_update_stats_checked(
                                &mut attention_batch_update,
                                layer_update,
                            )?;
                        }
                        if config.attention_vo_oracle {
                            let (v_oracle, o_oracle) =
                                mini_transformer_attention_vo_oracle_update_i8_checked(
                                    &mut candidate_model,
                                    tokens,
                                    batch_windows,
                                    config.seq_len,
                                    config.learning_rate,
                                )?;
                            add_linear_weight_update_stats_checked(
                                &mut attention_batch_update.v,
                                v_oracle,
                            )?;
                            add_linear_weight_update_stats_checked(
                                &mut attention_batch_update.o,
                                o_oracle,
                            )?;
                            attention_batch_update.gradient_saturation_count =
                                attention_batch_update
                                    .gradient_saturation_count
                                    .saturating_add(v_oracle.gradient_saturation_count)
                                    .saturating_add(o_oracle.gradient_saturation_count);
                            attention_batch_update.zero_delta_count = attention_batch_update
                                .zero_delta_count
                                .saturating_add(v_oracle.zero_delta_count)
                                .saturating_add(o_oracle.zero_delta_count);
                            attention_batch_update.weight_delta_l1 = attention_batch_update
                                .weight_delta_l1
                                .saturating_add(v_oracle.weight_delta_l1)
                                .saturating_add(o_oracle.weight_delta_l1);
                        }
                        batch_attention_saturation_count =
                            attention_batch_update.gradient_saturation_count;
                        batch_attention_zero_delta_count = attention_batch_update.zero_delta_count;
                        batch_attention_delta_l1 = attention_batch_update.weight_delta_l1;
                        batch_attention_q_delta_l1 = attention_batch_update.q.weight_delta_l1;
                        batch_attention_k_delta_l1 = attention_batch_update.k.weight_delta_l1;
                        batch_attention_v_delta_l1 = attention_batch_update.v.weight_delta_l1;
                        batch_attention_o_delta_l1 = attention_batch_update.o.weight_delta_l1;
                        batch_attention_update_for_controller = Some(attention_batch_update);
                        batch_attention_accumulator_batch_count = 1;
                        batch_attention_accumulator_window_count = accepted_windows_in_batch;
                    }
                    if use_embedding_accumulator {
                        let embedding_learning_rate_shift = if transformer_layers > 1 {
                            batch_apply_config
                                .embedding_learning_rate_shift
                                .saturating_add(
                                    MINI_TRANSFORMER_STACKED_EMBEDDING_LEARNING_RATE_EXTRA_SHIFT,
                                )
                                .min(MAX_RIGHT_SHIFT)
                        } else {
                            batch_apply_config.embedding_learning_rate_shift
                        };
                        let embedding_batch_update =
                            apply_mini_transformer_embedding_gradient_i64_to_i16_with_position_policy(
                                &mut embedding_gradient,
                                &mut candidate_model.embeddings,
                                &mut candidate_model.position_embeddings,
                                config.position_policy,
                                batch_apply_config.learning_rate,
                                embedding_learning_rate_shift,
                            )?;
                        batch_embedding_saturation_count =
                            embedding_batch_update.gradient_saturation_count;
                        batch_embedding_zero_delta_count = embedding_batch_update.zero_delta_count;
                        batch_embedding_delta_l1 = embedding_batch_update.weight_delta_l1;
                        batch_embedding_accumulator_batch_count = 1;
                        batch_embedding_accumulator_window_count = accepted_windows_in_batch;
                    }

                    let batch_valid = mini_transformer_validate_batch_windows(
                        &candidate_model,
                        tokens,
                        batch_windows,
                        config.seq_len,
                        config.attention_kind,
                        config.position_policy,
                    )
                    .and_then(|_| {
                        mini_transformer_validate_guard_windows(
                            &candidate_model,
                            tokens,
                            &starts,
                            config.seq_len,
                            config.attention_kind,
                            config.position_policy,
                            epoch,
                            batch_end_index.saturating_sub(1).min(starts.len() - 1),
                            config.epochs,
                        )
                    })
                    .is_ok();
                    let mut batch_loss_regressed = false;
                    if batch_valid && config.reject_loss_regression {
                        let loss_guard_starts = mini_transformer_loss_guard_starts(
                            &starts,
                            batch_start_index,
                            batch_end_index,
                        );
                        let before_loss = mini_transformer_total_probability_error_q15_with_attention_and_position_policy(
                            tokens,
                            &loss_guard_starts,
                            &batch_model_checkpoint,
                            config.seq_len,
                            config.attention_kind,
                            config.position_policy,
                        )?;
                        let after_loss = mini_transformer_total_probability_error_q15_with_attention_and_position_policy(
                            tokens,
                            &loss_guard_starts,
                            &candidate_model,
                            config.seq_len,
                            config.attention_kind,
                            config.position_policy,
                        );
                        batch_loss_regressed = match after_loss {
                            Ok(after_loss) => mini_transformer_loss_guard_regressed(
                                before_loss,
                                after_loss,
                                loss_guard_starts.len(),
                            ),
                            Err(TrainError::CoreRejected(_)) => true,
                            Err(error) => return Err(error),
                        };
                    }

                    if batch_valid && !batch_loss_regressed {
                        model = candidate_model;
                        output_head_saturation_count = output_head_saturation_count
                            .saturating_add(batch_output_head_saturation_count);
                        output_head_zero_delta_count = output_head_zero_delta_count
                            .saturating_add(batch_output_head_zero_delta_count);
                        output_head_delta_l1 =
                            output_head_delta_l1.saturating_add(batch_output_head_delta_l1);
                        mlp_saturation_count =
                            mlp_saturation_count.saturating_add(batch_mlp_saturation_count);
                        mlp_zero_delta_count =
                            mlp_zero_delta_count.saturating_add(batch_mlp_zero_delta_count);
                        mlp_delta_l1 = mlp_delta_l1.saturating_add(batch_mlp_delta_l1);
                        attention_saturation_count = attention_saturation_count
                            .saturating_add(batch_attention_saturation_count);
                        attention_zero_delta_count = attention_zero_delta_count
                            .saturating_add(batch_attention_zero_delta_count);
                        attention_delta_l1 =
                            attention_delta_l1.saturating_add(batch_attention_delta_l1);
                        attention_q_delta_l1 =
                            attention_q_delta_l1.saturating_add(batch_attention_q_delta_l1);
                        attention_k_delta_l1 =
                            attention_k_delta_l1.saturating_add(batch_attention_k_delta_l1);
                        attention_v_delta_l1 =
                            attention_v_delta_l1.saturating_add(batch_attention_v_delta_l1);
                        attention_o_delta_l1 =
                            attention_o_delta_l1.saturating_add(batch_attention_o_delta_l1);
                        embedding_saturation_count = embedding_saturation_count
                            .saturating_add(batch_embedding_saturation_count);
                        embedding_zero_delta_count = embedding_zero_delta_count
                            .saturating_add(batch_embedding_zero_delta_count);
                        embedding_delta_l1 =
                            embedding_delta_l1.saturating_add(batch_embedding_delta_l1);
                        output_head_accumulator_batch_count = output_head_accumulator_batch_count
                            .saturating_add(batch_output_head_accumulator_batch_count);
                        output_head_accumulator_window_count = output_head_accumulator_window_count
                            .saturating_add(batch_output_head_accumulator_window_count);
                        mlp_accumulator_batch_count = mlp_accumulator_batch_count
                            .saturating_add(batch_mlp_accumulator_batch_count);
                        mlp_accumulator_window_count = mlp_accumulator_window_count
                            .saturating_add(batch_mlp_accumulator_window_count);
                        attention_accumulator_batch_count = attention_accumulator_batch_count
                            .saturating_add(batch_attention_accumulator_batch_count);
                        attention_accumulator_window_count = attention_accumulator_window_count
                            .saturating_add(batch_attention_accumulator_window_count);
                        embedding_accumulator_batch_count = embedding_accumulator_batch_count
                            .saturating_add(batch_embedding_accumulator_batch_count);
                        embedding_accumulator_window_count = embedding_accumulator_window_count
                            .saturating_add(batch_embedding_accumulator_window_count);
                        if let Some(update) = batch_attention_update_for_controller.as_ref() {
                            adaptive_attention_shifts.observe_accepted(
                                LinearWeightUpdateStats {
                                    gradient_saturation_count: batch_output_head_saturation_count,
                                    zero_delta_count: batch_output_head_zero_delta_count,
                                    weight_delta_l1: batch_output_head_delta_l1,
                                },
                                GatedMlpWeightUpdateStats {
                                    down: LinearWeightUpdateStats {
                                        gradient_saturation_count: batch_mlp_saturation_count,
                                        zero_delta_count: batch_mlp_zero_delta_count,
                                        weight_delta_l1: batch_mlp_delta_l1,
                                    },
                                    up: empty_linear_weight_update_stats(),
                                    gate: empty_linear_weight_update_stats(),
                                },
                                SoftmaxUpdateStats {
                                    gradient_saturation_count: batch_embedding_saturation_count,
                                    zero_delta_count: batch_embedding_zero_delta_count,
                                    weight_delta_l1: batch_embedding_delta_l1,
                                },
                                update,
                                accepted_batch_count.saturating_add(1),
                                adaptive_shift_controller_enabled,
                                config,
                                &mut adaptive_shift_events,
                            );
                        }
                        accepted_batch_count = accepted_batch_count.saturating_add(1);
                        emit_mini_transformer_committed_binary_steps(
                            &steps,
                            steps_before_batch,
                            &mut binary_trace,
                        )?;
                    } else {
                        model = batch_model_checkpoint;
                        updates = updates_before_batch;
                        steps.truncate(steps_before_batch);
                        rollback_count = rollback_count.saturating_add(1);
                        rejected_window_count =
                            rejected_window_count.saturating_add(accepted_windows_in_batch);
                        rejected_batch_count = rejected_batch_count.saturating_add(1);
                        if batch_loss_regressed {
                            loss_regression_rejected_batch_count =
                                loss_regression_rejected_batch_count.saturating_add(1);
                        } else {
                            adaptive_attention_shifts.observe_rejected(
                                rejected_batch_count,
                                adaptive_shift_controller_enabled,
                                config,
                                &mut adaptive_shift_events,
                            );
                        }
                        if use_output_head_accumulator {
                            output_head_gradient = output_head_gradient_checkpoint;
                            output_head_gradient.clear();
                        }
                        if use_mlp_accumulator {
                            mlp_weight_gradients = mlp_weight_gradients_checkpoint;
                            mini_transformer_clear_gated_mlp_weight_gradient_i64_layers(
                                &mut mlp_weight_gradients,
                            );
                        }
                        if use_attention_accumulator {
                            attention_weight_gradients = attention_weight_gradients_checkpoint;
                            mini_transformer_clear_attention_weight_gradient_i64_layers(
                                &mut attention_weight_gradients,
                            );
                        }
                        if use_embedding_accumulator {
                            embedding_gradient = embedding_gradient_checkpoint;
                            embedding_gradient.clear();
                        }
                    }
                } else {
                    if use_output_head_accumulator {
                        output_head_gradient.clear();
                    }
                    if use_mlp_accumulator {
                        mini_transformer_clear_gated_mlp_weight_gradient_i64_layers(
                            &mut mlp_weight_gradients,
                        );
                    }
                    if use_attention_accumulator {
                        mini_transformer_clear_attention_weight_gradient_i64_layers(
                            &mut attention_weight_gradients,
                        );
                    }
                    if use_embedding_accumulator {
                        embedding_gradient.clear();
                    }
                    accepted_batch_count = accepted_batch_count.saturating_add(1);
                    emit_mini_transformer_committed_binary_steps(
                        &steps,
                        steps_before_batch,
                        &mut binary_trace,
                    )?;
                }
            } else {
                if use_output_head_accumulator {
                    output_head_gradient.clear();
                }
                if use_mlp_accumulator {
                    mini_transformer_clear_gated_mlp_weight_gradient_i64_layers(
                        &mut mlp_weight_gradients,
                    );
                }
                if use_attention_accumulator {
                    mini_transformer_clear_attention_weight_gradient_i64_layers(
                        &mut attention_weight_gradients,
                    );
                }
                if use_embedding_accumulator {
                    embedding_gradient.clear();
                }
                rejected_batch_count = rejected_batch_count.saturating_add(1);
                adaptive_attention_shifts.observe_rejected(
                    rejected_batch_count,
                    adaptive_shift_controller_enabled,
                    config,
                    &mut adaptive_shift_events,
                );
            }
            output_head_carry_l1 = if use_output_head_accumulator {
                output_head_gradient.residual_l1()
            } else {
                0
            };
            mlp_carry_l1 = if use_mlp_accumulator {
                mini_transformer_gated_mlp_weight_gradient_i64_layers_residual_l1(
                    &mlp_weight_gradients,
                )
            } else {
                0
            };
            embedding_carry_l1 = if use_embedding_accumulator {
                embedding_gradient.residual_l1(config.position_policy)
            } else {
                0
            };
            if use_attention_accumulator {
                attention_q_carry_l1 =
                    mini_transformer_attention_weight_gradient_i64_layers_projection_residual_l1(
                        &attention_weight_gradients,
                        MiniTransformerAttentionProjection::Query,
                    );
                attention_k_carry_l1 =
                    mini_transformer_attention_weight_gradient_i64_layers_projection_residual_l1(
                        &attention_weight_gradients,
                        MiniTransformerAttentionProjection::Key,
                    );
                attention_v_carry_l1 =
                    mini_transformer_attention_weight_gradient_i64_layers_projection_residual_l1(
                        &attention_weight_gradients,
                        MiniTransformerAttentionProjection::Value,
                    );
                attention_o_carry_l1 =
                    mini_transformer_attention_weight_gradient_i64_layers_projection_residual_l1(
                        &attention_weight_gradients,
                        MiniTransformerAttentionProjection::Output,
                    );
                attention_carry_l1 =
                    mini_transformer_attention_weight_gradient_i64_layers_residual_l1(
                        &attention_weight_gradients,
                    );
            } else {
                attention_q_carry_l1 = 0;
                attention_k_carry_l1 = 0;
                attention_v_carry_l1 = 0;
                attention_o_carry_l1 = 0;
                attention_carry_l1 = 0;
            }
            let observed_batch_count = accepted_batch_count.saturating_add(rejected_batch_count);
            if progress_interval_batches > 0
                && observed_batch_count > 0
                && observed_batch_count.is_multiple_of(progress_interval_batches)
            {
                progress(&mini_transformer_training_progress_trace(
                    config,
                    tokens.len(),
                    token_hash,
                    window_hash,
                    starts.len(),
                    examined_windows,
                    updates,
                    accepted_batch_count,
                    rejected_batch_count,
                    rollback_count,
                    rejected_window_count,
                    output_head_delta_l1,
                    mlp_delta_l1,
                    embedding_delta_l1,
                    attention_delta_l1,
                    attention_q_delta_l1,
                    attention_k_delta_l1,
                    attention_v_delta_l1,
                    attention_o_delta_l1,
                    output_head_carry_l1,
                    mlp_carry_l1,
                    embedding_carry_l1,
                    attention_carry_l1,
                    attention_q_carry_l1,
                    attention_k_carry_l1,
                    attention_v_carry_l1,
                    attention_o_carry_l1,
                    &adaptive_attention_shifts,
                    &model,
                ))?;
            }
            batch_start_index = batch_end_index;
        }
    }

    let final_eval = mini_transformer_eval_summary_with_attention_and_position_policy(
        tokens,
        &starts,
        &model,
        config.seq_len,
        config.attention_kind,
        config.position_policy,
    )?;
    let final_total_error = final_eval.mistakes;
    let final_probability_error_q15 = final_eval.probability_error_q15;
    let final_mistakes = final_eval.mistakes;
    let final_correct = starts.len() - final_mistakes;
    let final_accuracy_per_mille = final_correct * 1000 / starts.len();
    let final_logits_hash = final_eval.logits_hash;
    if progress_interval_batches > 0 {
        progress(&mini_transformer_training_progress_trace(
            config,
            tokens.len(),
            token_hash,
            window_hash,
            starts.len(),
            examined_windows,
            updates,
            accepted_batch_count,
            rejected_batch_count,
            rollback_count,
            rejected_window_count,
            output_head_delta_l1,
            mlp_delta_l1,
            embedding_delta_l1,
            attention_delta_l1,
            attention_q_delta_l1,
            attention_k_delta_l1,
            attention_v_delta_l1,
            attention_o_delta_l1,
            output_head_carry_l1,
            mlp_carry_l1,
            embedding_carry_l1,
            attention_carry_l1,
            attention_q_carry_l1,
            attention_k_carry_l1,
            attention_v_carry_l1,
            attention_o_carry_l1,
            &adaptive_attention_shifts,
            &model,
        ))?;
    }

    let trace = MiniTransformerMlpTrainingTrace {
        trace_detail,
        config,
        token_count: tokens.len(),
        token_hash,
        window_hash,
        windows: starts.len(),
        examined_windows,
        updates,
        accepted_batch_count,
        rejected_batch_count,
        output_head_accumulator_batch_count,
        output_head_accumulator_window_count,
        mlp_accumulator_batch_count,
        mlp_accumulator_window_count,
        attention_accumulator_batch_count,
        attention_accumulator_window_count,
        embedding_accumulator_batch_count,
        embedding_accumulator_window_count,
        rollback_count,
        rejected_window_count,
        loss_regression_rejected_batch_count,
        final_invalid_forward_count: final_eval.invalid_forward_count,
        initial_model_hash,
        final_model_hash: model.model_hash(),
        initial_embedding_hash,
        final_embedding_hash: model.embedding_hash(),
        initial_output_head_hash,
        final_output_head_hash: model.output_head_hash(),
        initial_mlp_hash,
        final_mlp_hash: model.mlp_hash(),
        initial_attention_hash,
        final_attention_hash: model.attention_hash(),
        initial_attention_q_hash,
        final_attention_q_hash: model.attention_q_hash(),
        initial_attention_k_hash,
        final_attention_k_hash: model.attention_k_hash(),
        initial_attention_v_hash,
        final_attention_v_hash: model.attention_v_hash(),
        initial_attention_o_hash,
        final_attention_o_hash: model.attention_o_hash(),
        initial_total_error,
        final_total_error,
        initial_probability_error_q15,
        final_probability_error_q15,
        initial_mistakes,
        final_mistakes,
        output_head_saturation_count,
        mlp_saturation_count,
        embedding_saturation_count,
        attention_saturation_count,
        residual_saturation_count,
        output_head_zero_delta_count,
        mlp_zero_delta_count,
        embedding_zero_delta_count,
        attention_zero_delta_count,
        output_head_delta_l1,
        mlp_delta_l1,
        embedding_delta_l1,
        attention_delta_l1,
        attention_q_delta_l1,
        attention_k_delta_l1,
        attention_v_delta_l1,
        attention_o_delta_l1,
        output_head_carry_l1,
        mlp_carry_l1,
        embedding_carry_l1,
        attention_carry_l1,
        attention_q_carry_l1,
        attention_k_carry_l1,
        attention_v_carry_l1,
        attention_o_carry_l1,
        adaptive_rule_shift_adjustment_count: adaptive_attention_shifts.rule_adjustment_count,
        adaptive_rule_update_count: adaptive_attention_shifts.rule_update_count,
        adaptive_rule_event_count: adaptive_attention_shifts.rule_event_count,
        adaptive_holographic_shift_adjustment_count: adaptive_attention_shifts
            .holographic_adjustment_count,
        adaptive_holographic_update_count: adaptive_attention_shifts.total_memory_updates(),
        adaptive_holographic_hash: adaptive_attention_shifts.memory_hash(),
        adaptive_attention_shift_adjustment_count: adaptive_attention_shifts.adjustment_count,
        adaptive_attention_holographic_update_count: adaptive_attention_shifts
            .attention_memory_updates(),
        adaptive_attention_holographic_hash: adaptive_attention_shifts.attention_memory_hash(),
        final_output_learning_rate_shift: adaptive_attention_shifts.output_learning_rate_shift,
        final_mlp_learning_rate_shift: adaptive_attention_shifts.mlp_learning_rate_shift,
        final_embedding_learning_rate_shift: adaptive_attention_shifts
            .embedding_learning_rate_shift,
        final_attention_learning_rate_shift: adaptive_attention_shifts
            .attention_learning_rate_shift,
        final_attention_q_learning_rate_shift: adaptive_attention_shifts
            .attention_q_learning_rate_shift,
        final_attention_qk_learning_rate_shift: adaptive_attention_shifts
            .attention_qk_learning_rate_shift,
        final_accuracy_per_mille,
        final_logits_hash,
        adaptive_shift_events,
        steps,
    };

    for event in &trace.adaptive_shift_events {
        binary_trace(MiniTransformerBinaryTraceRecord::AdaptiveShift(event))?;
    }
    binary_trace(MiniTransformerBinaryTraceRecord::FinalSummary(&trace))?;

    Ok(MiniTransformerMlpTrainingRun { trace, model })
}

pub(super) fn validate_mini_transformer_batch_mode(
    config: MiniTransformerMlpTrainConfig,
) -> Result<(), TrainError> {
    match config.batch_mode {
        MiniTransformerBatchMode::Serial => Ok(()),
        MiniTransformerBatchMode::MapReduce => {
            if config.batch_windows <= 1
                || config.attention_vo_error_feedback
                || config.attention_vo_oracle
                || config.reject_loss_regression
                || config.attention_kind.uses_incremental_state()
            {
                return Err(TrainError::InvalidConfig);
            }

            Ok(())
        }
    }
}

pub(super) fn mini_transformer_effective_map_reduce_workers(
    config: MiniTransformerMlpTrainConfig,
) -> usize {
    if config.batch_mode != MiniTransformerBatchMode::MapReduce {
        return 1;
    }
    if config.map_reduce_workers == 0 {
        std::thread::available_parallelism()
            .map(|parallelism| parallelism.get())
            .unwrap_or(1)
            .max(1)
    } else {
        config.map_reduce_workers.max(1)
    }
}

pub(super) struct MiniTransformerMapReduceBatchResult {
    pub(super) accepted_window_count: usize,
    pub(super) output_head_gradient: LinearWeightGradientI64,
    pub(super) mlp_weight_gradients: Vec<GatedMlpWeightGradientI64>,
    pub(super) attention_weight_gradients: Vec<MiniTransformerAttentionWeightGradientI64>,
    pub(super) rms_weight_gradients: Vec<MiniTransformerRmsWeightGradientI64>,
    pub(super) embedding_gradient: MiniTransformerEmbeddingGradientI64,
    pub(super) mlp_saturation_count: usize,
    pub(super) attention_saturation_count: usize,
    pub(super) residual_saturation_count: usize,
    pub(super) steps: Vec<MiniTransformerMlpTrainingStepTrace>,
}

impl MiniTransformerMapReduceBatchResult {
    pub(super) fn new(
        config: MiniTransformerMlpTrainConfig,
        transformer_layers: usize,
    ) -> Result<Self, TrainError> {
        Ok(Self {
            accepted_window_count: 0,
            output_head_gradient: LinearWeightGradientI64::new(
                MINI_TRANSFORMER_D_MODEL,
                BYTE_VOCAB,
            )
            .ok_or(TrainError::InvalidConfig)?,
            mlp_weight_gradients: mini_transformer_new_gated_mlp_weight_gradients(
                transformer_layers,
            )?,
            attention_weight_gradients: mini_transformer_new_attention_weight_gradients(
                transformer_layers,
            )?,
            rms_weight_gradients: (0..transformer_layers)
                .map(|_| MiniTransformerRmsWeightGradientI64::new())
                .collect(),
            embedding_gradient: MiniTransformerEmbeddingGradientI64::new(config.seq_len)
                .ok_or(TrainError::InvalidConfig)?,
            mlp_saturation_count: 0,
            attention_saturation_count: 0,
            residual_saturation_count: 0,
            steps: Vec::new(),
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mini_transformer_map_reduce_batch(
    tokens: &[u8],
    starts: &[usize],
    target_frequency_weights_q15: &[i16; BYTE_VOCAB],
    batch_start_index: usize,
    batch_end_index: usize,
    epoch: usize,
    model: &MiniTransformerMlpModel,
    config: MiniTransformerMlpTrainConfig,
    updates_before_batch: usize,
    trace_detail: MiniTransformerTraceDetail,
    trace_sample_interval: usize,
) -> Result<MiniTransformerMapReduceBatchResult, TrainError> {
    if batch_start_index >= batch_end_index
        || batch_end_index > starts.len()
        || config.batch_mode != MiniTransformerBatchMode::MapReduce
    {
        return Err(TrainError::InvalidConfig);
    }

    let batch_len = batch_end_index - batch_start_index;
    let worker_count = mini_transformer_effective_map_reduce_workers(config)
        .min(batch_len)
        .max(1);
    if worker_count == 1 {
        return mini_transformer_map_reduce_worker_batch(
            tokens,
            starts,
            target_frequency_weights_q15,
            batch_start_index,
            batch_end_index,
            batch_start_index,
            epoch,
            model,
            config,
            updates_before_batch,
            trace_detail,
            trace_sample_interval,
        );
    }

    let chunk_size = batch_len.div_ceil(worker_count);
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        let mut chunk_start = batch_start_index;
        while chunk_start < batch_end_index {
            let start = chunk_start;
            let end = start.saturating_add(chunk_size).min(batch_end_index);
            handles.push(scope.spawn(move || {
                mini_transformer_map_reduce_worker_batch(
                    tokens,
                    starts,
                    target_frequency_weights_q15,
                    start,
                    end,
                    batch_start_index,
                    epoch,
                    model,
                    config,
                    updates_before_batch,
                    trace_detail,
                    trace_sample_interval,
                )
            }));
            chunk_start = end;
        }

        let mut result =
            MiniTransformerMapReduceBatchResult::new(config, model.transformer_layers())?;
        for handle in handles {
            let worker = match handle.join() {
                Ok(worker) => worker?,
                Err(payload) => std::panic::resume_unwind(payload),
            };
            mini_transformer_merge_map_reduce_batch_result(&mut result, worker)?;
        }
        Ok(result)
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mini_transformer_map_reduce_worker_batch(
    tokens: &[u8],
    starts: &[usize],
    target_frequency_weights_q15: &[i16; BYTE_VOCAB],
    range_start_index: usize,
    range_end_index: usize,
    batch_start_index: usize,
    epoch: usize,
    model: &MiniTransformerMlpModel,
    config: MiniTransformerMlpTrainConfig,
    updates_before_batch: usize,
    trace_detail: MiniTransformerTraceDetail,
    trace_sample_interval: usize,
) -> Result<MiniTransformerMapReduceBatchResult, TrainError> {
    if range_start_index > range_end_index || range_end_index > starts.len() || config.seq_len == 0
    {
        return Err(TrainError::InvalidConfig);
    }

    let mut result = MiniTransformerMapReduceBatchResult::new(config, model.transformer_layers())?;
    let mut model_for_backward = model.clone();
    let last_start = (config.seq_len - 1)
        .checked_mul(MINI_TRANSFORMER_D_MODEL)
        .ok_or(TrainError::InvalidConfig)?;
    let last_end = last_start
        .checked_add(MINI_TRANSFORMER_D_MODEL)
        .ok_or(TrainError::InvalidConfig)?;
    let mut workspace = MiniTransformerHostTrainCoreWorkspaceBuffers::new(config.seq_len)?;
    workspace.validate_host_training_step_shape(config.seq_len)?;

    for (window_index, &window_start) in starts
        .iter()
        .enumerate()
        .take(range_end_index)
        .skip(range_start_index)
    {
        let context_end = window_start
            .checked_add(config.seq_len)
            .ok_or(TrainError::InvalidConfig)?;
        if context_end >= tokens.len() {
            return Err(TrainError::InvalidConfig);
        }
        let context = &tokens[window_start..context_end];
        let target_token = tokens[context_end];
        let cache_before = mini_transformer_forward_for_attention_and_position(
            model,
            context,
            config.attention_kind,
            config.position_policy,
        )
        .map_err(|_| TrainError::CoreRejected("mini_transformer_map_reduce_forward"))?;
        let mut gradient_q15 =
            byte_vocab_softmax_gradient_q15(&cache_before.probabilities_q15, target_token);
        apply_byte_argmax_margin_gradient_q15(
            &mut gradient_q15,
            &cache_before.logits_q8,
            target_token,
            config.argmax_margin_weight_q15,
        );
        let target_frequency_weight_q15 = target_frequency_weights_q15[usize::from(target_token)];
        let weighted_gradient_q15 =
            byte_scale_gradient_q15(&gradient_q15, target_frequency_weight_q15);
        let grad_output_q15 = byte_gradient_i32_to_i16(&weighted_gradient_q15);
        workspace.reset_host_training_step();
        linear_backward_input_i16_i8_i16_per_channel_checked(
            &grad_output_q15,
            LinearBackwardInputI16I8Params {
                weights: &model.output_weights,
                forward_scales: &MINI_TRANSFORMER_OUTPUT_SCALES,
                grad_input_scales: &MINI_TRANSFORMER_OUTPUT_GRAD_INPUT_SCALES,
                input_dim: MINI_TRANSFORMER_D_MODEL,
                output_dim: BYTE_VOCAB,
            },
            LinearBackwardInputWorkspace {
                scaled_grad_output: &mut workspace.output_scaled_grad,
            },
            &mut workspace.grad_last_features,
        )
        .ok_or(TrainError::CoreRejected(
            "mini_transformer_map_reduce_output_head_backward_input",
        ))?;
        accumulate_linear_weight_gradient_i64_prescaled(
            &cache_before.output_features,
            &workspace.output_scaled_grad,
            &mut result.output_head_gradient,
        )?;

        if cache_before.layers.len() != result.mlp_weight_gradients.len()
            || cache_before.layers.len() != result.attention_weight_gradients.len()
            || cache_before.layers.len() != result.rms_weight_gradients.len()
        {
            return Err(TrainError::InvalidConfig);
        }
        let (
            mlp_input_saturation_count,
            attention_gradient_saturation_count,
            gradient_residual_saturation_count,
            embedding_gradient_saturation_count,
        ) = if cache_before.layers.len() > 1 {
            let total = config
                .seq_len
                .checked_mul(MINI_TRANSFORMER_D_MODEL)
                .ok_or(TrainError::InvalidConfig)?;
            let mut grad_block_output = vec![0_i16; total];
            grad_block_output[last_start..last_end].copy_from_slice(&workspace.grad_last_features);
            let mut stacked_mlp_input_saturation_count = 0_usize;
            let mut stacked_attention_gradient_saturation_count = 0_usize;
            let mut stacked_gradient_residual_saturation_count = 0_usize;
            let mut stacked_input_gradient_saturation_count = 0_usize;

            for layer_index in (0..cache_before.layers.len()).rev() {
                let layer_runtime_config = mini_transformer_stacked_layer_runtime_config(
                    config,
                    layer_index,
                    cache_before.layers.len(),
                );
                let block_accumulation = mini_transformer_block_backward_accumulate_i64_checked(
                    &cache_before.layers[layer_index],
                    &grad_block_output,
                    &mut model_for_backward,
                    layer_index,
                    layer_runtime_config,
                    &mut workspace,
                    &mut result.mlp_weight_gradients[layer_index],
                    &mut result.attention_weight_gradients[layer_index],
                    &mut result.rms_weight_gradients[layer_index],
                )?;
                stacked_mlp_input_saturation_count = stacked_mlp_input_saturation_count
                    .saturating_add(block_accumulation.mlp_input_saturation_count);
                stacked_attention_gradient_saturation_count =
                    stacked_attention_gradient_saturation_count
                        .saturating_add(block_accumulation.attention_gradient_saturation_count);
                stacked_gradient_residual_saturation_count =
                    stacked_gradient_residual_saturation_count
                        .saturating_add(block_accumulation.gradient_residual_saturation_count);
                stacked_input_gradient_saturation_count = stacked_input_gradient_saturation_count
                    .saturating_add(block_accumulation.input_gradient_saturation_count);
                grad_block_output = block_accumulation.grad_input;
            }

            workspace.grad_embedding_output[..total].copy_from_slice(&grad_block_output);
            accumulate_mini_transformer_embedding_gradient_i64_with_position_policy(
                context,
                &workspace.grad_embedding_output,
                config.position_policy,
                &mut result.embedding_gradient,
            )?;
            (
                stacked_mlp_input_saturation_count,
                stacked_attention_gradient_saturation_count,
                stacked_gradient_residual_saturation_count,
                stacked_input_gradient_saturation_count,
            )
        } else {
            let block_cache = cache_before
                .layers
                .last()
                .ok_or(TrainError::InvalidConfig)?;
            let rms_weights = if model.rms_norm_enabled() {
                let range = model.rms_weight_range(0)?;
                Some((
                    model.attention_rms_weights[range.clone()].to_vec(),
                    model.mlp_rms_weights[range].to_vec(),
                ))
            } else {
                None
            };
            workspace.grad_mlp_output[last_start..last_end]
                .copy_from_slice(&workspace.grad_last_features);
            let mlp_input_saturation_count = gated_mlp_backward_input_i16_q15_checked(
                &workspace.grad_mlp_output,
                mini_transformer_final_mlp_params(model, config.seq_len)?,
                &cache_before.mlp_up,
                &cache_before.mlp_gate,
                GatedMlpBackwardScales {
                    down_to_hidden: &MINI_TRANSFORMER_HIDDEN_GRAD_INPUT_SCALES,
                    up_to_input: &MINI_TRANSFORMER_D_MODEL_GRAD_INPUT_SCALES,
                    gate_to_input: &MINI_TRANSFORMER_D_MODEL_GRAD_INPUT_SCALES,
                },
                GatedMlpBackwardWorkspace {
                    scaled_grad_output: &mut workspace.mlp_scaled_grad,
                    grad_gated: &mut workspace.mlp_input_grad_gated,
                    grad_up: &mut workspace.mlp_input_grad_up,
                    grad_gate: &mut workspace.mlp_input_grad_gate,
                    grad_up_input: &mut workspace.mlp_input_grad_up_input,
                    grad_gate_input: &mut workspace.mlp_input_grad_gate_input,
                },
                &mut workspace.grad_mlp_input,
            )
            .ok_or(TrainError::CoreRejected(
                "mini_transformer_map_reduce_mlp_backward_input",
            ))?;

            let mut grad_mlp_residual = vec![0_i16; workspace.grad_mlp_input.len()];
            let mlp_rms_saturation = if let Some((_, mlp_weights)) = &rms_weights {
                mini_transformer_rms_norm_backward_rows(
                    &block_cache.attention_residual,
                    mlp_weights,
                    &workspace.grad_mlp_input,
                    &mut grad_mlp_residual,
                    &mut result.rms_weight_gradients[0].mlp,
                )?
            } else {
                grad_mlp_residual.copy_from_slice(&workspace.grad_mlp_input);
                0
            };
            let gradient_residual_saturation_count = add_i16_residual_rows_checked(
                &workspace.grad_mlp_output,
                &grad_mlp_residual,
                &mut workspace.grad_attention_output,
            )?;

            accumulate_gated_mlp_weight_gradient_i64(
                &cache_before.mlp_norm,
                &workspace.grad_mlp_output,
                &cache_before.mlp_gated,
                &workspace.mlp_input_grad_up,
                &workspace.mlp_input_grad_gate,
                GatedMlpWeightUpdateParams {
                    up_scales: &MINI_TRANSFORMER_HIDDEN_SCALES,
                    gate_scales: &MINI_TRANSFORMER_HIDDEN_SCALES,
                    down_scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
                    down_to_hidden_scales: &MINI_TRANSFORMER_HIDDEN_GRAD_INPUT_SCALES,
                    seq_len: config.seq_len,
                    d_model: MINI_TRANSFORMER_D_MODEL,
                    hidden_dim: MINI_TRANSFORMER_HIDDEN_DIM,
                    learning_rate: config.learning_rate,
                    learning_rate_shift: config.mlp_learning_rate_shift,
                },
                &mut result.mlp_weight_gradients[0],
                &mut workspace.mlp_scaled_grad,
            )?;

            let attention_update = mini_transformer_attention_update_i8_checked(
                block_cache,
                &mut model_for_backward,
                0,
                config,
                &mut workspace,
                Some(&mut result.attention_weight_gradients[0]),
            )?;
            let mut grad_attention_input = vec![0_i16; workspace.grad_attention_norm_input.len()];
            let attention_rms_saturation = if let Some((attention_weights, _)) = &rms_weights {
                mini_transformer_rms_norm_backward_rows(
                    &block_cache.block_input,
                    attention_weights,
                    &workspace.grad_attention_norm_input,
                    &mut grad_attention_input,
                    &mut result.rms_weight_gradients[0].attention,
                )?
            } else {
                grad_attention_input.copy_from_slice(&workspace.grad_attention_norm_input);
                0
            };
            let embedding_gradient_saturation_count = add_i16_residual_rows_checked(
                &workspace.grad_attention_output,
                &grad_attention_input,
                &mut workspace.grad_embedding_output,
            )?;
            accumulate_mini_transformer_embedding_gradient_i64_with_position_policy(
                context,
                &workspace.grad_embedding_output,
                config.position_policy,
                &mut result.embedding_gradient,
            )?;
            (
                mlp_input_saturation_count.saturating_add(mlp_rms_saturation),
                attention_update.gradient_saturation_count,
                gradient_residual_saturation_count,
                embedding_gradient_saturation_count.saturating_add(attention_rms_saturation),
            )
        };

        result.accepted_window_count = result.accepted_window_count.saturating_add(1);
        result.mlp_saturation_count = result
            .mlp_saturation_count
            .saturating_add(mlp_input_saturation_count);
        result.attention_saturation_count = result
            .attention_saturation_count
            .saturating_add(attention_gradient_saturation_count);
        result.residual_saturation_count = result
            .residual_saturation_count
            .saturating_add(gradient_residual_saturation_count)
            .saturating_add(embedding_gradient_saturation_count)
            .saturating_add(cache_before.residual_saturation_count)
            .saturating_add(cache_before.residual_saturation_count);

        let update_index = updates_before_batch
            .saturating_add(window_index.saturating_sub(batch_start_index))
            .saturating_add(1);
        if mini_transformer_should_record_step(trace_detail, update_index, trace_sample_interval) {
            let predicted_token_before = byte_argmax_i32(&cache_before.logits_q8);
            let output_head_hash = model.output_head_hash();
            let mlp_hash = model.mlp_hash();
            let attention_hash = model.attention_hash();
            let embedding_hash = model.embedding_hash();
            result.steps.push(MiniTransformerMlpTrainingStepTrace {
                update_index,
                epoch,
                window_index,
                window_start,
                first_token: tokens[window_start],
                last_token: tokens[window_start + config.seq_len - 1],
                target_token,
                predicted_token_before,
                predicted_token_after: predicted_token_before,
                target_probability_before_q15: cache_before.probabilities_q15
                    [usize::from(target_token)],
                target_probability_after_q15: cache_before.probabilities_q15
                    [usize::from(target_token)],
                embedding_cache_hash: hash_i16_slice(&cache_before.embedding_output),
                attention_cache_hash: hash_i16_slice(&cache_before.attention_output),
                mlp_cache_hash: hash_i16_slice(&cache_before.mlp_gated),
                block_output_hash_before: hash_i16_slice(&cache_before.block_output),
                block_output_hash_after: hash_i16_slice(&cache_before.block_output),
                output_head_hash_before: output_head_hash,
                output_head_hash_after: output_head_hash,
                mlp_hash_before: mlp_hash,
                mlp_hash_after: mlp_hash,
                attention_hash_before: attention_hash,
                attention_hash_after: attention_hash,
                embedding_hash_before: embedding_hash,
                embedding_hash_after: embedding_hash,
                output_head_saturation_count: 0,
                mlp_saturation_count: mlp_input_saturation_count,
                embedding_saturation_count: 0,
                attention_saturation_count: attention_gradient_saturation_count,
                residual_saturation_count: gradient_residual_saturation_count
                    + embedding_gradient_saturation_count
                    + cache_before.residual_saturation_count
                    + cache_before.residual_saturation_count,
                output_head_zero_delta_count: 0,
                mlp_zero_delta_count: 0,
                embedding_zero_delta_count: 0,
                attention_zero_delta_count: 0,
                output_head_delta_l1: 0,
                mlp_delta_l1: 0,
                embedding_delta_l1: 0,
                attention_delta_l1: 0,
                attention_q_delta_l1: 0,
                attention_k_delta_l1: 0,
                attention_v_delta_l1: 0,
                attention_o_delta_l1: 0,
            });
        }
    }

    Ok(result)
}

pub(super) fn mini_transformer_merge_map_reduce_batch_result(
    target: &mut MiniTransformerMapReduceBatchResult,
    source: MiniTransformerMapReduceBatchResult,
) -> Result<(), TrainError> {
    target.accepted_window_count = target
        .accepted_window_count
        .checked_add(source.accepted_window_count)
        .ok_or(TrainError::CoreRejected(
            "mini_transformer_map_reduce_window_count",
        ))?;
    mini_transformer_merge_linear_weight_gradient_i64(
        &mut target.output_head_gradient,
        &source.output_head_gradient,
    )?;
    mini_transformer_merge_gated_mlp_weight_gradient_i64_layers(
        &mut target.mlp_weight_gradients,
        &source.mlp_weight_gradients,
    )?;
    mini_transformer_merge_attention_weight_gradient_i64_layers(
        &mut target.attention_weight_gradients,
        &source.attention_weight_gradients,
    )?;
    mini_transformer_merge_rms_weight_gradient_i64_layers(
        &mut target.rms_weight_gradients,
        &source.rms_weight_gradients,
    )?;
    mini_transformer_merge_embedding_gradient_i64(
        &mut target.embedding_gradient,
        &source.embedding_gradient,
    )?;
    target.mlp_saturation_count = target
        .mlp_saturation_count
        .saturating_add(source.mlp_saturation_count);
    target.attention_saturation_count = target
        .attention_saturation_count
        .saturating_add(source.attention_saturation_count);
    target.residual_saturation_count = target
        .residual_saturation_count
        .saturating_add(source.residual_saturation_count);
    target.steps.extend(source.steps);
    Ok(())
}

pub(super) fn mini_transformer_merge_linear_weight_gradient_i64(
    target: &mut LinearWeightGradientI64,
    source: &LinearWeightGradientI64,
) -> Result<(), TrainError> {
    if target.input_dim != source.input_dim
        || target.output_dim != source.output_dim
        || target.accumulators.len() != source.accumulators.len()
    {
        return Err(TrainError::InvalidConfig);
    }
    target.sample_count =
        target
            .sample_count
            .checked_add(source.sample_count)
            .ok_or(TrainError::CoreRejected(
                "mini_transformer_map_reduce_sample_count",
            ))?;
    for (target, source) in target
        .accumulators
        .iter_mut()
        .zip(source.accumulators.iter())
    {
        *target = target.checked_add(*source).ok_or(TrainError::CoreRejected(
            "mini_transformer_map_reduce_accumulator",
        ))?;
    }
    Ok(())
}

pub(super) fn mini_transformer_merge_rms_vector_gradient_i64(
    target: &mut MiniTransformerRmsVectorGradientI64,
    source: &MiniTransformerRmsVectorGradientI64,
) -> Result<(), TrainError> {
    if target.accumulators.len() != source.accumulators.len() {
        return Err(TrainError::InvalidConfig);
    }
    target.sample_count = target
        .sample_count
        .checked_add(source.sample_count)
        .ok_or(TrainError::CoreRejected("RMSNorm sample count overflow"))?;
    for (target, source) in target
        .accumulators
        .iter_mut()
        .zip(source.accumulators.iter())
    {
        *target = target
            .checked_add(*source)
            .ok_or(TrainError::CoreRejected("RMSNorm gradient overflow"))?;
    }
    Ok(())
}

pub(super) fn mini_transformer_merge_rms_weight_gradient_i64_layers(
    target: &mut [MiniTransformerRmsWeightGradientI64],
    source: &[MiniTransformerRmsWeightGradientI64],
) -> Result<(), TrainError> {
    if target.len() != source.len() {
        return Err(TrainError::InvalidConfig);
    }
    for (target, source) in target.iter_mut().zip(source.iter()) {
        mini_transformer_merge_rms_vector_gradient_i64(&mut target.attention, &source.attention)?;
        mini_transformer_merge_rms_vector_gradient_i64(&mut target.mlp, &source.mlp)?;
    }
    Ok(())
}

pub(super) fn mini_transformer_merge_gated_mlp_weight_gradient_i64(
    target: &mut GatedMlpWeightGradientI64,
    source: &GatedMlpWeightGradientI64,
) -> Result<(), TrainError> {
    mini_transformer_merge_linear_weight_gradient_i64(&mut target.down, &source.down)?;
    mini_transformer_merge_linear_weight_gradient_i64(&mut target.up, &source.up)?;
    mini_transformer_merge_linear_weight_gradient_i64(&mut target.gate, &source.gate)?;
    Ok(())
}

pub(super) fn mini_transformer_new_gated_mlp_weight_gradients(
    layer_count: usize,
) -> Result<Vec<GatedMlpWeightGradientI64>, TrainError> {
    if layer_count == 0 {
        return Err(TrainError::InvalidConfig);
    }
    let mut gradients = Vec::with_capacity(layer_count);
    for _ in 0..layer_count {
        gradients.push(
            GatedMlpWeightGradientI64::new(MINI_TRANSFORMER_D_MODEL, MINI_TRANSFORMER_HIDDEN_DIM)
                .ok_or(TrainError::InvalidConfig)?,
        );
    }
    Ok(gradients)
}

pub(super) fn mini_transformer_clear_gated_mlp_weight_gradient_i64_layers(
    gradients: &mut [GatedMlpWeightGradientI64],
) {
    for gradient in gradients {
        gradient.clear();
    }
}

pub(super) fn mini_transformer_gated_mlp_weight_gradient_i64_layers_residual_l1(
    gradients: &[GatedMlpWeightGradientI64],
) -> u64 {
    gradients.iter().fold(0_u64, |total, gradient| {
        total.saturating_add(gradient.residual_l1())
    })
}

pub(super) fn mini_transformer_merge_gated_mlp_weight_gradient_i64_layers(
    target: &mut [GatedMlpWeightGradientI64],
    source: &[GatedMlpWeightGradientI64],
) -> Result<(), TrainError> {
    if target.len() != source.len() || target.is_empty() {
        return Err(TrainError::InvalidConfig);
    }
    for (target, source) in target.iter_mut().zip(source.iter()) {
        mini_transformer_merge_gated_mlp_weight_gradient_i64(target, source)?;
    }
    Ok(())
}

pub(super) fn mini_transformer_merge_attention_weight_gradient_i64(
    target: &mut MiniTransformerAttentionWeightGradientI64,
    source: &MiniTransformerAttentionWeightGradientI64,
) -> Result<(), TrainError> {
    mini_transformer_merge_linear_weight_gradient_i64(&mut target.q, &source.q)?;
    mini_transformer_merge_linear_weight_gradient_i64(&mut target.k, &source.k)?;
    mini_transformer_merge_linear_weight_gradient_i64(&mut target.v, &source.v)?;
    mini_transformer_merge_linear_weight_gradient_i64(&mut target.o, &source.o)?;
    Ok(())
}

pub(super) fn mini_transformer_new_attention_weight_gradients(
    layer_count: usize,
) -> Result<Vec<MiniTransformerAttentionWeightGradientI64>, TrainError> {
    if layer_count == 0 {
        return Err(TrainError::InvalidConfig);
    }
    let mut gradients = Vec::with_capacity(layer_count);
    for _ in 0..layer_count {
        gradients.push(
            MiniTransformerAttentionWeightGradientI64::new(MINI_TRANSFORMER_D_MODEL)
                .ok_or(TrainError::InvalidConfig)?,
        );
    }
    Ok(gradients)
}

pub(super) fn mini_transformer_clear_attention_weight_gradient_i64_layers(
    gradients: &mut [MiniTransformerAttentionWeightGradientI64],
) {
    for gradient in gradients {
        gradient.clear();
    }
}

pub(super) fn mini_transformer_attention_weight_gradient_i64_layers_residual_l1(
    gradients: &[MiniTransformerAttentionWeightGradientI64],
) -> u64 {
    gradients.iter().fold(0_u64, |total, gradient| {
        total.saturating_add(gradient.residual_l1())
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MiniTransformerAttentionProjection {
    Query,
    Key,
    Value,
    Output,
}

pub(super) fn mini_transformer_attention_weight_gradient_i64_layers_projection_residual_l1(
    gradients: &[MiniTransformerAttentionWeightGradientI64],
    projection: MiniTransformerAttentionProjection,
) -> u64 {
    gradients.iter().fold(0_u64, |total, gradient| {
        let projection_l1 = match projection {
            MiniTransformerAttentionProjection::Query => gradient.q.residual_l1(),
            MiniTransformerAttentionProjection::Key => gradient.k.residual_l1(),
            MiniTransformerAttentionProjection::Value => gradient.v.residual_l1(),
            MiniTransformerAttentionProjection::Output => gradient.o.residual_l1(),
        };
        total.saturating_add(projection_l1)
    })
}

pub(super) fn mini_transformer_merge_attention_weight_gradient_i64_layers(
    target: &mut [MiniTransformerAttentionWeightGradientI64],
    source: &[MiniTransformerAttentionWeightGradientI64],
) -> Result<(), TrainError> {
    if target.len() != source.len() || target.is_empty() {
        return Err(TrainError::InvalidConfig);
    }
    for (target, source) in target.iter_mut().zip(source.iter()) {
        mini_transformer_merge_attention_weight_gradient_i64(target, source)?;
    }
    Ok(())
}

pub(super) fn mini_transformer_merge_embedding_gradient_i64(
    target: &mut MiniTransformerEmbeddingGradientI64,
    source: &MiniTransformerEmbeddingGradientI64,
) -> Result<(), TrainError> {
    if target.token_accumulators.len() != source.token_accumulators.len()
        || target.position_accumulators.len() != source.position_accumulators.len()
        || target.token_residuals.len() != source.token_residuals.len()
        || target.position_residuals.len() != source.position_residuals.len()
    {
        return Err(TrainError::InvalidConfig);
    }
    target.sample_count =
        target
            .sample_count
            .checked_add(source.sample_count)
            .ok_or(TrainError::CoreRejected(
                "mini_transformer_map_reduce_embedding_sample_count",
            ))?;
    for (target, source) in target
        .token_accumulators
        .iter_mut()
        .zip(source.token_accumulators.iter())
    {
        *target = target.checked_add(*source).ok_or(TrainError::CoreRejected(
            "mini_transformer_map_reduce_embedding_accumulator",
        ))?;
    }
    for (target, source) in target
        .position_accumulators
        .iter_mut()
        .zip(source.position_accumulators.iter())
    {
        *target = target.checked_add(*source).ok_or(TrainError::CoreRejected(
            "mini_transformer_map_reduce_position_accumulator",
        ))?;
    }
    Ok(())
}
