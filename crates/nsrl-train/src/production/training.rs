use core::{cmp::Reverse, ops::Range};
use std::thread;

use nsrl_core::{
    GatedMlpI16Params, GatedMlpWorkspace, LinearAttentionWorkspace, RmsNormBackwardWorkspace,
    SelfAttentionI16Params, SoftmaxNormalization, base2_exp_neg_q15, base2_exp_neg_q47,
    base2_softmax_i32_q15, base2_softmax_i32_q31_with_normalization,
    gated_activation_backward_i16_q15, gated_mlp_i16_q15_checked, hard_silu_derivative_q15,
    hard_silu_q15, linear_attention_i16_q15_checked, rms_norm_backward_i16_q15_checked,
    rms_norm_i16_q15_checked, round_shift_rhu_i64, saturate_i8, saturate_i16,
};
use nsrl_corpus::subword::{BOS_TOKEN_ID, EOS_TOKEN_ID};

use super::alignment::{
    DocumentWindow, ProductionGradientAlignmentConfig, SurfaceEval, can_perturb, evaluate_surface,
    select_surfaces, set_parameter_delta,
};
use super::{
    FNV_OFFSET, FNV_PRIME, PRODUCTION_RMS_EPSILON, ProductionModelV1, TrainError, checked_product,
    fnv1a, linear_params, scale_shifts, scales, spread_document_windows,
};

const OPTIMIZER_MAGIC: &[u8; 8] = b"NSRLPO2\n";
const OPTIMIZER_VERSION: u32 = 2;
const GROUP_NAMES: [&str; 13] = [
    "embeddings",
    "attention_rms",
    "mlp_rms",
    "final_rms",
    "q",
    "k",
    "v",
    "o",
    "up",
    "gate",
    "down",
    "output",
    "bias",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProductionGradientProposalLane {
    NormalizedRescued,
    MassCorrectedNormalized,
    ReciprocalFreeRescued,
    ReciprocalFreeLateRhu,
    ReciprocalFreeLateStochastic,
    SystematicFixedMassK15,
    SystematicFixedMassK16,
    SystematicFixedMassK18,
    MassCorrectedNormalizedNoRescue,
}

impl ProductionGradientProposalLane {
    // Keep the original public audit set stable so v1/v2 traces and coordinate
    // selection remain byte-replayable.
    pub const ALL: [Self; 5] = [
        Self::NormalizedRescued,
        Self::MassCorrectedNormalized,
        Self::ReciprocalFreeRescued,
        Self::ReciprocalFreeLateRhu,
        Self::ReciprocalFreeLateStochastic,
    ];
    pub const WITH_CAUSAL_NO_RESCUE: [Self; 6] = [
        Self::NormalizedRescued,
        Self::MassCorrectedNormalized,
        Self::ReciprocalFreeRescued,
        Self::ReciprocalFreeLateRhu,
        Self::ReciprocalFreeLateStochastic,
        Self::MassCorrectedNormalizedNoRescue,
    ];
    pub const WITH_SYSTEMATIC: [Self; 8] = [
        Self::NormalizedRescued,
        Self::MassCorrectedNormalized,
        Self::ReciprocalFreeRescued,
        Self::ReciprocalFreeLateRhu,
        Self::ReciprocalFreeLateStochastic,
        Self::SystematicFixedMassK15,
        Self::SystematicFixedMassK16,
        Self::SystematicFixedMassK18,
    ];
    pub const WITH_SYSTEMATIC_AND_CAUSAL_NO_RESCUE: [Self; 9] = [
        Self::NormalizedRescued,
        Self::MassCorrectedNormalized,
        Self::ReciprocalFreeRescued,
        Self::ReciprocalFreeLateRhu,
        Self::ReciprocalFreeLateStochastic,
        Self::SystematicFixedMassK15,
        Self::SystematicFixedMassK16,
        Self::SystematicFixedMassK18,
        Self::MassCorrectedNormalizedNoRescue,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NormalizedRescued => "normalized-rescued-rhu",
            Self::MassCorrectedNormalized => "mass-corrected-normalized-rhu",
            Self::ReciprocalFreeRescued => "reciprocal-free-rescued-rhu",
            Self::ReciprocalFreeLateRhu => "reciprocal-free-late-rhu",
            Self::ReciprocalFreeLateStochastic => "reciprocal-free-late-stochastic",
            Self::SystematicFixedMassK15 => "systematic-fixed-mass-k15-token-id-rhu",
            Self::SystematicFixedMassK16 => "systematic-fixed-mass-k16-token-id-rhu",
            Self::SystematicFixedMassK18 => "systematic-fixed-mass-k18-token-id-rhu",
            Self::MassCorrectedNormalizedNoRescue => "mass-corrected-normalized-no-rescue-rhu",
        }
    }

    pub(super) const fn stable_index(self) -> usize {
        match self {
            Self::NormalizedRescued => 0,
            Self::MassCorrectedNormalized => 1,
            Self::ReciprocalFreeRescued => 2,
            Self::ReciprocalFreeLateRhu => 3,
            Self::ReciprocalFreeLateStochastic => 4,
            Self::SystematicFixedMassK15 => 5,
            Self::SystematicFixedMassK16 => 6,
            Self::SystematicFixedMassK18 => 7,
            Self::MassCorrectedNormalizedNoRescue => 8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum GradientSource {
    NormalizedProbability,
    MassCorrectedNormalizedProbability,
    ReciprocalFreeWeights,
    SystematicFixedMassK15,
    SystematicFixedMassK16,
    SystematicFixedMassK18,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BackwardQuantizationMode {
    RescuedRhu,
    PlainRhu,
    LateRhu,
    LateStochastic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct GradientProposalSpec {
    pub source: GradientSource,
    pub quantization: BackwardQuantizationMode,
    pub stochastic_seed: u64,
}

impl GradientProposalSpec {
    pub const fn lane(lane: ProductionGradientProposalLane, stochastic_seed: u64) -> Self {
        match lane {
            ProductionGradientProposalLane::NormalizedRescued => Self {
                source: GradientSource::NormalizedProbability,
                quantization: BackwardQuantizationMode::RescuedRhu,
                stochastic_seed,
            },
            ProductionGradientProposalLane::MassCorrectedNormalized => Self {
                source: GradientSource::MassCorrectedNormalizedProbability,
                quantization: BackwardQuantizationMode::RescuedRhu,
                stochastic_seed,
            },
            ProductionGradientProposalLane::ReciprocalFreeRescued => Self {
                source: GradientSource::ReciprocalFreeWeights,
                quantization: BackwardQuantizationMode::RescuedRhu,
                stochastic_seed,
            },
            ProductionGradientProposalLane::ReciprocalFreeLateRhu => Self {
                source: GradientSource::ReciprocalFreeWeights,
                quantization: BackwardQuantizationMode::LateRhu,
                stochastic_seed,
            },
            ProductionGradientProposalLane::ReciprocalFreeLateStochastic => Self {
                source: GradientSource::ReciprocalFreeWeights,
                quantization: BackwardQuantizationMode::LateStochastic,
                stochastic_seed,
            },
            ProductionGradientProposalLane::SystematicFixedMassK15 => Self {
                source: GradientSource::SystematicFixedMassK15,
                quantization: BackwardQuantizationMode::PlainRhu,
                stochastic_seed,
            },
            ProductionGradientProposalLane::SystematicFixedMassK16 => Self {
                source: GradientSource::SystematicFixedMassK16,
                quantization: BackwardQuantizationMode::PlainRhu,
                stochastic_seed,
            },
            ProductionGradientProposalLane::SystematicFixedMassK18 => Self {
                source: GradientSource::SystematicFixedMassK18,
                quantization: BackwardQuantizationMode::PlainRhu,
                stochastic_seed,
            },
            ProductionGradientProposalLane::MassCorrectedNormalizedNoRescue => Self {
                source: GradientSource::MassCorrectedNormalizedProbability,
                quantization: BackwardQuantizationMode::PlainRhu,
                stochastic_seed,
            },
        }
    }

    pub const fn natural_reference(source: GradientSource) -> Self {
        Self {
            source,
            quantization: BackwardQuantizationMode::PlainRhu,
            stochastic_seed: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductionFullTrainConfig {
    pub context_tokens: usize,
    pub max_windows: usize,
    pub spread_windows: bool,
    pub targets_per_window: usize,
    pub training_workers: usize,
    pub epochs: usize,
    pub matrix_learning_rate_shift: u8,
    pub q_learning_rate_shift: Option<u8>,
    pub k_learning_rate_shift: Option<u8>,
    pub v_learning_rate_shift: Option<u8>,
    pub o_learning_rate_shift: Option<u8>,
    pub up_learning_rate_shift: Option<u8>,
    pub gate_learning_rate_shift: Option<u8>,
    pub down_learning_rate_shift: Option<u8>,
    pub vector_learning_rate_shift: u8,
    pub embedding_learning_rate_shift: u8,
    pub embedding_learning_rate_boost_shift: u8,
    pub output_learning_rate_shift: u8,
    pub final_rms_learning_rate_shift: Option<u8>,
    pub output_backward_shift: Option<u8>,
    pub probability_gradient_fractional_bits: u8,
    pub probability_normalization: SoftmaxNormalization,
    pub batch_windows: usize,
    pub max_optimizer_steps: usize,
    pub evaluation_windows: usize,
}

impl Default for ProductionFullTrainConfig {
    fn default() -> Self {
        Self {
            context_tokens: 4,
            max_windows: 8,
            spread_windows: false,
            targets_per_window: 1,
            training_workers: 1,
            epochs: 2,
            matrix_learning_rate_shift: 16,
            q_learning_rate_shift: None,
            k_learning_rate_shift: None,
            v_learning_rate_shift: None,
            o_learning_rate_shift: None,
            up_learning_rate_shift: None,
            gate_learning_rate_shift: None,
            down_learning_rate_shift: None,
            vector_learning_rate_shift: 10,
            embedding_learning_rate_shift: 4,
            embedding_learning_rate_boost_shift: 0,
            output_learning_rate_shift: 24,
            final_rms_learning_rate_shift: None,
            output_backward_shift: None,
            probability_gradient_fractional_bits: 15,
            probability_normalization: SoftmaxNormalization::LegacyQ31Lut,
            batch_windows: 4,
            max_optimizer_steps: usize::MAX,
            evaluation_windows: usize::MAX,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionOptimizerStateV2 {
    pub tokenizer_hash: u64,
    pub token_stream_hash: u64,
    pub bound_model_hash: u64,
    pub step: u64,
    pub schedule_hash: u64,
    pub next_epoch: u64,
    pub next_window: u64,
    pub residuals: Vec<i64>,
}

impl ProductionOptimizerStateV2 {
    pub fn new(
        model: &ProductionModelV1,
        token_stream_hash: u64,
        config: ProductionFullTrainConfig,
    ) -> Self {
        Self {
            tokenizer_hash: model.tokenizer_hash,
            token_stream_hash,
            bound_model_hash: model.model_hash(),
            step: 0,
            schedule_hash: schedule_hash(config),
            next_epoch: 0,
            next_window: 0,
            residuals: vec![0; model.parameter_count()],
        }
    }

    pub fn validate_binding(
        &self,
        model: &ProductionModelV1,
        token_stream_hash: u64,
        config: ProductionFullTrainConfig,
    ) -> Result<(), TrainError> {
        if self.tokenizer_hash != model.tokenizer_hash
            || self.token_stream_hash != token_stream_hash
            || self.bound_model_hash != model.model_hash()
            || self.schedule_hash != schedule_hash(config)
            || self.residuals.len() != model.parameter_count()
        {
            return Err(TrainError::InvalidModel(
                "production optimizer binding mismatch",
            ));
        }
        Ok(())
    }

    pub fn state_hash(&self) -> u64 {
        let mut hash = FNV_OFFSET;
        let mut write = |bytes: &[u8]| {
            for &byte in bytes {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(FNV_PRIME);
            }
        };
        write(OPTIMIZER_MAGIC);
        write(&OPTIMIZER_VERSION.to_le_bytes());
        write(&self.tokenizer_hash.to_le_bytes());
        write(&self.token_stream_hash.to_le_bytes());
        write(&self.bound_model_hash.to_le_bytes());
        write(&self.step.to_le_bytes());
        write(&self.schedule_hash.to_le_bytes());
        write(&self.next_epoch.to_le_bytes());
        write(&self.next_window.to_le_bytes());
        write(&(self.residuals.len() as u64).to_le_bytes());
        for residual in &self.residuals {
            write(&residual.to_le_bytes());
        }
        hash
    }

    pub fn try_to_bytes(&self) -> Result<Vec<u8>, TrainError> {
        if self.tokenizer_hash == 0
            || self.token_stream_hash == 0
            || self.bound_model_hash == 0
            || self.residuals.is_empty()
        {
            return Err(TrainError::InvalidModel(
                "invalid production optimizer state",
            ));
        }
        let mut bytes = self.bytes_without_checksum();
        bytes.extend_from_slice(&fnv1a(&bytes).to_le_bytes());
        Ok(bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TrainError> {
        if bytes.len() < 84 || &bytes[..8] != OPTIMIZER_MAGIC {
            return Err(TrainError::InvalidModel(
                "bad production optimizer artifact",
            ));
        }
        let checksum_offset = bytes.len() - 8;
        let expected = u64::from_le_bytes(
            bytes[checksum_offset..]
                .try_into()
                .map_err(|_| TrainError::InvalidModel("bad production optimizer checksum"))?,
        );
        if fnv1a(&bytes[..checksum_offset]) != expected {
            return Err(TrainError::InvalidModel(
                "bad production optimizer checksum",
            ));
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if version != OPTIMIZER_VERSION {
            return Err(TrainError::InvalidModel(
                "unsupported production optimizer version",
            ));
        }
        let residual_count = usize::try_from(u64::from_le_bytes(bytes[68..76].try_into().unwrap()))
            .map_err(|_| TrainError::InvalidModel("production optimizer residual overflow"))?;
        if bytes.len() != 84_usize.saturating_add(residual_count.saturating_mul(8)) {
            return Err(TrainError::InvalidModel(
                "production optimizer residual length mismatch",
            ));
        }
        let residuals = bytes[76..checksum_offset]
            .chunks_exact(8)
            .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        Ok(Self {
            tokenizer_hash: u64::from_le_bytes(bytes[12..20].try_into().unwrap()),
            token_stream_hash: u64::from_le_bytes(bytes[20..28].try_into().unwrap()),
            bound_model_hash: u64::from_le_bytes(bytes[28..36].try_into().unwrap()),
            step: u64::from_le_bytes(bytes[36..44].try_into().unwrap()),
            schedule_hash: u64::from_le_bytes(bytes[44..52].try_into().unwrap()),
            next_epoch: u64::from_le_bytes(bytes[52..60].try_into().unwrap()),
            next_window: u64::from_le_bytes(bytes[60..68].try_into().unwrap()),
            residuals,
        })
    }

    fn bytes_without_checksum(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(84 + self.residuals.len() * 8);
        bytes.extend_from_slice(OPTIMIZER_MAGIC);
        bytes.extend_from_slice(&OPTIMIZER_VERSION.to_le_bytes());
        bytes.extend_from_slice(&self.tokenizer_hash.to_le_bytes());
        bytes.extend_from_slice(&self.token_stream_hash.to_le_bytes());
        bytes.extend_from_slice(&self.bound_model_hash.to_le_bytes());
        bytes.extend_from_slice(&self.step.to_le_bytes());
        bytes.extend_from_slice(&self.schedule_hash.to_le_bytes());
        bytes.extend_from_slice(&self.next_epoch.to_le_bytes());
        bytes.extend_from_slice(&self.next_window.to_le_bytes());
        bytes.extend_from_slice(&(self.residuals.len() as u64).to_le_bytes());
        for residual in &self.residuals {
            bytes.extend_from_slice(&residual.to_le_bytes());
        }
        bytes
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductionFullTrainTrace {
    pub profile: &'static str,
    pub parameter_count: usize,
    pub tokenizer_hash: u64,
    pub token_stream_hash: u64,
    pub context_tokens: usize,
    pub windows: usize,
    pub spread_windows: bool,
    pub targets_per_window: usize,
    pub training_workers: usize,
    pub supervised_targets: usize,
    pub embedding_learning_rate_boost_shift: u8,
    pub epochs: usize,
    pub initial_mistakes: usize,
    pub final_mistakes: usize,
    pub optimizer_steps: usize,
    pub total_optimizer_step: u64,
    pub batch_windows: usize,
    pub evaluation_windows: usize,
    pub start_epoch: u64,
    pub start_window: u64,
    pub next_epoch: u64,
    pub next_window: u64,
    pub schedule_complete: bool,
    pub learning_rate_shifts: [u8; 13],
    pub forward_shifts: [u8; 6],
    pub output_backward_shift: u8,
    pub probability_gradient_fractional_bits: u8,
    pub probability_normalization: SoftmaxNormalization,
    pub probability_gradient_shift_delta: u8,
    pub effective_output_backward_shift: u8,
    pub effective_output_learning_rate_shift: u8,
    pub effective_bias_learning_rate_shift: u8,
    pub gradient_saturation_count: usize,
    pub residual_saturation_count: usize,
    pub weight_saturation_count: usize,
    pub movement_l1: [u64; 13],
    pub gradient_nonzero_count: [u64; 13],
    pub residual_carry_count: [u64; 13],
    pub update_nonzero_count: [u64; 13],
    pub saturation_by_group: [u64; 13],
    pub residual_saturation_by_group: [u64; 13],
    pub backward_ste_rescue_count: u64,
    pub backward_quantization_count: u64,
    pub initial_model_hash: u64,
    pub final_model_hash: u64,
    pub optimizer_state_hash: u64,
}

impl ProductionFullTrainTrace {
    pub fn to_json_line(self) -> String {
        let movement = GROUP_NAMES
            .iter()
            .zip(self.movement_l1)
            .map(|(name, value)| format!("\"{name}\":{value}"))
            .collect::<Vec<_>>()
            .join(",");
        let moved = GROUP_NAMES
            .iter()
            .zip(self.movement_l1)
            .filter(|(_, value)| *value > 0)
            .map(|(name, _)| format!("\"{name}\""))
            .collect::<Vec<_>>()
            .join(",");
        let render_groups = |values: [u64; 13]| {
            GROUP_NAMES
                .iter()
                .zip(values)
                .map(|(name, value)| format!("\"{name}\":{value}"))
                .collect::<Vec<_>>()
                .join(",")
        };
        let learning_rate_shifts = GROUP_NAMES
            .iter()
            .zip(self.learning_rate_shifts)
            .map(|(name, value)| format!("\"{name}\":{value}"))
            .collect::<Vec<_>>()
            .join(",");
        let mut output = format!(
            concat!(
                "{{\"schema\":\"nsrl.production_full_train_smoke.v1\",",
                "\"profile\":\"{}\",\"parameter_count\":{},",
                "\"bindings\":{{\"tokenizer_hash\":\"0x{:016x}\",\"token_stream_hash\":\"0x{:016x}\"}},",
                "\"training\":{{\"optimizer\":\"integer_residual_sgd\",\"backward\":\"full_quantized_straight_through\",\"context_tokens\":{},\"windows\":{},\"evaluation_windows\":{},\"epochs\":{},\"batch_windows\":{},\"optimizer_steps\":{},\"total_optimizer_step\":{},\"initial_mistakes\":{},\"final_mistakes\":{},\"learning_rate_shifts\":{{{}}},\"output_backward_shift\":{},\"probability_gradient_fractional_bits\":{},\"probability_normalization\":\"{}\",\"probability_gradient_shift_delta\":{},\"effective_output_backward_shift\":{},\"effective_output_learning_rate_shift\":{},\"effective_bias_learning_rate_shift\":{}}},",
                "\"forward_shifts\":{{\"qkv\":{},\"o\":{},\"up\":{},\"gate\":{},\"down\":{},\"output\":{}}},",
                "\"cursor\":{{\"start_epoch\":{},\"start_window\":{},\"next_epoch\":{},\"next_window\":{},\"schedule_complete\":{}}},",
                "\"movement_l1\":{{{}}},\"moved_parameter_groups\":[{}],",
                "\"diagnostics\":{{\"gradient_nonzero_count\":{{{}}},\"residual_carry_count\":{{{}}},\"update_nonzero_count\":{{{}}},\"saturation_by_group\":{{{}}},\"residual_saturation_by_group\":{{{}}},\"backward_ste_rescue_count\":{},\"backward_quantization_count\":{},\"backward_ste_rescue_per_million\":{}}},",
                "\"health\":{{\"gradient_saturation_count\":{},\"residual_saturation_count\":{},\"weight_saturation_count\":{}}},",
                "\"hashes\":{{\"initial_model\":\"0x{:016x}\",\"final_model\":\"0x{:016x}\",\"optimizer_state\":\"0x{:016x}\"}},",
                "\"gates\":{{\"all_parameter_groups_moved\":{},\"model_hash_changed\":{},\"resumable_optimizer_state\":true,\"batched_residual_updates\":true}},",
                "\"known_non_claims\":[\"bounded_full_backward_smoke_not_scaling_run\",\"straight_through_backward_derivatives\",\"not_open_generation_quality\"]}}\n"
            ),
            self.profile,
            self.parameter_count,
            self.tokenizer_hash,
            self.token_stream_hash,
            self.context_tokens,
            self.windows,
            self.evaluation_windows,
            self.epochs,
            self.batch_windows,
            self.optimizer_steps,
            self.total_optimizer_step,
            self.initial_mistakes,
            self.final_mistakes,
            learning_rate_shifts,
            self.output_backward_shift,
            self.probability_gradient_fractional_bits,
            self.probability_normalization.as_str(),
            self.probability_gradient_shift_delta,
            self.effective_output_backward_shift,
            self.effective_output_learning_rate_shift,
            self.effective_bias_learning_rate_shift,
            self.forward_shifts[0],
            self.forward_shifts[1],
            self.forward_shifts[2],
            self.forward_shifts[3],
            self.forward_shifts[4],
            self.forward_shifts[5],
            self.start_epoch,
            self.start_window,
            self.next_epoch,
            self.next_window,
            self.schedule_complete,
            movement,
            moved,
            render_groups(self.gradient_nonzero_count),
            render_groups(self.residual_carry_count),
            render_groups(self.update_nonzero_count),
            render_groups(self.saturation_by_group),
            render_groups(self.residual_saturation_by_group),
            self.backward_ste_rescue_count,
            self.backward_quantization_count,
            self.backward_ste_rescue_count.saturating_mul(1_000_000)
                / self.backward_quantization_count.max(1),
            self.gradient_saturation_count,
            self.residual_saturation_count,
            self.weight_saturation_count,
            self.initial_model_hash,
            self.final_model_hash,
            self.optimizer_state_hash,
            self.movement_l1.iter().all(|&value| value > 0),
            self.initial_model_hash != self.final_model_hash,
        );
        if self.spread_windows {
            output = output.replace(
                ",\"evaluation_windows\"",
                ",\"window_selection\":\"deterministic_uniform_target_rank_over_all_documents\",\"evaluation_windows\"",
            );
        }
        if self.targets_per_window > 1 {
            output = output.replace(
                ",\"evaluation_windows\"",
                &format!(
                    ",\"target_policy\":\"causal_suffix_mean_v1\",\"targets_per_window\":{},\"target_mean_shift\":{},\"mean_reduction\":\"parameter_update_power_of_two_shift\",\"supervised_targets\":{},\"evaluation_windows\"",
                    self.targets_per_window,
                    self.targets_per_window.trailing_zeros(),
                    self.supervised_targets,
                ),
            );
        }
        if self.embedding_learning_rate_boost_shift > 0 {
            output = output.replace(
                ",\"learning_rate_shifts\"",
                &format!(
                    ",\"embedding_learning_rate_boost_shift\":{},\"learning_rate_shifts\"",
                    self.embedding_learning_rate_boost_shift,
                ),
            );
        }
        if self.training_workers > 1 {
            output = output.replace(
                ",\"learning_rate_shifts\"",
                &format!(
                    ",\"training_workers\":{},\"learning_rate_shifts\"",
                    self.training_workers,
                ),
            );
        }
        output
    }
}

#[derive(Clone)]
struct LayerCache {
    input: Vec<i16>,
    attention_input: Vec<i16>,
    q: Vec<i16>,
    k: Vec<i16>,
    v: Vec<i16>,
    context: Vec<i16>,
    attention_residual: Vec<i16>,
    mlp_input: Vec<i16>,
    up: Vec<i16>,
    gate: Vec<i16>,
    gated: Vec<i16>,
}

struct ForwardCache {
    layers: Vec<LayerCache>,
    final_hidden: Vec<i16>,
    target_rows: usize,
    features: Vec<i16>,
    logits: Vec<i32>,
    probabilities: Vec<i32>,
}

pub(super) struct CoarseGradientSnapshot {
    pub residuals: Vec<i64>,
    pub output_gradient_vector_hash: u64,
    pub output_gradient_sum: i64,
    pub output_gradient_l1: u64,
    pub output_gradient_max_abs: u64,
    pub backward_ste_rescue_count: u64,
    pub backward_quantization_count: u64,
    pub stochastic_round_up_count: u64,
    pub gradient_saturation_count: usize,
    pub residual_saturation_count: usize,
}

#[derive(Default)]
struct UpdateStats {
    movement: [u64; 13],
    gradient_nonzero: [u64; 13],
    residual_carry: [u64; 13],
    update_nonzero: [u64; 13],
    saturation_by_group: [u64; 13],
    residual_saturation_by_group: [u64; 13],
    backward_ste_rescue: u64,
    backward_quantization: u64,
    gradient_saturation: usize,
    residual_saturation: usize,
    weight_saturation: usize,
}

impl UpdateStats {
    fn merge(&mut self, other: Self) {
        for (left, right) in self.movement.iter_mut().zip(other.movement) {
            *left = left.saturating_add(right);
        }
        for (left, right) in self.gradient_nonzero.iter_mut().zip(other.gradient_nonzero) {
            *left = left.saturating_add(right);
        }
        for (left, right) in self.residual_carry.iter_mut().zip(other.residual_carry) {
            *left = left.saturating_add(right);
        }
        for (left, right) in self.update_nonzero.iter_mut().zip(other.update_nonzero) {
            *left = left.saturating_add(right);
        }
        for (left, right) in self
            .saturation_by_group
            .iter_mut()
            .zip(other.saturation_by_group)
        {
            *left = left.saturating_add(right);
        }
        for (left, right) in self
            .residual_saturation_by_group
            .iter_mut()
            .zip(other.residual_saturation_by_group)
        {
            *left = left.saturating_add(right);
        }
        self.backward_ste_rescue = self
            .backward_ste_rescue
            .saturating_add(other.backward_ste_rescue);
        self.backward_quantization = self
            .backward_quantization
            .saturating_add(other.backward_quantization);
        self.gradient_saturation = self
            .gradient_saturation
            .saturating_add(other.gradient_saturation);
        self.residual_saturation = self
            .residual_saturation
            .saturating_add(other.residual_saturation);
        self.weight_saturation = self
            .weight_saturation
            .saturating_add(other.weight_saturation);
    }
}

struct BackwardQuantizer {
    mode: BackwardQuantizationMode,
    seed: u64,
    operation: u64,
    stochastic_round_up_count: u64,
}

impl BackwardQuantizer {
    const fn new(mode: BackwardQuantizationMode, seed: u64) -> Self {
        Self {
            mode,
            seed,
            operation: 0,
            stochastic_round_up_count: 0,
        }
    }

    const fn late_quantization(&self) -> bool {
        matches!(
            self.mode,
            BackwardQuantizationMode::LateRhu | BackwardQuantizationMode::LateStochastic
        )
    }

    fn shift(&mut self, value: i64, shift: u8, stats: &mut UpdateStats) -> i64 {
        stats.backward_quantization = stats.backward_quantization.saturating_add(1);
        let rounded = if self.mode == BackwardQuantizationMode::LateStochastic {
            self.stochastic_shift(value, shift)
        } else {
            round_shift_rhu_i64(value, shift)
        };
        if self.mode == BackwardQuantizationMode::RescuedRhu && rounded == 0 && value != 0 {
            stats.backward_ste_rescue = stats.backward_ste_rescue.saturating_add(1);
            value.signum()
        } else {
            rounded
        }
    }

    fn ratio(
        &mut self,
        numerator: i64,
        denominator: i64,
        stats: &mut UpdateStats,
    ) -> Result<i64, TrainError> {
        stats.backward_quantization = stats.backward_quantization.saturating_add(1);
        if denominator <= 0 {
            return Err(TrainError::InvalidConfig);
        }
        let rounded = if self.mode == BackwardQuantizationMode::LateStochastic {
            self.stochastic_ratio(numerator, denominator)?
        } else {
            nearest_ratio(numerator, denominator)?
        };
        if self.mode == BackwardQuantizationMode::RescuedRhu && rounded == 0 && numerator != 0 {
            stats.backward_ste_rescue = stats.backward_ste_rescue.saturating_add(1);
            Ok(numerator.signum())
        } else {
            Ok(rounded)
        }
    }

    fn stochastic_shift(&mut self, value: i64, shift: u8) -> i64 {
        if shift == 0 {
            return value;
        }
        let shift = shift.min(62);
        let denominator = 1_u64 << shift;
        let quotient = value >> shift;
        let remainder = (value & ((1_i64 << shift) - 1)) as u64;
        quotient + i64::from(self.choose_round_up(remainder, denominator))
    }

    fn stochastic_ratio(&mut self, numerator: i64, denominator: i64) -> Result<i64, TrainError> {
        let quotient = numerator.div_euclid(denominator);
        let remainder = numerator.rem_euclid(denominator) as u64;
        let denominator = denominator as u64;
        quotient
            .checked_add(i64::from(self.choose_round_up(remainder, denominator)))
            .ok_or(TrainError::CoreRejected(
                "production_training_stochastic_ratio",
            ))
    }

    fn choose_round_up(&mut self, remainder: u64, denominator: u64) -> bool {
        let operation = self.operation;
        self.operation = self.operation.wrapping_add(1);
        if remainder == 0 {
            return false;
        }
        let sample = sample_below(self.seed, operation, denominator);
        let round_up = sample < remainder;
        self.stochastic_round_up_count = self
            .stochastic_round_up_count
            .saturating_add(u64::from(round_up));
        round_up
    }
}

struct ParameterRanges {
    embeddings: Range<usize>,
    attention_rms: Range<usize>,
    mlp_rms: Range<usize>,
    final_rms: Range<usize>,
    q: Range<usize>,
    k: Range<usize>,
    v: Range<usize>,
    o: Range<usize>,
    up: Range<usize>,
    gate: Range<usize>,
    down: Range<usize>,
    output: Range<usize>,
    bias: Range<usize>,
}

impl ParameterRanges {
    fn new(model: &ProductionModelV1) -> Self {
        let mut cursor = 0;
        let mut take = |len: usize| {
            let range = cursor..cursor + len;
            cursor += len;
            range
        };
        let ranges = Self {
            embeddings: take(model.embeddings.len()),
            attention_rms: take(model.attention_rms_weights.len()),
            mlp_rms: take(model.mlp_rms_weights.len()),
            final_rms: take(model.final_rms_weights.len()),
            q: take(model.q_weights.len()),
            k: take(model.k_weights.len()),
            v: take(model.v_weights.len()),
            o: take(model.o_weights.len()),
            up: take(model.up_weights.len()),
            gate: take(model.gate_weights.len()),
            down: take(model.down_weights.len()),
            output: take(model.output_weights.len()),
            bias: take(model.output_bias_q8.len()),
        };
        debug_assert_eq!(cursor, model.parameter_count());
        ranges
    }
}

pub fn train_production_full_smoke(
    model: &mut ProductionModelV1,
    tokens: &[u32],
    token_stream_hash: u64,
    config: ProductionFullTrainConfig,
    state: Option<ProductionOptimizerStateV2>,
) -> Result<(ProductionFullTrainTrace, ProductionOptimizerStateV2), TrainError> {
    model.validate()?;
    if config.context_tokens == 0
        || config.context_tokens > model.config.context_tokens
        || config.max_windows == 0
        || config.targets_per_window == 0
        || config.targets_per_window > config.context_tokens
        || !config.targets_per_window.is_power_of_two()
        || config.training_workers == 0
        || config.training_workers > 256
        || config.embedding_learning_rate_boost_shift
            > config
                .embedding_learning_rate_shift
                .saturating_add(target_mean_shift(config))
        || config.epochs == 0
        || config.batch_windows == 0
        || config.max_optimizer_steps == 0
        || config.evaluation_windows == 0
        || config.output_backward_shift.is_some_and(|shift| shift > 30)
        || !(15..=31).contains(&config.probability_gradient_fractional_bits)
        || config
            .output_backward_shift
            .unwrap_or(model.scales.output_shift)
            .saturating_add(config.probability_gradient_fractional_bits - 15)
            > 62
        || config
            .output_learning_rate_shift
            .saturating_add(config.probability_gradient_fractional_bits - 15)
            .saturating_add(target_mean_shift(config))
            > 62
        || config
            .vector_learning_rate_shift
            .saturating_add(config.probability_gradient_fractional_bits - 15)
            .saturating_add(target_mean_shift(config))
            > 62
        || effective_learning_rate_shifts(config)
            .iter()
            .any(|&shift| shift > 62)
    {
        return Err(TrainError::InvalidConfig);
    }
    let windows = if config.spread_windows {
        spread_document_windows(tokens, config.context_tokens, config.max_windows)
    } else {
        document_windows(tokens, config.context_tokens, config.max_windows)
    };
    if windows.is_empty() {
        return Err(TrainError::InvalidConfig);
    }
    let initial_model_hash = model.model_hash();
    let evaluation_windows = config.evaluation_windows.min(windows.len());
    let initial_mistakes = evaluate_mistakes(model, &windows[..evaluation_windows])?;
    let mut state =
        state.unwrap_or_else(|| ProductionOptimizerStateV2::new(model, token_stream_hash, config));
    state.validate_binding(model, token_stream_hash, config)?;
    if state.next_epoch > config.epochs as u64
        || state.next_window > windows.len() as u64
        || (state.next_epoch == config.epochs as u64 && state.next_window != 0)
    {
        return Err(TrainError::InvalidModel(
            "production optimizer cursor mismatch",
        ));
    }
    let start_epoch = state.next_epoch;
    let start_window = state.next_window;
    let ranges = ParameterRanges::new(model);
    let mut stats = UpdateStats::default();
    let mut optimizer_steps = 0_usize;
    let mut supervised_targets = 0_usize;
    while state.next_epoch < config.epochs as u64 && optimizer_steps < config.max_optimizer_steps {
        let batch_start = state.next_window as usize;
        let batch_end = batch_start
            .saturating_add(config.batch_windows)
            .min(windows.len());
        for (index, (context, target)) in windows[batch_start..batch_end].iter().enumerate() {
            let targets = causal_suffix_targets(context, *target, config.targets_per_window)?;
            let cache = forward_cache_for_target_rows(
                model,
                context,
                targets.len(),
                config.training_workers,
                config.probability_gradient_fractional_bits,
                config.probability_normalization,
            )?;
            backward_update_targets(
                model,
                context,
                &targets,
                cache,
                config,
                &ranges,
                &mut state.residuals,
                index + 1 == batch_end - batch_start,
                &mut stats,
            )?;
            supervised_targets = supervised_targets.saturating_add(targets.len());
        }
        state.step = state.step.checked_add(1).ok_or(TrainError::InvalidConfig)?;
        optimizer_steps = optimizer_steps.saturating_add(1);
        if batch_end == windows.len() {
            state.next_epoch = state.next_epoch.saturating_add(1);
            state.next_window = 0;
        } else {
            state.next_window = batch_end as u64;
        }
    }
    state.bound_model_hash = model.model_hash();
    let final_mistakes = evaluate_mistakes(model, &windows[..evaluation_windows])?;
    let trace = ProductionFullTrainTrace {
        profile: model.config.profile_id().unwrap_or("custom"),
        parameter_count: model.parameter_count(),
        tokenizer_hash: model.tokenizer_hash,
        token_stream_hash,
        context_tokens: config.context_tokens,
        windows: windows.len(),
        spread_windows: config.spread_windows,
        targets_per_window: config.targets_per_window,
        training_workers: config.training_workers,
        supervised_targets,
        embedding_learning_rate_boost_shift: config.embedding_learning_rate_boost_shift,
        epochs: config.epochs,
        initial_mistakes,
        final_mistakes,
        optimizer_steps,
        total_optimizer_step: state.step,
        batch_windows: config.batch_windows,
        evaluation_windows,
        start_epoch,
        start_window,
        next_epoch: state.next_epoch,
        next_window: state.next_window,
        schedule_complete: state.next_epoch == config.epochs as u64,
        learning_rate_shifts: effective_learning_rate_shifts(config),
        forward_shifts: scale_shifts(model.scales),
        output_backward_shift: config
            .output_backward_shift
            .unwrap_or(model.scales.output_shift),
        probability_gradient_fractional_bits: config.probability_gradient_fractional_bits,
        probability_normalization: config.probability_normalization,
        probability_gradient_shift_delta: config.probability_gradient_fractional_bits - 15,
        effective_output_backward_shift: config
            .output_backward_shift
            .unwrap_or(model.scales.output_shift)
            .saturating_add(config.probability_gradient_fractional_bits - 15),
        effective_output_learning_rate_shift: config
            .output_learning_rate_shift
            .saturating_add(config.probability_gradient_fractional_bits - 15)
            .saturating_add(target_mean_shift(config)),
        effective_bias_learning_rate_shift: config
            .vector_learning_rate_shift
            .saturating_add(config.probability_gradient_fractional_bits - 15)
            .saturating_add(target_mean_shift(config)),
        gradient_saturation_count: stats.gradient_saturation,
        residual_saturation_count: stats.residual_saturation,
        weight_saturation_count: stats.weight_saturation,
        movement_l1: stats.movement,
        gradient_nonzero_count: stats.gradient_nonzero,
        residual_carry_count: stats.residual_carry,
        update_nonzero_count: stats.update_nonzero,
        saturation_by_group: stats.saturation_by_group,
        residual_saturation_by_group: stats.residual_saturation_by_group,
        backward_ste_rescue_count: stats.backward_ste_rescue,
        backward_quantization_count: stats.backward_quantization,
        initial_model_hash,
        final_model_hash: model.model_hash(),
        optimizer_state_hash: state.state_hash(),
    };
    Ok((trace, state))
}

fn forward_cache(
    model: &ProductionModelV1,
    context_tokens: &[u32],
    probability_gradient_fractional_bits: u8,
    probability_normalization: SoftmaxNormalization,
) -> Result<ForwardCache, TrainError> {
    forward_cache_for_target_rows(
        model,
        context_tokens,
        1,
        1,
        probability_gradient_fractional_bits,
        probability_normalization,
    )
}

fn forward_cache_for_target_rows(
    model: &ProductionModelV1,
    context_tokens: &[u32],
    target_rows: usize,
    training_workers: usize,
    probability_gradient_fractional_bits: u8,
    probability_normalization: SoftmaxNormalization,
) -> Result<ForwardCache, TrainError> {
    let config = model.config;
    if context_tokens.is_empty()
        || context_tokens.len() > config.context_tokens
        || target_rows == 0
        || target_rows > context_tokens.len()
        || training_workers == 0
        || context_tokens
            .iter()
            .any(|&token| token as usize >= config.vocab_size)
    {
        return Err(TrainError::InvalidConfig);
    }
    let mut hidden = Vec::with_capacity(context_tokens.len() * config.d_model);
    for &token in context_tokens {
        let start = token as usize * config.d_model;
        hidden.extend_from_slice(&model.embeddings[start..start + config.d_model]);
    }
    let seq_len = context_tokens.len();
    let total = checked_product(seq_len, config.d_model)?;
    let matrix = checked_product(config.d_model, config.d_model)?;
    let up_matrix = checked_product(config.d_model, config.hidden_dim)?;
    let down_matrix = checked_product(config.hidden_dim, config.d_model)?;
    let qkv_scales = scales(config.d_model, model.scales.qkv_shift);
    let o_scales = scales(config.d_model, model.scales.o_shift);
    let up_scales = scales(config.hidden_dim, model.scales.up_shift);
    let gate_scales = scales(config.hidden_dim, model.scales.gate_shift);
    let down_scales = scales(config.d_model, model.scales.down_shift);
    let mut layers = Vec::with_capacity(config.layers);
    for layer in 0..config.layers {
        let input = hidden.clone();
        let rms = layer * config.d_model..(layer + 1) * config.d_model;
        let attention_input = rms_rows(
            &hidden,
            &model.attention_rms_weights[rms.clone()],
            config.d_model,
        )?;
        let range = layer * matrix..(layer + 1) * matrix;
        let params = SelfAttentionI16Params {
            q: linear_params(
                &model.q_weights[range.clone()],
                &qkv_scales,
                config.d_model,
                config.d_model,
            ),
            k: linear_params(
                &model.k_weights[range.clone()],
                &qkv_scales,
                config.d_model,
                config.d_model,
            ),
            v: linear_params(
                &model.v_weights[range.clone()],
                &qkv_scales,
                config.d_model,
                config.d_model,
            ),
            o: linear_params(
                &model.o_weights[range],
                &o_scales,
                config.d_model,
                config.d_model,
            ),
            seq_len,
            d_model: config.d_model,
            heads: config.heads,
            causal: true,
        };
        let head_dim = config.d_model / config.heads;
        let mut q = vec![0_i16; total];
        let mut k = vec![0_i16; total];
        let mut v = vec![0_i16; total];
        let mut attention_context = vec![0_i16; total];
        let mut state_kv = vec![0_i64; config.heads * head_dim * head_dim];
        let mut key_sums = vec![0_i64; config.heads * head_dim];
        let mut attention_output = vec![0_i16; total];
        linear_attention_i16_q15_checked(
            &attention_input,
            params,
            LinearAttentionWorkspace {
                q: &mut q,
                k: &mut k,
                v: &mut v,
                context: &mut attention_context,
                state_kv: &mut state_kv,
                key_sums: &mut key_sums,
            },
            &mut attention_output,
        )
        .ok_or(TrainError::CoreRejected("production_training_attention"))?;
        let attention_residual = add_rows(&hidden, &attention_output);
        let mlp_input = rms_rows(
            &attention_residual,
            &model.mlp_rms_weights[rms],
            config.d_model,
        )?;
        let up_range = layer * up_matrix..(layer + 1) * up_matrix;
        let down_range = layer * down_matrix..(layer + 1) * down_matrix;
        let mlp_params = GatedMlpI16Params {
            up: linear_params(
                &model.up_weights[up_range.clone()],
                &up_scales,
                config.d_model,
                config.hidden_dim,
            ),
            gate: linear_params(
                &model.gate_weights[up_range],
                &gate_scales,
                config.d_model,
                config.hidden_dim,
            ),
            down: linear_params(
                &model.down_weights[down_range],
                &down_scales,
                config.hidden_dim,
                config.d_model,
            ),
            seq_len,
            d_model: config.d_model,
            hidden_dim: config.hidden_dim,
        };
        let mut up = vec![0_i16; seq_len * config.hidden_dim];
        let mut gate = vec![0_i16; seq_len * config.hidden_dim];
        let mut gated = vec![0_i16; seq_len * config.hidden_dim];
        let mut mlp_output = vec![0_i16; total];
        gated_mlp_i16_q15_checked(
            &mlp_input,
            mlp_params,
            GatedMlpWorkspace {
                up: &mut up,
                gate: &mut gate,
                gated: &mut gated,
            },
            &mut mlp_output,
        )
        .ok_or(TrainError::CoreRejected("production_training_mlp"))?;
        hidden = add_rows(&attention_residual, &mlp_output);
        layers.push(LayerCache {
            input,
            attention_input,
            q,
            k,
            v,
            context: attention_context,
            attention_residual,
            mlp_input,
            up,
            gate,
            gated,
        });
    }
    let final_hidden = hidden;
    let first_target_row = seq_len - target_rows;
    let features = rms_rows(
        &final_hidden[first_target_row * config.d_model..],
        &model.final_rms_weights,
        config.d_model,
    )?;
    let (logits, probabilities) = output_rows(
        model,
        &features,
        training_workers,
        probability_gradient_fractional_bits,
        probability_normalization,
    )?;
    Ok(ForwardCache {
        layers,
        final_hidden,
        target_rows,
        features,
        logits,
        probabilities,
    })
}

fn output_gradient(
    cache: &ForwardCache,
    target: usize,
    config: ProductionFullTrainConfig,
    source: GradientSource,
    stochastic_seed: u64,
) -> Result<(Vec<i32>, u8), TrainError> {
    if cache.target_rows != 1 {
        return Err(TrainError::InvalidConfig);
    }
    output_gradient_row(cache, 0, target, config, source, stochastic_seed)
}

fn output_gradient_row(
    cache: &ForwardCache,
    row: usize,
    target: usize,
    config: ProductionFullTrainConfig,
    source: GradientSource,
    stochastic_seed: u64,
) -> Result<(Vec<i32>, u8), TrainError> {
    if row >= cache.target_rows
        || cache.logits.len() != cache.probabilities.len()
        || !cache.logits.len().is_multiple_of(cache.target_rows)
    {
        return Err(TrainError::InvalidConfig);
    }
    let vocabulary = cache.logits.len() / cache.target_rows;
    if target >= vocabulary {
        return Err(TrainError::InvalidConfig);
    }
    let range = row * vocabulary..(row + 1) * vocabulary;
    let logits = &cache.logits[range.clone()];
    let probabilities = &cache.probabilities[range];
    match source {
        GradientSource::NormalizedProbability => {
            let mut gradient = probabilities.to_vec();
            let gradient_scale = ((1_u64 << config.probability_gradient_fractional_bits) - 1)
                .try_into()
                .map_err(|_| TrainError::InvalidConfig)?;
            gradient[target] = gradient[target].saturating_sub(gradient_scale);
            Ok((gradient, config.probability_gradient_fractional_bits - 15))
        }
        GradientSource::MassCorrectedNormalizedProbability => {
            let mut gradient = probabilities.to_vec();
            let non_target_mass = gradient
                .iter()
                .enumerate()
                .filter(|&(index, _)| index != target)
                .try_fold(0_i64, |sum, (_, &value)| {
                    sum.checked_add(i64::from(value))
                        .ok_or(TrainError::CoreRejected(
                            "production_probability_mass_overflow",
                        ))
                })?;
            gradient[target] = i32::try_from(-non_target_mass).map_err(|_| {
                TrainError::CoreRejected("production_mass_corrected_gradient_exceeds_i32")
            })?;
            Ok((gradient, config.probability_gradient_fractional_bits - 15))
        }
        GradientSource::ReciprocalFreeWeights => {
            let maximum = logits
                .iter()
                .copied()
                .max()
                .ok_or(TrainError::InvalidConfig)?;
            let mut gradient = logits
                .iter()
                .map(|&logit| i32::from(base2_exp_neg_q15(logit.saturating_sub(maximum))))
                .collect::<Vec<_>>();
            let non_target_mass = gradient
                .iter()
                .enumerate()
                .filter(|&(index, _)| index != target)
                .try_fold(0_i64, |sum, (_, &value)| {
                    sum.checked_add(i64::from(value))
                        .ok_or(TrainError::CoreRejected(
                            "production_exponent_weight_mass_overflow",
                        ))
                })?;
            gradient[target] = i32::try_from(-non_target_mass).map_err(|_| {
                TrainError::CoreRejected("production_reciprocal_free_gradient_exceeds_i32")
            })?;
            // The raw exponent residual is approximately vocab_size times a
            // normalized Q15 residual on a uniform surface. A fixed power-of-
            // two shift prevents downstream i16 saturation without introducing
            // a per-example reciprocal. The remaining exponent mass variation
            // is intentionally measured as sample reweighting by the audit.
            let vocabulary_shift = usize::BITS - (gradient.len() - 1).leading_zeros();
            Ok((gradient, vocabulary_shift as u8))
        }
        GradientSource::SystematicFixedMassK15 => {
            systematic_fixed_mass_gradient(logits, target, 1_u32 << 15, stochastic_seed)
        }
        GradientSource::SystematicFixedMassK16 => {
            systematic_fixed_mass_gradient(logits, target, 1_u32 << 16, stochastic_seed)
        }
        GradientSource::SystematicFixedMassK18 => {
            systematic_fixed_mass_gradient(logits, target, 1_u32 << 18, stochastic_seed)
        }
    }
}

fn systematic_fixed_mass_gradient(
    logits: &[i32],
    target: usize,
    mass: u32,
    stochastic_seed: u64,
) -> Result<(Vec<i32>, u8), TrainError> {
    let maximum = logits
        .iter()
        .copied()
        .max()
        .ok_or(TrainError::InvalidConfig)?;
    let weights = logits
        .iter()
        .map(|&logit| base2_exp_neg_q47(logit.saturating_sub(maximum)))
        .collect::<Vec<_>>();
    let total = weights.iter().try_fold(0_u64, |sum, &weight| {
        sum.checked_add(weight).ok_or(TrainError::CoreRejected(
            "production_systematic_fixed_mass_weight_overflow",
        ))
    })?;
    if total == 0 || target >= weights.len() {
        return Err(TrainError::InvalidConfig);
    }
    let phase = sample_below(stochastic_seed, 0x5359_5354_454d_4154, total);
    let mut prefix = 0_u64;
    let mut previous = 0_u64;
    let mut apportioned = Vec::with_capacity(weights.len());
    for weight in weights {
        prefix = prefix.checked_add(weight).ok_or(TrainError::CoreRejected(
            "production_systematic_fixed_mass_prefix_overflow",
        ))?;
        let cumulative =
            (u128::from(mass) * u128::from(prefix) + u128::from(phase)) / u128::from(total);
        let cumulative = u64::try_from(cumulative).map_err(|_| {
            TrainError::CoreRejected("production_systematic_fixed_mass_cumulative_overflow")
        })?;
        apportioned.push(i32::try_from(cumulative - previous).map_err(|_| {
            TrainError::CoreRejected("production_systematic_fixed_mass_coordinate_overflow")
        })?);
        previous = cumulative;
    }
    if previous != u64::from(mass) {
        return Err(TrainError::CoreRejected(
            "production_systematic_fixed_mass_mass_mismatch",
        ));
    }
    apportioned[target] = apportioned[target]
        .checked_sub(i32::try_from(mass).map_err(|_| TrainError::InvalidConfig)?)
        .ok_or(TrainError::CoreRejected(
            "production_systematic_fixed_mass_target_overflow",
        ))?;
    if apportioned
        .iter()
        .map(|&value| i64::from(value))
        .sum::<i64>()
        != 0
    {
        return Err(TrainError::CoreRejected(
            "production_systematic_fixed_mass_zero_sum_mismatch",
        ));
    }
    let fractional_bits = mass.trailing_zeros() as u8;
    Ok((apportioned, fractional_bits.saturating_sub(15)))
}

pub(super) fn coarse_gradients_for_window_with_spec(
    model: &mut ProductionModelV1,
    context_tokens: &[u32],
    target: usize,
    config: ProductionFullTrainConfig,
    spec: GradientProposalSpec,
) -> Result<CoarseGradientSnapshot, TrainError> {
    let cache = forward_cache(
        model,
        context_tokens,
        config.probability_gradient_fractional_bits,
        config.probability_normalization,
    )?;
    let (output_gradient, _) =
        output_gradient(&cache, target, config, spec.source, spec.stochastic_seed)?;
    let output_gradient_vector_hash = output_gradient.iter().fold(FNV_OFFSET, |hash, value| {
        value.to_le_bytes().into_iter().fold(hash, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME)
        })
    });
    let output_gradient_sum = output_gradient.iter().try_fold(0_i64, |sum, &value| {
        sum.checked_add(i64::from(value))
            .ok_or(TrainError::CoreRejected(
                "production_output_gradient_sum_overflow",
            ))
    })?;
    let output_gradient_l1 = output_gradient.iter().try_fold(0_u64, |sum, &value| {
        sum.checked_add(i64::from(value).unsigned_abs())
            .ok_or(TrainError::CoreRejected(
                "production_output_gradient_l1_overflow",
            ))
    })?;
    let output_gradient_max_abs = output_gradient
        .iter()
        .map(|&value| i64::from(value).unsigned_abs())
        .max()
        .unwrap_or(0);
    let ranges = ParameterRanges::new(model);
    let mut residuals = vec![0_i64; model.parameter_count()];
    let mut stats = UpdateStats::default();
    let mut quantizer = BackwardQuantizer::new(spec.quantization, spec.stochastic_seed);
    backward_update_with_spec(
        model,
        context_tokens,
        &[target],
        cache,
        config,
        &ranges,
        &mut residuals,
        false,
        &mut stats,
        spec,
        &mut quantizer,
    )?;
    Ok(CoarseGradientSnapshot {
        residuals,
        output_gradient_vector_hash,
        output_gradient_sum,
        output_gradient_l1,
        output_gradient_max_abs,
        backward_ste_rescue_count: stats.backward_ste_rescue,
        backward_quantization_count: stats.backward_quantization,
        stochastic_round_up_count: quantizer.stochastic_round_up_count,
        gradient_saturation_count: stats.gradient_saturation,
        residual_saturation_count: stats.residual_saturation,
    })
}

#[allow(clippy::too_many_arguments)]
fn backward_update_targets(
    model: &mut ProductionModelV1,
    context_tokens: &[u32],
    targets: &[usize],
    cache: ForwardCache,
    config: ProductionFullTrainConfig,
    ranges: &ParameterRanges,
    residuals: &mut [i64],
    apply_updates: bool,
    stats: &mut UpdateStats,
) -> Result<(), TrainError> {
    let spec = GradientProposalSpec::lane(ProductionGradientProposalLane::NormalizedRescued, 0);
    let mut quantizer = BackwardQuantizer::new(spec.quantization, spec.stochastic_seed);
    backward_update_with_spec(
        model,
        context_tokens,
        targets,
        cache,
        config,
        ranges,
        residuals,
        apply_updates,
        stats,
        spec,
        &mut quantizer,
    )
}

#[allow(clippy::too_many_arguments)]
fn backward_update_with_spec(
    model: &mut ProductionModelV1,
    context_tokens: &[u32],
    targets: &[usize],
    cache: ForwardCache,
    config: ProductionFullTrainConfig,
    ranges: &ParameterRanges,
    residuals: &mut [i64],
    apply_updates: bool,
    stats: &mut UpdateStats,
    spec: GradientProposalSpec,
    quantizer: &mut BackwardQuantizer,
) -> Result<(), TrainError> {
    let c = model.config;
    let seq_len = context_tokens.len();
    let total = seq_len * c.d_model;
    if targets.is_empty()
        || targets.len() != cache.target_rows
        || targets.len() > seq_len
        || cache.features.len() != targets.len() * c.d_model
    {
        return Err(TrainError::InvalidConfig);
    }
    let mut gradients = Vec::with_capacity(targets.len());
    let mut precision_shift = None;
    for (row, &target) in targets.iter().enumerate() {
        let (gradient, row_precision_shift) = output_gradient_row(
            &cache,
            row,
            target,
            config,
            spec.source,
            spec.stochastic_seed.wrapping_add(row as u64),
        )?;
        if precision_shift
            .replace(row_precision_shift)
            .is_some_and(|value| value != row_precision_shift)
        {
            return Err(TrainError::InvalidConfig);
        }
        gradients.push(gradient);
    }
    let precision_shift = precision_shift.ok_or(TrainError::InvalidConfig)?;
    let feature_accumulators =
        output_feature_accumulators(model, &gradients, config.training_workers)?;
    let mut grad_features = vec![0_i16; targets.len() * c.d_model];
    for (gradient, accumulator) in grad_features.iter_mut().zip(feature_accumulators) {
        *gradient = saturate_i16(
            quantizer.shift(
                accumulator,
                config
                    .output_backward_shift
                    .unwrap_or(model.scales.output_shift)
                    .saturating_add(precision_shift),
                stats,
            ),
        );
    }
    update_output_parameters(
        model,
        &gradients,
        &cache.features,
        config,
        precision_shift,
        ranges,
        residuals,
        apply_updates,
        stats,
    )?;
    let final_start = (seq_len - targets.len()) * c.d_model;
    let mut grad_hidden = vec![0_i16; total];
    let mut final_gamma_grad = vec![0_i64; c.d_model];
    rms_backward_rows(
        &cache.final_hidden[final_start..],
        &model.final_rms_weights,
        &grad_features,
        c.d_model,
        &mut grad_hidden[final_start..],
        &mut final_gamma_grad,
        stats,
    )?;
    update_i16_slice(
        &mut model.final_rms_weights,
        &final_gamma_grad,
        vector_group_shift(config.vector_learning_rate_shift, 3)
            .saturating_add(target_mean_shift(config)),
        &mut residuals[ranges.final_rms.clone()],
        apply_updates,
        3,
        stats,
    );

    let matrix = c.d_model * c.d_model;
    let up_matrix = c.d_model * c.hidden_dim;
    let down_matrix = c.hidden_dim * c.d_model;
    for layer in (0..c.layers).rev() {
        let item = &cache.layers[layer];
        let mut grad_attention_residual = grad_hidden.clone();
        let grad_mlp_output = grad_hidden;
        let down_range = layer * down_matrix..(layer + 1) * down_matrix;
        let down_residual_range =
            ranges.down.start + down_range.start..ranges.down.start + down_range.end;
        let mut grad_gated = linear_backward_input(
            &grad_mlp_output,
            &model.down_weights[down_range.clone()],
            c.hidden_dim,
            c.d_model,
            model.scales.down_shift,
            stats,
            quantizer,
        );
        let mut grad_up = vec![0_i16; seq_len * c.hidden_dim];
        let mut grad_gate = vec![0_i16; seq_len * c.hidden_dim];
        for index in 0..grad_gated.len() {
            let up_numerator =
                i64::from(grad_gated[index]) * i64::from(hard_silu_q15(item.gate[index]));
            let gate_numerator = i64::from(grad_gated[index])
                * i64::from(item.up[index])
                * i64::from(hard_silu_derivative_q15(item.gate[index]));
            if quantizer.late_quantization() {
                grad_up[index] = saturate_i16(quantizer.shift(up_numerator, 15, stats));
                grad_gate[index] = saturate_i16(quantizer.shift(gate_numerator, 30, stats));
            } else {
                stats.backward_quantization = stats.backward_quantization.saturating_add(2);
                let grad = gated_activation_backward_i16_q15(
                    item.up[index],
                    item.gate[index],
                    grad_gated[index],
                );
                grad_up[index] = if quantizer.mode == BackwardQuantizationMode::RescuedRhu
                    && grad.up == 0
                    && up_numerator != 0
                {
                    stats.backward_ste_rescue = stats.backward_ste_rescue.saturating_add(1);
                    saturate_i16(up_numerator.signum())
                } else {
                    grad.up
                };
                grad_gate[index] = if quantizer.mode == BackwardQuantizationMode::RescuedRhu
                    && grad.gate == 0
                    && gate_numerator != 0
                {
                    stats.backward_ste_rescue = stats.backward_ste_rescue.saturating_add(1);
                    saturate_i16(gate_numerator.signum())
                } else {
                    grad.gate
                };
            }
        }
        let up_range = layer * up_matrix..(layer + 1) * up_matrix;
        let up_residual_range = ranges.up.start + up_range.start..ranges.up.start + up_range.end;
        let gate_residual_range =
            ranges.gate.start + up_range.start..ranges.gate.start + up_range.end;
        let up_input = linear_backward_input(
            &grad_up,
            &model.up_weights[up_range.clone()],
            c.d_model,
            c.hidden_dim,
            model.scales.up_shift,
            stats,
            quantizer,
        );
        let gate_input = linear_backward_input(
            &grad_gate,
            &model.gate_weights[up_range.clone()],
            c.d_model,
            c.hidden_dim,
            model.scales.gate_shift,
            stats,
            quantizer,
        );
        let grad_mlp_input = add_rows(&up_input, &gate_input);
        let rms_range = layer * c.d_model..(layer + 1) * c.d_model;
        let mlp_rms_residual_range =
            ranges.mlp_rms.start + rms_range.start..ranges.mlp_rms.start + rms_range.end;
        let attention_rms_residual_range = ranges.attention_rms.start + rms_range.start
            ..ranges.attention_rms.start + rms_range.end;
        let mut grad_mlp_residual = vec![0_i16; total];
        let mut mlp_gamma_grad = vec![0_i64; c.d_model];
        rms_backward_rows(
            &item.attention_residual,
            &model.mlp_rms_weights[rms_range.clone()],
            &grad_mlp_input,
            c.d_model,
            &mut grad_mlp_residual,
            &mut mlp_gamma_grad,
            stats,
        )?;
        add_rows_in_place(&mut grad_attention_residual, &grad_mlp_residual, stats);

        update_i8_matrix_rows(
            &item.gated,
            &grad_mlp_output,
            &mut model.down_weights[down_range],
            c.hidden_dim,
            c.d_model,
            matrix_learning_rate_shift(config, 10),
            &mut residuals[down_residual_range],
            apply_updates,
            10,
            stats,
        );
        update_i8_matrix_rows(
            &item.mlp_input,
            &grad_up,
            &mut model.up_weights[up_range.clone()],
            c.d_model,
            c.hidden_dim,
            matrix_learning_rate_shift(config, 8),
            &mut residuals[up_residual_range],
            apply_updates,
            8,
            stats,
        );
        update_i8_matrix_rows(
            &item.mlp_input,
            &grad_gate,
            &mut model.gate_weights[up_range],
            c.d_model,
            c.hidden_dim,
            matrix_learning_rate_shift(config, 9),
            &mut residuals[gate_residual_range],
            apply_updates,
            9,
            stats,
        );
        update_i16_slice(
            &mut model.mlp_rms_weights[rms_range.clone()],
            &mlp_gamma_grad,
            vector_group_shift(config.vector_learning_rate_shift, 2)
                .saturating_add(target_mean_shift(config)),
            &mut residuals[mlp_rms_residual_range],
            apply_updates,
            2,
            stats,
        );

        let attention_range = layer * matrix..(layer + 1) * matrix;
        let q_residual_range =
            ranges.q.start + attention_range.start..ranges.q.start + attention_range.end;
        let k_residual_range =
            ranges.k.start + attention_range.start..ranges.k.start + attention_range.end;
        let v_residual_range =
            ranges.v.start + attention_range.start..ranges.v.start + attention_range.end;
        let o_residual_range =
            ranges.o.start + attention_range.start..ranges.o.start + attention_range.end;
        let grad_context = linear_backward_input(
            &grad_attention_residual,
            &model.o_weights[attention_range.clone()],
            c.d_model,
            c.d_model,
            model.scales.o_shift,
            stats,
            quantizer,
        );
        let grad_context = scale_gradient(&grad_context, 8, stats);
        let (grad_q, grad_k, grad_v) = linear_attention_backward(
            c.d_model,
            c.heads,
            &item.q,
            &item.k,
            &item.v,
            &item.context,
            &grad_context,
            stats,
            quantizer,
        )?;
        let q_input = linear_backward_input(
            &grad_q,
            &model.q_weights[attention_range.clone()],
            c.d_model,
            c.d_model,
            model.scales.qkv_shift,
            stats,
            quantizer,
        );
        let k_input = linear_backward_input(
            &grad_k,
            &model.k_weights[attention_range.clone()],
            c.d_model,
            c.d_model,
            model.scales.qkv_shift,
            stats,
            quantizer,
        );
        let v_input = linear_backward_input(
            &grad_v,
            &model.v_weights[attention_range.clone()],
            c.d_model,
            c.d_model,
            model.scales.qkv_shift,
            stats,
            quantizer,
        );
        let grad_attention_input = add_rows(&add_rows(&q_input, &k_input), &v_input);
        let mut grad_input_norm = vec![0_i16; total];
        let mut attention_gamma_grad = vec![0_i64; c.d_model];
        rms_backward_rows(
            &item.input,
            &model.attention_rms_weights[rms_range.clone()],
            &grad_attention_input,
            c.d_model,
            &mut grad_input_norm,
            &mut attention_gamma_grad,
            stats,
        )?;
        grad_hidden = grad_attention_residual.clone();
        add_rows_in_place(&mut grad_hidden, &grad_input_norm, stats);

        update_i8_matrix_rows(
            &item.context,
            &grad_attention_residual,
            &mut model.o_weights[attention_range.clone()],
            c.d_model,
            c.d_model,
            matrix_learning_rate_shift(config, 7),
            &mut residuals[o_residual_range],
            apply_updates,
            7,
            stats,
        );
        update_i8_matrix_rows(
            &item.attention_input,
            &grad_q,
            &mut model.q_weights[attention_range.clone()],
            c.d_model,
            c.d_model,
            matrix_learning_rate_shift(config, 4),
            &mut residuals[q_residual_range],
            apply_updates,
            4,
            stats,
        );
        update_i8_matrix_rows(
            &item.attention_input,
            &grad_k,
            &mut model.k_weights[attention_range.clone()],
            c.d_model,
            c.d_model,
            matrix_learning_rate_shift(config, 5),
            &mut residuals[k_residual_range],
            apply_updates,
            5,
            stats,
        );
        update_i8_matrix_rows(
            &item.attention_input,
            &grad_v,
            &mut model.v_weights[attention_range],
            c.d_model,
            c.d_model,
            matrix_learning_rate_shift(config, 6),
            &mut residuals[v_residual_range],
            apply_updates,
            6,
            stats,
        );
        update_i16_slice(
            &mut model.attention_rms_weights[rms_range],
            &attention_gamma_grad,
            vector_group_shift(config.vector_learning_rate_shift, 1)
                .saturating_add(target_mean_shift(config)),
            &mut residuals[attention_rms_residual_range],
            apply_updates,
            1,
            stats,
        );
        grad_gated.clear();
    }
    for (row, &token) in context_tokens.iter().enumerate() {
        let embedding =
            &mut model.embeddings[token as usize * c.d_model..(token as usize + 1) * c.d_model];
        for dim in 0..c.d_model {
            update_i16(
                &mut embedding[dim],
                i64::from(grad_hidden[row * c.d_model + dim]),
                config
                    .embedding_learning_rate_shift
                    .saturating_add(target_mean_shift(config))
                    .saturating_sub(config.embedding_learning_rate_boost_shift),
                &mut residuals[ranges.embeddings.start + token as usize * c.d_model + dim],
                apply_updates,
                0,
                stats,
            );
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn update_output_parameter_chunk(
    output_weights: &mut [i16],
    output_bias: &mut [i32],
    output_residuals: &mut [i64],
    bias_residuals: &mut [i64],
    first_token: usize,
    d_model: usize,
    gradients: &[Vec<i32>],
    features: &[i16],
    output_shift: u8,
    bias_shift: u8,
    apply_updates: bool,
) -> UpdateStats {
    let mut stats = UpdateStats::default();
    for local_token in 0..output_bias.len() {
        let token = first_token + local_token;
        let bias_gradient = gradients
            .iter()
            .fold(0_i64, |sum, row| sum.saturating_add(i64::from(row[token])));
        update_i32(
            &mut output_bias[local_token],
            bias_gradient,
            bias_shift,
            &mut bias_residuals[local_token],
            apply_updates,
            12,
            &mut stats,
        );
        for dim in 0..d_model {
            let gradient = gradients
                .iter()
                .enumerate()
                .fold(0_i64, |sum, (row, grad_logits)| {
                    sum.saturating_add(
                        i64::from(grad_logits[token]) * i64::from(features[row * d_model + dim]),
                    )
                });
            let index = local_token * d_model + dim;
            update_i16(
                &mut output_weights[index],
                gradient,
                output_shift,
                &mut output_residuals[index],
                apply_updates,
                11,
                &mut stats,
            );
        }
    }
    stats
}

#[allow(clippy::too_many_arguments)]
fn update_output_parameters(
    model: &mut ProductionModelV1,
    gradients: &[Vec<i32>],
    features: &[i16],
    config: ProductionFullTrainConfig,
    precision_shift: u8,
    ranges: &ParameterRanges,
    residuals: &mut [i64],
    apply_updates: bool,
    stats: &mut UpdateStats,
) -> Result<(), TrainError> {
    if ranges.output.end != ranges.bias.start {
        return Err(TrainError::InvalidModel(
            "production output parameter ranges are not contiguous",
        ));
    }
    let d_model = model.config.d_model;
    let vocabulary = model.config.vocab_size;
    let workers = config.training_workers.min(vocabulary).max(1);
    let tokens_per_worker = vocabulary.div_ceil(workers);
    let output_shift = config
        .output_learning_rate_shift
        .saturating_add(precision_shift)
        .saturating_add(target_mean_shift(config));
    let bias_shift = config
        .vector_learning_rate_shift
        .saturating_add(precision_shift)
        .saturating_add(target_mean_shift(config));
    let parameter_residuals = &mut residuals[ranges.output.start..ranges.bias.end];
    let (output_residuals, bias_residuals) = parameter_residuals.split_at_mut(ranges.output.len());
    if workers == 1 {
        stats.merge(update_output_parameter_chunk(
            &mut model.output_weights,
            &mut model.output_bias_q8,
            output_residuals,
            bias_residuals,
            0,
            d_model,
            gradients,
            features,
            output_shift,
            bias_shift,
            apply_updates,
        ));
        return Ok(());
    }
    let output_chunk_len = tokens_per_worker * d_model;
    let chunk_stats = thread::scope(|scope| {
        let chunks = model
            .output_weights
            .chunks_mut(output_chunk_len)
            .zip(model.output_bias_q8.chunks_mut(tokens_per_worker))
            .zip(output_residuals.chunks_mut(output_chunk_len))
            .zip(bias_residuals.chunks_mut(tokens_per_worker));
        let handles = chunks
            .enumerate()
            .map(
                |(worker, (((weights, bias), weight_residuals), bias_residuals))| {
                    scope.spawn(move || {
                        update_output_parameter_chunk(
                            weights,
                            bias,
                            weight_residuals,
                            bias_residuals,
                            worker * tokens_per_worker,
                            d_model,
                            gradients,
                            features,
                            output_shift,
                            bias_shift,
                            apply_updates,
                        )
                    })
                },
            )
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .map_err(|_| TrainError::CoreRejected("production_training_worker_panicked"))
            })
            .collect::<Result<Vec<_>, _>>()
    })?;
    for chunk in chunk_stats {
        stats.merge(chunk);
    }
    Ok(())
}

fn linear_backward_input(
    grad: &[i16],
    weights: &[i8],
    input_dim: usize,
    output_dim: usize,
    shift: u8,
    stats: &mut UpdateStats,
    quantizer: &mut BackwardQuantizer,
) -> Vec<i16> {
    let rows = grad.len() / output_dim;
    let mut result = vec![0_i16; rows * input_dim];
    for row in 0..rows {
        for input in 0..input_dim {
            let mut acc = 0_i64;
            for output in 0..output_dim {
                acc = acc.saturating_add(
                    i64::from(grad[row * output_dim + output])
                        * i64::from(weights[output * input_dim + input]),
                );
            }
            let wide = quantizer.shift(acc, shift, stats);
            result[row * input_dim + input] = saturate_i16(wide);
            stats.gradient_saturation +=
                usize::from(wide < i64::from(i16::MIN) || wide > i64::from(i16::MAX));
        }
    }
    result
}

fn scale_gradient(values: &[i16], left_shift: u8, stats: &mut UpdateStats) -> Vec<i16> {
    values
        .iter()
        .map(|&value| {
            let wide = i64::from(value) << left_shift;
            stats.gradient_saturation = stats.gradient_saturation.saturating_add(usize::from(
                wide < i64::from(i16::MIN) || wide > i64::from(i16::MAX),
            ));
            saturate_i16(wide)
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn update_i8_matrix_rows(
    input: &[i16],
    grad: &[i16],
    weights: &mut [i8],
    input_dim: usize,
    output_dim: usize,
    shift: u8,
    residuals: &mut [i64],
    apply_updates: bool,
    group: usize,
    stats: &mut UpdateStats,
) {
    let rows = input.len() / input_dim;
    for output in 0..output_dim {
        for input_index in 0..input_dim {
            let mut gradient = 0_i64;
            for row in 0..rows {
                gradient = gradient.saturating_add(
                    i64::from(grad[row * output_dim + output])
                        * i64::from(input[row * input_dim + input_index]),
                );
            }
            let index = output * input_dim + input_index;
            stats.gradient_nonzero[group] =
                stats.gradient_nonzero[group].saturating_add(u64::from(gradient != 0));
            accumulate_residual(&mut residuals[index], gradient, group, stats);
            if !apply_updates {
                continue;
            }
            let update = round_shift_rhu_i64(residuals[index], shift);
            stats.residual_carry[group] = stats.residual_carry[group]
                .saturating_add(u64::from(update == 0 && residuals[index] != 0));
            let previous = weights[index];
            let wide = i64::from(previous).saturating_sub(update);
            let next = saturate_i8(wide);
            let saturated = wide < i64::from(i8::MIN) || wide > i64::from(i8::MAX);
            stats.weight_saturation += usize::from(saturated);
            stats.saturation_by_group[group] =
                stats.saturation_by_group[group].saturating_add(u64::from(saturated));
            stats.update_nonzero[group] =
                stats.update_nonzero[group].saturating_add(u64::from(next != previous));
            stats.movement[group] = stats.movement[group]
                .saturating_add((i64::from(next) - i64::from(previous)).unsigned_abs());
            weights[index] = next;
            consume_residual(&mut residuals[index], update, shift);
        }
    }
}

fn rms_backward_rows(
    input: &[i16],
    weights: &[i16],
    grad: &[i16],
    d_model: usize,
    output: &mut [i16],
    gamma_grad: &mut [i64],
    stats: &mut UpdateStats,
) -> Result<(), TrainError> {
    for row in 0..input.len() / d_model {
        let range = row * d_model..(row + 1) * d_model;
        rms_backward_row(
            &input[range.clone()],
            weights,
            &grad[range.clone()],
            &mut output[range],
            gamma_grad,
            stats,
        )?;
    }
    Ok(())
}

fn rms_backward_row(
    input: &[i16],
    weights: &[i16],
    grad: &[i16],
    output: &mut [i16],
    gamma_grad: &mut [i64],
    stats: &mut UpdateStats,
) -> Result<(), TrainError> {
    let mut normalized = vec![0_i32; input.len()];
    let mut scaled = vec![0_i32; input.len()];
    stats.gradient_saturation = stats.gradient_saturation.saturating_add(
        rms_norm_backward_i16_q15_checked(
            input,
            weights,
            grad,
            PRODUCTION_RMS_EPSILON,
            RmsNormBackwardWorkspace {
                normalized_q15: &mut normalized,
                scaled_grad_q15: &mut scaled,
            },
            output,
            gamma_grad,
        )
        .ok_or(TrainError::CoreRejected("production_training_rms_backward"))?,
    );
    Ok(())
}

type AttentionGradients = (Vec<i16>, Vec<i16>, Vec<i16>);

#[inline]
fn production_linear_attention_phi_i64(value: i16) -> i64 {
    i64::from(value) + 32769
}

#[inline]
fn production_linear_attention_phi_derivative_i64(_value: i16) -> i64 {
    1
}

#[allow(clippy::too_many_arguments)]
fn linear_attention_backward(
    d_model: usize,
    heads: usize,
    q: &[i16],
    k: &[i16],
    v: &[i16],
    context: &[i16],
    grad_context: &[i16],
    stats: &mut UpdateStats,
    quantizer: &mut BackwardQuantizer,
) -> Result<AttentionGradients, TrainError> {
    let seq_len = q.len() / d_model;
    let head_dim = d_model / heads;
    let state_len = head_dim * head_dim;
    let mut gq = vec![0_i64; q.len()];
    let mut gk = vec![0_i64; q.len()];
    let mut gv = vec![0_i64; q.len()];
    for head in 0..heads {
        let offset = head * head_dim;
        let mut prefixes = vec![0_i64; seq_len * state_len];
        let mut prefix_sums = vec![0_i64; seq_len * head_dim];
        let mut denominators = vec![0_i64; seq_len];
        let mut state = vec![0_i64; state_len];
        let mut sums = vec![0_i64; head_dim];
        for token in 0..seq_len {
            let base = token * d_model + offset;
            for kd in 0..head_dim {
                let phi = production_linear_attention_phi_i64(k[base + kd]);
                sums[kd] = sums[kd].saturating_add(phi);
                for vd in 0..head_dim {
                    state[kd * head_dim + vd] =
                        state[kd * head_dim + vd].saturating_add(phi * i64::from(v[base + vd]));
                }
            }
            denominators[token] = (0..head_dim)
                .map(|d| production_linear_attention_phi_i64(q[base + d]) * sums[d])
                .sum();
            if denominators[token] <= 0 {
                return Err(TrainError::CoreRejected(
                    "production_training_attention_denominator",
                ));
            }
            prefixes[token * state_len..(token + 1) * state_len].copy_from_slice(&state);
            prefix_sums[token * head_dim..(token + 1) * head_dim].copy_from_slice(&sums);
        }
        let mut grad_state = vec![0_i64; state_len];
        let mut grad_sums = vec![0_i64; head_dim];
        for token in (0..seq_len).rev() {
            let base = token * d_model + offset;
            let denominator = denominators[token];
            let dot_grad_context = (0..head_dim).fold(0_i64, |acc, vd| {
                acc.saturating_add(
                    i64::from(grad_context[base + vd]) * i64::from(context[base + vd]),
                )
            });
            for kd in 0..head_dim {
                let mut numerator = 0_i64;
                for vd in 0..head_dim {
                    numerator = numerator.saturating_add(
                        i64::from(grad_context[base + vd])
                            * prefixes[token * state_len + kd * head_dim + vd],
                    );
                }
                numerator = numerator.saturating_sub(
                    dot_grad_context.saturating_mul(prefix_sums[token * head_dim + kd]),
                );
                let phi_gradient = quantizer.ratio(numerator, denominator, stats)?;
                gq[base + kd] =
                    gq[base + kd].saturating_add(phi_gradient.saturating_mul(
                        production_linear_attention_phi_derivative_i64(q[base + kd]),
                    ));
            }
            for kd in 0..head_dim {
                let phi_q = production_linear_attention_phi_i64(q[base + kd]);
                for vd in 0..head_dim {
                    let product = i64::from(grad_context[base + vd])
                        .saturating_mul(phi_q)
                        .saturating_mul(1_i64 << 15);
                    grad_state[kd * head_dim + vd] = grad_state[kd * head_dim + vd]
                        .saturating_add(quantizer.ratio(product, denominator, stats)?);
                }
                grad_sums[kd] = grad_sums[kd].saturating_add(quantizer.ratio(
                    dot_grad_context.saturating_mul(phi_q).saturating_neg(),
                    denominator,
                    stats,
                )?);
            }
            if quantizer.late_quantization() {
                for vd in 0..head_dim {
                    let mut value_numerator = 0_i64;
                    for kd in 0..head_dim {
                        let phi_k = production_linear_attention_phi_i64(k[base + kd]);
                        value_numerator = value_numerator
                            .saturating_add(grad_state[kd * head_dim + vd].saturating_mul(phi_k));
                    }
                    gv[base + vd] =
                        gv[base + vd].saturating_add(quantizer.shift(value_numerator, 15, stats));
                }
                for kd in 0..head_dim {
                    let mut key_numerator = 0_i64;
                    for vd in 0..head_dim {
                        key_numerator = key_numerator.saturating_add(
                            grad_state[kd * head_dim + vd].saturating_mul(i64::from(v[base + vd])),
                        );
                    }
                    let phi_gradient = quantizer
                        .shift(key_numerator, 15, stats)
                        .saturating_add(grad_sums[kd]);
                    gk[base + kd] = gk[base + kd].saturating_add(phi_gradient.saturating_mul(
                        production_linear_attention_phi_derivative_i64(k[base + kd]),
                    ));
                }
            } else {
                for kd in 0..head_dim {
                    let phi_k = production_linear_attention_phi_i64(k[base + kd]);
                    let mut key_grad = 0_i64;
                    for vd in 0..head_dim {
                        let sg = grad_state[kd * head_dim + vd];
                        gv[base + vd] = gv[base + vd].saturating_add(quantizer.shift(
                            sg.saturating_mul(phi_k),
                            15,
                            stats,
                        ));
                        key_grad = key_grad.saturating_add(quantizer.shift(
                            sg.saturating_mul(i64::from(v[base + vd])),
                            15,
                            stats,
                        ));
                    }
                    let phi_gradient = key_grad.saturating_add(grad_sums[kd]);
                    gk[base + kd] = gk[base + kd].saturating_add(phi_gradient.saturating_mul(
                        production_linear_attention_phi_derivative_i64(k[base + kd]),
                    ));
                }
            }
        }
    }
    Ok((
        gq.into_iter().map(saturate_i16).collect(),
        gk.into_iter().map(saturate_i16).collect(),
        gv.into_iter().map(saturate_i16).collect(),
    ))
}

fn nearest_ratio(numerator: i64, denominator: i64) -> Result<i64, TrainError> {
    if denominator <= 0 {
        return Err(TrainError::InvalidConfig);
    }
    let half = denominator / 2;
    let rounded = if numerator >= 0 {
        numerator.checked_add(half).map(|v| v / denominator)
    } else {
        numerator
            .checked_neg()
            .and_then(|v| v.checked_add(half))
            .map(|v| -(v / denominator))
    }
    .ok_or(TrainError::CoreRejected("production_training_ratio"))?;
    Ok(rounded)
}

fn update_i16_slice(
    values: &mut [i16],
    gradients: &[i64],
    shift: u8,
    residuals: &mut [i64],
    apply_updates: bool,
    group: usize,
    stats: &mut UpdateStats,
) {
    for ((value, &gradient), residual) in values.iter_mut().zip(gradients).zip(residuals) {
        update_i16(
            value,
            gradient,
            shift,
            residual,
            apply_updates,
            group,
            stats,
        );
    }
}
fn update_i16(
    value: &mut i16,
    gradient: i64,
    shift: u8,
    residual: &mut i64,
    apply_updates: bool,
    group: usize,
    stats: &mut UpdateStats,
) {
    stats.gradient_nonzero[group] =
        stats.gradient_nonzero[group].saturating_add(u64::from(gradient != 0));
    accumulate_residual(residual, gradient, group, stats);
    if !apply_updates {
        return;
    }
    let update = round_shift_rhu_i64(*residual, shift);
    stats.residual_carry[group] =
        stats.residual_carry[group].saturating_add(u64::from(update == 0 && *residual != 0));
    let previous = *value;
    let wide = i64::from(previous).saturating_sub(update);
    let next = saturate_i16(wide);
    let saturated = wide < i64::from(i16::MIN) || wide > i64::from(i16::MAX);
    stats.weight_saturation += usize::from(saturated);
    stats.saturation_by_group[group] =
        stats.saturation_by_group[group].saturating_add(u64::from(saturated));
    stats.update_nonzero[group] =
        stats.update_nonzero[group].saturating_add(u64::from(next != previous));
    stats.movement[group] = stats.movement[group]
        .saturating_add((i64::from(next) - i64::from(previous)).unsigned_abs());
    *value = next;
    consume_residual(residual, update, shift);
}
fn update_i32(
    value: &mut i32,
    gradient: i64,
    shift: u8,
    residual: &mut i64,
    apply_updates: bool,
    group: usize,
    stats: &mut UpdateStats,
) {
    stats.gradient_nonzero[group] =
        stats.gradient_nonzero[group].saturating_add(u64::from(gradient != 0));
    accumulate_residual(residual, gradient, group, stats);
    if !apply_updates {
        return;
    }
    let update = round_shift_rhu_i64(*residual, shift);
    stats.residual_carry[group] =
        stats.residual_carry[group].saturating_add(u64::from(update == 0 && *residual != 0));
    let previous = *value;
    let wide = i64::from(previous).saturating_sub(update);
    let next = wide.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
    let saturated = wide < i64::from(i32::MIN) || wide > i64::from(i32::MAX);
    stats.weight_saturation += usize::from(saturated);
    stats.saturation_by_group[group] =
        stats.saturation_by_group[group].saturating_add(u64::from(saturated));
    stats.update_nonzero[group] =
        stats.update_nonzero[group].saturating_add(u64::from(next != previous));
    stats.movement[group] = stats.movement[group]
        .saturating_add((i64::from(next) - i64::from(previous)).unsigned_abs());
    *value = next;
    consume_residual(residual, update, shift);
}

fn consume_residual(residual: &mut i64, update: i64, shift: u8) {
    let consumed = i128::from(update) * (1_i128 << shift);
    let remaining = i128::from(*residual) - consumed;
    *residual = remaining.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64;
}

fn accumulate_residual(residual: &mut i64, gradient: i64, group: usize, stats: &mut UpdateStats) {
    if let Some(next) = residual.checked_add(gradient) {
        *residual = next;
    } else {
        *residual = residual.saturating_add(gradient);
        stats.residual_saturation = stats.residual_saturation.saturating_add(1);
        stats.residual_saturation_by_group[group] =
            stats.residual_saturation_by_group[group].saturating_add(1);
    }
}
fn add_rows(left: &[i16], right: &[i16]) -> Vec<i16> {
    left.iter()
        .zip(right)
        .map(|(&a, &b)| saturate_i16(i64::from(a) + i64::from(b)))
        .collect()
}
fn add_rows_in_place(left: &mut [i16], right: &[i16], stats: &mut UpdateStats) {
    for (a, &b) in left.iter_mut().zip(right) {
        let wide = i64::from(*a) + i64::from(b);
        stats.gradient_saturation +=
            usize::from(wide < i64::from(i16::MIN) || wide > i64::from(i16::MAX));
        *a = saturate_i16(wide);
    }
}
fn rms_rows(input: &[i16], weights: &[i16], d_model: usize) -> Result<Vec<i16>, TrainError> {
    let mut out = vec![0_i16; input.len()];
    for (a, b) in input
        .chunks_exact(d_model)
        .zip(out.chunks_exact_mut(d_model))
    {
        rms_norm_i16_q15_checked(a, weights, PRODUCTION_RMS_EPSILON, b)
            .ok_or(TrainError::CoreRejected("production_training_rms"))?;
    }
    Ok(out)
}
fn output_logits(model: &ProductionModelV1, features: &[i16]) -> Vec<i32> {
    (0..model.config.vocab_size)
        .map(|token| {
            let start = token * model.config.d_model;
            let acc = features.iter().enumerate().fold(0_i64, |sum, (d, &f)| {
                sum.saturating_add(i64::from(f) * i64::from(model.output_weights[start + d]))
            });
            ((acc >> model.scales.output_shift).clamp(i64::from(i32::MIN), i64::from(i32::MAX))
                as i32)
                .saturating_add(model.output_bias_q8[token])
        })
        .collect()
}

fn output_row(
    model: &ProductionModelV1,
    features: &[i16],
    probability_gradient_fractional_bits: u8,
    probability_normalization: SoftmaxNormalization,
) -> Result<(Vec<i32>, Vec<i32>), TrainError> {
    let logits = output_logits(model, features);
    let probabilities = if probability_gradient_fractional_bits == 15
        && probability_normalization == SoftmaxNormalization::LegacyQ31Lut
    {
        let mut q15 = vec![0_i16; model.config.vocab_size];
        base2_softmax_i32_q15(&logits, &mut q15)
            .ok_or(TrainError::CoreRejected("production_training_softmax"))?;
        q15.into_iter().map(i32::from).collect()
    } else {
        let mut q31 = vec![0_u32; model.config.vocab_size];
        base2_softmax_i32_q31_with_normalization(&logits, &mut q31, probability_normalization)
            .ok_or(TrainError::CoreRejected("production_training_softmax_wide"))?;
        q31.into_iter()
            .map(|probability| {
                quantize_probability_q31(probability, probability_gradient_fractional_bits)
            })
            .collect()
    };
    Ok((logits, probabilities))
}

fn output_rows(
    model: &ProductionModelV1,
    features: &[i16],
    training_workers: usize,
    probability_gradient_fractional_bits: u8,
    probability_normalization: SoftmaxNormalization,
) -> Result<(Vec<i32>, Vec<i32>), TrainError> {
    let target_rows = features.len() / model.config.d_model;
    let workers = training_workers.min(target_rows).max(1);
    let rows_per_worker = target_rows.div_ceil(workers);
    let feature_chunk_len = rows_per_worker * model.config.d_model;
    let chunks = if workers == 1 {
        vec![features]
    } else {
        features.chunks(feature_chunk_len).collect::<Vec<_>>()
    };
    let results = if workers == 1 {
        vec![
            chunks[0]
                .chunks_exact(model.config.d_model)
                .map(|row| {
                    output_row(
                        model,
                        row,
                        probability_gradient_fractional_bits,
                        probability_normalization,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?,
        ]
    } else {
        thread::scope(|scope| {
            let handles = chunks
                .into_iter()
                .map(|chunk| {
                    scope.spawn(move || {
                        chunk
                            .chunks_exact(model.config.d_model)
                            .map(|row| {
                                output_row(
                                    model,
                                    row,
                                    probability_gradient_fractional_bits,
                                    probability_normalization,
                                )
                            })
                            .collect::<Result<Vec<_>, _>>()
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| {
                    handle.join().map_err(|_| {
                        TrainError::CoreRejected("production_training_worker_panicked")
                    })?
                })
                .collect::<Result<Vec<_>, _>>()
        })?
    };
    let mut logits = Vec::with_capacity(target_rows * model.config.vocab_size);
    let mut probabilities = Vec::with_capacity(target_rows * model.config.vocab_size);
    for chunk in results {
        for (row_logits, row_probabilities) in chunk {
            logits.extend(row_logits);
            probabilities.extend(row_probabilities);
        }
    }
    Ok((logits, probabilities))
}

fn output_feature_accumulators(
    model: &ProductionModelV1,
    gradients: &[Vec<i32>],
    training_workers: usize,
) -> Result<Vec<i64>, TrainError> {
    let workers = training_workers.min(gradients.len()).max(1);
    let rows_per_worker = gradients.len().div_ceil(workers);
    let accumulate_rows = |rows: &[Vec<i32>]| {
        let mut accumulators = Vec::with_capacity(rows.len() * model.config.d_model);
        for grad_logits in rows {
            for dim in 0..model.config.d_model {
                let mut accumulator = 0_i64;
                for (token, &gradient) in grad_logits.iter().enumerate() {
                    accumulator = accumulator.saturating_add(
                        i64::from(gradient)
                            * i64::from(model.output_weights[token * model.config.d_model + dim]),
                    );
                }
                accumulators.push(accumulator);
            }
        }
        accumulators
    };
    if workers == 1 {
        return Ok(accumulate_rows(gradients));
    }
    let chunks = gradients.chunks(rows_per_worker).collect::<Vec<_>>();
    let results = thread::scope(|scope| {
        let handles = chunks
            .into_iter()
            .map(|chunk| scope.spawn(move || accumulate_rows(chunk)))
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .map_err(|_| TrainError::CoreRejected("production_training_worker_panicked"))
            })
            .collect::<Result<Vec<_>, _>>()
    })?;
    Ok(results.into_iter().flatten().collect())
}

fn evaluate_mistakes(
    model: &ProductionModelV1,
    windows: &[(Vec<u32>, u32)],
) -> Result<usize, TrainError> {
    let mut n = 0;
    for (context, target) in windows {
        let cache = forward_cache(model, context, 15, SoftmaxNormalization::LegacyQ31Lut)?;
        let predicted = cache
            .probabilities
            .iter()
            .enumerate()
            .max_by_key(|&(i, v)| (*v, core::cmp::Reverse(i)))
            .map(|(i, _)| i)
            .unwrap_or(0);
        n += usize::from(predicted != *target as usize);
    }
    Ok(n)
}

fn causal_suffix_targets(
    context: &[u32],
    final_target: u32,
    targets_per_window: usize,
) -> Result<Vec<usize>, TrainError> {
    if targets_per_window == 0 || targets_per_window > context.len() {
        return Err(TrainError::InvalidConfig);
    }
    let first_context_target = context.len() - targets_per_window + 1;
    let mut targets = context[first_context_target..]
        .iter()
        .map(|&token| token as usize)
        .collect::<Vec<_>>();
    targets.push(final_target as usize);
    debug_assert_eq!(targets.len(), targets_per_window);
    Ok(targets)
}

fn document_windows(tokens: &[u32], context: usize, max: usize) -> Vec<(Vec<u32>, u32)> {
    let mut windows = Vec::new();
    let mut doc = Vec::new();
    let mut active = false;
    for &token in tokens {
        if token == BOS_TOKEN_ID {
            doc.clear();
            active = true
        } else if token == EOS_TOKEN_ID {
            if active && doc.len() > context {
                for start in 0..doc.len() - context {
                    windows.push((doc[start..start + context].to_vec(), doc[start + context]));
                    if windows.len() >= max {
                        return windows;
                    }
                }
            }
            doc.clear();
            active = false
        } else if active {
            doc.push(token)
        }
    }
    windows
}

fn schedule_hash(c: ProductionFullTrainConfig) -> u64 {
    let mut bytes = Vec::new();
    for value in [c.context_tokens, c.max_windows, c.epochs, c.batch_windows] {
        bytes.extend_from_slice(&(value as u64).to_le_bytes())
    }
    if projection_shift_overrides(c).iter().all(Option::is_none)
        && c.output_backward_shift.is_none()
    {
        bytes.extend_from_slice(&[
            c.matrix_learning_rate_shift,
            c.vector_learning_rate_shift,
            c.embedding_learning_rate_shift,
            c.output_learning_rate_shift,
        ]);
    } else {
        bytes.extend_from_slice(&effective_learning_rate_shifts(c));
        bytes.push(c.output_backward_shift.unwrap_or(u8::MAX));
    }
    if c.probability_gradient_fractional_bits != 15 {
        bytes.extend_from_slice(&[0xff, c.probability_gradient_fractional_bits]);
    }
    if c.probability_normalization != SoftmaxNormalization::LegacyQ31Lut {
        let method = match c.probability_normalization {
            SoftmaxNormalization::LegacyQ31Lut => 0,
            SoftmaxNormalization::Q47Lut => 1,
            SoftmaxNormalization::Q47Newton1 => 2,
            SoftmaxNormalization::Q47Exact => 3,
        };
        bytes.extend_from_slice(&[0xfe, method]);
    }
    if c.spread_windows {
        bytes.extend_from_slice(&[0xfd, 1]);
    }
    if c.targets_per_window > 1 {
        bytes.extend_from_slice(&[0xfc]);
        bytes.extend_from_slice(&(c.targets_per_window as u64).to_le_bytes());
    }
    if c.embedding_learning_rate_boost_shift > 0 {
        bytes.extend_from_slice(&[0xfb, c.embedding_learning_rate_boost_shift]);
    }
    bytes.iter().fold(FNV_OFFSET, |mut hash, &byte| {
        hash ^= u64::from(byte);
        hash.wrapping_mul(FNV_PRIME)
    })
}

fn projection_shift_overrides(config: ProductionFullTrainConfig) -> [Option<u8>; 7] {
    [
        config.q_learning_rate_shift,
        config.k_learning_rate_shift,
        config.v_learning_rate_shift,
        config.o_learning_rate_shift,
        config.up_learning_rate_shift,
        config.gate_learning_rate_shift,
        config.down_learning_rate_shift,
    ]
}

fn sample_below(seed: u64, operation: u64, upper: u64) -> u64 {
    debug_assert!(upper > 0);
    let rejection_ceiling = u64::MAX - (u64::MAX % upper);
    let mut attempt = 0_u64;
    loop {
        let sample = splitmix64(
            seed ^ operation.rotate_left(23) ^ attempt.wrapping_mul(0xd134_2543_de82_ef95),
        );
        if sample < rejection_ceiling {
            return sample % upper;
        }
        attempt = attempt.wrapping_add(1);
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut mixed = value;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    mixed ^ (mixed >> 31)
}

fn quantize_probability_q31(probability_q31: u32, fractional_bits: u8) -> i32 {
    let shift = 31_u8.saturating_sub(fractional_bits);
    let rounded = round_shift_rhu_i64(i64::from(probability_q31), shift);
    let maximum = (1_u64 << fractional_bits) - 1;
    u64::try_from(rounded).unwrap_or(0).min(maximum) as i32
}

fn matrix_group_shift(base: u8, group: usize) -> u8 {
    match group {
        4 | 7..=9 => base,
        5 => base.saturating_add(4).min(62),
        6 => base.saturating_add(8).min(62),
        10 => base.saturating_sub(8),
        _ => base,
    }
}

fn matrix_learning_rate_shift(config: ProductionFullTrainConfig, group: usize) -> u8 {
    let explicit = match group {
        4 => config.q_learning_rate_shift,
        5 => config.k_learning_rate_shift,
        6 => config.v_learning_rate_shift,
        7 => config.o_learning_rate_shift,
        8 => config.up_learning_rate_shift,
        9 => config.gate_learning_rate_shift,
        10 => config.down_learning_rate_shift,
        _ => None,
    };
    explicit
        .unwrap_or_else(|| matrix_group_shift(config.matrix_learning_rate_shift, group))
        .saturating_add(target_mean_shift(config))
}

pub(super) fn effective_learning_rate_shifts(config: ProductionFullTrainConfig) -> [u8; 13] {
    let mean_shift = target_mean_shift(config);
    [
        config
            .embedding_learning_rate_shift
            .saturating_add(mean_shift)
            .saturating_sub(config.embedding_learning_rate_boost_shift),
        vector_group_shift(config.vector_learning_rate_shift, 1).saturating_add(mean_shift),
        vector_group_shift(config.vector_learning_rate_shift, 2).saturating_add(mean_shift),
        config
            .final_rms_learning_rate_shift
            .unwrap_or_else(|| vector_group_shift(config.vector_learning_rate_shift, 3))
            .saturating_add(mean_shift),
        matrix_learning_rate_shift(config, 4),
        matrix_learning_rate_shift(config, 5),
        matrix_learning_rate_shift(config, 6),
        matrix_learning_rate_shift(config, 7),
        matrix_learning_rate_shift(config, 8),
        matrix_learning_rate_shift(config, 9),
        matrix_learning_rate_shift(config, 10),
        config.output_learning_rate_shift.saturating_add(mean_shift),
        config.vector_learning_rate_shift.saturating_add(mean_shift),
    ]
}

fn target_mean_shift(config: ProductionFullTrainConfig) -> u8 {
    config.targets_per_window.max(1).trailing_zeros() as u8
}

fn vector_group_shift(base: u8, group: usize) -> u8 {
    match group {
        1 | 3 => base.saturating_sub(4),
        2 => base.saturating_sub(10),
        _ => base,
    }
}

#[cfg(test)]
mod tests {
    use nsrl_core::SoftmaxNormalization;

    use super::{
        BackwardQuantizationMode, BackwardQuantizer, FNV_OFFSET, FNV_PRIME,
        ProductionFullTrainConfig, UpdateStats, accumulate_residual, causal_suffix_targets,
        effective_learning_rate_shifts, schedule_hash, spread_document_windows,
        systematic_fixed_mass_gradient,
    };

    #[test]
    fn systematic_fixed_mass_lanes_are_exact_replayable_and_zero_sum() {
        let logits = [0_i32, -1, -256, -4096, -8192];
        for mass in [1_u32 << 15, 1_u32 << 16, 1_u32 << 18] {
            let (left, shift) =
                systematic_fixed_mass_gradient(&logits, 2, mass, 0x1234).expect("systematic");
            let (right, _) =
                systematic_fixed_mass_gradient(&logits, 2, mass, 0x1234).expect("replay");
            assert_eq!(left, right);
            assert_eq!(left.iter().map(|&value| i64::from(value)).sum::<i64>(), 0);
            assert_eq!(shift, mass.trailing_zeros() as u8 - 15);
            assert_eq!(
                left.iter()
                    .enumerate()
                    .filter(|(index, _)| *index != 2)
                    .map(|(_, &value)| i64::from(value))
                    .sum::<i64>(),
                -i64::from(left[2]),
            );
        }
    }

    #[test]
    fn default_effective_learning_rate_shifts_preserve_legacy_schedule() {
        let config = ProductionFullTrainConfig::default();
        assert_eq!(
            effective_learning_rate_shifts(config),
            [4, 6, 0, 6, 16, 20, 24, 16, 16, 16, 8, 24, 10]
        );
        let mut legacy_bytes = Vec::new();
        for value in [
            config.context_tokens,
            config.max_windows,
            config.epochs,
            config.batch_windows,
        ] {
            legacy_bytes.extend_from_slice(&(value as u64).to_le_bytes());
        }
        legacy_bytes.extend_from_slice(&[16, 10, 4, 24]);
        let legacy_hash = legacy_bytes.iter().fold(FNV_OFFSET, |mut hash, &byte| {
            hash ^= u64::from(byte);
            hash.wrapping_mul(FNV_PRIME)
        });
        assert_eq!(schedule_hash(config), legacy_hash);
    }

    #[test]
    fn projection_overrides_are_independent_and_bind_the_optimizer_schedule() {
        let base = ProductionFullTrainConfig::default();
        let tuned = ProductionFullTrainConfig {
            q_learning_rate_shift: Some(21),
            k_learning_rate_shift: Some(30),
            down_learning_rate_shift: Some(24),
            ..base
        };
        assert_eq!(
            effective_learning_rate_shifts(tuned),
            [4, 6, 0, 6, 21, 30, 24, 16, 16, 16, 24, 24, 10]
        );
        assert_ne!(schedule_hash(base), schedule_hash(tuned));
        let backward_tuned = ProductionFullTrainConfig {
            output_backward_shift: Some(8),
            ..base
        };
        assert_ne!(schedule_hash(base), schedule_hash(backward_tuned));
        let wide_probability = ProductionFullTrainConfig {
            probability_gradient_fractional_bits: 19,
            ..base
        };
        assert_ne!(schedule_hash(base), schedule_hash(wide_probability));
        let normalized_probability = ProductionFullTrainConfig {
            probability_gradient_fractional_bits: 23,
            probability_normalization: SoftmaxNormalization::Q47Newton1,
            ..base
        };
        assert_ne!(
            schedule_hash(wide_probability),
            schedule_hash(normalized_probability)
        );
        let spread = ProductionFullTrainConfig {
            spread_windows: true,
            ..base
        };
        assert_ne!(schedule_hash(base), schedule_hash(spread));
        let sequence = ProductionFullTrainConfig {
            targets_per_window: 4,
            ..base
        };
        assert_ne!(schedule_hash(base), schedule_hash(sequence));
        assert_eq!(
            effective_learning_rate_shifts(sequence),
            [6, 8, 2, 8, 18, 22, 26, 18, 18, 18, 10, 26, 12]
        );
        let boosted_embedding = ProductionFullTrainConfig {
            targets_per_window: 8,
            embedding_learning_rate_shift: 0,
            embedding_learning_rate_boost_shift: 1,
            ..base
        };
        assert_eq!(effective_learning_rate_shifts(boosted_embedding)[0], 2);
        assert_ne!(schedule_hash(sequence), schedule_hash(boosted_embedding));
        let parallel = ProductionFullTrainConfig {
            training_workers: 4,
            ..base
        };
        assert_eq!(schedule_hash(base), schedule_hash(parallel));
    }

    #[test]
    fn causal_suffix_targets_align_each_selected_row_with_its_next_token() {
        let context = [10, 11, 12, 13];
        assert_eq!(causal_suffix_targets(&context, 14, 1).unwrap(), [14]);
        assert_eq!(
            causal_suffix_targets(&context, 14, 3).unwrap(),
            [12, 13, 14]
        );
        assert_eq!(
            causal_suffix_targets(&context, 14, 4).unwrap(),
            [11, 12, 13, 14]
        );
        assert!(causal_suffix_targets(&context, 14, 0).is_err());
        assert!(causal_suffix_targets(&context, 14, 5).is_err());
    }

    #[test]
    fn spread_windows_select_the_first_middle_and_last_global_ranks() {
        let tokens = [
            super::BOS_TOKEN_ID,
            10,
            11,
            12,
            13,
            super::EOS_TOKEN_ID,
            super::BOS_TOKEN_ID,
            20,
            21,
            22,
            23,
            super::EOS_TOKEN_ID,
            super::BOS_TOKEN_ID,
            30,
            31,
            32,
            33,
            super::EOS_TOKEN_ID,
        ];
        assert_eq!(
            spread_document_windows(&tokens, 2, 3),
            vec![(vec![10, 11], 12), (vec![20, 21], 22), (vec![31, 32], 33)]
        );
        assert_eq!(spread_document_windows(&tokens, 2, 99).len(), 6);
    }

    #[test]
    fn output_backward_dead_zone_rescue_preserves_nonzero_signal() {
        let mut stats = UpdateStats::default();
        let mut quantizer = BackwardQuantizer::new(BackwardQuantizationMode::RescuedRhu, 0);
        assert_eq!(quantizer.shift(1, 12, &mut stats), 1);
        assert_eq!(quantizer.shift(-1, 12, &mut stats), -1);
        assert_eq!(stats.backward_quantization, 2);
        assert_eq!(stats.backward_ste_rescue, 2);
    }

    #[test]
    fn residual_accumulation_reports_integer_overflow_by_group() {
        let mut stats = UpdateStats::default();
        let mut residual = i64::MAX;
        accumulate_residual(&mut residual, 1, 4, &mut stats);
        assert_eq!(residual, i64::MAX);
        assert_eq!(stats.residual_saturation, 1);
        assert_eq!(stats.residual_saturation_by_group[4], 1);
    }
}

// ─── Direct-search with trainable final RMS feature scales ───

/// Tunes the per-dimension final RMS gamma together with the output head by
/// finite-difference coordinate search. Transformer layers remain frozen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectFeatureTrainConfig {
    pub context_tokens: usize,
    pub train_windows: usize,
    pub dev_windows: usize,
    pub head_candidates_per_round: usize,
    pub final_rms_candidates_per_round: usize,
    pub max_rounds: usize,
    pub min_train_nll_delta: i64,
    pub probability_gradient_fractional_bits: u8,
    pub probability_normalization: SoftmaxNormalization,
    pub sample_seed: u64,
}

impl Default for DirectFeatureTrainConfig {
    fn default() -> Self {
        Self {
            context_tokens: 64,
            train_windows: 512,
            dev_windows: 256,
            head_candidates_per_round: 16,
            final_rms_candidates_per_round: 8,
            max_rounds: 500,
            min_train_nll_delta: 0,
            probability_gradient_fractional_bits: 23,
            probability_normalization: SoftmaxNormalization::Q47Newton1,
            sample_seed: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectTrainWindowBinding {
    pub document: usize,
    pub context_start: usize,
    pub context_tokens: usize,
    pub target_offset: usize,
    pub target_token: u32,
}

fn direct_window_bindings(windows: &[DocumentWindow]) -> Vec<DirectTrainWindowBinding> {
    windows
        .iter()
        .map(|window| DirectTrainWindowBinding {
            document: window.document,
            context_start: window.context_start,
            context_tokens: window.context.len(),
            target_offset: window.context_start.saturating_add(window.context.len()),
            target_token: window.target,
        })
        .collect()
}

fn push_direct_window_bindings_json(json: &mut String, bindings: &[DirectTrainWindowBinding]) {
    use std::fmt::Write;

    json.push('[');
    for (index, binding) in bindings.iter().enumerate() {
        if index != 0 {
            json.push(',');
        }
        write!(
            json,
            concat!(
                "{{\"document\":{},\"context_start\":{},",
                "\"context_tokens\":{},\"target_offset\":{},",
                "\"target_token\":{}}}"
            ),
            binding.document,
            binding.context_start,
            binding.context_tokens,
            binding.target_offset,
            binding.target_token,
        )
        .expect("writing direct train window binding JSON cannot fail");
    }
    json.push(']');
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectFeatureMoveKind {
    NoOp,
    OutputWeight,
    OutputBias,
    FinalRmsScale,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectFeatureTrainRound {
    pub round: usize,
    pub kind: DirectFeatureMoveKind,
    pub coordinate: usize,
    pub applied_delta: i8,
    pub train_nll_q20_before: u64,
    pub train_nll_q20_after: u64,
    pub dev_nll_q20_before: u64,
    pub dev_nll_q20_after: u64,
    pub dev_mistakes: usize,
    pub function_visible: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectFeatureTrainTrace {
    pub profile: &'static str,
    pub parameter_count: usize,
    pub tokenizer_hash: u64,
    pub token_stream_hash: u64,
    pub context_tokens: usize,
    pub train_windows: usize,
    pub dev_windows: usize,
    pub head_candidates_per_round: usize,
    pub final_rms_candidates_per_round: usize,
    pub max_rounds: usize,
    pub probability_gradient_fractional_bits: u8,
    pub probability_normalization: &'static str,
    pub sample_seed: u64,
    pub initial_model_hash: u64,
    pub final_model_hash: u64,
    pub initial_train_nll_q20: u64,
    pub final_train_nll_q20: u64,
    pub initial_dev_nll_q20: u64,
    pub final_dev_nll_q20: u64,
    pub initial_dev_mistakes: usize,
    pub final_dev_mistakes: usize,
    pub rounds: usize,
    pub output_rounds: usize,
    pub bias_rounds: usize,
    pub final_rms_rounds: usize,
    pub train_bindings: Vec<DirectTrainWindowBinding>,
    pub dev_bindings: Vec<DirectTrainWindowBinding>,
    pub round_traces: Vec<DirectFeatureTrainRound>,
}

impl DirectFeatureTrainTrace {
    pub fn to_json_line(&self) -> String {
        let mut json = String::new();
        use std::fmt::Write;
        write!(
            json,
            concat!(
                "{{\"schema\":\"nsrl.direct_feature_train.v1\",",
                "\"objective\":\"integer_base2_softmax_nll_q20\",",
                "\"method\":\"gradient_ranked_probe_scored_coordinate_descent_with_final_rms_scale\",",
                "\"claims\":{{\"trunk\":\"layers_frozen_final_rms_gamma_trainable\",",
                "\"training\":\"backprop_gradient_ranked_finite_difference_full_train_verified\",",
                "\"gradient_use\":\"candidate_ranking_only_not_update_rule\"}},",
                "\"profile\":\"{}\",\"parameter_count\":{},",
                "\"bindings\":{{\"tokenizer_hash\":\"0x{:016x}\",",
                "\"token_stream_hash\":\"0x{:016x}\"}},",
                "\"training\":{{\"context_tokens\":{},\"train_windows\":{},",
                "\"dev_windows\":{},\"head_candidates_per_round\":{},",
                "\"final_rms_candidates_per_round\":{},\"max_rounds\":{},",
                "\"probability_gradient_fractional_bits\":{},",
                "\"probability_normalization\":\"{}\",\"sample_seed\":{}}},",
                "\"quality\":{{\"initial_train_nll_q20\":{},",
                "\"final_train_nll_q20\":{},",
                "\"initial_dev_nll_q20\":{},\"final_dev_nll_q20\":{},",
                "\"initial_dev_mistakes\":{},\"final_dev_mistakes\":{}}},",
                "\"hashes\":{{\"initial_model\":\"0x{:016x}\",",
                "\"final_model\":\"0x{:016x}\"}},",
                "\"stats\":{{\"rounds\":{},\"output_rounds\":{},",
                "\"bias_rounds\":{},\"final_rms_rounds\":{}}},\"rounds\":["
            ),
            self.profile,
            self.parameter_count,
            self.tokenizer_hash,
            self.token_stream_hash,
            self.context_tokens,
            self.train_windows,
            self.dev_windows,
            self.head_candidates_per_round,
            self.final_rms_candidates_per_round,
            self.max_rounds,
            self.probability_gradient_fractional_bits,
            self.probability_normalization,
            self.sample_seed,
            self.initial_train_nll_q20,
            self.final_train_nll_q20,
            self.initial_dev_nll_q20,
            self.final_dev_nll_q20,
            self.initial_dev_mistakes,
            self.final_dev_mistakes,
            self.initial_model_hash,
            self.final_model_hash,
            self.rounds,
            self.output_rounds,
            self.bias_rounds,
            self.final_rms_rounds,
        )
        .expect("writing json");
        for (i, r) in self.round_traces.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            let kind = match r.kind {
                DirectFeatureMoveKind::NoOp => "none",
                DirectFeatureMoveKind::OutputWeight => "output_weight",
                DirectFeatureMoveKind::OutputBias => "output_bias",
                DirectFeatureMoveKind::FinalRmsScale => "final_rms_scale",
            };
            write!(
                json,
                concat!(
                    "{{\"round\":{},\"kind\":\"{}\",\"coordinate\":{},",
                    "\"applied_delta\":{},\"train_nll_q20_before\":{},",
                    "\"train_nll_q20_after\":{},\"dev_nll_q20_before\":{},",
                    "\"dev_nll_q20_after\":{},\"dev_mistakes\":{},",
                    "\"function_visible\":{}}}"
                ),
                r.round,
                kind,
                r.coordinate,
                r.applied_delta,
                r.train_nll_q20_before,
                r.train_nll_q20_after,
                r.dev_nll_q20_before,
                r.dev_nll_q20_after,
                r.dev_mistakes,
                r.function_visible,
            )
            .expect("writing round");
        }
        json.push_str("],\"window_selection\":\"different_document_else_nonoverlap_context_plus_target\",\"window_bindings\":{\"train\":");
        push_direct_window_bindings_json(&mut json, &self.train_bindings);
        json.push_str(",\"dev\":");
        push_direct_window_bindings_json(&mut json, &self.dev_bindings);
        json.push_str("}}\n");
        json
    }
}

fn select_direct_train_dev_surfaces(
    windows: &[DocumentWindow],
    train_windows: usize,
    dev_windows: usize,
    insufficient_reason: &'static str,
) -> Result<(Vec<DocumentWindow>, Vec<DocumentWindow>), TrainError> {
    let required = train_windows
        .checked_add(dev_windows)
        .ok_or(TrainError::InvalidConfig)?;
    if windows.len() < required {
        return Err(TrainError::CoreRejected(insufficient_reason));
    }
    select_surfaces(
        windows,
        ProductionGradientAlignmentConfig {
            proposal_windows: train_windows,
            transfer_windows: dev_windows,
            documents_per_surface: 0,
            rescue_stratified_sampling: false,
            include_mass_corrected_no_rescue: false,
            include_systematic_fixed_mass: false,
            coordinates_per_group: 1,
            sample_seed: 0,
        },
    )
    .map_err(|_| TrainError::CoreRejected(insufficient_reason))
}

pub fn train_production_direct_feature(
    model: &mut ProductionModelV1,
    tokens: &[u32],
    token_stream_hash: u64,
    config: DirectFeatureTrainConfig,
) -> Result<DirectFeatureTrainTrace, TrainError> {
    model.validate()?;
    if config.context_tokens == 0
        || config.context_tokens > model.config.context_tokens
        || config.train_windows == 0
        || config.dev_windows == 0
        || config.head_candidates_per_round == 0
        || config.final_rms_candidates_per_round == 0
        || config.max_rounds == 0
        || config.min_train_nll_delta < 0
        || !(15..=31).contains(&config.probability_gradient_fractional_bits)
    {
        return Err(TrainError::InvalidConfig);
    }
    let all_windows =
        super::alignment::document_windows_with_coordinates(tokens, config.context_tokens);
    let (train_surface, dev_surface) = select_direct_train_dev_surfaces(
        &all_windows,
        config.train_windows,
        config.dev_windows,
        "direct_feature_insufficient_windows",
    )?;
    let train_windows = train_surface.as_slice();
    let dev_windows = dev_surface.as_slice();
    let initial_model_hash = model.model_hash();
    let initial_train = evaluate_surface(model, train_windows)?;
    let initial_dev = evaluate_surface(model, dev_windows)?;
    let initial_dev_mistakes = count_mistakes(model, dev_windows)?;
    // Score candidates on a small probe to keep latency tractable
    let probe_windows = train_windows.len().min(12);
    let score_surface = &train_windows[..probe_windows];
    let output_weight_count = checked_product(model.config.vocab_size, model.config.d_model)?;
    let bias_offset = output_weight_count;
    let final_rms_dim = model.config.d_model;
    let mut round_traces = Vec::with_capacity(config.max_rounds);
    let mut output_rounds = 0_usize;
    let mut bias_rounds = 0_usize;
    let mut final_rms_rounds = 0_usize;
    for round_index in 0..config.max_rounds {
        let full_train = evaluate_surface(model, train_windows)?;
        let full_dev = evaluate_surface(model, dev_windows)?;
        let dev_mistakes = count_mistakes(model, dev_windows)?;
        // --- Evaluate head candidates ---
        let top_head = select_top_gradient_candidates(
            model,
            train_windows,
            &DirectHeadTrainConfig {
                context_tokens: config.context_tokens,
                train_windows: config.train_windows,
                dev_windows: config.dev_windows,
                candidates_per_round: config.head_candidates_per_round,
                max_rounds: 1,
                min_train_nll_delta: 0,
                probability_gradient_fractional_bits: config.probability_gradient_fractional_bits,
                probability_normalization: config.probability_normalization,
                sample_seed: config.sample_seed.wrapping_add(round_index as u64),
            },
            output_weight_count,
        )?;
        // When gradient candidates are empty (e.g. zero output weights),
        // fall back to random head candidates.
        let head_candidates: Vec<(usize, i64)> = if top_head.is_empty() {
            let mut rand_head = Vec::new();
            let mut rng = splitmix64(
                config
                    .sample_seed
                    .wrapping_add(model.model_hash())
                    .wrapping_add((round_index as u64).rotate_left(7)),
            );
            for i in 0..config.head_candidates_per_round {
                let is_bias = i % 3 == 0;
                if is_bias {
                    let local = (rng as usize) % model.config.vocab_size;
                    rand_head.push((bias_offset + local, 0));
                } else {
                    let local = (rng as usize) % output_weight_count;
                    rand_head.push((local, 0));
                }
                rng = splitmix64(rng);
            }
            rand_head
        } else {
            top_head
        };
        let score = evaluate_surface(model, score_surface)?;
        let mut best_kind = None;
        let mut best_coord = 0_usize;
        let mut best_delta = 0_i8;
        let mut best_score_delta = 0_i64;
        // Step sizes: large enough to be visible through the output shift
        let weight_step: i8 =
            (1_i32 << (model.scales.output_shift.saturating_sub(6))).min(64) as i8;
        let final_rms_step: i8 = 64;
        // Head candidates: output weights + bias
        for &(global, _gradient) in &head_candidates {
            let is_bias = global >= bias_offset;
            let group: usize = if is_bias { 12 } else { 11 };
            let local = if is_bias {
                global - bias_offset
            } else {
                global
            };
            let step: i8 = if is_bias { 1 } else { weight_step.max(4) };
            if !can_perturb(model, group, local, step) || !can_perturb(model, group, local, -step) {
                continue;
            }
            let plus = evaluate_parameter_delta(model, group, local, step, score_surface)?;
            let minus = evaluate_parameter_delta(model, group, local, -step, score_surface)?;
            let (better, delta) = best_dir(score.nll_q20, plus.nll_q20, minus.nll_q20, step);
            if delta > best_score_delta {
                best_kind = Some(if is_bias {
                    DirectFeatureMoveKind::OutputBias
                } else {
                    DirectFeatureMoveKind::OutputWeight
                });
                best_coord = local;
                best_delta = better;
                best_score_delta = delta;
            }
        }
        // Final RMS gamma candidates.
        let mut final_rms_candidates = Vec::new();
        let seed = splitmix64(
            config
                .sample_seed
                .wrapping_add(model.model_hash())
                .wrapping_add((round_index as u64).rotate_left(13)),
        );
        let mut rng = seed;
        for dim in 0..final_rms_dim {
            if can_perturb(model, 3, dim, final_rms_step)
                && can_perturb(model, 3, dim, -final_rms_step)
            {
                final_rms_candidates.push((dim, rng));
                rng = splitmix64(rng);
            }
        }
        final_rms_candidates.sort_unstable_by_key(|&(_, key)| key);
        final_rms_candidates.truncate(config.final_rms_candidates_per_round);
        for (dim, _key) in &final_rms_candidates {
            let dim = *dim;
            let step: i8 = final_rms_step;
            let plus = evaluate_parameter_delta(model, 3, dim, step, score_surface)?;
            let minus = evaluate_parameter_delta(model, 3, dim, -step, score_surface)?;
            let (better, delta) = best_dir(score.nll_q20, plus.nll_q20, minus.nll_q20, step);
            if delta > best_score_delta {
                best_kind = Some(DirectFeatureMoveKind::FinalRmsScale);
                best_coord = dim;
                best_delta = better;
                best_score_delta = delta;
            }
        }
        if best_kind.is_none() || best_score_delta <= config.min_train_nll_delta {
            round_traces.push(DirectFeatureTrainRound {
                round: round_index,
                kind: DirectFeatureMoveKind::NoOp,
                coordinate: 0,
                applied_delta: 0,
                train_nll_q20_before: full_train.nll_q20,
                train_nll_q20_after: full_train.nll_q20,
                dev_nll_q20_before: full_dev.nll_q20,
                dev_nll_q20_after: full_dev.nll_q20,
                dev_mistakes,
                function_visible: false,
            });
            break;
        }
        let kind = best_kind.expect("positive score delta has a move kind");
        let (group, local) = match kind {
            DirectFeatureMoveKind::NoOp => unreachable!("no-op is not a candidate move"),
            DirectFeatureMoveKind::OutputWeight => (11, best_coord),
            DirectFeatureMoveKind::OutputBias => (12, best_coord),
            DirectFeatureMoveKind::FinalRmsScale => (3, best_coord),
        };
        let post_train = evaluate_parameter_delta(model, group, local, best_delta, train_windows)?;
        let full_train_delta = signed_nll_improvement(full_train.nll_q20, post_train.nll_q20);
        if full_train_delta <= config.min_train_nll_delta {
            round_traces.push(DirectFeatureTrainRound {
                round: round_index,
                kind,
                coordinate: best_coord,
                applied_delta: 0,
                train_nll_q20_before: full_train.nll_q20,
                train_nll_q20_after: full_train.nll_q20,
                dev_nll_q20_before: full_dev.nll_q20,
                dev_nll_q20_after: full_dev.nll_q20,
                dev_mistakes,
                function_visible: false,
            });
            break;
        }
        shift_parameter(model, group, local, best_delta)?;
        let post_dev = evaluate_surface(model, dev_windows)?;
        let post_mistakes = count_mistakes(model, dev_windows)?;
        let function_visible = post_train.logits != full_train.logits;
        match kind {
            DirectFeatureMoveKind::NoOp => unreachable!("no-op was not applied"),
            DirectFeatureMoveKind::OutputWeight => output_rounds = output_rounds.saturating_add(1),
            DirectFeatureMoveKind::OutputBias => bias_rounds = bias_rounds.saturating_add(1),
            DirectFeatureMoveKind::FinalRmsScale => {
                final_rms_rounds = final_rms_rounds.saturating_add(1)
            }
        }
        round_traces.push(DirectFeatureTrainRound {
            round: round_index,
            kind,
            coordinate: best_coord,
            applied_delta: best_delta,
            train_nll_q20_before: full_train.nll_q20,
            train_nll_q20_after: post_train.nll_q20,
            dev_nll_q20_before: full_dev.nll_q20,
            dev_nll_q20_after: post_dev.nll_q20,
            dev_mistakes: post_mistakes,
            function_visible,
        });
    }
    let final_train = evaluate_surface(model, train_windows)?;
    let final_dev = evaluate_surface(model, dev_windows)?;
    let final_mistakes = count_mistakes(model, dev_windows)?;
    Ok(DirectFeatureTrainTrace {
        profile: model.config.profile_id().unwrap_or("custom"),
        parameter_count: model.parameter_count(),
        tokenizer_hash: model.tokenizer_hash,
        token_stream_hash,
        context_tokens: config.context_tokens,
        train_windows: config.train_windows,
        dev_windows: config.dev_windows,
        head_candidates_per_round: config.head_candidates_per_round,
        final_rms_candidates_per_round: config.final_rms_candidates_per_round,
        max_rounds: config.max_rounds,
        probability_gradient_fractional_bits: config.probability_gradient_fractional_bits,
        probability_normalization: config.probability_normalization.as_str(),
        sample_seed: config.sample_seed,
        initial_model_hash,
        final_model_hash: model.model_hash(),
        initial_train_nll_q20: initial_train.nll_q20,
        final_train_nll_q20: final_train.nll_q20,
        initial_dev_nll_q20: initial_dev.nll_q20,
        final_dev_nll_q20: final_dev.nll_q20,
        initial_dev_mistakes,
        final_dev_mistakes: final_mistakes,
        rounds: round_traces.len(),
        output_rounds,
        bias_rounds,
        final_rms_rounds,
        train_bindings: direct_window_bindings(train_windows),
        dev_bindings: direct_window_bindings(dev_windows),
        round_traces,
    })
}

fn best_dir(current: u64, plus: u64, minus: u64, step: i8) -> (i8, i64) {
    if plus <= minus && plus < current {
        (step, signed_nll_improvement(current, plus))
    } else if minus < current {
        (-step, signed_nll_improvement(current, minus))
    } else {
        (0, 0)
    }
}

// ─── Direct-search output-head trainer (original) ───

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectHeadTrainConfig {
    pub context_tokens: usize,
    pub train_windows: usize,
    pub dev_windows: usize,
    pub candidates_per_round: usize,
    pub max_rounds: usize,
    pub min_train_nll_delta: i64,
    pub probability_gradient_fractional_bits: u8,
    pub probability_normalization: SoftmaxNormalization,
    pub sample_seed: u64,
}

impl Default for DirectHeadTrainConfig {
    fn default() -> Self {
        Self {
            context_tokens: 64,
            train_windows: 512,
            dev_windows: 256,
            candidates_per_round: 32,
            max_rounds: 500,
            min_train_nll_delta: 0,
            probability_gradient_fractional_bits: 23,
            probability_normalization: SoftmaxNormalization::Q47Newton1,
            sample_seed: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectHeadTrainRound {
    pub round: usize,
    pub candidates_evaluated: usize,
    pub candidates_with_descent: usize,
    pub best_delta_train_nll_q20: i64,
    pub best_delta_dev_nll_q20: i64,
    pub output_weight_coordinate: Option<usize>,
    pub output_bias_coordinate: Option<usize>,
    pub applied_delta: i8,
    pub train_nll_q20_after: u64,
    pub dev_nll_q20_after: u64,
    pub dev_mistakes: usize,
    pub output_gradient_sum: i64,
    pub weight_saturation_count: usize,
    pub function_visible: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectHeadTrainTrace {
    pub profile: &'static str,
    pub parameter_count: usize,
    pub tokenizer_hash: u64,
    pub token_stream_hash: u64,
    pub context_tokens: usize,
    pub train_windows: usize,
    pub dev_windows: usize,
    pub candidates_per_round: usize,
    pub max_rounds: usize,
    pub probability_gradient_fractional_bits: u8,
    pub probability_normalization: &'static str,
    pub sample_seed: u64,
    pub initial_model_hash: u64,
    pub final_model_hash: u64,
    pub initial_train_nll_q20: u64,
    pub final_train_nll_q20: u64,
    pub initial_dev_nll_q20: u64,
    pub final_dev_nll_q20: u64,
    pub initial_dev_mistakes: usize,
    pub final_dev_mistakes: usize,
    pub rounds: usize,
    pub total_descent_steps: usize,
    pub total_candidates_evaluated: usize,
    pub output_rounds: usize,
    pub bias_rounds: usize,
    pub rounds_with_descent: usize,
    pub train_bindings: Vec<DirectTrainWindowBinding>,
    pub dev_bindings: Vec<DirectTrainWindowBinding>,
    pub round_traces: Vec<DirectHeadTrainRound>,
}

impl DirectHeadTrainTrace {
    pub fn to_json_line(&self) -> String {
        let mut json = String::new();
        use std::fmt::Write;
        write!(
            json,
            concat!(
                "{{\"schema\":\"nsrl.direct_head_train.v1\",",
                "\"objective\":\"integer_base2_softmax_nll_q20\",",
                "\"method\":\"gradient_ranked_probe_scored_coordinate_descent\",",
                "\"claims\":{{\"trunk\":\"frozen_no_backprop_through_layers\",",
                "\"output_head\":\"unit_finite_difference_full_train_verified\",",
                "\"gradient_use\":\"candidate_ranking_only_not_update_rule\"}},",
                "\"profile\":\"{}\",\"parameter_count\":{},",
                "\"bindings\":{{\"tokenizer_hash\":\"0x{:016x}\",",
                "\"token_stream_hash\":\"0x{:016x}\"}},",
                "\"training\":{{\"context_tokens\":{},\"train_windows\":{},",
                "\"dev_windows\":{},\"candidates_per_round\":{},",
                "\"max_rounds\":{},",
                "\"probability_gradient_fractional_bits\":{},",
                "\"probability_normalization\":\"{}\",\"sample_seed\":{}}},",
                "\"quality\":{{\"initial_train_nll_q20\":{},",
                "\"final_train_nll_q20\":{},",
                "\"initial_dev_nll_q20\":{},\"final_dev_nll_q20\":{},",
                "\"initial_dev_mistakes\":{},\"final_dev_mistakes\":{}}},",
                "\"hashes\":{{\"initial_model\":\"0x{:016x}\",",
                "\"final_model\":\"0x{:016x}\"}},",
                "\"stats\":{{\"rounds\":{},\"rounds_with_descent\":{},",
                "\"output_rounds\":{},\"bias_rounds\":{},",
                "\"total_descent_steps\":{},",
                "\"total_candidates_evaluated\":{}}},\"rounds\":["
            ),
            self.profile,
            self.parameter_count,
            self.tokenizer_hash,
            self.token_stream_hash,
            self.context_tokens,
            self.train_windows,
            self.dev_windows,
            self.candidates_per_round,
            self.max_rounds,
            self.probability_gradient_fractional_bits,
            self.probability_normalization,
            self.sample_seed,
            self.initial_train_nll_q20,
            self.final_train_nll_q20,
            self.initial_dev_nll_q20,
            self.final_dev_nll_q20,
            self.initial_dev_mistakes,
            self.final_dev_mistakes,
            self.initial_model_hash,
            self.final_model_hash,
            self.rounds,
            self.rounds_with_descent,
            self.output_rounds,
            self.bias_rounds,
            self.total_descent_steps,
            self.total_candidates_evaluated,
        )
        .expect("writing direct head train JSON cannot fail");
        for (index, round) in self.round_traces.iter().enumerate() {
            if index > 0 {
                json.push(',');
            }
            write!(
                json,
                "{{\"round\":{},\"candidates_evaluated\":{},",
                round.round, round.candidates_evaluated
            )
            .expect("write round");
            write!(
                json,
                "\"candidates_with_descent\":{},",
                round.candidates_with_descent
            )
            .expect("write");
            write!(
                json,
                "\"best_delta_train_nll_q20\":{},",
                round.best_delta_train_nll_q20
            )
            .expect("w");
            write!(
                json,
                "\"best_delta_dev_nll_q20\":{},",
                round.best_delta_dev_nll_q20
            )
            .expect("w");
            write!(json, "\"output_weight_coordinate\":").expect("w");
            if let Some(coord) = round.output_weight_coordinate {
                write!(json, "{coord}").expect("w");
            } else {
                json.push_str("null");
            }
            write!(json, ",\"output_bias_coordinate\":").expect("w");
            if let Some(coord) = round.output_bias_coordinate {
                write!(json, "{coord}").expect("w");
            } else {
                json.push_str("null");
            }
            write!(
                json,
                concat!(
                    ",\"applied_delta\":{},\"train_nll_q20_after\":{},",
                    "\"dev_nll_q20_after\":{},\"dev_mistakes\":{},",
                    "\"output_gradient_sum\":{},",
                    "\"weight_saturation_count\":{},",
                    "\"function_visible\":{}}}"
                ),
                round.applied_delta,
                round.train_nll_q20_after,
                round.dev_nll_q20_after,
                round.dev_mistakes,
                round.output_gradient_sum,
                round.weight_saturation_count,
                round.function_visible,
            )
            .expect("writing round tail");
        }
        json.push_str("],\"window_selection\":\"different_document_else_nonoverlap_context_plus_target\",\"window_bindings\":{\"train\":");
        push_direct_window_bindings_json(&mut json, &self.train_bindings);
        json.push_str(",\"dev\":");
        push_direct_window_bindings_json(&mut json, &self.dev_bindings);
        json.push_str("}}\n");
        json
    }
}

pub fn train_production_direct_head_search(
    model: &mut ProductionModelV1,
    tokens: &[u32],
    token_stream_hash: u64,
    config: DirectHeadTrainConfig,
) -> Result<DirectHeadTrainTrace, TrainError> {
    model.validate()?;
    if config.context_tokens == 0
        || config.context_tokens > model.config.context_tokens
        || config.train_windows == 0
        || config.dev_windows == 0
        || config.candidates_per_round == 0
        || config.max_rounds == 0
        || config.min_train_nll_delta < 0
        || !(15..=31).contains(&config.probability_gradient_fractional_bits)
    {
        return Err(TrainError::InvalidConfig);
    }
    let all_windows =
        super::alignment::document_windows_with_coordinates(tokens, config.context_tokens);
    let (train_surface, dev_surface) = select_direct_train_dev_surfaces(
        &all_windows,
        config.train_windows,
        config.dev_windows,
        "direct_head_search_insufficient_windows",
    )?;
    let train_windows = train_surface.as_slice();
    let dev_windows = dev_surface.as_slice();
    let initial_model_hash = model.model_hash();
    let initial_train = evaluate_surface(model, train_windows)?;
    let initial_dev = evaluate_surface(model, dev_windows)?;
    let initial_dev_mistakes = count_mistakes(model, dev_windows)?;
    let initial_train_nll = initial_train.nll_q20;
    let initial_dev_nll = initial_dev.nll_q20;
    let output_weight_count = checked_product(model.config.vocab_size, model.config.d_model)?;
    let bias_offset = output_weight_count;
    let mut round_traces = Vec::with_capacity(config.max_rounds);
    let mut total_descent_steps = 0_usize;
    let mut total_candidates_evaluated = 0_usize;
    let mut output_rounds = 0_usize;
    let mut bias_rounds = 0_usize;
    let mut rounds_with_descent = 0_usize;
    // Score candidates on a small probe subset, then verify on full train
    let probe_windows = train_windows.len().min(16);
    let score_surface = &train_windows[..probe_windows];
    for round_index in 0..config.max_rounds {
        let current_score = evaluate_surface(model, score_surface)?;
        let top_candidates =
            select_top_gradient_candidates(model, train_windows, &config, output_weight_count)?;
        let mut best_coord = None;
        let mut best_is_bias = false;
        let mut best_delta = 0_i8;
        let mut best_probe_delta = 0_i64;
        let mut candidates_with_descent = 0_usize;
        let mut round_candidates_evaluated = 0_usize;
        let mut output_gradient_sum = 0_i64;
        for &(global, gradient) in &top_candidates {
            let gradient_magnitude = gradient.unsigned_abs().min(i64::MAX as u64) as i64;
            output_gradient_sum = output_gradient_sum.saturating_add(gradient_magnitude);
            let is_bias = global >= bias_offset;
            let group: usize = if is_bias { 12 } else { 11 };
            let local = if is_bias {
                global - bias_offset
            } else {
                global
            };
            if !can_perturb(model, group, local, 1) || !can_perturb(model, group, local, -1) {
                continue;
            }
            let plus_score = evaluate_parameter_delta(model, group, local, 1, score_surface)?;
            let minus_score = evaluate_parameter_delta(model, group, local, -1, score_surface)?;
            round_candidates_evaluated = round_candidates_evaluated.saturating_add(1);
            total_candidates_evaluated = total_candidates_evaluated.saturating_add(1);
            let (candidate_delta, probe_delta) = best_dir(
                current_score.nll_q20,
                plus_score.nll_q20,
                minus_score.nll_q20,
                1,
            );
            if candidate_delta != 0 {
                candidates_with_descent = candidates_with_descent.saturating_add(1);
                if probe_delta > best_probe_delta {
                    best_coord = Some(global);
                    best_is_bias = is_bias;
                    best_delta = candidate_delta;
                    best_probe_delta = probe_delta;
                }
            }
        }
        let current_train = evaluate_surface(model, train_windows)?;
        let current_dev = evaluate_surface(model, dev_windows)?;
        let current_dev_mistakes = count_mistakes(model, dev_windows)?;
        let round_train_nll = current_train.nll_q20;
        let round_dev_nll = current_dev.nll_q20;
        if let Some(global) = best_coord {
            let is_bias = global >= bias_offset;
            let group: usize = if is_bias { 12 } else { 11 };
            let local = if is_bias {
                global - bias_offset
            } else {
                global
            };
            let post_train =
                evaluate_parameter_delta(model, group, local, best_delta, train_windows)?;
            let full_train_delta = signed_nll_improvement(round_train_nll, post_train.nll_q20);
            if full_train_delta <= config.min_train_nll_delta {
                round_traces.push(DirectHeadTrainRound {
                    round: round_index,
                    candidates_evaluated: round_candidates_evaluated,
                    candidates_with_descent,
                    best_delta_train_nll_q20: full_train_delta,
                    best_delta_dev_nll_q20: 0,
                    output_weight_coordinate: if best_is_bias { None } else { Some(local) },
                    output_bias_coordinate: if best_is_bias { Some(local) } else { None },
                    applied_delta: 0,
                    train_nll_q20_after: round_train_nll,
                    dev_nll_q20_after: round_dev_nll,
                    dev_mistakes: current_dev_mistakes,
                    output_gradient_sum,
                    weight_saturation_count: 0,
                    function_visible: false,
                });
                break;
            }
            shift_parameter(model, group, local, best_delta)?;
            if is_bias {
                bias_rounds = bias_rounds.saturating_add(1);
            } else {
                output_rounds = output_rounds.saturating_add(1);
            }
            rounds_with_descent = rounds_with_descent.saturating_add(1);
            total_descent_steps = total_descent_steps.saturating_add(1);
            let post_dev = evaluate_surface(model, dev_windows)?;
            let post_dev_mistakes = count_mistakes(model, dev_windows)?;
            let best_dev_delta = signed_nll_improvement(round_dev_nll, post_dev.nll_q20);
            let function_visible = post_dev.logits != current_dev.logits;
            round_traces.push(DirectHeadTrainRound {
                round: round_index,
                candidates_evaluated: round_candidates_evaluated,
                candidates_with_descent,
                best_delta_train_nll_q20: full_train_delta,
                best_delta_dev_nll_q20: best_dev_delta,
                output_weight_coordinate: if best_is_bias { None } else { Some(local) },
                output_bias_coordinate: if best_is_bias { Some(local) } else { None },
                applied_delta: best_delta,
                train_nll_q20_after: post_train.nll_q20,
                dev_nll_q20_after: post_dev.nll_q20,
                dev_mistakes: post_dev_mistakes,
                output_gradient_sum,
                weight_saturation_count: 0,
                function_visible,
            });
        } else {
            round_traces.push(DirectHeadTrainRound {
                round: round_index,
                candidates_evaluated: round_candidates_evaluated,
                candidates_with_descent: 0,
                best_delta_train_nll_q20: 0,
                best_delta_dev_nll_q20: 0,
                output_weight_coordinate: None,
                output_bias_coordinate: None,
                applied_delta: 0,
                train_nll_q20_after: round_train_nll,
                dev_nll_q20_after: round_dev_nll,
                dev_mistakes: current_dev_mistakes,
                output_gradient_sum,
                weight_saturation_count: 0,
                function_visible: false,
            });
            break;
        }
    }
    let final_train = evaluate_surface(model, train_windows)?;
    let final_dev = evaluate_surface(model, dev_windows)?;
    let final_dev_mistakes = count_mistakes(model, dev_windows)?;
    Ok(DirectHeadTrainTrace {
        profile: model.config.profile_id().unwrap_or("custom"),
        parameter_count: model.parameter_count(),
        tokenizer_hash: model.tokenizer_hash,
        token_stream_hash,
        context_tokens: config.context_tokens,
        train_windows: config.train_windows,
        dev_windows: config.dev_windows,
        candidates_per_round: config.candidates_per_round,
        max_rounds: config.max_rounds,
        probability_gradient_fractional_bits: config.probability_gradient_fractional_bits,
        probability_normalization: config.probability_normalization.as_str(),
        sample_seed: config.sample_seed,
        initial_model_hash,
        final_model_hash: model.model_hash(),
        initial_train_nll_q20: initial_train_nll,
        final_train_nll_q20: final_train.nll_q20,
        initial_dev_nll_q20: initial_dev_nll,
        final_dev_nll_q20: final_dev.nll_q20,
        initial_dev_mistakes,
        final_dev_mistakes,
        rounds: round_traces.len(),
        total_descent_steps,
        total_candidates_evaluated,
        output_rounds,
        bias_rounds,
        rounds_with_descent,
        train_bindings: direct_window_bindings(train_windows),
        dev_bindings: direct_window_bindings(dev_windows),
        round_traces,
    })
}

fn count_mistakes(
    model: &ProductionModelV1,
    windows: &[DocumentWindow],
) -> Result<usize, TrainError> {
    let mut mistakes = 0_usize;
    for window in windows {
        let forward = super::forward_production_model(model, &window.context)?;
        let predicted = forward
            .logits_q8
            .iter()
            .enumerate()
            .max_by_key(|&(index, &value)| (value, Reverse(index)))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        if predicted != window.target as usize {
            mistakes = mistakes.saturating_add(1);
        }
    }
    Ok(mistakes)
}

fn select_top_gradient_candidates(
    model: &ProductionModelV1,
    windows: &[DocumentWindow],
    config: &DirectHeadTrainConfig,
    output_weight_count: usize,
) -> Result<Vec<(usize, i64)>, TrainError> {
    let training = ProductionFullTrainConfig {
        context_tokens: config.context_tokens,
        max_windows: windows.len(),
        epochs: 1,
        probability_gradient_fractional_bits: config.probability_gradient_fractional_bits,
        probability_normalization: config.probability_normalization,
        batch_windows: 1,
        max_optimizer_steps: 1,
        evaluation_windows: 0,
        ..ProductionFullTrainConfig::default()
    };
    let total_head_params = output_weight_count.saturating_add(model.config.vocab_size);
    let mut gradient_sum = vec![0_i64; total_head_params];
    let ranges = ParameterRanges::new(model);
    if ranges.output.len() != output_weight_count || ranges.bias.len() != model.config.vocab_size {
        return Err(TrainError::InvalidModel(
            "direct head gradient range mismatch",
        ));
    }
    let mut working = model.clone();
    let spec = GradientProposalSpec::lane(
        ProductionGradientProposalLane::MassCorrectedNormalized,
        splitmix64(
            config
                .sample_seed
                .wrapping_add(model.model_hash())
                .wrapping_add(0x9e37_79b9_7f4a_7c15),
        ),
    );
    for window in windows.iter().take(config.train_windows) {
        let snapshot = coarse_gradients_for_window_with_spec(
            &mut working,
            &window.context,
            window.target as usize,
            training,
            spec,
        )?;
        for (local, sum) in gradient_sum[..output_weight_count].iter_mut().enumerate() {
            *sum = sum.saturating_add(snapshot.residuals[ranges.output.start + local]);
        }
        for (local, sum) in gradient_sum[output_weight_count..].iter_mut().enumerate() {
            *sum = sum.saturating_add(snapshot.residuals[ranges.bias.start + local]);
        }
    }
    let mut indexed = gradient_sum
        .into_iter()
        .enumerate()
        .filter(|&(_, value)| value != 0)
        .collect::<Vec<_>>();
    indexed.sort_unstable_by_key(|&(_, value)| {
        -(value.unsigned_abs() as i128).min(i64::MAX as i128) as i64
    });
    indexed.truncate(config.candidates_per_round);
    Ok(indexed)
}

fn signed_nll_improvement(before: u64, after: u64) -> i64 {
    if before >= after {
        i64::try_from(before - after).unwrap_or(i64::MAX)
    } else {
        -i64::try_from(after - before).unwrap_or(i64::MAX)
    }
}

fn evaluate_parameter_delta(
    model: &mut ProductionModelV1,
    group: usize,
    index: usize,
    delta: i8,
    windows: &[DocumentWindow],
) -> Result<SurfaceEval, TrainError> {
    let inverse = delta.checked_neg().ok_or(TrainError::CoreRejected(
        "direct_search_delta_not_invertible",
    ))?;
    shift_parameter(model, group, index, delta)?;
    let evaluation = evaluate_surface(model, windows);
    shift_parameter(model, group, index, inverse)?;
    evaluation
}

fn shift_parameter(
    model: &mut ProductionModelV1,
    group: usize,
    index: usize,
    delta: i8,
) -> Result<(), TrainError> {
    if group >= 13 {
        return Err(TrainError::CoreRejected("direct_search_invalid_group"));
    }
    if !can_perturb(model, group, index, delta) {
        return Err(TrainError::CoreRejected(
            "direct_search_parameter_delta_out_of_range",
        ));
    }
    set_parameter_delta(model, group, index, delta)
}

#[cfg(test)]
mod direct_search_tests {
    use super::*;
    use crate::production::ProductionModelConfig;

    fn tiny_model() -> ProductionModelV1 {
        let config = ProductionModelConfig {
            vocab_size: 320,
            d_model: 16,
            heads: 4,
            layers: 1,
            hidden_dim: 32,
            context_tokens: 8,
        };
        let mut model =
            ProductionModelV1::new_initial(config, 0x1234, 0x5678).expect("tiny production model");
        model
            .initialize_output_weights(8)
            .expect("active output head");
        model
    }

    fn one_window() -> Vec<DocumentWindow> {
        vec![DocumentWindow {
            document: 0,
            context_start: 0,
            context: vec![300, 301],
            target: 302,
        }]
    }

    fn token_fixture() -> Vec<u32> {
        vec![
            BOS_TOKEN_ID,
            300,
            301,
            302,
            303,
            EOS_TOKEN_ID,
            BOS_TOKEN_ID,
            304,
            305,
            306,
            307,
            EOS_TOKEN_ID,
        ]
    }

    #[test]
    fn production_attention_backward_preserves_v1_affine_feature_contract() {
        assert_eq!(production_linear_attention_phi_i64(i16::MIN), 1);
        assert_eq!(production_linear_attention_phi_i64(-1), 32_768);
        assert_eq!(production_linear_attention_phi_i64(0), 32_769);
        assert_eq!(production_linear_attention_phi_i64(7), 32_776);
        assert_eq!(production_linear_attention_phi_derivative_i64(-1), 1);
        assert_eq!(production_linear_attention_phi_derivative_i64(0), 1);
        assert_eq!(production_linear_attention_phi_derivative_i64(1), 1);
    }

    #[test]
    fn direct_train_dev_surfaces_are_document_disjoint_when_possible() {
        let windows =
            super::super::alignment::document_windows_with_coordinates(&token_fixture(), 2);
        let (train, dev) = select_direct_train_dev_surfaces(&windows, 2, 2, "insufficient")
            .expect("separated direct train/dev surfaces");
        assert!(train.iter().all(|source| {
            dev.iter()
                .all(|held_out| source.document != held_out.document)
        }));
    }

    #[test]
    fn direct_search_parameter_evaluation_always_restores_the_model() {
        let mut model = tiny_model();
        model.output_weights[0] = i16::MIN + 32;
        let initial_hash = model.model_hash();
        evaluate_parameter_delta(&mut model, 11, 0, 64, &one_window())
            .expect("positive in-range probe");
        assert_eq!(model.model_hash(), initial_hash);

        assert!(evaluate_parameter_delta(&mut model, 11, 0, -64, &one_window()).is_err());
        assert_eq!(model.model_hash(), initial_hash);

        let invalid_window = vec![DocumentWindow {
            document: 0,
            context_start: 0,
            context: vec![300, 301],
            target: model.config.vocab_size as u32,
        }];
        assert!(evaluate_parameter_delta(&mut model, 11, 0, 1, &invalid_window).is_err());
        assert_eq!(model.model_hash(), initial_hash);
    }

    #[test]
    fn head_candidate_ranking_reads_output_and_bias_residual_ranges() {
        let model = tiny_model();
        let windows = one_window();
        let config = DirectHeadTrainConfig {
            context_tokens: 2,
            train_windows: 1,
            dev_windows: 1,
            candidates_per_round: 24,
            max_rounds: 1,
            ..DirectHeadTrainConfig::default()
        };
        let output_weight_count = model.output_weights.len();
        let actual = select_top_gradient_candidates(&model, &windows, &config, output_weight_count)
            .expect("head candidates");

        let training = ProductionFullTrainConfig {
            context_tokens: config.context_tokens,
            max_windows: windows.len(),
            epochs: 1,
            probability_gradient_fractional_bits: config.probability_gradient_fractional_bits,
            probability_normalization: config.probability_normalization,
            batch_windows: 1,
            max_optimizer_steps: 1,
            evaluation_windows: 0,
            ..ProductionFullTrainConfig::default()
        };
        let spec = GradientProposalSpec::lane(
            ProductionGradientProposalLane::MassCorrectedNormalized,
            splitmix64(
                config
                    .sample_seed
                    .wrapping_add(model.model_hash())
                    .wrapping_add(0x9e37_79b9_7f4a_7c15),
            ),
        );
        let mut working = model.clone();
        let snapshot = coarse_gradients_for_window_with_spec(
            &mut working,
            &windows[0].context,
            windows[0].target as usize,
            training,
            spec,
        )
        .expect("coarse gradient snapshot");
        let ranges = ParameterRanges::new(&model);
        let mut expected = snapshot.residuals[ranges.output.clone()]
            .iter()
            .copied()
            .chain(snapshot.residuals[ranges.bias.clone()].iter().copied())
            .enumerate()
            .filter(|&(_, value)| value != 0)
            .collect::<Vec<_>>();
        expected.sort_unstable_by_key(|&(_, value)| {
            -(value.unsigned_abs() as i128).min(i64::MAX as i128) as i64
        });
        expected.truncate(config.candidates_per_round);

        assert!(!expected.is_empty());
        assert_eq!(actual, expected);
    }

    #[test]
    fn direct_trainers_preserve_full_train_descent_and_initial_metrics() {
        let tokens = token_fixture();
        let windows = super::super::alignment::document_windows_with_coordinates(&tokens, 2);

        let mut head_model = tiny_model();
        let head_trace = train_production_direct_head_search(
            &mut head_model,
            &tokens,
            0x1111,
            DirectHeadTrainConfig {
                context_tokens: 2,
                train_windows: 2,
                dev_windows: 2,
                candidates_per_round: 8,
                max_rounds: 2,
                ..DirectHeadTrainConfig::default()
            },
        )
        .expect("direct head train");
        assert!(head_trace.final_train_nll_q20 <= head_trace.initial_train_nll_q20);
        assert_eq!(
            head_trace.total_candidates_evaluated,
            head_trace
                .round_traces
                .iter()
                .map(|round| round.candidates_evaluated)
                .sum()
        );
        assert_eq!(head_trace.train_bindings[0].document, 0);
        assert_eq!(head_trace.dev_bindings[0].document, 1);
        assert!(head_trace.to_json_line().contains(
            "\"window_selection\":\"different_document_else_nonoverlap_context_plus_target\""
        ));

        let mut feature_model = tiny_model();
        let initial_dev_mistakes =
            count_mistakes(&feature_model, &windows[2..4]).expect("initial dev mistakes");
        let feature_trace = train_production_direct_feature(
            &mut feature_model,
            &tokens,
            0x2222,
            DirectFeatureTrainConfig {
                context_tokens: 2,
                train_windows: 2,
                dev_windows: 2,
                head_candidates_per_round: 8,
                final_rms_candidates_per_round: 4,
                max_rounds: 3,
                ..DirectFeatureTrainConfig::default()
            },
        )
        .expect("direct feature train");
        assert!(feature_trace.final_train_nll_q20 <= feature_trace.initial_train_nll_q20);
        assert_eq!(feature_trace.initial_dev_mistakes, initial_dev_mistakes);
        assert_eq!(feature_trace.head_candidates_per_round, 8);
        assert_eq!(feature_trace.final_rms_candidates_per_round, 4);
        assert_eq!(feature_trace.max_rounds, 3);
        let json = feature_trace.to_json_line();
        assert!(json.contains("\"final_rms_candidates_per_round\":4"));
        assert!(json.contains("\"max_rounds\":3"));
        assert!(json.contains("\"gradient_use\":\"candidate_ranking_only_not_update_rule\""));
        assert!(json.contains(
            "\"window_selection\":\"different_document_else_nonoverlap_context_plus_target\""
        ));
        assert_eq!(feature_trace.train_bindings[0].document, 0);
        assert_eq!(feature_trace.dev_bindings[0].document, 1);
        assert!(!json.contains("no_backprop"));
        assert!(!json.contains("residual_feature"));
    }

    #[test]
    fn direct_mistake_count_uses_canonical_lowest_index_tie_break() {
        let mut model = tiny_model();
        model.output_weights.fill(0);
        model.output_bias_q8.fill(0);
        let correct_tie = vec![DocumentWindow {
            document: 0,
            context_start: 0,
            context: vec![300, 301],
            target: 0,
        }];
        let wrong_tie = vec![DocumentWindow {
            target: (model.config.vocab_size - 1) as u32,
            ..correct_tie[0].clone()
        }];
        assert_eq!(count_mistakes(&model, &correct_tie).expect("tie count"), 0);
        assert_eq!(count_mistakes(&model, &wrong_tie).expect("tie count"), 1);
    }

    #[test]
    fn signed_nll_improvement_reports_regressions() {
        assert_eq!(signed_nll_improvement(10, 7), 3);
        assert_eq!(signed_nll_improvement(7, 10), -3);
        assert_eq!(signed_nll_improvement(9, 9), 0);
        assert_eq!(signed_nll_improvement(u64::MAX, 0), i64::MAX);
        assert_eq!(signed_nll_improvement(0, u64::MAX), -i64::MAX);
    }

    #[test]
    fn direct_search_uses_a_stable_direction_when_both_neighbors_tie_on_descent() {
        assert_eq!(best_dir(10, 9, 9, 4), (4, 1));
        assert_eq!(best_dir(10, 10, 10, 4), (0, 0));
    }
}
