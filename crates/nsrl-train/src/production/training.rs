use core::ops::Range;

use nsrl_core::{
    GatedMlpI16Params, GatedMlpWorkspace, LinearAttentionWorkspace, RmsNormBackwardWorkspace,
    SelfAttentionI16Params, SoftmaxNormalization, base2_exp_neg_q15, base2_exp_neg_q47,
    base2_softmax_i32_q15, base2_softmax_i32_q31_with_normalization,
    gated_activation_backward_i16_q15, gated_mlp_i16_q15_checked, hard_silu_derivative_q15,
    hard_silu_q15, linear_attention_i16_q15_checked, rms_norm_backward_i16_q15_checked,
    rms_norm_i16_q15_checked, round_shift_rhu_i64, saturate_i8, saturate_i16,
};
use nsrl_corpus::subword::{BOS_TOKEN_ID, EOS_TOKEN_ID};

use super::{
    FNV_OFFSET, FNV_PRIME, PRODUCTION_RMS_EPSILON, ProductionModelV1, TrainError, checked_product,
    fnv1a, linear_params, scale_shifts, scales,
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
    pub output_learning_rate_shift: u8,
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
            output_learning_rate_shift: 24,
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
        format!(
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
        )
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
            > 62
        || config
            .vector_learning_rate_shift
            .saturating_add(config.probability_gradient_fractional_bits - 15)
            > 62
        || effective_learning_rate_shifts(config)
            .iter()
            .any(|&shift| shift > 62)
    {
        return Err(TrainError::InvalidConfig);
    }
    let windows = document_windows(tokens, config.context_tokens, config.max_windows);
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
    while state.next_epoch < config.epochs as u64 && optimizer_steps < config.max_optimizer_steps {
        let batch_start = state.next_window as usize;
        let batch_end = batch_start
            .saturating_add(config.batch_windows)
            .min(windows.len());
        for (index, (context, target)) in windows[batch_start..batch_end].iter().enumerate() {
            let cache = forward_cache(
                model,
                context,
                config.probability_gradient_fractional_bits,
                config.probability_normalization,
            )?;
            backward_update(
                model,
                context,
                *target as usize,
                cache,
                config,
                &ranges,
                &mut state.residuals,
                index + 1 == batch_end - batch_start,
                &mut stats,
            )?;
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
            .saturating_add(config.probability_gradient_fractional_bits - 15),
        effective_bias_learning_rate_shift: config
            .vector_learning_rate_shift
            .saturating_add(config.probability_gradient_fractional_bits - 15),
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
    let config = model.config;
    if context_tokens.is_empty()
        || context_tokens.len() > config.context_tokens
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
    let last = (seq_len - 1) * config.d_model;
    let mut features = vec![0_i16; config.d_model];
    rms_norm_i16_q15_checked(
        &final_hidden[last..last + config.d_model],
        &model.final_rms_weights,
        PRODUCTION_RMS_EPSILON,
        &mut features,
    )
    .ok_or(TrainError::CoreRejected("production_training_final_rms"))?;
    let logits = output_logits(model, &features);
    let probabilities = if probability_gradient_fractional_bits == 15
        && probability_normalization == SoftmaxNormalization::LegacyQ31Lut
    {
        let mut q15 = vec![0_i16; config.vocab_size];
        base2_softmax_i32_q15(&logits, &mut q15)
            .ok_or(TrainError::CoreRejected("production_training_softmax"))?;
        q15.into_iter().map(i32::from).collect()
    } else {
        let mut q31 = vec![0_u32; config.vocab_size];
        base2_softmax_i32_q31_with_normalization(&logits, &mut q31, probability_normalization)
            .ok_or(TrainError::CoreRejected("production_training_softmax_wide"))?;
        q31.into_iter()
            .map(|probability| {
                quantize_probability_q31(probability, probability_gradient_fractional_bits)
            })
            .collect()
    };
    Ok(ForwardCache {
        layers,
        final_hidden,
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
    if target >= cache.probabilities.len() || cache.logits.len() != cache.probabilities.len() {
        return Err(TrainError::InvalidConfig);
    }
    match source {
        GradientSource::NormalizedProbability => {
            let mut gradient = cache.probabilities.clone();
            let gradient_scale = ((1_u64 << config.probability_gradient_fractional_bits) - 1)
                .try_into()
                .map_err(|_| TrainError::InvalidConfig)?;
            gradient[target] = gradient[target].saturating_sub(gradient_scale);
            Ok((gradient, config.probability_gradient_fractional_bits - 15))
        }
        GradientSource::MassCorrectedNormalizedProbability => {
            let mut gradient = cache.probabilities.clone();
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
            let maximum = cache
                .logits
                .iter()
                .copied()
                .max()
                .ok_or(TrainError::InvalidConfig)?;
            let mut gradient = cache
                .logits
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
            systematic_fixed_mass_gradient(&cache.logits, target, 1_u32 << 15, stochastic_seed)
        }
        GradientSource::SystematicFixedMassK16 => {
            systematic_fixed_mass_gradient(&cache.logits, target, 1_u32 << 16, stochastic_seed)
        }
        GradientSource::SystematicFixedMassK18 => {
            systematic_fixed_mass_gradient(&cache.logits, target, 1_u32 << 18, stochastic_seed)
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
        target,
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
fn backward_update(
    model: &mut ProductionModelV1,
    context_tokens: &[u32],
    target: usize,
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
        target,
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
    target: usize,
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
    let (grad_logits, precision_shift) =
        output_gradient(&cache, target, config, spec.source, spec.stochastic_seed)?;
    let mut grad_features = vec![0_i16; c.d_model];
    for (dim, grad_feature) in grad_features.iter_mut().enumerate() {
        let mut acc = 0_i64;
        for (token, &grad) in grad_logits.iter().enumerate() {
            acc = acc.saturating_add(
                i64::from(grad) * i64::from(model.output_weights[token * c.d_model + dim]),
            );
        }
        *grad_feature = saturate_i16(
            quantizer.shift(
                acc,
                config
                    .output_backward_shift
                    .unwrap_or(model.scales.output_shift)
                    .saturating_add(precision_shift),
                stats,
            ),
        );
    }
    for (token, &grad) in grad_logits.iter().enumerate() {
        update_i32(
            &mut model.output_bias_q8[token],
            i64::from(grad),
            config
                .vector_learning_rate_shift
                .saturating_add(precision_shift),
            &mut residuals[ranges.bias.start + token],
            apply_updates,
            12,
            stats,
        );
        for (dim, &feature) in cache.features.iter().enumerate() {
            update_i16(
                &mut model.output_weights[token * c.d_model + dim],
                i64::from(grad) * i64::from(feature),
                config
                    .output_learning_rate_shift
                    .saturating_add(precision_shift),
                &mut residuals[ranges.output.start + token * c.d_model + dim],
                apply_updates,
                11,
                stats,
            );
        }
    }
    let final_start = total - c.d_model;
    let mut grad_hidden = vec![0_i16; total];
    let mut final_gamma_grad = vec![0_i64; c.d_model];
    rms_backward_row(
        &cache.final_hidden[final_start..],
        &model.final_rms_weights,
        &grad_features,
        &mut grad_hidden[final_start..],
        &mut final_gamma_grad,
        stats,
    )?;
    update_i16_slice(
        &mut model.final_rms_weights,
        &final_gamma_grad,
        vector_group_shift(config.vector_learning_rate_shift, 3),
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
            vector_group_shift(config.vector_learning_rate_shift, 2),
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
            vector_group_shift(config.vector_learning_rate_shift, 1),
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
                config.embedding_learning_rate_shift,
                &mut residuals[ranges.embeddings.start + token as usize * c.d_model + dim],
                apply_updates,
                0,
                stats,
            );
        }
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
                let phi = i64::from(k[base + kd]) + 32769;
                sums[kd] = sums[kd].saturating_add(phi);
                for vd in 0..head_dim {
                    state[kd * head_dim + vd] =
                        state[kd * head_dim + vd].saturating_add(phi * i64::from(v[base + vd]));
                }
            }
            denominators[token] = (0..head_dim)
                .map(|d| (i64::from(q[base + d]) + 32769) * sums[d])
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
                gq[base + kd] =
                    gq[base + kd].saturating_add(quantizer.ratio(numerator, denominator, stats)?);
            }
            for kd in 0..head_dim {
                let phi_q = i64::from(q[base + kd]) + 32769;
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
                        let phi_k = i64::from(k[base + kd]) + 32769;
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
                    gk[base + kd] = gk[base + kd]
                        .saturating_add(quantizer.shift(key_numerator, 15, stats))
                        .saturating_add(grad_sums[kd]);
                }
            } else {
                for kd in 0..head_dim {
                    let phi_k = i64::from(k[base + kd]) + 32769;
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
                    gk[base + kd] = gk[base + kd]
                        .saturating_add(key_grad)
                        .saturating_add(grad_sums[kd]);
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
    explicit.unwrap_or_else(|| matrix_group_shift(config.matrix_learning_rate_shift, group))
}

pub(super) fn effective_learning_rate_shifts(config: ProductionFullTrainConfig) -> [u8; 13] {
    [
        config.embedding_learning_rate_shift,
        vector_group_shift(config.vector_learning_rate_shift, 1),
        vector_group_shift(config.vector_learning_rate_shift, 2),
        vector_group_shift(config.vector_learning_rate_shift, 3),
        matrix_learning_rate_shift(config, 4),
        matrix_learning_rate_shift(config, 5),
        matrix_learning_rate_shift(config, 6),
        matrix_learning_rate_shift(config, 7),
        matrix_learning_rate_shift(config, 8),
        matrix_learning_rate_shift(config, 9),
        matrix_learning_rate_shift(config, 10),
        config.output_learning_rate_shift,
        config.vector_learning_rate_shift,
    ]
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
        ProductionFullTrainConfig, UpdateStats, accumulate_residual,
        effective_learning_rate_shifts, schedule_hash, systematic_fixed_mass_gradient,
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
