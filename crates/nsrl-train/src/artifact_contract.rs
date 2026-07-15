//! Frozen identifiers and byte-layout constants for persisted training artifacts.
//!
//! Keep serialization code elsewhere; this module is the compatibility boundary
//! reviewed before any schema, magic, or record-size change is accepted.

pub const MINI_TRANSFORMER_MLP_SCHEMA: &str = "nsrl.training_mini_transformer_mlp_trace.v1";
pub const MINI_TRANSFORMER_ADAM_SCHEMA: &str =
    "nsrl.training_mini_transformer_integer_adam_trace.v1";
pub const MINI_TRANSFORMER_SWARM_SCHEMA: &str = "nsrl.training_mini_transformer_swarm_trace.v1";
pub const MINI_TRANSFORMER_SWARM_WORKER_SCHEMA: &str =
    "nsrl.training_mini_transformer_swarm_worker_artifact.v1";
pub const MINI_TRANSFORMER_SWARM_PROGRESS_SCHEMA: &str =
    "nsrl.training_mini_transformer_swarm_progress.v1";
pub const MINI_TRANSFORMER_SWARM_SCALING_SCHEMA: &str =
    "nsrl.training_mini_transformer_swarm_scaling_trace.v1";
pub const MINI_TRANSFORMER_SWARM_EXPERT_MANIFEST_SCHEMA: &str =
    "nsrl.mini_transformer_swarm_expert_manifest.v1";
pub const MINI_TRANSFORMER_SWARM_ROUTE_SCHEMA: &str = "nsrl.mini_transformer_swarm_route_trace.v1";
pub const MINI_TRANSFORMER_SWARM_ROUTED_GENERATION_SCHEMA: &str =
    "nsrl.mini_transformer_swarm_routed_generation_trace.v1";
pub const MINI_TRANSFORMER_GENERATION_SCHEMA: &str = "nsrl.mini_transformer_generation_trace.v1";
pub const MINI_TRANSFORMER_SWARM_GENERATION_SCHEMA: &str =
    "nsrl.mini_transformer_swarm_generation_trace.v1";
pub const MINI_TRANSFORMER_BINARY_TRACE_SCHEMA: &str =
    "nsrl.training_mini_transformer_mlp_binary_trace.v1";

pub const AUTHORITY: &str = "deterministic_training_replay";
pub const GENERATION_AUTHORITY: &str = "deterministic_integer_generation";
pub const MINI_TRANSFORMER_MLP_TASK: &str = "wiki_bard_mini_transformer_mlp_first";
pub const BYTE_TOKENIZER_ID: &str = "byte_identity_u8_v1";
pub const ASCII_LOWER_TOKENIZER_ID: &str = "byte_ascii_lower_text_u8_v1";

pub const MINI_TRANSFORMER_MODEL_ID: &str = "mini_transformer_byte_qkvo_mlp_v1";
pub const MINI_TRANSFORMER_MODEL_MAGIC: &[u8; 8] = b"NSRLMT5\n";
pub const MINI_TRANSFORMER_V6_MODEL_MAGIC: &[u8; 8] = b"NSRLMT6\n";
pub const PRODUCTION_MODEL_V1_MAGIC: &[u8; 8] = b"NSRLPM1\n";
pub(crate) const MINI_TRANSFORMER_LEGACY_MODEL_MAGIC: &[u8; 8] = b"NSRLMT4\n";
pub(crate) const MINI_TRANSFORMER_LEGACY_V4_D_MODEL: usize = 32;
pub(crate) const MINI_TRANSFORMER_LEGACY_V4_HEADS: usize = 2;
pub(crate) const MINI_TRANSFORMER_LEGACY_V4_HIDDEN_DIM: usize = 64;
pub const MINI_TRANSFORMER_ADAM_STATE_MAGIC: &[u8; 8] = b"NSRLAD2\n";
pub const MINI_TRANSFORMER_BLOCK_EXPERT_MAGIC: &[u8; 8] = b"NSRLBE2\n";
pub const MINI_TRANSFORMER_SWARM_MODEL_ID: &str = "mini_transformer_swarm_qkvo_mlp_v1";
pub const MINI_TRANSFORMER_SWARM_MODEL_MAGIC: &[u8; 8] = b"NSRLSW1\n";
pub const MINI_TRANSFORMER_SWARM_WORKER_ARTIFACT_MAGIC: &[u8; 8] = b"NSRLWK1\n";
pub const MINI_TRANSFORMER_SWARM_CAPABILITY_TAGS: &[&str] = &[
    "byte_generation",
    "mini_transformer_mlp",
    "integer_q15",
    "swarm_ensemble",
    "deterministic_router_candidate",
];

pub const MINI_TRANSFORMER_BINARY_TRACE_MAGIC: &[u8; 4] = b"NSRL";
pub const MINI_TRANSFORMER_BINARY_TRACE_VERSION: u8 = 1;
pub const MINI_TRANSFORMER_BINARY_TRACE_SCHEMA_ID: u8 = 1;
pub const MINI_TRANSFORMER_BINARY_TRACE_HEADER_LEN: usize = 16;
pub const MINI_TRANSFORMER_BINARY_STEP_SAMPLE_RECORD_LEN: usize = 32;
pub const MINI_TRANSFORMER_BINARY_ADAPTIVE_SHIFT_RECORD_LEN: usize = 22;
pub const MINI_TRANSFORMER_BINARY_FINAL_SUMMARY_RECORD_LEN: usize = 561;
pub const MINI_TRANSFORMER_BINARY_TAG_STEP_SAMPLE: u8 = 0x01;
pub const MINI_TRANSFORMER_BINARY_TAG_ADAPTIVE_SHIFT: u8 = 0x02;
pub const MINI_TRANSFORMER_BINARY_TAG_FINAL_SUMMARY: u8 = 0x7f;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persisted_artifact_magics_are_distinct() {
        let magics = [
            MINI_TRANSFORMER_MODEL_MAGIC,
            MINI_TRANSFORMER_V6_MODEL_MAGIC,
            PRODUCTION_MODEL_V1_MAGIC,
            MINI_TRANSFORMER_LEGACY_MODEL_MAGIC,
            MINI_TRANSFORMER_ADAM_STATE_MAGIC,
            MINI_TRANSFORMER_BLOCK_EXPERT_MAGIC,
            MINI_TRANSFORMER_SWARM_MODEL_MAGIC,
            MINI_TRANSFORMER_SWARM_WORKER_ARTIFACT_MAGIC,
        ];
        for (index, magic) in magics.iter().enumerate() {
            assert!(magics[index + 1..].iter().all(|other| magic != other));
        }
    }
}
