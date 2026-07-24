//! Mini-transformer tests — win.
use super::*;

#[test]
fn byte_window_starts_remain_sequential() {
    let starts = byte_window_starts(1000, 4, 10, 0, Some(5));
    assert_eq!(starts, vec![0, 10, 20, 30, 40]);
}

#[test]
fn mini_transformer_window_starts_spread_capped_runs() {
    let starts = mini_transformer_window_starts(1000, 4, 10, 0, Some(5));
    assert_eq!(starts, vec![0, 250, 500, 740, 990]);
}

#[test]
fn mini_transformer_window_starts_keep_full_runs_sequential() {
    let sequential = byte_window_starts(1000, 4, 10, 0, None);
    let distributed = mini_transformer_window_starts(1000, 4, 10, 0, None);
    assert_eq!(distributed, sequential);
}

#[test]
fn mini_transformer_filtered_window_starts_cap_after_target_filter() {
    let mut tokens = vec![b'a'; 40];
    for target_index in [4_usize, 10, 16, 22, 28, 34] {
        tokens[target_index] = b'Z';
    }

    let starts = mini_transformer_filtered_window_starts(
        tokens.len(),
        &tokens,
        MiniTransformerMlpTrainConfig {
            seq_len: 4,
            stride: 1,
            window_offset: 0,
            max_windows: Some(3),
            target_token_min: b'Z',
            target_token_max: b'Z',
            ..MiniTransformerMlpTrainConfig::default()
        },
    );

    assert_eq!(starts, vec![0, 18, 30]);
    assert!(starts.iter().all(|&start| tokens[start + 4] == b'Z'));
}

#[test]
fn mini_transformer_filtered_window_starts_can_target_marker_segment() {
    let tokens = [
        0, 1, 2, b's', b'e', 3, b'A', b'B', 5, 1, 2, b'x', 3, b'C', 4, b'i', 5,
    ];
    let starts = mini_transformer_filtered_window_starts(
        tokens.len(),
        &tokens,
        MiniTransformerMlpTrainConfig {
            seq_len: 1,
            stride: 1,
            window_offset: 0,
            max_windows: None,
            target_token_min: b'A',
            target_token_max: b'Z',
            target_segment: MiniTransformerTargetSegment::after_marker_before_any(3, &[4, 5])
                .expect("segment"),
            ..MiniTransformerMlpTrainConfig::default()
        },
    );

    assert_eq!(starts, vec![5, 6, 12]);
    assert_eq!(
        starts
            .iter()
            .map(|&start| tokens[start + 1])
            .collect::<Vec<_>>(),
        vec![b'A', b'B', b'C']
    );
}

#[test]
fn mini_transformer_filtered_window_starts_can_target_sequence_segment() {
    let tokens = [
        1, 2, b'p', 3, b'S', b'o', b'B', b'a', b':', b' ', b'H', 5, 1, 2, b'q', 3, b'S', b'o',
        b'C', b'a', b'm', b':', 5,
    ];
    let starts = mini_transformer_filtered_window_starts(
        tokens.len(),
        &tokens,
        MiniTransformerMlpTrainConfig {
            seq_len: 1,
            stride: 1,
            window_offset: 0,
            max_windows: None,
            target_token_min: b'A',
            target_token_max: b'z',
            target_segment: MiniTransformerTargetSegment::after_sequence_before_any(
                &[3, b'S', b'o'],
                &[b':', 4, 5],
            )
            .expect("segment"),
            ..MiniTransformerMlpTrainConfig::default()
        },
    );

    assert_eq!(starts, vec![5, 6, 17, 18, 19]);
    assert_eq!(
        starts
            .iter()
            .map(|&start| tokens[start + 1])
            .collect::<Vec<_>>(),
        vec![b'B', b'a', b'C', b'a', b'm']
    );
}

#[test]
fn mini_transformer_filtered_window_starts_can_target_first_after_sequence() {
    let tokens = [
        1, 3, b'H', b'e', b' ', b'm', b'a', 5, 1, 3, b'H', b'e', b' ', b'i', b's', 5,
    ];
    let starts = mini_transformer_filtered_window_starts(
        tokens.len(),
        &tokens,
        MiniTransformerMlpTrainConfig {
            seq_len: 1,
            stride: 1,
            window_offset: 0,
            max_windows: None,
            target_token_min: b'a',
            target_token_max: b'z',
            target_segment: MiniTransformerTargetSegment::first_after_sequence_before_any(
                b"He ",
                &[4, 5],
            )
            .expect("segment"),
            ..MiniTransformerMlpTrainConfig::default()
        },
    );

    assert_eq!(starts, vec![4, 12]);
    assert_eq!(
        starts
            .iter()
            .map(|&start| tokens[start + 1])
            .collect::<Vec<_>>(),
        vec![b'm', b'i']
    );
}

#[test]
fn mini_transformer_loss_guard_starts_mix_batch_and_global_points() {
    let starts: Vec<usize> = (0..32).map(|index| index * 10).collect();
    let guarded = mini_transformer_loss_guard_starts(&starts, 5, 7);

    assert!(guarded.contains(&50));
    assert!(guarded.contains(&60));
    assert_eq!(guarded.first().copied(), Some(50));
    assert_eq!(guarded.get(1).copied(), Some(60));
    assert!(guarded.contains(&0));
    assert!(guarded.contains(&310));
    assert_eq!(guarded.len(), 17);
}

#[test]
fn mini_transformer_loss_guard_ignores_small_regressions() {
    assert!(!mini_transformer_loss_guard_regressed(100_000, 117_000, 17));
    assert!(mini_transformer_loss_guard_regressed(100_000, 118_000, 17));
}

#[test]
fn attention_vo_oracle_does_not_increase_configured_loss() {
    let tokens = b"To be or not to be, that is the question. To be or not to be. ";
    let seq_len = 4;
    let starts = byte_window_starts(tokens.len(), seq_len, 1, 0, Some(4));
    let mut model = MiniTransformerMlpModel::new_initial_with_seq_len(seq_len);
    if MINI_TRANSFORMER_D_MODEL > MINI_TRANSFORMER_ATTENTION_VO_ORACLE_MAX_D_MODEL {
        assert_eq!(
            mini_transformer_attention_vo_oracle_update_i8_checked(
                &mut model, tokens, &starts, seq_len, 1,
            ),
            Err(TrainError::InvalidConfig)
        );
        return;
    }
    let before = mini_transformer_total_probability_error_q15(tokens, &starts, &model, seq_len)
        .expect("before loss");
    let (v, o) = mini_transformer_attention_vo_oracle_update_i8_checked(
        &mut model, tokens, &starts, seq_len, 1,
    )
    .expect("oracle update");
    let after = mini_transformer_total_probability_error_q15(tokens, &starts, &model, seq_len)
        .expect("after loss");

    assert!(after <= before);
    assert_eq!(v.gradient_saturation_count, 0);
    assert_eq!(o.gradient_saturation_count, 0);
    assert_eq!(
        v.zero_delta_count + v.weight_delta_l1 as usize,
        MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_D_MODEL
    );
    assert_eq!(
        o.zero_delta_count + o.weight_delta_l1 as usize,
        MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_D_MODEL
    );
}
