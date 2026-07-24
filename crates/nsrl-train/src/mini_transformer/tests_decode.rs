//! Mini-transformer tests — decode.
use super::*;

#[test]
fn byte_target_frequency_weights_only_downweight_common_targets() {
    let tokens = [b'x', b'a', b'y', b'a', b'z', b'a', b'w', b'b'];
    let weights = byte_target_frequency_weights_q15(&tokens, &[0, 2, 4, 6], 1, 2, 4096)
        .expect("byte target frequency weights");

    assert!(weights[usize::from(b'a')] < i16::MAX);
    assert!(weights[usize::from(b'a')] >= 4096);
    assert_eq!(weights[usize::from(b'b')], i16::MAX);
    assert_eq!(weights[usize::from(b'c')], i16::MAX);

    let disabled = byte_target_frequency_weights_q15(&tokens, &[0, 2, 4, 6], 1, 0, 4096)
        .expect("disabled byte target frequency weights");
    assert!(disabled.iter().all(|&weight| weight == i16::MAX));
}

#[test]
fn byte_argmax_margin_gradient_pushes_target_against_best_competitor() {
    let mut gradient = [0_i32; BYTE_VOCAB];
    let mut logits = [0_i32; BYTE_VOCAB];
    logits[usize::from(b'a')] = 10;
    logits[usize::from(b'b')] = 12;
    logits[usize::from(b'c')] = 12;

    apply_byte_argmax_margin_gradient_q15(&mut gradient, &logits, b'a', i16::MAX);

    assert!(gradient[usize::from(b'a')] < 0);
    assert!(gradient[usize::from(b'b')] > 0);
    assert_eq!(gradient[usize::from(b'c')], 0);

    let pushed_target = gradient[usize::from(b'a')];
    let pushed_competitor = gradient[usize::from(b'b')];
    logits[usize::from(b'a')] = 13;
    apply_byte_argmax_margin_gradient_q15(&mut gradient, &logits, b'a', i16::MAX);
    assert_eq!(gradient[usize::from(b'a')], pushed_target);
    assert_eq!(gradient[usize::from(b'b')], pushed_competitor);
}

#[test]
fn sample_decode_is_deterministic_and_can_escape_argmax() {
    let logits = [0_i32; BYTE_VOCAB];
    let mut probabilities = [0_i16; BYTE_VOCAB];
    for probability in probabilities.iter_mut().take(4) {
        *probability = 8192;
    }
    let decode = DecodeConfig {
        strategy: DecodeStrategy::Sample,
        sample_seed: 7,
        top_k: 4,
        ..DecodeConfig::greedy()
    };

    let left =
        select_byte_from_row(&logits, &probabilities, decode, 3, b"context").expect("sample left");
    let right =
        select_byte_from_row(&logits, &probabilities, decode, 3, b"context").expect("sample right");

    assert_eq!(left, right);
    assert!(left < 4);
    assert!((0..64).any(|seed| {
        let decode = DecodeConfig {
            strategy: DecodeStrategy::Sample,
            sample_seed: seed,
            top_k: 4,
            ..DecodeConfig::greedy()
        };
        select_byte_from_row(&logits, &probabilities, decode, 0, b"context")
            .is_ok_and(|token| token != 0 && token < 4)
    }));
}

#[test]
fn printable_decode_filters_control_bytes() {
    let mut logits = [0_i32; BYTE_VOCAB];
    let mut probabilities = [0_i16; BYTE_VOCAB];
    logits[0] = 1000;
    probabilities[0] = 20_000;
    logits[usize::from(b'A')] = 900;
    probabilities[usize::from(b'A')] = 10_000;
    let decode = DecodeConfig {
        printable_only: true,
        ..DecodeConfig::greedy()
    };

    let token = select_byte_from_row(&logits, &probabilities, decode, 0, b"context")
        .expect("printable decode");

    assert_eq!(token, b'A');
}

#[test]
fn ascii_lower_decode_filters_outside_curriculum_bytes() {
    let mut logits = [0_i32; BYTE_VOCAB];
    let probabilities = [1_i16; BYTE_VOCAB];
    logits[usize::from(b'Z')] = 1000;
    logits[usize::from(b'@')] = 900;
    logits[usize::from(b'z')] = 800;
    let decode = DecodeConfig {
        ascii_lower_only: true,
        ..DecodeConfig::greedy()
    };

    let token = select_byte_from_row(&logits, &probabilities, decode, 0, b"context")
        .expect("ascii lower decode");

    assert_eq!(token, b'z');
}

#[test]
fn max_repeat_run_decode_breaks_greedy_loop() {
    let mut logits = [0_i32; BYTE_VOCAB];
    let probabilities = [1_i16; BYTE_VOCAB];
    logits[usize::from(b'a')] = 1000;
    logits[usize::from(b'b')] = 900;
    let decode = DecodeConfig {
        max_repeat_run: 3,
        ..DecodeConfig::greedy()
    };

    let token = select_byte_from_row(&logits, &probabilities, decode, 0, b"aaa")
        .expect("run-capped decode");

    assert_eq!(token, b'b');
}

#[test]
fn strict_adjacency_decode_rejects_unseen_successors() {
    let priors = ByteDecodePriors::from_tokens(b"ababab").expect("priors");
    let mut logits = [0_i32; BYTE_VOCAB];
    let probabilities = [1_i16; BYTE_VOCAB];
    logits[usize::from(b'z')] = 1000;
    logits[usize::from(b'b')] = 900;
    let decode = DecodeConfig {
        strict_adjacency: true,
        ..DecodeConfig::greedy()
    };

    let selection =
        select_byte_from_row_with_priors(&logits, &probabilities, decode, 0, b"a", Some(&priors))
            .expect("strict adjacency decode");

    assert_eq!(selection.token, b'b');
    assert_eq!(selection.candidate_count, 1);
    assert_eq!(selection.rejected_candidates.adjacency, BYTE_VOCAB - 1);
}

#[test]
fn corpus_prior_can_rerank_greedy_decode() {
    let priors = ByteDecodePriors::from_tokens(b"ababab").expect("priors");
    let mut logits = [0_i32; BYTE_VOCAB];
    let probabilities = [1_i16; BYTE_VOCAB];
    logits[usize::from(b'z')] = 1000;
    logits[usize::from(b'b')] = 900;
    let decode = DecodeConfig {
        corpus_prior: true,
        corpus_prior_logit_shift: 7,
        ..DecodeConfig::greedy()
    };

    let selection =
        select_byte_from_row_with_priors(&logits, &probabilities, decode, 0, b"a", Some(&priors))
            .expect("corpus prior decode");

    assert_eq!(selection.token, b'b');
    assert_eq!(selection.candidate_count, BYTE_VOCAB);
    assert_eq!(selection.rejected_candidates.adjacency, 0);
}

#[test]
fn corpus_prior_decode_requires_priors() {
    let logits = [0_i32; BYTE_VOCAB];
    let probabilities = [1_i16; BYTE_VOCAB];
    let decode = DecodeConfig {
        corpus_prior: true,
        ..DecodeConfig::greedy()
    };

    assert!(
        select_byte_from_row_with_priors(&logits, &probabilities, decode, 0, b"a", None).is_err()
    );
}
