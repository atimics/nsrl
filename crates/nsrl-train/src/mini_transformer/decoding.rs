//! Byte decoding policies, constraints, and deterministic sampling.

use super::*;

#[cfg(test)]
pub(super) fn select_byte_from_row(
    logits_q8: &[i32; BYTE_VOCAB],
    probabilities_q15: &[i16; BYTE_VOCAB],
    decode: DecodeConfig,
    step_index: usize,
    context: &[u8],
) -> Result<u8, TrainError> {
    Ok(select_byte_from_row_with_priors(
        logits_q8,
        probabilities_q15,
        decode,
        step_index,
        context,
        None,
    )?
    .token)
}

pub(super) fn select_byte_from_row_with_priors(
    logits_q8: &[i32; BYTE_VOCAB],
    probabilities_q15: &[i16; BYTE_VOCAB],
    decode: DecodeConfig,
    step_index: usize,
    context: &[u8],
    decode_priors: Option<&ByteDecodePriors>,
) -> Result<DecodeSelection, TrainError> {
    validate_decode_priors(decode, decode_priors)?;
    match decode.strategy {
        DecodeStrategy::Greedy => Ok(select_greedy_selection(
            logits_q8,
            probabilities_q15,
            decode,
            context,
            decode_priors,
        )),
        DecodeStrategy::Sample => sample_byte_from_probabilities_q15(
            logits_q8,
            probabilities_q15,
            decode,
            step_index,
            context,
            decode_priors,
        ),
    }
}

fn sample_byte_from_probabilities_q15(
    logits_q8: &[i32; BYTE_VOCAB],
    probabilities_q15: &[i16; BYTE_VOCAB],
    decode: DecodeConfig,
    step_index: usize,
    context: &[u8],
    decode_priors: Option<&ByteDecodePriors>,
) -> Result<DecodeSelection, TrainError> {
    let candidate_set = decode_candidates(logits_q8, decode, context, decode_priors);
    let candidates = candidate_set.candidates;
    let rejected_candidates = candidate_set.rejected_candidates;

    let mut mass = 0_u64;
    for &candidate in candidates.iter() {
        mass = mass.saturating_add(decode_candidate_weight_q15(
            probabilities_q15,
            candidate,
            decode,
            context,
            decode_priors,
        ));
    }
    if mass == 0 {
        return Ok(decode_fallback_selection(
            logits_q8,
            probabilities_q15,
            decode,
            context,
            decode_priors,
        ));
    }

    let mut threshold = decode_sample_u64(decode.sample_seed, step_index, context) % mass;
    for &candidate in candidates.iter() {
        let weight = decode_candidate_weight_q15(
            probabilities_q15,
            candidate,
            decode,
            context,
            decode_priors,
        );
        if threshold < weight {
            return Ok(DecodeSelection {
                token: candidate as u8,
                candidate_count: candidates.len(),
                rejected_candidates,
            });
        }
        threshold -= weight;
    }

    Ok(decode_fallback_selection(
        logits_q8,
        probabilities_q15,
        decode,
        context,
        decode_priors,
    ))
}

fn select_greedy_selection(
    logits_q8: &[i32; BYTE_VOCAB],
    probabilities_q15: &[i16; BYTE_VOCAB],
    decode: DecodeConfig,
    context: &[u8],
    decode_priors: Option<&ByteDecodePriors>,
) -> DecodeSelection {
    if !decode_has_constraints(decode) {
        return DecodeSelection {
            token: byte_argmax_i32(logits_q8),
            candidate_count: BYTE_VOCAB,
            rejected_candidates: DecodeRejectStats::default(),
        };
    }

    decode_fallback_selection(logits_q8, probabilities_q15, decode, context, decode_priors)
}

fn decode_fallback_selection(
    logits_q8: &[i32; BYTE_VOCAB],
    probabilities_q15: &[i16; BYTE_VOCAB],
    decode: DecodeConfig,
    context: &[u8],
    decode_priors: Option<&ByteDecodePriors>,
) -> DecodeSelection {
    let candidate_set = decode_candidates(logits_q8, decode, context, decode_priors);
    let candidates = candidate_set.candidates;
    let token = if decode.repeat_window == 0
        && decode.repeat_penalty_shift == 0
        && !decode.corpus_prior
    {
        candidates
            .first()
            .copied()
            .unwrap_or_else(|| usize::from(byte_argmax_i32(logits_q8))) as u8
    } else {
        candidates
            .iter()
            .copied()
            .max_by_key(|&candidate| {
                (
                    decode_candidate_weight_q15(
                        probabilities_q15,
                        candidate,
                        decode,
                        context,
                        decode_priors,
                    ),
                    decode_effective_logit_q8(logits_q8, candidate, decode, context, decode_priors),
                    core::cmp::Reverse(candidate),
                )
            })
            .unwrap_or_else(|| usize::from(byte_argmax_i32(logits_q8))) as u8
    };
    DecodeSelection {
        token,
        candidate_count: candidates.len(),
        rejected_candidates: candidate_set.rejected_candidates,
    }
}

fn decode_candidates(
    logits_q8: &[i32; BYTE_VOCAB],
    decode: DecodeConfig,
    context: &[u8],
    decode_priors: Option<&ByteDecodePriors>,
) -> DecodeCandidateSet {
    let top_k = if decode.top_k == 0 || decode.top_k > BYTE_VOCAB {
        BYTE_VOCAB
    } else {
        decode.top_k
    };
    let mut rejected_candidates = DecodeRejectStats::default();
    let mut candidates = Vec::with_capacity(BYTE_VOCAB);
    for candidate in 0..BYTE_VOCAB {
        let token = candidate as u8;
        if decode.printable_only && !is_printable_decode_byte(token) {
            rejected_candidates.non_printable += 1;
            continue;
        }
        if decode.ascii_lower_only && !is_ascii_lower_text_decode_byte(token) {
            rejected_candidates.outside_ascii_lower += 1;
            continue;
        }
        if decode.max_repeat_run > 0
            && would_exceed_repeat_run(token, context, decode.max_repeat_run)
        {
            rejected_candidates.repeat_run += 1;
            continue;
        }
        if would_repeat_ngram(token, context, decode.no_repeat_ngram_order) {
            rejected_candidates.repeat_ngram += 1;
            continue;
        }
        if decode.strict_adjacency
            && let (Some(priors), Some(&previous)) = (decode_priors, context.last())
            && !priors.allows_transition(previous, token)
        {
            rejected_candidates.adjacency += 1;
            continue;
        }
        candidates.push(candidate);
    }
    if candidates.len() > top_k {
        rejected_candidates.top_k_truncated += candidates.len() - top_k;
        candidates.select_nth_unstable_by(top_k, |&left, &right| {
            compare_byte_decode_candidates(left, right, logits_q8, decode, context, decode_priors)
        });
        candidates.truncate(top_k);
    }
    candidates.sort_unstable_by(|&left, &right| {
        compare_byte_decode_candidates(left, right, logits_q8, decode, context, decode_priors)
    });
    if candidates.is_empty() {
        candidates.push(usize::from(byte_argmax_i32(logits_q8)));
    }
    DecodeCandidateSet {
        candidates,
        rejected_candidates,
    }
}

fn compare_byte_decode_candidates(
    left: usize,
    right: usize,
    logits_q8: &[i32; BYTE_VOCAB],
    decode: DecodeConfig,
    context: &[u8],
    decode_priors: Option<&ByteDecodePriors>,
) -> core::cmp::Ordering {
    decode_effective_logit_q8(logits_q8, right, decode, context, decode_priors)
        .cmp(&decode_effective_logit_q8(
            logits_q8,
            left,
            decode,
            context,
            decode_priors,
        ))
        .then_with(|| left.cmp(&right))
}

fn decode_candidate_weight_q15(
    probabilities_q15: &[i16; BYTE_VOCAB],
    candidate: usize,
    decode: DecodeConfig,
    context: &[u8],
    decode_priors: Option<&ByteDecodePriors>,
) -> u64 {
    let mut weight = i32::from(probabilities_q15[candidate]).max(0) as u64;
    if decode.corpus_prior
        && let (Some(priors), Some(&previous)) = (decode_priors, context.last())
    {
        let prior_q15 = priors.transition_probability_q15(previous, candidate as u8);
        let bonus = (weight.saturating_mul(u64::from(prior_q15))) >> Q15_SHIFT;
        weight = weight.saturating_add(bonus);
    }
    if decode.repeat_window > 0 && decode.repeat_penalty_shift > 0 {
        let repeat_count = recent_byte_count(candidate as u8, context, decode.repeat_window);
        let penalty_shift = repeat_count
            .saturating_mul(usize::from(decode.repeat_penalty_shift))
            .min(63);
        weight >>= penalty_shift;
    }
    weight
}

fn decode_has_constraints(decode: DecodeConfig) -> bool {
    decode.printable_only
        || decode.ascii_lower_only
        || decode.max_repeat_run > 0
        || decode.no_repeat_ngram_order > 1
        || decode.corpus_prior
        || decode.strict_adjacency
        || (decode.repeat_window > 0 && decode.repeat_penalty_shift > 0)
}

pub(super) fn validate_decode_priors(
    decode: DecodeConfig,
    decode_priors: Option<&ByteDecodePriors>,
) -> Result<(), TrainError> {
    if decode.corpus_prior
        && (decode.corpus_prior_order == 0
            || decode.corpus_prior_order > DEFAULT_CORPUS_PRIOR_ORDER)
    {
        return Err(TrainError::InvalidConfig);
    }
    if (decode.corpus_prior || decode.strict_adjacency) && decode_priors.is_none() {
        return Err(TrainError::InvalidConfig);
    }
    Ok(())
}

fn decode_effective_logit_q8(
    logits_q8: &[i32; BYTE_VOCAB],
    candidate: usize,
    decode: DecodeConfig,
    context: &[u8],
    decode_priors: Option<&ByteDecodePriors>,
) -> i32 {
    let mut logit = logits_q8[candidate];
    if decode.corpus_prior
        && let (Some(priors), Some(&previous)) = (decode_priors, context.last())
    {
        let prior_q15 = i32::from(priors.transition_probability_q15(previous, candidate as u8));
        let shift = decode.corpus_prior_logit_shift.min(30);
        logit = logit.saturating_add(prior_q15 >> shift);
    }
    logit
}

fn is_printable_decode_byte(byte: u8) -> bool {
    byte == b'\n' || (b' '..=b'~').contains(&byte)
}

fn is_ascii_lower_text_decode_byte(byte: u8) -> bool {
    matches!(
        byte,
        b'a'..=b'z'
            | b'0'..=b'9'
            | b'.'
            | b','
            | b';'
            | b':'
            | b'?'
            | b'!'
            | b'\''
            | b'-'
            | b' '
    )
}

fn recent_byte_count(candidate: u8, context: &[u8], repeat_window: usize) -> usize {
    context
        .iter()
        .rev()
        .take(repeat_window)
        .filter(|&&byte| byte == candidate)
        .count()
}

fn would_exceed_repeat_run(candidate: u8, context: &[u8], max_repeat_run: usize) -> bool {
    let run_len = context
        .iter()
        .rev()
        .take_while(|&&byte| byte == candidate)
        .count();
    run_len >= max_repeat_run
}

fn would_repeat_ngram<T: Copy + Eq>(candidate: T, context: &[T], ngram_order: usize) -> bool {
    if ngram_order < 2 || context.len() + 1 < ngram_order {
        return false;
    }

    let prefix_len = ngram_order - 1;
    let suffix_start = context.len() - prefix_len;
    let suffix = &context[suffix_start..];
    let search_end = context.len() + 1 - ngram_order;
    for start in 0..search_end {
        if &context[start..start + prefix_len] == suffix && context[start + prefix_len] == candidate
        {
            return true;
        }
    }
    false
}

fn decode_sample_u64(seed: u64, step_index: usize, context: &[u8]) -> u64 {
    let mut hasher = StableHasher::new();
    hasher.update_bytes(&seed.to_le_bytes());
    hasher.update_usize(step_index);
    hasher.update_u8_slice(context);
    splitmix64(hasher.finish())
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
