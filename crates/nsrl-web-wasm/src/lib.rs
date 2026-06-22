#![deny(unsafe_code)]

use std::collections::HashMap;

use nsrl_corpus::{LEXEME_PAGE_BOUNDARY, encode_lexeme_prompt_tokens};
use nsrl_train::{
    DecodeConfig, DecodeStrategy, LEXEME_SENTENCE_STOP_TOKEN_CAP, LexemeDecodePriors,
    LexemeGenerationConfig, LexemeQualityWeightProfile, LexemeSoftmaxModel,
    LexemeSoftmaxTrainConfig, generate_lexeme_softmax_with_priors,
    run_lexeme_softmax_training_from_softmax_model_and_quality,
};
use wasm_bindgen::prelude::*;

const DEFAULT_MAX_NEW_TOKENS: usize = 72;
const MAX_BROWSER_TOKENS: usize = 180;
const DEFAULT_FINE_TUNE_BATCH_WINDOWS: usize = 2;
const DEFAULT_FINE_TUNE_LR_SHIFT: u8 = 24;
const DEFAULT_FINE_TUNE_MAX_LR_SHIFT: u8 = 26;
const DEFAULT_TOP_K: usize = 12;
const DEFAULT_REPEAT_WINDOW: usize = 80;
const DEFAULT_REPEAT_PENALTY_SHIFT: u8 = 3;
const DEFAULT_MAX_REPEAT_RUN: usize = 2;
const DEFAULT_NO_REPEAT_NGRAM: usize = 3;
const DEFAULT_CORPUS_PRIOR_ORDER: u8 = 2;
const DEFAULT_CORPUS_PRIOR_SHIFT: u8 = 7;

#[wasm_bindgen]
pub struct NsrlChat {
    model: LexemeSoftmaxModel,
    vocab_entries: Vec<String>,
    vocab_lookup: HashMap<String, u16>,
    decode_priors: Option<LexemeDecodePriors>,
    turn_count: u64,
}

#[wasm_bindgen]
impl NsrlChat {
    #[wasm_bindgen(constructor)]
    pub fn new(
        model_bytes: &[u8],
        vocab_tsv: &str,
        token_bytes: &[u8],
    ) -> Result<NsrlChat, JsValue> {
        console_error_panic_hook::set_once();

        let model = LexemeSoftmaxModel::from_bytes(model_bytes).map_err(js_error)?;
        let vocab = parse_lexeme_vocab(vocab_tsv)?;
        if vocab.entries.len() < model.vocab_size {
            return Err(JsValue::from_str(
                "vocab has fewer entries than the model expects",
            ));
        }
        let tokens = read_lexeme_tokens(token_bytes)?;
        let decode_priors = if tokens.len() >= 2 {
            Some(LexemeDecodePriors::from_tokens(&tokens, model.vocab_size).map_err(js_error)?)
        } else {
            None
        };

        Ok(Self {
            model,
            vocab_entries: vocab.entries,
            vocab_lookup: vocab.lookup,
            decode_priors,
            turn_count: 0,
        })
    }

    pub fn reply(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
        sample_seed: u32,
        top_k: usize,
    ) -> Result<String, JsValue> {
        let prompt_tokens = encode_lexeme_prompt_tokens(prompt.as_bytes(), &self.vocab_lookup);
        let mut config = generation_config(
            max_new_tokens,
            u64::from(sample_seed).wrapping_add(self.turn_count.wrapping_mul(7919)),
            top_k,
            self.decode_priors.is_some(),
        );
        config.decode.top_k = config.decode.top_k.min(self.model.vocab_size);
        configure_sentence_stops(&mut config, &self.vocab_entries);

        let generation = generate_lexeme_softmax_with_priors(
            &self.model,
            &prompt_tokens,
            config,
            self.decode_priors.as_ref(),
        )
        .map_err(js_error)?;
        self.turn_count = self.turn_count.wrapping_add(1);

        let text = render_lexeme_tokens(&generation.generated_tokens, &self.vocab_entries);
        let text = clean_generated_text(&text);

        Ok(format!(
            "{{\"text\":\"{}\",\"generated_tokens\":{},\"distinct_per_mille\":{},\"mean_probability_q15\":{},\"stopped_on_sentence_terminal\":{},\"model_hash\":\"0x{:016x}\",\"embedding_hash\":\"0x{:016x}\",\"output_weight_hash\":\"0x{:016x}\"}}",
            escape_json(&text),
            generation.metrics.generated_token_count,
            generation.metrics.distinct_token_per_mille,
            generation.metrics.mean_selected_probability_q15,
            if generation.stopped_on_sentence_terminal {
                "true"
            } else {
                "false"
            },
            generation.model_hash,
            generation.embedding_hash,
            generation.output_weight_hash,
        ))
    }

    pub fn adapt_and_reply(
        &mut self,
        history: &str,
        latest_message: &str,
        max_new_tokens: usize,
        sample_seed: u32,
        top_k: usize,
        fine_tune_max_windows: usize,
    ) -> Result<String, JsValue> {
        let adaptation = self.fine_tune_on_history(history, fine_tune_max_windows)?;
        let prompt_tokens =
            encode_lexeme_prompt_tokens(latest_message.as_bytes(), &self.vocab_lookup);
        let mut config = generation_config(
            max_new_tokens,
            u64::from(sample_seed).wrapping_add(self.turn_count.wrapping_mul(7919)),
            top_k,
            self.decode_priors.is_some(),
        );
        config.decode.top_k = config.decode.top_k.min(self.model.vocab_size);
        configure_sentence_stops(&mut config, &self.vocab_entries);

        let generation = generate_lexeme_softmax_with_priors(
            &self.model,
            &prompt_tokens,
            config,
            self.decode_priors.as_ref(),
        )
        .map_err(js_error)?;
        self.turn_count = self.turn_count.wrapping_add(1);

        let text = render_lexeme_tokens(&generation.generated_tokens, &self.vocab_entries);
        let text = clean_generated_text(&text);

        Ok(format!(
            "{{\"text\":\"{}\",\"adapted\":{},\"history_tokens\":{},\"fine_tune_windows\":{},\"fine_tune_updates\":{},\"generated_tokens\":{},\"distinct_per_mille\":{},\"mean_probability_q15\":{},\"stopped_on_sentence_terminal\":{},\"model_hash_before\":\"0x{:016x}\",\"model_hash_after\":\"0x{:016x}\",\"embedding_hash\":\"0x{:016x}\",\"output_weight_hash\":\"0x{:016x}\"}}",
            escape_json(&text),
            if adaptation.adapted { "true" } else { "false" },
            adaptation.history_tokens,
            adaptation.windows,
            adaptation.updates,
            generation.metrics.generated_token_count,
            generation.metrics.distinct_token_per_mille,
            generation.metrics.mean_selected_probability_q15,
            if generation.stopped_on_sentence_terminal {
                "true"
            } else {
                "false"
            },
            adaptation.model_hash_before,
            self.model.model_hash(),
            generation.embedding_hash,
            generation.output_weight_hash,
        ))
    }

    pub fn model_card(&self) -> String {
        format!(
            "{{\"model\":\"crowley_bard_lexeme_wasm\",\"vocab_size\":{},\"context_seq_len\":{},\"embedding_dim\":{},\"hidden_dim\":{},\"has_corpus_priors\":{},\"default_max_new_tokens\":{},\"default_top_k\":{},\"fine_tune_lr_shift\":{},\"fine_tune_batch_windows\":{}}}",
            self.model.vocab_size,
            self.model.seq_len,
            self.model.embedding_dim,
            self.model.hidden_dim,
            if self.decode_priors.is_some() {
                "true"
            } else {
                "false"
            },
            DEFAULT_MAX_NEW_TOKENS,
            DEFAULT_TOP_K,
            DEFAULT_FINE_TUNE_LR_SHIFT,
            DEFAULT_FINE_TUNE_BATCH_WINDOWS,
        )
    }

    pub fn export_model(&self) -> Result<Vec<u8>, JsValue> {
        self.model.try_to_bytes().map_err(js_error)
    }

    pub fn import_model(&mut self, model_bytes: &[u8]) -> Result<(), JsValue> {
        let model = LexemeSoftmaxModel::from_bytes(model_bytes).map_err(js_error)?;
        if model.vocab_size != self.model.vocab_size
            || model.seq_len != self.model.seq_len
            || model.embedding_dim != self.model.embedding_dim
            || model.context_features != self.model.context_features
        {
            return Err(JsValue::from_str(
                "saved model shape does not match the loaded web bundle",
            ));
        }
        self.model = model;
        Ok(())
    }
}

impl NsrlChat {
    fn fine_tune_on_history(
        &mut self,
        history: &str,
        fine_tune_max_windows: usize,
    ) -> Result<AdaptationStats, JsValue> {
        let history_tokens = encode_lexeme_prompt_tokens(history.as_bytes(), &self.vocab_lookup);
        let model_hash_before = self.model.model_hash();
        if history_tokens.len() <= self.model.seq_len {
            return Ok(AdaptationStats {
                adapted: false,
                history_tokens: history_tokens.len(),
                windows: 0,
                updates: 0,
                model_hash_before,
            });
        }

        let token_bytes = encode_u16_tokens(&history_tokens);
        let max_windows = if fine_tune_max_windows == 0 {
            None
        } else {
            Some(fine_tune_max_windows)
        };
        let config = LexemeSoftmaxTrainConfig {
            epochs: 1,
            seq_len: self.model.seq_len,
            stride: 1,
            max_windows,
            batch_windows: DEFAULT_FINE_TUNE_BATCH_WINDOWS,
            learning_rate_shift: DEFAULT_FINE_TUNE_LR_SHIFT,
            max_learning_rate_shift: DEFAULT_FINE_TUNE_MAX_LR_SHIFT,
            max_weight_delta: 1,
            context_features: self.model.context_features,
            train_embeddings: false,
            hidden_dim: self.model.hidden_dim,
            adapter_logit_shift: self.model.adapter_logit_shift,
            ..LexemeSoftmaxTrainConfig::default()
        };
        let run = run_lexeme_softmax_training_from_softmax_model_and_quality(
            &token_bytes,
            self.model.clone(),
            config,
            None,
        )
        .map_err(js_error)?;
        let windows = run.trace.windows;
        let updates = run.trace.updates;
        self.model = run.model;

        Ok(AdaptationStats {
            adapted: true,
            history_tokens: history_tokens.len(),
            windows,
            updates,
            model_hash_before,
        })
    }
}

struct AdaptationStats {
    adapted: bool,
    history_tokens: usize,
    windows: usize,
    updates: usize,
    model_hash_before: u64,
}

struct Vocab {
    entries: Vec<String>,
    lookup: HashMap<String, u16>,
}

fn parse_lexeme_vocab(input: &str) -> Result<Vocab, JsValue> {
    let mut entries = Vec::new();
    let mut lookup = HashMap::new();
    entries.resize(256, String::new());
    for (id, entry) in entries.iter_mut().enumerate().take(256) {
        *entry = String::from_utf8_lossy(&[id as u8]).to_string();
    }

    let mut lines = input.lines();
    match lines.next() {
        Some("token_id\tlexeme\tcount") => {}
        _ => {
            return Err(JsValue::from_str(
                "vocab must start with token_id<TAB>lexeme<TAB>count",
            ));
        }
    }

    for line in lines {
        let mut parts = line.splitn(3, '\t');
        let Some(id) = parts.next() else { continue };
        let Some(lexeme) = parts.next() else { continue };
        let id = id
            .parse::<usize>()
            .map_err(|_| JsValue::from_str("vocab contains a non-numeric token id"))?;
        if id > usize::from(u16::MAX) {
            return Err(JsValue::from_str("vocab token id exceeds u16"));
        }
        if entries.len() <= id {
            entries.resize(id + 1, String::new());
        }
        entries[id] = lexeme.to_string();
        lookup.insert(lexeme.to_string(), id as u16);
    }

    Ok(Vocab { entries, lookup })
}

fn read_lexeme_tokens(bytes: &[u8]) -> Result<Vec<u16>, JsValue> {
    if bytes.len() % 2 != 0 {
        return Err(JsValue::from_str(
            "lexeme token stream must be little-endian u16 data",
        ));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect())
}

fn encode_u16_tokens(tokens: &[u16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(tokens.len() * 2);
    for &token in tokens {
        bytes.extend_from_slice(&token.to_le_bytes());
    }
    bytes
}

fn generation_config(
    max_new_tokens: usize,
    sample_seed: u64,
    top_k: usize,
    has_decode_priors: bool,
) -> LexemeGenerationConfig {
    let mut config = LexemeGenerationConfig {
        max_new_tokens: if max_new_tokens == 0 {
            DEFAULT_MAX_NEW_TOKENS
        } else {
            max_new_tokens.min(MAX_BROWSER_TOKENS)
        },
        decode: DecodeConfig {
            strategy: DecodeStrategy::Sample,
            sample_seed,
            top_k: if top_k == 0 { DEFAULT_TOP_K } else { top_k },
            repeat_window: DEFAULT_REPEAT_WINDOW,
            repeat_penalty_shift: DEFAULT_REPEAT_PENALTY_SHIFT,
            max_repeat_run: DEFAULT_MAX_REPEAT_RUN,
            no_repeat_ngram_order: DEFAULT_NO_REPEAT_NGRAM,
            corpus_prior: has_decode_priors,
            corpus_prior_order: DEFAULT_CORPUS_PRIOR_ORDER,
            corpus_prior_logit_shift: DEFAULT_CORPUS_PRIOR_SHIFT,
            ..DecodeConfig::greedy()
        },
        quality_weight_profile: LexemeQualityWeightProfile::Off,
        stop_on_sentence_terminal: true,
        sentence_terminal_token_count: 0,
        sentence_terminal_tokens: [0; LEXEME_SENTENCE_STOP_TOKEN_CAP],
    };
    config.decode.top_k = config.decode.top_k.min(64);
    config
}

fn configure_sentence_stops(config: &mut LexemeGenerationConfig, vocab: &[String]) {
    if !config.stop_on_sentence_terminal {
        return;
    }
    let mut tokens = [0_u16; LEXEME_SENTENCE_STOP_TOKEN_CAP];
    let mut count = 0_usize;
    for (index, text) in vocab.iter().enumerate() {
        if !is_sentence_terminal_lexeme(text) {
            continue;
        }
        let Ok(token) = u16::try_from(index) else {
            continue;
        };
        if count == LEXEME_SENTENCE_STOP_TOKEN_CAP {
            break;
        }
        tokens[count] = token;
        count += 1;
    }
    config.sentence_terminal_tokens = tokens;
    config.sentence_terminal_token_count = count;
}

fn render_lexeme_tokens(tokens: &[u16], vocab: &[String]) -> String {
    let mut out = String::new();
    let mut previous_was_byte_word = false;
    for &token in tokens {
        if token <= u16::from(u8::MAX) {
            append_rendered_byte_token(&mut out, token as u8, previous_was_byte_word);
            previous_was_byte_word = is_render_byte_word(token as u8);
            continue;
        }

        previous_was_byte_word = false;
        let text = vocab
            .get(usize::from(token))
            .filter(|value| !value.is_empty())
            .cloned()
            .unwrap_or_else(|| String::from("?"));
        if text == LEXEME_PAGE_BOUNDARY {
            if !out.is_empty() && !out.ends_with('\n') {
                out.push('\n');
            }
            continue;
        }
        append_rendered_lexeme(&mut out, &text);
    }
    out
}

fn append_rendered_byte_token(out: &mut String, byte: u8, previous_was_byte_word: bool) {
    if is_render_byte_word(byte) {
        if !previous_was_byte_word
            && !out.is_empty()
            && !out.ends_with(' ')
            && !out.ends_with('\'')
            && !out.ends_with('-')
        {
            out.push(' ');
        }
        out.push(char::from(byte));
        return;
    }

    let text = String::from_utf8_lossy(&[byte]).to_string();
    append_rendered_lexeme(out, &text);
}

fn append_rendered_lexeme(out: &mut String, text: &str) {
    if text.is_empty() {
        return;
    }
    if text == " " {
        if !out.ends_with(' ') {
            out.push(' ');
        }
        return;
    }
    if is_render_punctuation(text) {
        while out.ends_with(' ') {
            out.pop();
        }
        out.push_str(text);
        return;
    }
    if !out.is_empty() && !out.ends_with(' ') && !out.ends_with('\'') && !out.ends_with('-') {
        out.push(' ');
    }
    out.push_str(text);
}

fn is_render_byte_word(byte: u8) -> bool {
    byte.is_ascii_alphanumeric()
}

fn is_render_punctuation(text: &str) -> bool {
    matches!(text, "." | "," | ";" | ":" | "?" | "!" | "'" | "-")
}

fn is_sentence_terminal_lexeme(text: &str) -> bool {
    text.ends_with('.') || text.ends_with('!') || text.ends_with('?')
}

fn clean_generated_text(input: &str) -> String {
    let mut text = input.trim().to_string();
    while text.starts_with([',', ';', ':']) {
        text.remove(0);
        text = text.trim_start().to_string();
    }
    if text.is_empty() {
        String::from("...")
    } else {
        text
    }
}

fn escape_json(input: &str) -> String {
    let mut out = String::new();
    for ch in input.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            value if value.is_control() => {
                use core::fmt::Write;
                let _ = write!(&mut out, "\\u{:04x}", value as u32);
            }
            value => out.push(value),
        }
    }
    out
}

fn js_error(error: impl core::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}
