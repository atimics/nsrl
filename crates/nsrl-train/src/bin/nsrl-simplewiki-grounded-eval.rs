#![deny(unsafe_code)]

use std::collections::{BTreeMap, HashMap};
use std::env;
use std::fs;
use std::path::PathBuf;

use nsrl_eval::{
    json_escape, read_u16_le_tokens, replay_check, source_grounding_metrics, stable_hash_bytes,
    stable_hex_u32, unix_timestamp,
};
use nsrl_train::{LexemeSoftmaxModel, run_lexeme_softmax_evaluate};

const SCHEMA: &str = "nsrl.simplewiki_grounded_eval.v1";

#[derive(Debug, Clone)]
struct Config {
    manifest_path: PathBuf,
    eval_tokens_path: Option<PathBuf>,
    bpt_max_windows: Option<usize>,
    timestamp: Option<u64>,
    expect_row_hash: Option<String>,
}

#[derive(Debug, Clone)]
struct Manifest {
    schema: String,
    run_id: String,
    out_dir: String,
    model_path: PathBuf,
    tokens_path: PathBuf,
    vocab_path: PathBuf,
    prompt: String,
    paragraph_text_path: PathBuf,
    choices_tsv_path: PathBuf,
}

#[derive(Debug, Clone)]
struct ChoiceRow {
    sentence: usize,
    candidate: usize,
    seed: u64,
    fragment_path: PathBuf,
    trace_path: PathBuf,
}

#[derive(Debug, Clone)]
struct SentenceEval {
    sentence: usize,
    candidate: usize,
    seed: u64,
    text: String,
    generated_tokens: Vec<u16>,
    stopped: bool,
    words: usize,
    sentence_marks: usize,
    source_exact_span: bool,
    source_longest_run: usize,
    source_bigram_per_mille: usize,
    source_trigram_per_mille: usize,
    repeated_bigram_count: usize,
    repeated_trigram_count: usize,
    max_token_run: usize,
    trace_bytes: usize,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("nsrl-simplewiki-grounded-eval: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args(env::args().skip(1))?;
    let manifest_text = fs::read_to_string(&config.manifest_path)?;
    let manifest = parse_manifest(&manifest_text)?;
    let source_tokens = read_u16_le_tokens(&manifest.tokens_path)?;
    let eval_tokens_path = config
        .eval_tokens_path
        .as_ref()
        .unwrap_or(&manifest.tokens_path)
        .clone();
    let eval_token_bytes = fs::read(&eval_tokens_path)?;
    let model_bytes = fs::read(&manifest.model_path)?;
    let artifact_bytes = model_bytes.len();
    let artifact_hash = stable_hex_u32(stable_hash_bytes(&model_bytes));
    let model = LexemeSoftmaxModel::from_bytes(&model_bytes)?;
    let model_hash = model.model_hash();
    let embedding_hash = model.embedding_hash();
    let hidden_weight_hash = model.hidden_weight_hash();
    let output_weight_hash = model.output_weight_hash();
    let bpt = run_lexeme_softmax_evaluate(
        &eval_token_bytes,
        model,
        usize_from_model_seq_len(&manifest.model_path, &model_bytes)?,
        1,
        0,
        config.bpt_max_windows,
    )?;

    let choices = read_choices_tsv(&manifest.choices_tsv_path)?;
    let mut sentences = Vec::with_capacity(choices.len());
    for choice in choices {
        sentences.push(evaluate_choice(choice, &source_tokens)?);
    }

    let paragraph = fs::read_to_string(&manifest.paragraph_text_path)?;
    let timestamp = match config.timestamp {
        Some(timestamp) => timestamp,
        None => unix_timestamp()?,
    };
    let row = eval_row(
        &config,
        &manifest,
        timestamp,
        &paragraph,
        source_tokens.len(),
        &eval_tokens_path,
        artifact_bytes,
        &artifact_hash,
        model_hash,
        embedding_hash,
        hidden_weight_hash,
        output_weight_hash,
        bpt.to_json_line().trim_end(),
        &sentences,
    )?;
    let replay = replay_check(&row, config.expect_row_hash.as_deref());
    if !replay.passed {
        return Err(format!(
            "replay row hash mismatch: expected {}, actual {}",
            replay.expected_hash.unwrap_or_else(|| "<none>".to_string()),
            replay.actual_hash
        )
        .into());
    }
    println!("{row}");
    Ok(())
}

fn usize_from_model_seq_len(
    path: &PathBuf,
    bytes: &[u8],
) -> Result<usize, Box<dyn std::error::Error>> {
    Ok(LexemeSoftmaxModel::from_bytes(bytes)
        .map_err(|error| format!("{}: {error}", path.display()))?
        .seq_len)
}

fn usage() {
    println!(
        "Usage: nsrl-simplewiki-grounded-eval --manifest PATH [--eval-tokens PATH] [--bpt-max-windows N] [--timestamp UNIX] [--expect-row-hash HASH]"
    );
}

fn parse_args<I>(mut args: I) -> Result<Config, Box<dyn std::error::Error>>
where
    I: Iterator<Item = String>,
{
    let mut manifest_path = None;
    let mut eval_tokens_path = None;
    let mut bpt_max_windows = Some(10_000_usize);
    let mut timestamp = None;
    let mut expect_row_hash = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                usage();
                std::process::exit(0);
            }
            "--manifest" => {
                manifest_path = Some(PathBuf::from(
                    args.next().ok_or("--manifest requires PATH")?,
                ));
            }
            "--eval-tokens" => {
                eval_tokens_path = Some(PathBuf::from(
                    args.next().ok_or("--eval-tokens requires PATH")?,
                ));
            }
            "--bpt-max-windows" => {
                bpt_max_windows = Some(args.next().ok_or("--bpt-max-windows requires N")?.parse()?);
            }
            "--all-bpt-windows" => {
                bpt_max_windows = None;
            }
            "--timestamp" => {
                timestamp = Some(args.next().ok_or("--timestamp requires UNIX")?.parse()?);
            }
            "--expect-row-hash" => {
                expect_row_hash = Some(args.next().ok_or("--expect-row-hash requires HASH")?);
            }
            _ => return Err(format!("unknown option: {arg}").into()),
        }
    }
    Ok(Config {
        manifest_path: manifest_path.ok_or("--manifest is required")?,
        eval_tokens_path,
        bpt_max_windows,
        timestamp,
        expect_row_hash,
    })
}

fn parse_manifest(text: &str) -> Result<Manifest, Box<dyn std::error::Error>> {
    Ok(Manifest {
        schema: json_string_field(text, "schema")?.ok_or("manifest missing schema")?,
        run_id: json_string_field(text, "run_id")?.ok_or("manifest missing run_id")?,
        out_dir: json_string_field(text, "out_dir")?.ok_or("manifest missing out_dir")?,
        model_path: PathBuf::from(
            json_string_field(text, "model")?.ok_or("manifest missing model")?,
        ),
        tokens_path: PathBuf::from(
            json_string_field(text, "tokens")?.ok_or("manifest missing tokens")?,
        ),
        vocab_path: PathBuf::from(
            json_string_field(text, "vocab")?.ok_or("manifest missing vocab")?,
        ),
        prompt: json_string_field(text, "prompt")?.ok_or("manifest missing prompt")?,
        paragraph_text_path: PathBuf::from(
            json_string_field(text, "paragraph_text")?.ok_or("manifest missing paragraph_text")?,
        ),
        choices_tsv_path: PathBuf::from(
            json_string_field(text, "choices_tsv")?.ok_or("manifest missing choices_tsv")?,
        ),
    })
}

fn read_choices_tsv(path: &PathBuf) -> Result<Vec<ChoiceRow>, Box<dyn std::error::Error>> {
    let text = fs::read_to_string(path)?;
    let mut lines = text.lines();
    let header = lines.next().ok_or("choices TSV is empty")?;
    let columns = header
        .split('\t')
        .enumerate()
        .map(|(index, name)| (name.to_string(), index))
        .collect::<BTreeMap<_, _>>();
    let mut rows = Vec::new();
    for (line_index, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let fields = line.split('\t').collect::<Vec<_>>();
        rows.push(ChoiceRow {
            sentence: parse_tsv_usize(path, line_index + 2, &fields, &columns, "sentence")?,
            candidate: parse_tsv_usize(path, line_index + 2, &fields, &columns, "candidate")?,
            seed: parse_tsv_u64(path, line_index + 2, &fields, &columns, "seed")?,
            fragment_path: PathBuf::from(tsv_field(
                path,
                line_index + 2,
                &fields,
                &columns,
                "fragment",
            )?),
            trace_path: PathBuf::from(tsv_field(path, line_index + 2, &fields, &columns, "trace")?),
        });
    }
    if rows.is_empty() {
        return Err(format!("{} has no selected choices", path.display()).into());
    }
    Ok(rows)
}

fn evaluate_choice(
    choice: ChoiceRow,
    source_tokens: &[u16],
) -> Result<SentenceEval, Box<dyn std::error::Error>> {
    let text = fs::read_to_string(&choice.fragment_path)?;
    let trace = fs::read_to_string(&choice.trace_path)?;
    let generated_tokens = generation_tokens(&trace)?;
    let grounding = source_grounding_metrics(source_tokens, &generated_tokens);
    Ok(SentenceEval {
        sentence: choice.sentence,
        candidate: choice.candidate,
        seed: choice.seed,
        text: normalize_space(&text),
        stopped: generation_stopped(&trace).unwrap_or(false),
        words: word_count(&text),
        sentence_marks: sentence_mark_count(&text),
        repeated_bigram_count: repeated_ngram_count(&generated_tokens, 2),
        repeated_trigram_count: repeated_ngram_count(&generated_tokens, 3),
        max_token_run: max_token_run(&generated_tokens),
        generated_tokens,
        source_exact_span: grounding.exact_span,
        source_longest_run: grounding.longest_run,
        source_bigram_per_mille: grounding.bigram_per_mille,
        source_trigram_per_mille: grounding.trigram_per_mille,
        trace_bytes: trace.len(),
    })
}

#[allow(clippy::too_many_arguments)]
fn eval_row(
    config: &Config,
    manifest: &Manifest,
    timestamp: u64,
    paragraph: &str,
    source_token_count: usize,
    eval_tokens_path: &PathBuf,
    artifact_bytes: usize,
    artifact_hash: &str,
    model_hash: u64,
    embedding_hash: u64,
    hidden_weight_hash: u64,
    output_weight_hash: u64,
    bpt_json: &str,
    sentences: &[SentenceEval],
) -> Result<String, Box<dyn std::error::Error>> {
    let mut out = String::new();
    out.push('{');
    json_pair(&mut out, "schema", SCHEMA, true);
    number_pair(&mut out, "timestamp_unix_s", timestamp, true);
    json_pair(&mut out, "source_manifest_schema", &manifest.schema, true);
    json_pair(&mut out, "run_id", &manifest.run_id, true);
    json_pair(&mut out, "out_dir", &manifest.out_dir, true);
    out.push_str("\"model\":{");
    json_pair(
        &mut out,
        "path",
        &manifest.model_path.display().to_string(),
        true,
    );
    json_pair(&mut out, "artifact_hash", artifact_hash, true);
    hash_pair(&mut out, "model_hash", model_hash, true);
    hash_pair(&mut out, "embedding_hash", embedding_hash, true);
    hash_pair(&mut out, "hidden_weight_hash", hidden_weight_hash, true);
    hash_pair(&mut out, "output_weight_hash", output_weight_hash, true);
    number_pair(
        &mut out,
        "artifact_bytes",
        u64::try_from(artifact_bytes)?,
        false,
    );
    out.push_str("},");
    out.push_str("\"data\":{");
    json_pair(
        &mut out,
        "tokens",
        &manifest.tokens_path.display().to_string(),
        true,
    );
    json_pair(
        &mut out,
        "eval_tokens",
        &eval_tokens_path.display().to_string(),
        true,
    );
    json_pair(
        &mut out,
        "vocab",
        &manifest.vocab_path.display().to_string(),
        true,
    );
    number_pair(
        &mut out,
        "source_token_count",
        u64::try_from(source_token_count)?,
        true,
    );
    number_pair(
        &mut out,
        "bpt_max_windows",
        u64::try_from(config.bpt_max_windows.unwrap_or(0))?,
        false,
    );
    out.push_str("},");
    out.push_str("\"bpt\":");
    out.push_str(bpt_json);
    out.push(',');
    push_grounding_summary(&mut out, sentences)?;
    out.push(',');
    push_repetition_summary(&mut out, sentences)?;
    out.push(',');
    push_trace_size_summary(
        &mut out,
        &config.manifest_path,
        manifest,
        paragraph,
        sentences,
    )?;
    out.push(',');
    push_sample(&mut out, manifest, paragraph, sentences)?;
    out.push('}');
    Ok(out)
}

fn push_grounding_summary(
    out: &mut String,
    sentences: &[SentenceEval],
) -> Result<(), Box<dyn std::error::Error>> {
    let count = sentences.len();
    let exact_count = sentences
        .iter()
        .filter(|sentence| sentence.source_exact_span)
        .count();
    let longest_total = sentences
        .iter()
        .map(|sentence| sentence.source_longest_run)
        .sum::<usize>();
    let bigram_total = sentences
        .iter()
        .map(|sentence| sentence.source_bigram_per_mille)
        .sum::<usize>();
    let trigram_total = sentences
        .iter()
        .map(|sentence| sentence.source_trigram_per_mille)
        .sum::<usize>();
    out.push_str("\"grounding\":{");
    number_pair(out, "selected_sentences", u64::try_from(count)?, true);
    number_pair(
        out,
        "source_exact_span_count",
        u64::try_from(exact_count)?,
        true,
    );
    number_pair(
        out,
        "source_exact_span_per_mille",
        u64::try_from(per_mille(exact_count, count))?,
        true,
    );
    number_pair(
        out,
        "mean_source_longest_run_q8",
        mean_q8(usize_to_u64(longest_total)?, usize_to_u64(count)?),
        true,
    );
    number_pair(
        out,
        "min_source_bigram_per_mille",
        u64::try_from(
            sentences
                .iter()
                .map(|sentence| sentence.source_bigram_per_mille)
                .min()
                .unwrap_or(0),
        )?,
        true,
    );
    number_pair(
        out,
        "mean_source_bigram_per_mille",
        u64::try_from(mean_usize(bigram_total, count))?,
        true,
    );
    number_pair(
        out,
        "min_source_trigram_per_mille",
        u64::try_from(
            sentences
                .iter()
                .map(|sentence| sentence.source_trigram_per_mille)
                .min()
                .unwrap_or(0),
        )?,
        true,
    );
    number_pair(
        out,
        "mean_source_trigram_per_mille",
        u64::try_from(mean_usize(trigram_total, count))?,
        false,
    );
    out.push('}');
    Ok(())
}

fn push_repetition_summary(
    out: &mut String,
    sentences: &[SentenceEval],
) -> Result<(), Box<dyn std::error::Error>> {
    let repeated_bigram_count = sentences
        .iter()
        .map(|sentence| sentence.repeated_bigram_count)
        .sum::<usize>();
    let repeated_trigram_count = sentences
        .iter()
        .map(|sentence| sentence.repeated_trigram_count)
        .sum::<usize>();
    let max_token_run = sentences
        .iter()
        .map(|sentence| sentence.max_token_run)
        .max()
        .unwrap_or(0);
    let words = sentences
        .iter()
        .map(|sentence| sentence.words)
        .sum::<usize>();
    let sentence_marks = sentences
        .iter()
        .map(|sentence| sentence.sentence_marks)
        .sum::<usize>();
    out.push_str("\"repetition\":{");
    number_pair(out, "words", u64::try_from(words)?, true);
    number_pair(out, "sentence_marks", u64::try_from(sentence_marks)?, true);
    number_pair(
        out,
        "repeated_bigram_count",
        u64::try_from(repeated_bigram_count)?,
        true,
    );
    number_pair(
        out,
        "repeated_trigram_count",
        u64::try_from(repeated_trigram_count)?,
        true,
    );
    number_pair(out, "max_token_run", u64::try_from(max_token_run)?, false);
    out.push('}');
    Ok(())
}

fn push_trace_size_summary(
    out: &mut String,
    manifest_path: &PathBuf,
    manifest: &Manifest,
    paragraph: &str,
    sentences: &[SentenceEval],
) -> Result<(), Box<dyn std::error::Error>> {
    let selected_trace_bytes = sentences
        .iter()
        .map(|sentence| sentence.trace_bytes)
        .sum::<usize>();
    let manifest_bytes = fs::metadata(manifest_path)?.len();
    let choices_bytes = fs::metadata(&manifest.choices_tsv_path)?.len();
    out.push_str("\"trace_size\":{");
    number_pair(
        out,
        "selected_generation_trace_bytes",
        u64::try_from(selected_trace_bytes)?,
        true,
    );
    number_pair(out, "manifest_bytes", manifest_bytes, true);
    number_pair(out, "choices_tsv_bytes", choices_bytes, true);
    number_pair(
        out,
        "paragraph_bytes",
        u64::try_from(paragraph.len())?,
        false,
    );
    out.push('}');
    Ok(())
}

fn push_sample(
    out: &mut String,
    manifest: &Manifest,
    paragraph: &str,
    sentences: &[SentenceEval],
) -> Result<(), Box<dyn std::error::Error>> {
    out.push_str("\"sample\":{");
    json_pair(out, "prompt", &manifest.prompt, true);
    json_pair(out, "paragraph", &normalize_space(paragraph), true);
    out.push_str("\"sentences\":[");
    for (index, sentence) in sentences.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push('{');
        number_pair(out, "sentence", u64::try_from(sentence.sentence)?, true);
        number_pair(out, "candidate", u64::try_from(sentence.candidate)?, true);
        number_pair(out, "seed", sentence.seed, true);
        bool_pair(out, "stopped", sentence.stopped, true);
        number_pair(
            out,
            "generated_tokens",
            u64::try_from(sentence.generated_tokens.len())?,
            true,
        );
        bool_pair(out, "source_exact_span", sentence.source_exact_span, true);
        number_pair(
            out,
            "source_trigram_per_mille",
            u64::try_from(sentence.source_trigram_per_mille)?,
            true,
        );
        json_pair(out, "text", &sentence.text, false);
        out.push('}');
    }
    out.push_str("]}");
    Ok(())
}

fn json_string_field(text: &str, key: &str) -> Result<Option<String>, Box<dyn std::error::Error>> {
    let pattern = format!("\"{key}\"");
    let Some(key_start) = text.find(&pattern) else {
        return Ok(None);
    };
    let after_key = &text[key_start + pattern.len()..];
    let colon = after_key.find(':').ok_or("malformed JSON field")?;
    let mut value = after_key[colon + 1..].trim_start().chars().peekable();
    if value.next() != Some('"') {
        return Ok(None);
    }
    let mut out = String::new();
    while let Some(ch) = value.next() {
        match ch {
            '"' => return Ok(Some(out)),
            '\\' => match value.next() {
                Some('"') => out.push('"'),
                Some('\\') => out.push('\\'),
                Some('/') => out.push('/'),
                Some('n') => out.push('\n'),
                Some('r') => out.push('\r'),
                Some('t') => out.push('\t'),
                Some(other) => out.push(other),
                None => return Err("unterminated JSON escape".into()),
            },
            other => out.push(other),
        }
    }
    Err("unterminated JSON string".into())
}

fn generation_tokens(trace: &str) -> Result<Vec<u16>, Box<dyn std::error::Error>> {
    let generation = trace
        .find("\"generation\"")
        .ok_or("trace missing generation object")?;
    let tokens = trace[generation..]
        .find("\"tokens\"")
        .ok_or("trace missing generation tokens")?
        + generation;
    let open = trace[tokens..]
        .find('[')
        .ok_or("trace generation tokens missing [")?
        + tokens;
    let close = trace[open..]
        .find(']')
        .ok_or("trace generation tokens missing ]")?
        + open;
    let mut out = Vec::new();
    for value in trace[open + 1..close].split(',') {
        let value = value.trim();
        if value.is_empty() {
            continue;
        }
        out.push(value.parse()?);
    }
    Ok(out)
}

fn generation_stopped(trace: &str) -> Option<bool> {
    let generation = trace.find("\"generation\"")?;
    let key = trace[generation..].find("\"stopped_on_sentence_terminal\"")? + generation;
    let colon = trace[key..].find(':')? + key;
    let value = trace[colon + 1..].trim_start();
    if value.starts_with("true") {
        Some(true)
    } else if value.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

fn parse_tsv_usize(
    path: &PathBuf,
    line: usize,
    fields: &[&str],
    columns: &BTreeMap<String, usize>,
    name: &str,
) -> Result<usize, Box<dyn std::error::Error>> {
    Ok(tsv_field(path, line, fields, columns, name)?.parse()?)
}

fn parse_tsv_u64(
    path: &PathBuf,
    line: usize,
    fields: &[&str],
    columns: &BTreeMap<String, usize>,
    name: &str,
) -> Result<u64, Box<dyn std::error::Error>> {
    Ok(tsv_field(path, line, fields, columns, name)?.parse()?)
}

fn tsv_field<'a>(
    path: &PathBuf,
    line: usize,
    fields: &'a [&str],
    columns: &BTreeMap<String, usize>,
    name: &str,
) -> Result<&'a str, Box<dyn std::error::Error>> {
    let index = *columns
        .get(name)
        .ok_or_else(|| format!("{} missing {name} column", path.display()))?;
    fields
        .get(index)
        .copied()
        .ok_or_else(|| format!("{} line {line} missing {name}", path.display()).into())
}

fn repeated_ngram_count(tokens: &[u16], order: usize) -> usize {
    if order == 0 || tokens.len() < order {
        return 0;
    }
    let mut counts = HashMap::<Vec<u16>, usize>::new();
    for window in tokens.windows(order) {
        *counts.entry(window.to_vec()).or_insert(0) += 1;
    }
    counts.values().filter(|&&count| count > 1).count()
}

fn max_token_run(tokens: &[u16]) -> usize {
    let mut max_run = 0_usize;
    let mut run = 0_usize;
    let mut previous = None;
    for &token in tokens {
        if previous == Some(token) {
            run += 1;
        } else {
            run = 1;
            previous = Some(token);
        }
        max_run = max_run.max(run);
    }
    max_run
}

fn word_count(text: &str) -> usize {
    let mut count = 0_usize;
    let mut in_word = false;
    for ch in text.chars() {
        let word_char = ch.is_ascii_alphabetic() || ch == '\'' || ch == '-';
        if word_char {
            if !in_word {
                count += 1;
                in_word = true;
            }
        } else {
            in_word = false;
        }
    }
    count
}

fn sentence_mark_count(text: &str) -> usize {
    text.chars()
        .filter(|ch| matches!(ch, '.' | '!' | '?'))
        .count()
}

fn normalize_space(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn mean_usize(total: usize, count: usize) -> usize {
    if count == 0 { 0 } else { total / count }
}

fn mean_q8(total: u64, count: u64) -> u64 {
    if count == 0 {
        0
    } else {
        ((u128::from(total) * 256) / u128::from(count)) as u64
    }
}

fn per_mille(count: usize, total: usize) -> usize {
    if total == 0 {
        0
    } else {
        count.saturating_mul(1000) / total
    }
}

fn usize_to_u64(value: usize) -> Result<u64, Box<dyn std::error::Error>> {
    Ok(u64::try_from(value)?)
}

fn json_pair(out: &mut String, key: &str, value: &str, comma: bool) {
    out.push('"');
    out.push_str(key);
    out.push_str("\":\"");
    out.push_str(&json_escape(value));
    out.push('"');
    if comma {
        out.push(',');
    }
}

fn hash_pair(out: &mut String, key: &str, value: u64, comma: bool) {
    json_pair(out, key, &format!("0x{value:016x}"), comma);
}

fn number_pair(out: &mut String, key: &str, value: u64, comma: bool) {
    out.push('"');
    out.push_str(key);
    out.push_str("\":");
    out.push_str(&value.to_string());
    if comma {
        out.push(',');
    }
}

fn bool_pair(out: &mut String, key: &str, value: bool, comma: bool) {
    out.push('"');
    out.push_str(key);
    out.push_str("\":");
    out.push_str(if value { "true" } else { "false" });
    if comma {
        out.push(',');
    }
}
