#![deny(unsafe_code)]

use std::collections::HashMap;
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

const DEFAULT_MODEL: &str = "data/processed/crowley-bard-aphorism-v2/experiments/v4096.seq8-mean-reduce-base15-lr25-o98304.nsrllm";
const DEFAULT_VOCAB: &str = "data/processed/crowley-bard-aphorism-v2/v4096.vocab.tsv";
const DEFAULT_TOKENS: &str = "data/processed/crowley-bard-aphorism-v2/v4096.tokens.u16";
const DEFAULT_PROMPT: &str = "the first omen today is";
const DEFAULT_RANKER: &str = "crlplrimes-proxy-v1";
const TRACE_SCHEMA: &str = "nsrl.swarm_cycle.v1";
const MEDIA_SCHEMA: &str = "nsrl.solomon_media_request.v1";

const DECODE_BANNED_TOKENS: &[&str] = &[
    "http",
    "https",
    "www",
    "nbsp",
    "class",
    "align",
    "style",
    "href",
    "span",
    "div",
    "category",
    "references",
    "website",
    "websites",
    "assistant",
    "chatgpt",
];

const FUNCTION_WORDS: &[&str] = &[
    "a", "all", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has", "hath",
    "have", "he", "her", "him", "his", "i", "in", "is", "it", "me", "my", "not", "of", "on", "or",
    "our", "she", "so", "that", "the", "thee", "their", "them", "thou", "thy", "to", "we", "with",
    "ye", "you", "your",
];

const GLUE_WORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has", "hath", "have",
    "in", "is", "it", "of", "on", "or", "so", "that", "the", "to", "with",
];

const FLAVOR_WORDS: &[&str] = &[
    "beauty", "blood", "burning", "darkness", "death", "delight", "dream", "eyes", "face", "fire",
    "god", "heaven", "holy", "joy", "kiss", "life", "light", "love", "night", "secret", "shadow",
    "silence", "soul", "star", "will", "world",
];

const CRUFT_WORDS: &[&str] = &[
    "align",
    "assistant",
    "category",
    "chatgpt",
    "class",
    "div",
    "href",
    "html",
    "http",
    "https",
    "nbsp",
    "references",
    "style",
    "website",
    "websites",
    "www",
];

const DANGLING_END_WORDS: &[&str] = &[
    "a", "and", "are", "as", "at", "but", "by", "for", "from", "hath", "have", "in", "is", "of",
    "or", "that", "the", "to", "with",
];

#[derive(Clone)]
struct Config {
    prompt: String,
    models: Vec<PathBuf>,
    vocab: PathBuf,
    tokens: PathBuf,
    generator: PathBuf,
    out_dir: PathBuf,
    candidates: usize,
    parallel: usize,
    max_new_tokens: usize,
    top_k: usize,
    min_score: i64,
    seed: String,
    ranker_label: String,
    write_media_request: bool,
}

#[derive(Clone)]
struct CandidateTask {
    index: usize,
    candidate_id: String,
    sample_seed: u64,
    model_path: PathBuf,
    raw_path: PathBuf,
    trace_path: PathBuf,
}

struct CandidateResult {
    index: usize,
    candidate_id: String,
    model_path: PathBuf,
    sample_seed: u64,
    raw_path: PathBuf,
    trace_path: PathBuf,
    raw_text: String,
    text: String,
    score: CandidateScore,
    error: Option<String>,
}

#[derive(Clone)]
struct CandidateScore {
    score: i64,
    accepted: bool,
    reasons: Vec<String>,
    chars: usize,
    words: usize,
    distinct_words: usize,
    distinct_ratio_q1000: i64,
    function_word_ratio_q1000: i64,
    glue_word_ratio_q1000: i64,
    max_function_word_run: usize,
    max_glue_word_run: usize,
    max_word_count: usize,
    repeated_bigram_count: usize,
    repeated_trigram_count: usize,
    punctuation_run_count: usize,
    sentence_terminal_count: usize,
    expressive_terminal_count: usize,
    flavor_hits: Vec<String>,
    cruft_hits: Vec<String>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("nsrl-swarm-cycle: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args()?;
    validate_inputs(&config)?;
    let candidates_dir = config.out_dir.join("candidates");
    fs::create_dir_all(&candidates_dir)?;

    let tasks = build_tasks(&config, &candidates_dir);
    let started_at = unix_seconds();
    let results = run_generation_swarm(&config, tasks)?;
    let mut sorted = results;
    sorted.sort_by_key(|candidate| candidate.index);
    let selected = sorted
        .iter()
        .filter(|candidate| candidate.error.is_none())
        .max_by(|left, right| {
            left.score
                .score
                .cmp(&right.score.score)
                .then_with(|| right.index.cmp(&left.index))
        })
        .ok_or("no successful candidates")?;
    let accepted = selected.score.score >= config.min_score && selected.score.accepted;
    let solomon_seed = stable_hex(&[
        config.seed.as_str(),
        selected.candidate_id.as_str(),
        selected.text.as_str(),
    ]);

    write_trace(
        &config,
        started_at,
        &sorted,
        selected,
        accepted,
        &solomon_seed,
    )?;
    fs::write(
        config.out_dir.join("selected.txt"),
        selected.text.as_bytes(),
    )?;
    write_candidates_tsv(&config, &sorted)?;
    if config.write_media_request {
        write_media_request(&config, selected, accepted, &solomon_seed)?;
    }

    println!("run_dir={}", config.out_dir.display());
    println!("ranker={}", config.ranker_label);
    println!("selected_candidate={}", selected.candidate_id);
    println!("score={}", selected.score.score);
    println!("accepted={}", accepted);
    println!("solomon_seed={solomon_seed}");
    println!("text={}", selected.text);

    Ok(())
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let now = unix_seconds();
    let mut config = Config {
        prompt: DEFAULT_PROMPT.to_string(),
        models: Vec::new(),
        vocab: PathBuf::from(DEFAULT_VOCAB),
        tokens: PathBuf::from(DEFAULT_TOKENS),
        generator: default_generator_path(),
        out_dir: PathBuf::from(format!("data/processed/swarm-cycles/run-{now}")),
        candidates: 32,
        parallel: thread::available_parallelism()
            .map(|value| value.get())
            .unwrap_or(1)
            .min(8),
        max_new_tokens: 60,
        top_k: 12,
        min_score: 48,
        seed: format!("swarm-cycle-{now}"),
        ranker_label: DEFAULT_RANKER.to_string(),
        write_media_request: true,
    };

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            "--prompt" => config.prompt = require_value(&mut args, "--prompt")?,
            "--model" => config
                .models
                .push(PathBuf::from(require_value(&mut args, "--model")?)),
            "--vocab" => config.vocab = PathBuf::from(require_value(&mut args, "--vocab")?),
            "--tokens" => config.tokens = PathBuf::from(require_value(&mut args, "--tokens")?),
            "--generator" => {
                config.generator = PathBuf::from(require_value(&mut args, "--generator")?)
            }
            "--out-dir" => config.out_dir = PathBuf::from(require_value(&mut args, "--out-dir")?),
            "--candidates" => config.candidates = parse_usize(&mut args, "--candidates", 1)?,
            "--parallel" => config.parallel = parse_usize(&mut args, "--parallel", 1)?,
            "--max-new-tokens" => {
                config.max_new_tokens = parse_usize(&mut args, "--max-new-tokens", 1)?
            }
            "--top-k" => config.top_k = parse_usize(&mut args, "--top-k", 1)?,
            "--min-score" => {
                config.min_score = require_value(&mut args, "--min-score")?.parse()?;
            }
            "--seed" => config.seed = require_value(&mut args, "--seed")?,
            "--ranker-label" => config.ranker_label = require_value(&mut args, "--ranker-label")?,
            "--no-media-request" => config.write_media_request = false,
            _ => return Err(format!("unknown argument: {arg}").into()),
        }
    }
    if config.models.is_empty() {
        config.models.push(PathBuf::from(DEFAULT_MODEL));
    }
    config.parallel = config.parallel.max(1).min(config.candidates.max(1));
    Ok(config)
}

fn require_value(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value").into())
}

fn parse_usize(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
    min: usize,
) -> Result<usize, Box<dyn std::error::Error>> {
    let value: usize = require_value(args, flag)?.parse()?;
    if value < min {
        return Err(format!("{flag} must be at least {min}").into());
    }
    Ok(value)
}

fn print_usage() {
    println!(
        "Usage: nsrl-swarm-cycle [--prompt TEXT] [--model PATH ...] [--vocab PATH] [--tokens PATH] \\
         [--generator PATH] [--out-dir PATH] [--candidates N] [--parallel N] \\
         [--max-new-tokens N] [--top-k N] [--min-score N] [--seed TEXT] \\
         [--ranker-label NAME] [--no-media-request]"
    );
}

fn default_generator_path() -> PathBuf {
    if let Ok(path) = env::var("NSRL_SWARM_GENERATOR") {
        return PathBuf::from(path);
    }
    if let Ok(exe) = env::current_exe() {
        let sibling = exe.with_file_name("nsrl-train");
        if sibling.exists() {
            return sibling;
        }
    }
    PathBuf::from("target/release/nsrl-train")
}

fn validate_inputs(config: &Config) -> Result<(), Box<dyn std::error::Error>> {
    require_file(&config.generator, "generator")?;
    require_file(&config.vocab, "vocab")?;
    require_file(&config.tokens, "tokens")?;
    for model in &config.models {
        require_file(model, "model")?;
    }
    Ok(())
}

fn require_file(path: &Path, label: &str) -> Result<(), Box<dyn std::error::Error>> {
    if !path.exists() {
        return Err(format!("{label} does not exist: {}", path.display()).into());
    }
    Ok(())
}

fn build_tasks(config: &Config, candidates_dir: &Path) -> Vec<CandidateTask> {
    (0..config.candidates)
        .map(|index| {
            let model_path = config.models[index % config.models.len()].clone();
            let candidate_id = format!("candidate-{:04}", index + 1);
            let sample_seed = stable_u64(&[
                config.seed.as_str(),
                &candidate_id,
                &model_path.display().to_string(),
            ]);
            CandidateTask {
                index,
                candidate_id,
                sample_seed,
                model_path,
                raw_path: candidates_dir.join(format!("candidate-{:04}.txt", index + 1)),
                trace_path: candidates_dir.join(format!("candidate-{:04}.trace.jsonl", index + 1)),
            }
        })
        .collect()
}

fn run_generation_swarm(
    config: &Config,
    tasks: Vec<CandidateTask>,
) -> Result<Vec<CandidateResult>, Box<dyn std::error::Error>> {
    let task_queue = Arc::new(Mutex::new((0..tasks.len()).rev().collect::<Vec<_>>()));
    let tasks = Arc::new(tasks);
    let results = Arc::new(Mutex::new(
        (0..tasks.len())
            .map(|_| None)
            .collect::<Vec<Option<CandidateResult>>>(),
    ));
    let config = Arc::new(config.clone());
    let mut handles = Vec::new();
    for _ in 0..config.parallel {
        let task_queue = Arc::clone(&task_queue);
        let tasks = Arc::clone(&tasks);
        let results = Arc::clone(&results);
        let config = Arc::clone(&config);
        handles.push(thread::spawn(move || {
            loop {
                let Some(index) = task_queue.lock().expect("task queue poisoned").pop() else {
                    break;
                };
                let task = tasks[index].clone();
                let result = generate_candidate(&config, task);
                results.lock().expect("results poisoned")[index] = Some(result);
            }
        }));
    }
    for handle in handles {
        handle.join().map_err(|_| "swarm worker thread panicked")?;
    }
    let mut out = Vec::new();
    for value in results.lock().expect("results poisoned").iter_mut() {
        out.push(value.take().ok_or("missing swarm result")?);
    }
    Ok(out)
}

fn generate_candidate(config: &Config, task: CandidateTask) -> CandidateResult {
    let _ = fs::remove_file(&task.raw_path);
    let _ = fs::remove_file(&task.trace_path);
    let mut command = Command::new(&config.generator);
    command
        .arg("--mode")
        .arg("lexeme-generate")
        .arg("--model")
        .arg(&task.model_path)
        .arg("--vocab")
        .arg(&config.vocab)
        .arg("--tokens")
        .arg(&config.tokens)
        .arg("--prompt")
        .arg(&config.prompt)
        .arg("--max-new-tokens")
        .arg(config.max_new_tokens.to_string())
        .arg("--decode")
        .arg("sample")
        .arg("--sample-seed")
        .arg(task.sample_seed.to_string())
        .arg("--top-k")
        .arg(config.top_k.to_string())
        .arg("--decode-profile")
        .arg("coherent-prose")
        .arg("--decode-function-word-run-cap")
        .arg("4")
        .arg("--corpus-prior")
        .arg("--corpus-prior-order")
        .arg("3")
        .arg("--corpus-prior-logit-shift")
        .arg("9")
        .arg("--decode-frequency-cap")
        .arg("600")
        .arg("--decode-frequency-min-q15")
        .arg("6144")
        .arg("--decode-frequency-logit-shift")
        .arg("5")
        .arg("--decode-local-frequency-cap")
        .arg("2")
        .arg("--decode-local-frequency-min-q15")
        .arg("8192")
        .arg("--decode-local-frequency-logit-shift")
        .arg("4")
        .arg("--decode-local-frequency-hard-cap")
        .arg("2")
        .arg("--repeat-window")
        .arg("64")
        .arg("--repeat-penalty-shift")
        .arg("4")
        .arg("--max-repeat-run")
        .arg("2")
        .arg("--no-repeat-ngram")
        .arg("3")
        .arg("--strict-adjacency")
        .arg("--quality-weight-profile")
        .arg("prose-aware")
        .arg("--generated-only")
        .arg("--text-out")
        .arg(&task.raw_path)
        .arg("--trace")
        .arg(&task.trace_path);
    for token in DECODE_BANNED_TOKENS {
        command.arg("--decode-ban-token").arg(token);
    }

    let output = command.output();
    let mut error = None;
    if let Err(err) = &output {
        error = Some(err.to_string());
    } else if let Ok(output) = &output {
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            error = Some(if stderr.is_empty() {
                format!("generator exited with {}", output.status)
            } else {
                stderr
            });
        }
    }
    let raw_text = fs::read_to_string(&task.raw_path).unwrap_or_default();
    let text = clean_generated_text(&raw_text);
    let mut score = score_candidate(&text);
    if error.is_some() {
        score.score -= 10_000;
        score.accepted = false;
        score.reasons.push("generator_error".to_string());
    }

    CandidateResult {
        index: task.index,
        candidate_id: task.candidate_id,
        model_path: task.model_path,
        sample_seed: task.sample_seed,
        raw_path: task.raw_path,
        trace_path: task.trace_path,
        raw_text,
        text,
        score,
        error,
    }
}

fn clean_generated_text(raw: &str) -> String {
    let mut normalized = String::new();
    let mut previous_space = false;
    for ch in raw.chars() {
        let ch = match ch {
            '\n' | '\r' | '\t' => ' ',
            value if value.is_control() => continue,
            value => value,
        };
        if ch.is_whitespace() {
            if !previous_space {
                normalized.push(' ');
            }
            previous_space = true;
        } else {
            normalized.push(ch);
            previous_space = false;
        }
    }
    let trimmed = normalized
        .trim_matches(|ch: char| ch.is_whitespace() || matches!(ch, '"' | '\'' | ',' | ';' | ':'))
        .to_string();
    trim_sentence_span(&trimmed, 56, 230)
}

fn trim_sentence_span(text: &str, min_chars: usize, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return ensure_terminal(text.trim().to_string());
    }
    let mut best_terminal = None;
    for (byte_index, ch) in text.char_indices() {
        let chars = text[..byte_index].chars().count() + 1;
        if chars > max_chars {
            break;
        }
        if chars >= min_chars && matches!(ch, '.' | '!' | '?') {
            best_terminal = Some(byte_index + ch.len_utf8());
        }
    }
    if let Some(end) = best_terminal {
        return text[..end].trim().to_string();
    }

    let mut end = text.len();
    for (byte_index, _) in text.char_indices() {
        if text[..byte_index].chars().count() > max_chars {
            end = byte_index;
            break;
        }
    }
    let clipped = &text[..end];
    let word_end = clipped
        .char_indices()
        .rev()
        .find(|(_, ch)| ch.is_whitespace())
        .map(|(index, _)| index)
        .unwrap_or(end);
    ensure_terminal(clipped[..word_end].trim().to_string())
}

fn ensure_terminal(mut text: String) -> String {
    if !text.is_empty() && !text.ends_with('.') && !text.ends_with('!') && !text.ends_with('?') {
        text.push('.');
    }
    text
}

fn score_candidate(text: &str) -> CandidateScore {
    let words = extract_words(text);
    let mut counts = HashMap::<String, usize>::new();
    for word in &words {
        *counts.entry(word.clone()).or_insert(0) += 1;
    }
    let distinct_words = counts.len();
    let max_word_count = counts.values().copied().max().unwrap_or(0);
    let repeated_bigram_count = repeated_ngram_count(&words, 2);
    let repeated_trigram_count = repeated_ngram_count(&words, 3);
    let function_word_count = words
        .iter()
        .filter(|word| contains_word(FUNCTION_WORDS, word))
        .count();
    let glue_word_count = words
        .iter()
        .filter(|word| contains_word(GLUE_WORDS, word))
        .count();
    let max_function_word_run = max_word_run(&words, FUNCTION_WORDS);
    let max_glue_word_run = max_word_run(&words, GLUE_WORDS);
    let flavor_hits = unique_hits(&words, FLAVOR_WORDS);
    let cruft_hits = unique_hits(&words, CRUFT_WORDS);
    let chars = text.chars().count();
    let word_count = words.len();
    let distinct_ratio_q1000 = ratio_q1000(distinct_words, word_count);
    let function_word_ratio_q1000 = ratio_q1000(function_word_count, word_count);
    let glue_word_ratio_q1000 = ratio_q1000(glue_word_count, word_count);
    let punctuation_run_count = count_punctuation_runs(text);
    let sentence_terminal_count = text
        .chars()
        .filter(|ch| matches!(ch, '.' | '!' | '?'))
        .count();
    let expressive_terminal_count = text.chars().filter(|ch| matches!(ch, '!' | '?')).count();
    let last_word = words.last().map(String::as_str).unwrap_or("");

    let mut reasons = Vec::new();
    if chars < 56 {
        reasons.push("too_short".to_string());
    }
    if chars > 230 {
        reasons.push("too_long".to_string());
    }
    if word_count < 8 {
        reasons.push("too_few_words".to_string());
    }
    if max_word_count > 4 {
        reasons.push("word_repeat_gt4".to_string());
    }
    if repeated_trigram_count > 0 {
        reasons.push("repeated_trigram".to_string());
    }
    if distinct_ratio_q1000 < 550 {
        reasons.push("low_distinct_ratio".to_string());
    }
    if function_word_ratio_q1000 > 580 {
        reasons.push("function_word_heavy".to_string());
    }
    if max_function_word_run > 4 {
        reasons.push("function_word_run".to_string());
    }
    if glue_word_ratio_q1000 > 420 {
        reasons.push("glue_word_heavy".to_string());
    }
    if max_glue_word_run > 4 {
        reasons.push("glue_word_run".to_string());
    }
    if !cruft_hits.is_empty() {
        reasons.push("cruft".to_string());
    }
    if punctuation_run_count > 0 {
        reasons.push("punctuation_run".to_string());
    }
    if sentence_terminal_count > 3 {
        reasons.push("terminal_heavy".to_string());
    }
    if expressive_terminal_count > 2 {
        reasons.push("expressive_punctuation_heavy".to_string());
    }
    if contains_word(DANGLING_END_WORDS, last_word) {
        reasons.push("dangling_end_word".to_string());
    }
    if word_count >= 12 && !text.ends_with('.') && !text.ends_with('!') && !text.ends_with('?') {
        reasons.push("no_sentence_terminal".to_string());
    }

    let mut score = 100_i64;
    if chars < 90 {
        score -= i64::try_from(90 - chars).unwrap_or(0);
    }
    if chars > 180 {
        score -= i64::try_from(chars - 180).unwrap_or(0) * 3 / 2;
    }
    if (90..=180).contains(&chars) {
        score += 30;
    }
    if (10..=32).contains(&word_count) {
        score += 18;
    }
    score += i64::try_from(flavor_hits.len().min(5) * 8).unwrap_or(0);
    score -= i64::try_from(max_word_count.saturating_sub(2) * 8).unwrap_or(0);
    score -= i64::try_from(repeated_bigram_count * 8).unwrap_or(0);
    score -= i64::try_from(repeated_trigram_count * 40).unwrap_or(0);
    if function_word_ratio_q1000 > 580 {
        score -= (function_word_ratio_q1000 - 580) / 5;
    }
    score -= i64::try_from(max_function_word_run.saturating_sub(3) * 16).unwrap_or(0);
    if glue_word_ratio_q1000 > 420 {
        score -= (glue_word_ratio_q1000 - 420) / 4;
    }
    score -= i64::try_from(max_glue_word_run.saturating_sub(4) * 10).unwrap_or(0);
    score -= i64::try_from(punctuation_run_count * 10).unwrap_or(0);
    score -= i64::try_from(sentence_terminal_count.saturating_sub(2) * 12).unwrap_or(0);
    score -= i64::try_from(expressive_terminal_count.saturating_sub(1) * 8).unwrap_or(0);
    score -= i64::try_from(cruft_hits.len() * 100).unwrap_or(0);
    if contains_word(DANGLING_END_WORDS, last_word) {
        score -= 40;
    }
    if word_count >= 12 && !text.ends_with('.') && !text.ends_with('!') && !text.ends_with('?') {
        score -= 30;
    }

    CandidateScore {
        score,
        accepted: reasons.is_empty(),
        reasons,
        chars,
        words: word_count,
        distinct_words,
        distinct_ratio_q1000,
        function_word_ratio_q1000,
        glue_word_ratio_q1000,
        max_function_word_run,
        max_glue_word_run,
        max_word_count,
        repeated_bigram_count,
        repeated_trigram_count,
        punctuation_run_count,
        sentence_terminal_count,
        expressive_terminal_count,
        flavor_hits,
        cruft_hits,
    }
}

fn extract_words(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_ascii_alphabetic() || ch == '\'' {
            current.push(ch.to_ascii_lowercase());
        } else if !current.is_empty() {
            words.push(trim_apostrophes(&current));
            current.clear();
        }
    }
    if !current.is_empty() {
        words.push(trim_apostrophes(&current));
    }
    words.retain(|word| !word.is_empty());
    words
}

fn trim_apostrophes(word: &str) -> String {
    word.trim_matches('\'').to_string()
}

fn repeated_ngram_count(words: &[String], order: usize) -> usize {
    if words.len() < order || order == 0 {
        return 0;
    }
    let mut counts = HashMap::<String, usize>::new();
    for window in words.windows(order) {
        let key = window.join(" ");
        *counts.entry(key).or_insert(0) += 1;
    }
    counts.values().filter(|&&count| count > 1).count()
}

fn max_word_run(words: &[String], set: &[&str]) -> usize {
    let mut best = 0;
    let mut current = 0;
    for word in words {
        if contains_word(set, word) {
            current += 1;
            best = best.max(current);
        } else {
            current = 0;
        }
    }
    best
}

fn contains_word(set: &[&str], word: &str) -> bool {
    set.iter().any(|value| *value == word)
}

fn unique_hits(words: &[String], set: &[&str]) -> Vec<String> {
    let mut hits = Vec::new();
    for word in words {
        if contains_word(set, word) && !hits.iter().any(|hit| hit == word) {
            hits.push(word.clone());
        }
    }
    hits
}

fn ratio_q1000(numerator: usize, denominator: usize) -> i64 {
    if denominator == 0 {
        0
    } else {
        i64::try_from((numerator * 1000 + denominator / 2) / denominator).unwrap_or(0)
    }
}

fn count_punctuation_runs(text: &str) -> usize {
    let mut runs = 0;
    let mut previous = false;
    for ch in text.chars() {
        let punctuation = matches!(ch, '!' | '?' | '.' | ',' | ';' | ':');
        if punctuation && previous {
            runs += 1;
        }
        previous = punctuation;
        if !punctuation {
            previous = false;
        }
    }
    runs
}

fn write_trace(
    config: &Config,
    started_at: u64,
    candidates: &[CandidateResult],
    selected: &CandidateResult,
    accepted: bool,
    solomon_seed: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut writer = BufWriter::new(File::create(config.out_dir.join("cycle.jsonl"))?);
    writeln!(
        writer,
        "{{\"schema\":{},\"record\":\"run\",\"run_id\":{},\"created_unix_seconds\":{},\"prompt\":{},\"ranker\":{},\"candidate_count\":{},\"parallel\":{},\"max_new_tokens\":{},\"top_k\":{},\"min_score\":{},\"models\":{},\"vocab\":{},\"tokens\":{}}}",
        json_string(TRACE_SCHEMA),
        json_string(&config.seed),
        started_at,
        json_string(&config.prompt),
        json_string(&config.ranker_label),
        config.candidates,
        config.parallel,
        config.max_new_tokens,
        config.top_k,
        config.min_score,
        json_string_array(&path_strings(&config.models)),
        json_string(&config.vocab.display().to_string()),
        json_string(&config.tokens.display().to_string()),
    )?;
    for candidate in candidates {
        writeln!(writer, "{}", candidate_json(config, candidate))?;
    }
    writeln!(
        writer,
        "{{\"schema\":{},\"record\":\"selection\",\"run_id\":{},\"ranker\":{},\"selected_candidate_id\":{},\"score\":{},\"accepted\":{},\"min_score\":{},\"text\":{},\"solomon_seed\":{},\"selected_trace\":{}}}",
        json_string(TRACE_SCHEMA),
        json_string(&config.seed),
        json_string(&config.ranker_label),
        json_string(&selected.candidate_id),
        selected.score.score,
        accepted,
        config.min_score,
        json_string(&selected.text),
        json_string(solomon_seed),
        json_string(&selected.trace_path.display().to_string()),
    )?;
    Ok(())
}

fn candidate_json(config: &Config, candidate: &CandidateResult) -> String {
    format!(
        "{{\"schema\":{},\"record\":\"candidate\",\"run_id\":{},\"candidate_id\":{},\"index\":{},\"model\":{},\"sample_seed\":{},\"status\":{},\"error\":{},\"score\":{},\"accepted\":{},\"reasons\":{},\"text\":{},\"raw_text\":{},\"paths\":{{\"text\":{},\"trace\":{}}},\"metrics\":{{\"chars\":{},\"words\":{},\"distinct_words\":{},\"distinct_ratio_q1000\":{},\"function_word_ratio_q1000\":{},\"glue_word_ratio_q1000\":{},\"max_function_word_run\":{},\"max_glue_word_run\":{},\"max_word_count\":{},\"repeated_bigram_count\":{},\"repeated_trigram_count\":{},\"punctuation_run_count\":{},\"sentence_terminal_count\":{},\"expressive_terminal_count\":{},\"flavor_hits\":{},\"cruft_hits\":{}}}}}",
        json_string(TRACE_SCHEMA),
        json_string(&config.seed),
        json_string(&candidate.candidate_id),
        candidate.index + 1,
        json_string(&candidate.model_path.display().to_string()),
        candidate.sample_seed,
        json_string(if candidate.error.is_some() {
            "error"
        } else {
            "ok"
        }),
        json_optional_string(candidate.error.as_deref()),
        candidate.score.score,
        candidate.score.accepted,
        json_string_array(&candidate.score.reasons),
        json_string(&candidate.text),
        json_string(&candidate.raw_text),
        json_string(&candidate.raw_path.display().to_string()),
        json_string(&candidate.trace_path.display().to_string()),
        candidate.score.chars,
        candidate.score.words,
        candidate.score.distinct_words,
        candidate.score.distinct_ratio_q1000,
        candidate.score.function_word_ratio_q1000,
        candidate.score.glue_word_ratio_q1000,
        candidate.score.max_function_word_run,
        candidate.score.max_glue_word_run,
        candidate.score.max_word_count,
        candidate.score.repeated_bigram_count,
        candidate.score.repeated_trigram_count,
        candidate.score.punctuation_run_count,
        candidate.score.sentence_terminal_count,
        candidate.score.expressive_terminal_count,
        json_string_array(&candidate.score.flavor_hits),
        json_string_array(&candidate.score.cruft_hits),
    )
}

fn write_candidates_tsv(
    config: &Config,
    candidates: &[CandidateResult],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut writer = BufWriter::new(File::create(config.out_dir.join("candidates.tsv"))?);
    writeln!(
        writer,
        "candidate_id\tindex\tscore\taccepted\tchars\twords\tmodel\tsample_seed\treasons\ttext"
    )?;
    for candidate in candidates {
        writeln!(
            writer,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            candidate.candidate_id,
            candidate.index + 1,
            candidate.score.score,
            candidate.score.accepted,
            candidate.score.chars,
            candidate.score.words,
            candidate.model_path.display(),
            candidate.sample_seed,
            candidate.score.reasons.join("|"),
            candidate.text.replace('\t', " "),
        )?;
    }
    Ok(())
}

fn write_media_request(
    config: &Config,
    selected: &CandidateResult,
    accepted: bool,
    solomon_seed: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut writer = BufWriter::new(File::create(
        config.out_dir.join("solomon-media-request.jsonl"),
    )?);
    writeln!(
        writer,
        "{{\"schema\":{},\"record\":\"media_request\",\"source_schema\":{},\"run_id\":{},\"candidate_id\":{},\"accepted\":{},\"seed\":{},\"prompt_text\":{},\"suggested_renderer\":\"solomon-illustrator\",\"source_trace\":{}}}",
        json_string(MEDIA_SCHEMA),
        json_string(TRACE_SCHEMA),
        json_string(&config.seed),
        json_string(&selected.candidate_id),
        accepted,
        json_string(solomon_seed),
        json_string(&selected.text),
        json_string(&config.out_dir.join("cycle.jsonl").display().to_string()),
    )?;
    Ok(())
}

fn path_strings(paths: &[PathBuf]) -> Vec<String> {
    paths
        .iter()
        .map(|path| path.display().to_string())
        .collect()
}

fn json_string_array(values: &[String]) -> String {
    let mut out = String::from("[");
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str(&json_string(value));
    }
    out.push(']');
    out
}

fn json_optional_string(value: Option<&str>) -> String {
    value.map(json_string).unwrap_or_else(|| "null".to_string())
}

fn json_string(value: &str) -> String {
    let mut out = String::from("\"");
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            value if value.is_control() => {
                out.push_str(&format!("\\u{:04x}", u32::from(value)));
            }
            value => out.push(value),
        }
    }
    out.push('"');
    out
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn stable_u64(parts: &[&str]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for part in parts {
        for byte in part.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash ^= 0xff;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn stable_hex(parts: &[&str]) -> String {
    format!("{:016x}", stable_u64(parts))
}
