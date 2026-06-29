#![deny(unsafe_code)]

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::PathBuf;

const TRAIN_SCHEMA: &str = "nsrl.solomon_multimodal_train_trace.v1";
const SAMPLE_SCHEMA: &str = "nsrl.solomon_multimodal_sample_trace.v1";
const MODEL_MAGIC: &[u8; 8] = b"NSRLMOD1";
const MODEL_VERSION: u32 = 1;

const BOS: u16 = 1;
const PROMPT: u16 = 2;
const TEXT: u16 = 3;
const IMAGE: u16 = 4;
const EOS: u16 = 5;
const TEXT_BASE: u16 = 16;
const TEXT_COUNT: u16 = 128;
const IMAGE_BASE: u16 = TEXT_BASE + TEXT_COUNT;
const IMAGE_BINS: u16 = 16;
const SIGNATURE_GRID: usize = 16;
const SIGNATURE_BINS: usize = SIGNATURE_GRID * SIGNATURE_GRID;
const VOCAB_SIZE: u16 = IMAGE_BASE + IMAGE_BINS;
const MAX_CONTEXT_TOKENS: usize = 64;
const CONTEXT_LENGTHS: &[usize] = &[1, 2, 4, 8, 16, 32, 64];

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Command {
    Train,
    Sample,
}

#[derive(Debug, Clone)]
struct Config {
    command: Command,
    tokens_path: PathBuf,
    model_path: PathBuf,
    model_out: PathBuf,
    out_dir: PathBuf,
    prompt: String,
    max_text_tokens: usize,
    top_k: usize,
    sample_seed: u64,
}

impl Default for Config {
    fn default() -> Self {
        let root = PathBuf::from("data/processed/key-solomon-goetia-multimodal-v1");
        Self {
            command: Command::Train,
            tokens_path: root.join("corpus.tokens.u16"),
            model_path: root.join("model.nsrlmod"),
            model_out: root.join("model.nsrlmod"),
            out_dir: root.join("sample"),
            prompt: String::from("king solomon seal"),
            max_text_tokens: 320,
            top_k: 1,
            sample_seed: 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct NextCount {
    token: u16,
    count: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct NgramRow {
    context: Vec<u16>,
    total: u32,
    next: Vec<NextCount>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SolomonMultimodalModel {
    token_count: u64,
    token_hash: u64,
    unigram_total: u32,
    unigram: Vec<NextCount>,
    contexts: Vec<NgramRow>,
}

#[derive(Debug, Clone)]
struct Sample {
    prompt: String,
    text: String,
    image_bins: [u8; SIGNATURE_BINS],
    generated_tokens: Vec<u16>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("nsrl-solomon-multimodal: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args(env::args().skip(1))?;
    match config.command {
        Command::Train => train_command(config),
        Command::Sample => sample_command(config),
    }
}

fn usage() {
    println!(
        "Usage: nsrl-solomon-multimodal train|sample [--tokens PATH] [--model PATH] [--model-out PATH] [--out-dir PATH] [--prompt TEXT] [--max-text-tokens N] [--top-k N] [--sample-seed N]"
    );
}

fn parse_args<I>(args: I) -> Result<Config, Box<dyn std::error::Error>>
where
    I: Iterator<Item = String>,
{
    let mut config = Config::default();
    let mut args = args.peekable();
    if let Some(arg) = args.peek()
        && !arg.starts_with("--")
    {
        config.command = parse_command(&args.next().unwrap())?;
    }
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                usage();
                std::process::exit(0);
            }
            "--tokens" => {
                config.tokens_path = PathBuf::from(args.next().ok_or("--tokens requires PATH")?);
            }
            "--model" => {
                config.model_path = PathBuf::from(args.next().ok_or("--model requires PATH")?);
            }
            "--model-out" => {
                config.model_out = PathBuf::from(args.next().ok_or("--model-out requires PATH")?);
            }
            "--out-dir" => {
                config.out_dir = PathBuf::from(args.next().ok_or("--out-dir requires PATH")?);
            }
            "--prompt" => {
                config.prompt = args.next().ok_or("--prompt requires TEXT")?;
            }
            "--max-text-tokens" => {
                config.max_text_tokens =
                    args.next().ok_or("--max-text-tokens requires N")?.parse()?;
            }
            "--top-k" => {
                config.top_k = args.next().ok_or("--top-k requires N")?.parse()?;
            }
            "--sample-seed" => {
                config.sample_seed = args.next().ok_or("--sample-seed requires N")?.parse()?;
            }
            value => return Err(format!("unknown option: {value}").into()),
        }
    }
    if config.max_text_tokens == 0 {
        return Err("--max-text-tokens must be positive".into());
    }
    Ok(config)
}

fn parse_command(value: &str) -> Result<Command, Box<dyn std::error::Error>> {
    match value {
        "train" => Ok(Command::Train),
        "sample" => Ok(Command::Sample),
        _ => Err(format!("unknown command: {value}; expected train or sample").into()),
    }
}

fn train_command(config: Config) -> Result<(), Box<dyn std::error::Error>> {
    let token_bytes = fs::read(&config.tokens_path)?;
    let tokens = read_u16_tokens(&token_bytes)?;
    let model = train_model(&tokens)?;
    if let Some(parent) = config.model_out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&config.model_out, model.try_to_bytes()?)?;
    println!(
        "{{\"schema\":\"{}\",\"model\":\"{}\",\"token_count\":{},\"token_hash\":\"0x{:016x}\",\"model_hash\":\"0x{:016x}\",\"unigram_tokens\":{},\"max_context_tokens\":{},\"context_rows\":{}}}",
        TRAIN_SCHEMA,
        json_escape(&config.model_out.display().to_string()),
        model.token_count,
        model.token_hash,
        model.model_hash()?,
        model.unigram.len(),
        MAX_CONTEXT_TOKENS,
        model.contexts.len(),
    );
    Ok(())
}

fn sample_command(config: Config) -> Result<(), Box<dyn std::error::Error>> {
    let model = SolomonMultimodalModel::from_bytes(&fs::read(&config.model_path)?)?;
    let sample = sample_model(
        &model,
        &config.prompt,
        config.max_text_tokens,
        config.top_k,
        config.sample_seed,
    );
    fs::create_dir_all(&config.out_dir)?;
    let text_path = config.out_dir.join("text.txt");
    let ink_path = config.out_dir.join("image.ink16.u8");
    let pgm_path = config.out_dir.join("image.pgm");
    let token_path = config.out_dir.join("image.tokens.tsv");
    let trace_path = config.out_dir.join("sample.json");
    fs::write(&text_path, format!("{}\n", sample.text))?;
    fs::write(&ink_path, image_ink_bytes(&sample.image_bins))?;
    fs::write(&pgm_path, image_pgm_bytes(&sample.image_bins))?;
    fs::write(&token_path, image_token_tsv(&sample.image_bins))?;
    fs::write(
        &trace_path,
        format!(
            "{{\n  \"schema\":\"{}\",\n  \"model\":\"{}\",\n  \"model_hash\":\"0x{:016x}\",\n  \"prompt\":\"{}\",\n  \"generated_text\":\"{}\",\n  \"generated_token_count\":{},\n  \"image_grid\":{},\n  \"image_bins\":{},\n  \"text_out\":\"{}\",\n  \"image_ink16_u8\":\"{}\",\n  \"image_pgm\":\"{}\"\n}}\n",
            SAMPLE_SCHEMA,
            json_escape(&config.model_path.display().to_string()),
            model.model_hash()?,
            json_escape(&sample.prompt),
            json_escape(&sample.text),
            sample.generated_tokens.len(),
            SIGNATURE_GRID,
            IMAGE_BINS,
            json_escape(&text_path.display().to_string()),
            json_escape(&ink_path.display().to_string()),
            json_escape(&pgm_path.display().to_string()),
        ),
    )?;
    println!(
        "{{\"schema\":\"{}\",\"out_dir\":\"{}\",\"model_hash\":\"0x{:016x}\",\"prompt\":\"{}\",\"generated_text\":\"{}\"}}",
        SAMPLE_SCHEMA,
        json_escape(&config.out_dir.display().to_string()),
        model.model_hash()?,
        json_escape(&sample.prompt),
        json_escape(&sample.text),
    );
    Ok(())
}

fn read_u16_tokens(bytes: &[u8]) -> Result<Vec<u16>, Box<dyn std::error::Error>> {
    if !bytes.len().is_multiple_of(2) {
        return Err("token file must contain little-endian u16 values".into());
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect())
}

fn train_model(tokens: &[u16]) -> Result<SolomonMultimodalModel, Box<dyn std::error::Error>> {
    if tokens.is_empty() {
        return Err("cannot train multimodal model from an empty corpus".into());
    }
    let mut unigram_counts = vec![0_u32; usize::from(VOCAB_SIZE)];
    let mut context_counts: BTreeMap<Vec<u16>, BTreeMap<u16, u32>> = BTreeMap::new();

    for &token in tokens {
        validate_token(token)?;
        let index = usize::from(token);
        unigram_counts[index] = unigram_counts[index].saturating_add(1);
    }
    let mut example_start = 0_usize;
    for index in 0..tokens.len() {
        if tokens[index] == BOS {
            example_start = index;
        }
        let example_position = index.saturating_sub(example_start);
        for context_len in context_lengths_for_position(example_position) {
            if context_len > example_position {
                continue;
            }
            let context = tokens[index - context_len..index].to_vec();
            increment_context(&mut context_counts, context, tokens[index]);
        }
    }

    let unigram_total = unigram_counts
        .iter()
        .fold(0_u32, |acc, &count| acc.saturating_add(count));
    Ok(SolomonMultimodalModel {
        token_count: u64::try_from(tokens.len())?,
        token_hash: hash_tokens(tokens),
        unigram_total,
        unigram: unigram_counts
            .iter()
            .enumerate()
            .filter_map(|(token, &count)| {
                if count == 0 {
                    None
                } else {
                    Some(NextCount {
                        token: u16::try_from(token).ok()?,
                        count,
                    })
                }
            })
            .collect(),
        contexts: context_counts
            .into_iter()
            .map(|(context, next)| NgramRow {
                context,
                total: sum_counts(&next),
                next: next_counts(next),
            })
            .collect(),
    })
}

fn validate_token(token: u16) -> Result<(), Box<dyn std::error::Error>> {
    if token >= VOCAB_SIZE {
        return Err(format!("token {token} is outside NSRLMOD1 vocab size {VOCAB_SIZE}").into());
    }
    Ok(())
}

fn increment_context(
    map: &mut BTreeMap<Vec<u16>, BTreeMap<u16, u32>>,
    context: Vec<u16>,
    next: u16,
) {
    let row = map.entry(context).or_default();
    let count = row.entry(next).or_default();
    *count = count.saturating_add(1);
}

fn sum_counts(map: &BTreeMap<u16, u32>) -> u32 {
    map.values()
        .fold(0_u32, |acc, &count| acc.saturating_add(count))
}

fn next_counts(map: BTreeMap<u16, u32>) -> Vec<NextCount> {
    map.into_iter()
        .map(|(token, count)| NextCount { token, count })
        .collect()
}

fn sample_model(
    model: &SolomonMultimodalModel,
    prompt: &str,
    max_text_tokens: usize,
    top_k: usize,
    sample_seed: u64,
) -> Sample {
    let normalized_prompt = normalize_text(prompt);
    let mut generated_tokens = Vec::new();
    generated_tokens.push(BOS);
    generated_tokens.push(PROMPT);
    generated_tokens.extend(encode_text_tokens(&normalized_prompt));
    generated_tokens.push(TEXT);

    let mut text_tokens = Vec::new();
    let seed = sample_seed ^ hash_text(&normalized_prompt);
    for step in 0..max_text_tokens {
        let token = model.next_token(&generated_tokens, Phase::Text, seed, step, top_k);
        if token == IMAGE || token == EOS {
            break;
        }
        if is_text_token(token) {
            generated_tokens.push(token);
            text_tokens.push(token);
        }
    }
    generated_tokens.push(IMAGE);

    let mut image_bins = [0_u8; SIGNATURE_BINS];
    for (index, bin) in image_bins.iter_mut().enumerate() {
        let token = model.next_token(
            &generated_tokens,
            Phase::Image,
            seed,
            max_text_tokens.saturating_add(index),
            top_k,
        );
        let image_token = if is_image_token(token) {
            token
        } else {
            model.default_image_token()
        };
        *bin = u8::try_from(image_token.saturating_sub(IMAGE_BASE)).unwrap_or(0);
        generated_tokens.push(image_token);
    }
    generated_tokens.push(EOS);

    Sample {
        prompt: normalized_prompt,
        text: decode_text_tokens(&text_tokens),
        image_bins,
        generated_tokens,
    }
}

#[derive(Debug, Clone, Copy)]
enum Phase {
    Text,
    Image,
}

impl SolomonMultimodalModel {
    fn next_token(
        &self,
        history: &[u16],
        phase: Phase,
        seed: u64,
        step: usize,
        top_k: usize,
    ) -> u16 {
        for context_len in context_lengths_for_position(history.len())
            .into_iter()
            .rev()
        {
            let start = history.len() - context_len;
            if let Some(row) = self.context_row(&history[start..])
                && let Some(token) = choose_from_counts(&row.next, phase, seed, step, top_k)
            {
                return token;
            }
        }
        choose_from_counts(&self.unigram, phase, seed, step, top_k).unwrap_or(match phase {
            Phase::Text => IMAGE,
            Phase::Image => IMAGE_BASE,
        })
    }

    fn context_row(&self, context: &[u16]) -> Option<&NgramRow> {
        self.contexts
            .binary_search_by(|row| row.context.as_slice().cmp(context))
            .ok()
            .and_then(|index| self.contexts.get(index))
    }

    fn default_image_token(&self) -> u16 {
        choose_from_counts(&self.unigram, Phase::Image, 0, 0, 1).unwrap_or(IMAGE_BASE)
    }

    fn try_to_bytes(&self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let mut out = self.bytes_without_hash()?;
        out.extend_from_slice(&self.model_hash()?.to_le_bytes());
        Ok(out)
    }

    fn bytes_without_hash(&self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let mut out = Vec::new();
        out.extend_from_slice(MODEL_MAGIC);
        push_u32(&mut out, MODEL_VERSION);
        push_u32(&mut out, u32::from(VOCAB_SIZE));
        push_u32(&mut out, u32::from(TEXT_BASE));
        push_u32(&mut out, u32::from(TEXT_COUNT));
        push_u32(&mut out, u32::from(IMAGE_BASE));
        push_u32(&mut out, u32::from(IMAGE_BINS));
        push_u32(&mut out, u32::try_from(SIGNATURE_GRID)?);
        push_u64(&mut out, self.token_count);
        push_u64(&mut out, self.token_hash);
        push_u32(&mut out, self.unigram_total);
        push_count_list(&mut out, &self.unigram)?;
        push_u32(&mut out, u32::try_from(MAX_CONTEXT_TOKENS)?);
        push_u32(&mut out, u32::try_from(self.contexts.len())?);
        for row in &self.contexts {
            push_u32(&mut out, u32::try_from(row.context.len())?);
            for &token in &row.context {
                push_u32(&mut out, u32::from(token));
            }
            push_u32(&mut out, row.total);
            push_count_list(&mut out, &row.next)?;
        }
        Ok(out)
    }

    fn model_hash(&self) -> Result<u64, Box<dyn std::error::Error>> {
        Ok(hash_bytes(&self.bytes_without_hash()?))
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        if bytes.len() < MODEL_MAGIC.len() + 8 || &bytes[..MODEL_MAGIC.len()] != MODEL_MAGIC {
            return Err("not an NSRLMOD1 model".into());
        }
        let expected_hash = read_u64_at_end(bytes)?;
        let actual_hash = hash_bytes(&bytes[..bytes.len() - 8]);
        if expected_hash != actual_hash {
            return Err(format!(
                "NSRLMOD1 hash mismatch: expected 0x{expected_hash:016x}, got 0x{actual_hash:016x}"
            )
            .into());
        }
        let mut cursor = Cursor::new(&bytes[MODEL_MAGIC.len()..bytes.len() - 8]);
        let version = cursor.read_u32()?;
        if version != MODEL_VERSION {
            return Err(format!("unsupported NSRLMOD1 version {version}").into());
        }
        expect_u32(&mut cursor, u32::from(VOCAB_SIZE), "vocab size")?;
        expect_u32(&mut cursor, u32::from(TEXT_BASE), "text base")?;
        expect_u32(&mut cursor, u32::from(TEXT_COUNT), "text count")?;
        expect_u32(&mut cursor, u32::from(IMAGE_BASE), "image base")?;
        expect_u32(&mut cursor, u32::from(IMAGE_BINS), "image bins")?;
        expect_u32(
            &mut cursor,
            u32::try_from(SIGNATURE_GRID)?,
            "signature grid",
        )?;
        let token_count = cursor.read_u64()?;
        let token_hash = cursor.read_u64()?;
        let unigram_total = cursor.read_u32()?;
        let unigram = cursor.read_count_list()?;
        let max_context = usize::try_from(cursor.read_u32()?)?;
        if max_context != MAX_CONTEXT_TOKENS {
            return Err(format!(
                "NSRLMOD1 max context mismatch: expected {MAX_CONTEXT_TOKENS}, got {max_context}"
            )
            .into());
        }
        let context_len = usize::try_from(cursor.read_u32()?)?;
        let mut contexts = Vec::with_capacity(context_len);
        for _ in 0..context_len {
            let token_count = usize::try_from(cursor.read_u32()?)?;
            if token_count == 0 || token_count > MAX_CONTEXT_TOKENS {
                return Err("NSRLMOD1 context row has invalid length".into());
            }
            let mut context = Vec::with_capacity(token_count);
            for _ in 0..token_count {
                context.push(read_token_u32(&mut cursor)?);
            }
            let total = cursor.read_u32()?;
            let next = cursor.read_count_list()?;
            contexts.push(NgramRow {
                context,
                total,
                next,
            });
        }
        if !cursor.is_empty() {
            return Err("trailing bytes in NSRLMOD1 body".into());
        }
        Ok(Self {
            token_count,
            token_hash,
            unigram_total,
            unigram,
            contexts,
        })
    }
}

fn context_lengths_for_position(position: usize) -> Vec<usize> {
    let mut lengths = Vec::new();
    if position > 0 && position <= MAX_CONTEXT_TOKENS {
        lengths.push(position);
    }
    for &length in CONTEXT_LENGTHS {
        if length <= position && !lengths.contains(&length) {
            lengths.push(length);
        }
    }
    lengths.sort_unstable();
    lengths
}

fn choose_from_counts(
    counts: &[NextCount],
    phase: Phase,
    seed: u64,
    step: usize,
    top_k: usize,
) -> Option<u16> {
    let mut candidates: Vec<NextCount> = counts
        .iter()
        .filter(|entry| allowed_token(entry.token, phase))
        .cloned()
        .collect();
    if candidates.is_empty() {
        return None;
    }
    candidates.sort_by(|left, right| {
        right
            .count
            .cmp(&left.count)
            .then_with(|| left.token.cmp(&right.token))
    });
    let limit = if top_k == 0 {
        1
    } else {
        top_k.min(candidates.len())
    };
    if limit == 1 {
        return candidates.first().map(|entry| entry.token);
    }
    let total = candidates.iter().take(limit).fold(0_u64, |acc, entry| {
        acc.saturating_add(u64::from(entry.count))
    });
    if total == 0 {
        return candidates.first().map(|entry| entry.token);
    }
    let mut draw = mix64(seed ^ u64::try_from(step).unwrap_or(0)) % total;
    for entry in candidates.iter().take(limit) {
        let count = u64::from(entry.count);
        if draw < count {
            return Some(entry.token);
        }
        draw = draw.saturating_sub(count);
    }
    candidates.first().map(|entry| entry.token)
}

fn allowed_token(token: u16, phase: Phase) -> bool {
    match phase {
        Phase::Text => is_text_token(token) || token == IMAGE || token == EOS,
        Phase::Image => is_image_token(token),
    }
}

fn is_text_token(token: u16) -> bool {
    (TEXT_BASE..TEXT_BASE + TEXT_COUNT).contains(&token)
}

fn is_image_token(token: u16) -> bool {
    (IMAGE_BASE..IMAGE_BASE + IMAGE_BINS).contains(&token)
}

fn encode_text_tokens(text: &str) -> Vec<u16> {
    normalize_text(text)
        .bytes()
        .map(|byte| TEXT_BASE + u16::from(byte.min(127)))
        .collect()
}

fn decode_text_tokens(tokens: &[u16]) -> String {
    let mut out = String::new();
    for &token in tokens {
        if is_text_token(token) {
            let byte = u8::try_from(token - TEXT_BASE).unwrap_or(b'?');
            if (32..=126).contains(&byte) {
                out.push(char::from(byte));
            }
        }
    }
    compact_spaces(&out)
}

fn normalize_text(text: &str) -> String {
    compact_spaces(
        &text
            .chars()
            .map(|ch| {
                if ch.is_ascii_graphic() || ch == ' ' {
                    ch
                } else {
                    ' '
                }
            })
            .collect::<String>(),
    )
}

fn compact_spaces(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn image_ink_bytes(image_bins: &[u8; SIGNATURE_BINS]) -> Vec<u8> {
    image_bins
        .iter()
        .map(|&bin| bin.saturating_mul(17))
        .collect()
}

fn image_pgm_bytes(image_bins: &[u8; SIGNATURE_BINS]) -> Vec<u8> {
    let image_size = 128_usize;
    let scale = image_size / SIGNATURE_GRID;
    let mut out = format!("P5\n{} {}\n255\n", image_size, image_size).into_bytes();
    for y in 0..image_size {
        let gy = y / scale;
        for x in 0..image_size {
            let gx = x / scale;
            let ink = image_bins[gy * SIGNATURE_GRID + gx].saturating_mul(17);
            out.push(u8::MAX.saturating_sub(ink));
        }
    }
    out
}

fn image_token_tsv(image_bins: &[u8; SIGNATURE_BINS]) -> String {
    let mut out = String::new();
    for y in 0..SIGNATURE_GRID {
        for x in 0..SIGNATURE_GRID {
            if x > 0 {
                out.push('\t');
            }
            out.push_str(&image_bins[y * SIGNATURE_GRID + x].to_string());
        }
        out.push('\n');
    }
    out
}

fn push_count_list(
    out: &mut Vec<u8>,
    counts: &[NextCount],
) -> Result<(), Box<dyn std::error::Error>> {
    push_u32(out, u32::try_from(counts.len())?);
    for entry in counts {
        push_u32(out, u32::from(entry.token));
        push_u32(out, entry.count);
    }
    Ok(())
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn expect_u32(
    cursor: &mut Cursor<'_>,
    expected: u32,
    label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let actual = cursor.read_u32()?;
    if actual != expected {
        return Err(format!("NSRLMOD1 {label} mismatch: expected {expected}, got {actual}").into());
    }
    Ok(())
}

fn read_token_u32(cursor: &mut Cursor<'_>) -> Result<u16, Box<dyn std::error::Error>> {
    let value = cursor.read_u32()?;
    let token = u16::try_from(value)?;
    validate_token(token)?;
    Ok(token)
}

fn read_u64_at_end(bytes: &[u8]) -> Result<u64, Box<dyn std::error::Error>> {
    if bytes.len() < 8 {
        return Err("buffer too short for u64".into());
    }
    let offset = bytes.len() - 8;
    Ok(u64::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
        bytes[offset + 4],
        bytes[offset + 5],
        bytes[offset + 6],
        bytes[offset + 7],
    ]))
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn is_empty(&self) -> bool {
        self.offset == self.bytes.len()
    }

    fn read_u32(&mut self) -> Result<u32, Box<dyn std::error::Error>> {
        let bytes = self.read_exact(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_u64(&mut self) -> Result<u64, Box<dyn std::error::Error>> {
        let bytes = self.read_exact(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_count_list(&mut self) -> Result<Vec<NextCount>, Box<dyn std::error::Error>> {
        let count = usize::try_from(self.read_u32()?)?;
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            let token = read_token_u32(self)?;
            let count = self.read_u32()?;
            out.push(NextCount { token, count });
        }
        Ok(out)
    }

    fn read_exact(&mut self, count: usize) -> Result<&'a [u8], Box<dyn std::error::Error>> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or("cursor offset overflow")?;
        if end > self.bytes.len() {
            return Err("truncated NSRLMOD1 model".into());
        }
        let bytes = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(bytes)
    }
}

fn hash_tokens(tokens: &[u16]) -> u64 {
    let mut hash = StableHasher::new();
    for &token in tokens {
        hash.write_u16(token);
    }
    hash.finish()
}

fn hash_text(text: &str) -> u64 {
    hash_bytes(text.as_bytes())
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut hash = StableHasher::new();
    hash.write_bytes(bytes);
    hash.finish()
}

struct StableHasher {
    state: u64,
}

impl StableHasher {
    fn new() -> Self {
        Self { state: FNV_OFFSET }
    }

    fn write_u16(&mut self, value: u16) {
        self.write_bytes(&value.to_le_bytes());
    }

    fn write_bytes(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.state ^= u64::from(byte);
            self.state = self.state.wrapping_mul(FNV_PRIME);
        }
    }

    fn finish(self) -> u64 {
        self.state
    }
}

fn mix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn json_escape(value: &str) -> String {
    let mut out = String::new();
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            value if value.is_control() => {
                use std::fmt::Write;
                let _ = write!(&mut out, "\\u{:04x}", value as u32);
            }
            value => out.push(value),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_round_trips_and_samples_joint_text_and_image() {
        let tokens = [
            BOS,
            PROMPT,
            TEXT_BASE + u16::from(b'a'),
            TEXT,
            TEXT_BASE + u16::from(b'h'),
            TEXT_BASE + u16::from(b'i'),
            IMAGE,
            IMAGE_BASE + 1,
            IMAGE_BASE + 2,
            EOS,
        ];
        let mut expanded = Vec::new();
        for _ in 0..SIGNATURE_BINS {
            expanded.extend_from_slice(&tokens[..7]);
            expanded.push(IMAGE_BASE + 3);
            expanded.push(EOS);
        }
        let model = train_model(&expanded).unwrap();
        let bytes = model.try_to_bytes().unwrap();
        let loaded = SolomonMultimodalModel::from_bytes(&bytes).unwrap();
        assert_eq!(model, loaded);
        let sample = sample_model(&loaded, "a", 8, 1, 1);
        assert!(!sample.text.is_empty());
        assert!(sample.image_bins.iter().all(|&bin| bin < 16));
    }

    #[test]
    fn text_tokens_are_printable_ascii_only() {
        let tokens = encode_text_tokens("Bael\nSeal");
        assert!(tokens.iter().all(|&token| is_text_token(token)));
        assert_eq!(decode_text_tokens(&tokens), "Bael Seal");
    }
}
