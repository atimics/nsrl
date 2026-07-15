use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

pub const SUBWORD_TOKENIZER_SCHEMA: &str = "nsrl.subword_tokenizer.v1";
pub const SUBWORD_TRAIN_TRACE_SCHEMA: &str = "nsrl.subword_train_trace.v1";
pub const SUBWORD_ENCODE_TRACE_SCHEMA: &str = "nsrl.subword_encode_trace.v1";
pub const SUBWORD_TOKENIZER_ID: &str = "deterministic_byte_bpe_v1";
pub const SUBWORD_TOKENIZER_MAGIC: [u8; 8] = *b"NSRLBPE1";
pub const SUBWORD_TOKEN_STREAM_MAGIC: [u8; 8] = *b"NSRLTOK1";
pub const BYTE_TOKEN_COUNT: u32 = 256;
pub const BOS_TOKEN_ID: u32 = 256;
pub const EOS_TOKEN_ID: u32 = 257;
pub const PAD_TOKEN_ID: u32 = 258;
pub const FIRST_MERGE_TOKEN_ID: u32 = 259;

const TOKENIZER_VERSION: u32 = 1;
const TOKENIZER_HEADER_BYTES: usize = 32;
const TOKEN_STREAM_HEADER_BYTES: usize = 24;
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubwordTrainConfig {
    pub target_vocab_size: usize,
    pub min_pair_frequency: usize,
}

impl Default for SubwordTrainConfig {
    fn default() -> Self {
        Self {
            target_vocab_size: 8_192,
            min_pair_frequency: 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubwordTokenizer {
    target_vocab_size: u32,
    min_pair_frequency: u32,
    source_hash: u64,
    merges: Vec<(u32, u32)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubwordTrainTrace {
    pub input_bytes: usize,
    pub source_hash: u64,
    pub target_vocab_size: usize,
    pub actual_vocab_size: usize,
    pub min_pair_frequency: usize,
    pub merge_count: usize,
    pub encoded_tokens: usize,
    pub compression_per_mille: usize,
    pub tokenizer_hash: u64,
    pub artifact_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubwordEncodeTrace {
    pub input_bytes: usize,
    pub input_hash: u64,
    pub token_count: usize,
    pub token_hash: u64,
    pub compression_per_mille: usize,
    pub tokenizer_hash: u64,
    pub vocab_size: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubwordError {
    InvalidConfig(&'static str),
    InvalidArtifact(&'static str),
    InvalidToken(u32),
}

impl core::fmt::Display for SubwordError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidConfig(message) | Self::InvalidArtifact(message) => {
                formatter.write_str(message)
            }
            Self::InvalidToken(token) => write!(formatter, "invalid subword token {token}"),
        }
    }
}

impl std::error::Error for SubwordError {}

impl SubwordTokenizer {
    pub fn train(
        bytes: &[u8],
        config: SubwordTrainConfig,
    ) -> Result<(Self, SubwordTrainTrace), SubwordError> {
        if bytes.is_empty() {
            return Err(SubwordError::InvalidConfig(
                "subword training requires non-empty input",
            ));
        }
        if config.target_vocab_size < FIRST_MERGE_TOKEN_ID as usize + 1 {
            return Err(SubwordError::InvalidConfig(
                "subword target vocabulary must leave room for a merge token",
            ));
        }
        if config.target_vocab_size > u32::MAX as usize || config.min_pair_frequency < 2 {
            return Err(SubwordError::InvalidConfig(
                "invalid subword vocabulary size or pair-frequency floor",
            ));
        }
        let mut sequence = bytes
            .iter()
            .map(|&byte| u32::from(byte))
            .collect::<Vec<_>>();
        let mut merges = Vec::new();
        while FIRST_MERGE_TOKEN_ID as usize + merges.len() < config.target_vocab_size {
            let mut counts = HashMap::<(u32, u32), usize>::new();
            for pair in sequence.windows(2) {
                *counts.entry((pair[0], pair[1])).or_default() += 1;
            }
            let selected = counts
                .into_iter()
                .filter(|(_, count)| *count >= config.min_pair_frequency)
                .max_by_key(|&((left, right), count)| {
                    (count, core::cmp::Reverse(left), core::cmp::Reverse(right))
                })
                .map(|(pair, _)| pair);
            let Some(pair) = selected else {
                break;
            };
            let merged_token = FIRST_MERGE_TOKEN_ID + merges.len() as u32;
            sequence = merge_pair(&sequence, pair, merged_token);
            merges.push(pair);
        }
        let tokenizer = Self {
            target_vocab_size: config.target_vocab_size as u32,
            min_pair_frequency: config.min_pair_frequency as u32,
            source_hash: hash_bytes(bytes),
            merges,
        };
        tokenizer.validate()?;
        let artifact = tokenizer.to_bytes();
        let trace = SubwordTrainTrace {
            input_bytes: bytes.len(),
            source_hash: tokenizer.source_hash,
            target_vocab_size: config.target_vocab_size,
            actual_vocab_size: tokenizer.vocab_size(),
            min_pair_frequency: config.min_pair_frequency,
            merge_count: tokenizer.merges.len(),
            encoded_tokens: sequence.len(),
            compression_per_mille: sequence.len().saturating_mul(1000) / bytes.len(),
            tokenizer_hash: hash_bytes(&artifact),
            artifact_bytes: artifact.len(),
        };
        Ok((tokenizer, trace))
    }

    pub fn vocab_size(&self) -> usize {
        FIRST_MERGE_TOKEN_ID as usize + self.merges.len()
    }

    pub fn target_vocab_size(&self) -> usize {
        self.target_vocab_size as usize
    }

    pub fn min_pair_frequency(&self) -> usize {
        self.min_pair_frequency as usize
    }

    pub fn source_hash(&self) -> u64 {
        self.source_hash
    }

    pub fn tokenizer_hash(&self) -> u64 {
        hash_bytes(&self.to_bytes())
    }

    pub fn merges(&self) -> &[(u32, u32)] {
        &self.merges
    }

    pub fn encode(&self, bytes: &[u8]) -> Vec<u32> {
        let tokens = bytes
            .iter()
            .map(|&byte| u32::from(byte))
            .collect::<Vec<_>>();
        apply_ranked_merges(tokens, &self.merges)
    }

    pub fn encode_with_trace(&self, bytes: &[u8]) -> (Vec<u32>, SubwordEncodeTrace) {
        let tokens = self.encode(bytes);
        let trace = SubwordEncodeTrace {
            input_bytes: bytes.len(),
            input_hash: hash_bytes(bytes),
            token_count: tokens.len(),
            token_hash: hash_tokens(&tokens),
            compression_per_mille: tokens.len().saturating_mul(1000) / bytes.len().max(1),
            tokenizer_hash: self.tokenizer_hash(),
            vocab_size: self.vocab_size(),
        };
        (tokens, trace)
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<Vec<u8>, SubwordError> {
        let pieces = self.pieces()?;
        let mut bytes = Vec::new();
        for &token in tokens {
            if matches!(token, BOS_TOKEN_ID | EOS_TOKEN_ID | PAD_TOKEN_ID) {
                continue;
            }
            let piece = pieces
                .get(token as usize)
                .ok_or(SubwordError::InvalidToken(token))?;
            bytes.extend_from_slice(piece);
        }
        Ok(bytes)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(TOKENIZER_HEADER_BYTES + self.merges.len() * 8);
        bytes.extend_from_slice(&SUBWORD_TOKENIZER_MAGIC);
        bytes.extend_from_slice(&TOKENIZER_VERSION.to_le_bytes());
        bytes.extend_from_slice(&self.target_vocab_size.to_le_bytes());
        bytes.extend_from_slice(&self.min_pair_frequency.to_le_bytes());
        bytes.extend_from_slice(&self.source_hash.to_le_bytes());
        bytes.extend_from_slice(&(self.merges.len() as u32).to_le_bytes());
        for &(left, right) in &self.merges {
            bytes.extend_from_slice(&left.to_le_bytes());
            bytes.extend_from_slice(&right.to_le_bytes());
        }
        bytes
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SubwordError> {
        if bytes.len() < TOKENIZER_HEADER_BYTES || bytes[..8] != SUBWORD_TOKENIZER_MAGIC {
            return Err(SubwordError::InvalidArtifact(
                "bad subword tokenizer header",
            ));
        }
        let version = read_u32(bytes, 8)?;
        if version != TOKENIZER_VERSION {
            return Err(SubwordError::InvalidArtifact(
                "unsupported subword tokenizer version",
            ));
        }
        let target_vocab_size = read_u32(bytes, 12)?;
        let min_pair_frequency = read_u32(bytes, 16)?;
        let source_hash = read_u64(bytes, 20)?;
        let merge_count = read_u32(bytes, 28)? as usize;
        let expected = TOKENIZER_HEADER_BYTES
            .checked_add(
                merge_count
                    .checked_mul(8)
                    .ok_or(SubwordError::InvalidArtifact(
                        "subword merge count overflow",
                    ))?,
            )
            .ok_or(SubwordError::InvalidArtifact(
                "subword artifact length overflow",
            ))?;
        if bytes.len() != expected {
            return Err(SubwordError::InvalidArtifact(
                "wrong subword tokenizer artifact length",
            ));
        }
        let mut merges = Vec::with_capacity(merge_count);
        for index in 0..merge_count {
            let offset = TOKENIZER_HEADER_BYTES + index * 8;
            merges.push((read_u32(bytes, offset)?, read_u32(bytes, offset + 4)?));
        }
        Self {
            target_vocab_size,
            min_pair_frequency,
            source_hash,
            merges,
        }
        .validated()
    }

    pub fn token_stream_bytes(&self, tokens: &[u32]) -> Result<Vec<u8>, SubwordError> {
        if tokens
            .iter()
            .any(|&token| token as usize >= self.vocab_size())
        {
            return Err(SubwordError::InvalidArtifact(
                "token stream contains an out-of-vocabulary token",
            ));
        }
        let mut bytes = Vec::with_capacity(TOKEN_STREAM_HEADER_BYTES + tokens.len() * 4);
        bytes.extend_from_slice(&SUBWORD_TOKEN_STREAM_MAGIC);
        bytes.extend_from_slice(&self.tokenizer_hash().to_le_bytes());
        bytes.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
        for &token in tokens {
            bytes.extend_from_slice(&token.to_le_bytes());
        }
        Ok(bytes)
    }

    pub fn tokens_from_stream_bytes(&self, bytes: &[u8]) -> Result<Vec<u32>, SubwordError> {
        if bytes.len() < TOKEN_STREAM_HEADER_BYTES || bytes[..8] != SUBWORD_TOKEN_STREAM_MAGIC {
            return Err(SubwordError::InvalidArtifact(
                "bad subword token stream header",
            ));
        }
        if read_u64(bytes, 8)? != self.tokenizer_hash() {
            return Err(SubwordError::InvalidArtifact(
                "subword token stream tokenizer hash mismatch",
            ));
        }
        let token_count = usize::try_from(read_u64(bytes, 16)?)
            .map_err(|_| SubwordError::InvalidArtifact("subword token count overflow"))?;
        let expected = TOKEN_STREAM_HEADER_BYTES
            .checked_add(
                token_count
                    .checked_mul(4)
                    .ok_or(SubwordError::InvalidArtifact(
                        "subword token stream length overflow",
                    ))?,
            )
            .ok_or(SubwordError::InvalidArtifact(
                "subword token stream length overflow",
            ))?;
        if bytes.len() != expected {
            return Err(SubwordError::InvalidArtifact(
                "wrong subword token stream length",
            ));
        }
        let mut tokens = Vec::with_capacity(token_count);
        for index in 0..token_count {
            let token = read_u32(bytes, TOKEN_STREAM_HEADER_BYTES + index * 4)?;
            if token as usize >= self.vocab_size() {
                return Err(SubwordError::InvalidToken(token));
            }
            tokens.push(token);
        }
        Ok(tokens)
    }

    fn pieces(&self) -> Result<Vec<Vec<u8>>, SubwordError> {
        let mut pieces = (0..BYTE_TOKEN_COUNT)
            .map(|byte| vec![byte as u8])
            .collect::<Vec<_>>();
        pieces.extend([Vec::new(), Vec::new(), Vec::new()]);
        for (index, &(left, right)) in self.merges.iter().enumerate() {
            let expected = FIRST_MERGE_TOKEN_ID + index as u32;
            if left >= expected || right >= expected {
                return Err(SubwordError::InvalidArtifact(
                    "subword merge references a future or special token",
                ));
            }
            let mut piece = pieces[left as usize].clone();
            piece.extend_from_slice(&pieces[right as usize]);
            pieces.push(piece);
        }
        Ok(pieces)
    }

    fn validate(&self) -> Result<(), SubwordError> {
        if self.target_vocab_size < FIRST_MERGE_TOKEN_ID + 1
            || self.min_pair_frequency < 2
            || self.vocab_size() > self.target_vocab_size as usize
        {
            return Err(SubwordError::InvalidArtifact(
                "invalid subword tokenizer configuration",
            ));
        }
        self.pieces().map(|_| ())
    }

    fn validated(self) -> Result<Self, SubwordError> {
        self.validate()?;
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy)]
struct MergeNode {
    token: u32,
    previous: Option<usize>,
    next: Option<usize>,
    alive: bool,
}

fn apply_ranked_merges(tokens: Vec<u32>, merges: &[(u32, u32)]) -> Vec<u32> {
    if tokens.len() < 2 || merges.is_empty() {
        return tokens;
    }
    let ranks = merges
        .iter()
        .copied()
        .enumerate()
        .map(|(rank, pair)| (pair, rank))
        .collect::<HashMap<_, _>>();
    let token_count = tokens.len();
    let mut nodes = tokens
        .into_iter()
        .enumerate()
        .map(|(index, token)| MergeNode {
            token,
            previous: index.checked_sub(1),
            next: (index + 1 < token_count).then_some(index + 1),
            alive: true,
        })
        .collect::<Vec<_>>();
    let mut queue = BinaryHeap::<Reverse<(usize, usize, usize)>>::new();
    for left in 0..nodes.len() - 1 {
        enqueue_ranked_pair(&nodes, &ranks, &mut queue, left, left + 1);
    }

    while let Some(Reverse((rank, left, right))) = queue.pop() {
        if !nodes[left].alive || !nodes[right].alive || nodes[left].next != Some(right) {
            continue;
        }
        let pair = (nodes[left].token, nodes[right].token);
        if ranks.get(&pair).copied() != Some(rank) {
            continue;
        }
        nodes[left].token = FIRST_MERGE_TOKEN_ID + rank as u32;
        let next = nodes[right].next;
        nodes[left].next = next;
        nodes[right].alive = false;
        if let Some(next) = next {
            nodes[next].previous = Some(left);
        }
        if let Some(previous) = nodes[left].previous {
            enqueue_ranked_pair(&nodes, &ranks, &mut queue, previous, left);
        }
        if let Some(next) = next {
            enqueue_ranked_pair(&nodes, &ranks, &mut queue, left, next);
        }
    }

    let mut output = Vec::new();
    let mut current = Some(0);
    while let Some(index) = current {
        if nodes[index].alive {
            output.push(nodes[index].token);
        }
        current = nodes[index].next;
    }
    output
}

fn enqueue_ranked_pair(
    nodes: &[MergeNode],
    ranks: &HashMap<(u32, u32), usize>,
    queue: &mut BinaryHeap<Reverse<(usize, usize, usize)>>,
    left: usize,
    right: usize,
) {
    if nodes[left].alive
        && nodes[right].alive
        && nodes[left].next == Some(right)
        && let Some(&rank) = ranks.get(&(nodes[left].token, nodes[right].token))
    {
        queue.push(Reverse((rank, left, right)));
    }
}

impl SubwordTrainTrace {
    pub fn to_json_line(self) -> String {
        format!(
            concat!(
                "{{\"schema\":\"{}\",\"tokenizer\":\"{}\",",
                "\"input\":{{\"bytes\":{},\"hash\":\"0x{:016x}\"}},",
                "\"vocabulary\":{{\"target\":{},\"actual\":{},\"byte_fallback_tokens\":256,\"special_tokens\":3,\"merges\":{},\"min_pair_frequency\":{}}},",
                "\"training_encoding\":{{\"tokens\":{},\"tokens_per_input_byte_per_mille\":{}}},",
                "\"artifact\":{{\"bytes\":{},\"hash\":\"0x{:016x}\"}}}}\n"
            ),
            SUBWORD_TRAIN_TRACE_SCHEMA,
            SUBWORD_TOKENIZER_ID,
            self.input_bytes,
            self.source_hash,
            self.target_vocab_size,
            self.actual_vocab_size,
            self.merge_count,
            self.min_pair_frequency,
            self.encoded_tokens,
            self.compression_per_mille,
            self.artifact_bytes,
            self.tokenizer_hash,
        )
    }
}

impl SubwordEncodeTrace {
    pub fn to_json_line(self) -> String {
        format!(
            concat!(
                "{{\"schema\":\"{}\",\"tokenizer\":\"{}\",",
                "\"tokenizer_hash\":\"0x{:016x}\",\"vocab_size\":{},",
                "\"input\":{{\"bytes\":{},\"hash\":\"0x{:016x}\"}},",
                "\"tokens\":{{\"count\":{},\"hash\":\"0x{:016x}\",\"tokens_per_input_byte_per_mille\":{}}}}}\n"
            ),
            SUBWORD_ENCODE_TRACE_SCHEMA,
            SUBWORD_TOKENIZER_ID,
            self.tokenizer_hash,
            self.vocab_size,
            self.input_bytes,
            self.input_hash,
            self.token_count,
            self.token_hash,
            self.compression_per_mille,
        )
    }
}

fn merge_pair(tokens: &[u32], pair: (u32, u32), merged_token: u32) -> Vec<u32> {
    let mut merged = Vec::with_capacity(tokens.len());
    let mut index = 0_usize;
    while index < tokens.len() {
        if index + 1 < tokens.len() && (tokens[index], tokens[index + 1]) == pair {
            merged.push(merged_token);
            index += 2;
        } else {
            merged.push(tokens[index]);
            index += 1;
        }
    }
    merged
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, SubwordError> {
    let value = bytes
        .get(offset..offset + 4)
        .ok_or(SubwordError::InvalidArtifact("truncated subword artifact"))?;
    Ok(u32::from_le_bytes([value[0], value[1], value[2], value[3]]))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, SubwordError> {
    let value = bytes
        .get(offset..offset + 8)
        .ok_or(SubwordError::InvalidArtifact("truncated subword artifact"))?;
    Ok(u64::from_le_bytes([
        value[0], value[1], value[2], value[3], value[4], value[5], value[6], value[7],
    ]))
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    bytes.iter().fold(FNV_OFFSET, |mut hash, &byte| {
        hash ^= u64::from(byte);
        hash.wrapping_mul(FNV_PRIME)
    })
}

fn hash_tokens(tokens: &[u32]) -> u64 {
    tokens.iter().fold(FNV_OFFSET, |mut hash, &token| {
        for byte in token.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode_reference(tokenizer: &SubwordTokenizer, bytes: &[u8]) -> Vec<u32> {
        let mut tokens = bytes
            .iter()
            .map(|&byte| u32::from(byte))
            .collect::<Vec<_>>();
        for (index, &pair) in tokenizer.merges.iter().enumerate() {
            tokens = merge_pair(&tokens, pair, FIRST_MERGE_TOKEN_ID + index as u32);
        }
        tokens
    }

    fn fixture() -> (SubwordTokenizer, SubwordTrainTrace) {
        SubwordTokenizer::train(
            b"integer transformers transform repeated integer patterns",
            SubwordTrainConfig {
                target_vocab_size: 288,
                min_pair_frequency: 2,
            },
        )
        .expect("train fixture tokenizer")
    }

    #[test]
    fn training_and_artifact_replay_are_deterministic() {
        let left = fixture();
        let right = fixture();
        assert_eq!(left, right);
        assert_eq!(left.0.to_bytes(), right.0.to_bytes());
        assert_eq!(
            SubwordTokenizer::from_bytes(&left.0.to_bytes()).expect("decode tokenizer"),
            left.0
        );
    }

    #[test]
    fn byte_fallback_round_trips_arbitrary_input() {
        let (tokenizer, _) = fixture();
        let bytes = [0, 1, 2, 127, 128, 254, 255, b'i', b'n', b't'];
        let tokens = tokenizer.encode(&bytes);
        assert_eq!(tokenizer.decode(&tokens).expect("decode bytes"), bytes);
    }

    #[test]
    fn ranked_encoder_matches_sequential_merge_replay() {
        let tokenizer = SubwordTokenizer {
            target_vocab_size: 266,
            min_pair_frequency: 2,
            source_hash: 0,
            merges: vec![
                (u32::from(b'a'), u32::from(b'a')),
                (FIRST_MERGE_TOKEN_ID, u32::from(b'a')),
                (u32::from(b'b'), u32::from(b'a')),
                (FIRST_MERGE_TOKEN_ID + 2, u32::from(b'b')),
                (FIRST_MERGE_TOKEN_ID + 1, FIRST_MERGE_TOKEN_ID + 3),
            ],
        };
        let alphabet = [b'a', b'b', b'c'];
        for encoded in 0_u32..3_u32.pow(8) {
            let mut value = encoded;
            let mut bytes = [0_u8; 8];
            for byte in &mut bytes {
                *byte = alphabet[(value % 3) as usize];
                value /= 3;
            }
            assert_eq!(
                tokenizer.encode(&bytes),
                encode_reference(&tokenizer, &bytes),
                "{bytes:?}"
            );
        }
    }

    #[test]
    fn special_tokens_do_not_alias_bytes_or_merges() {
        let (tokenizer, _) = fixture();
        assert_eq!(BYTE_TOKEN_COUNT, 256);
        assert_eq!([BOS_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID], [256, 257, 258]);
        assert!(
            tokenizer.merges().iter().all(|&(left, right)| {
                !matches!(left, 256..=258) && !matches!(right, 256..=258)
            })
        );
    }

    #[test]
    fn token_stream_binds_the_tokenizer_hash() {
        let (tokenizer, _) = fixture();
        let tokens = tokenizer.encode(b"integer integer");
        let stream = tokenizer
            .token_stream_bytes(&tokens)
            .expect("encode token stream");
        assert_eq!(
            tokenizer
                .tokens_from_stream_bytes(&stream)
                .expect("decode token stream"),
            tokens
        );
        let mut corrupt = stream;
        corrupt[8] ^= 1;
        assert!(tokenizer.tokens_from_stream_bytes(&corrupt).is_err());
    }

    #[test]
    fn learned_merges_compress_repeated_text() {
        let (tokenizer, trace) = fixture();
        assert!(!tokenizer.merges().is_empty());
        assert!(trace.encoded_tokens < trace.input_bytes);
        assert!(trace.compression_per_mille < 1000);
    }

    #[test]
    fn malformed_or_future_merge_is_rejected() {
        let (tokenizer, _) = fixture();
        let mut bytes = tokenizer.to_bytes();
        bytes[TOKENIZER_HEADER_BYTES..TOKENIZER_HEADER_BYTES + 4]
            .copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(SubwordTokenizer::from_bytes(&bytes).is_err());
    }
}
