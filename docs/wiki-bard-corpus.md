# Wiki-Bard Corpus

Wiki-Bard starts as a byte/character language model over two auditable sources:

- Project Gutenberg eBook 100, The Complete Works of William Shakespeare:
  `https://www.gutenberg.org/cache/epub/100/pg100.txt`
- Simple English Wikipedia latest pages/articles dump:
  `https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2`

The corpus pipeline intentionally emits a trace before training. A model can
only claim it was trained on Wiki-Bard data after the corpus hash, tokenizer
hash, model seed, and training trace all line up.

## Prepare Sources

```sh
mkdir -p data/raw data/processed

curl -L \
  https://www.gutenberg.org/cache/epub/100/pg100.txt \
  -o data/raw/shakespeare-gutenberg-100.txt

curl -L \
  https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2 \
  -o data/raw/simplewiki-latest-pages-articles.xml.bz2
```

Wikimedia publishes checksum files in the same directory:

```sh
curl -L \
  https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-sha1sums.txt \
  -o data/raw/simplewiki-latest-sha1sums.txt
```

## Build Corpus

For a fast smoke corpus:

```sh
bzip2 -dc data/raw/simplewiki-latest-pages-articles.xml.bz2 | \
  cargo run -p nsrl-corpus -- prepare \
    --shakespeare data/raw/shakespeare-gutenberg-100.txt \
    --simplewiki-xml - \
    --out data/processed/wiki-bard-corpus-smoke.txt \
    --trace data/processed/wiki-bard-corpus-smoke.trace.jsonl \
    --max-simplewiki-pages 128
```

For the full first corpus:

```sh
bzip2 -dc data/raw/simplewiki-latest-pages-articles.xml.bz2 | \
  cargo run -p nsrl-corpus -- prepare \
    --shakespeare data/raw/shakespeare-gutenberg-100.txt \
    --simplewiki-xml - \
    --out data/processed/wiki-bard-corpus.txt \
    --trace data/processed/wiki-bard-corpus.trace.jsonl
```

The output trace uses `nsrl.corpus_trace.v1` and records source URLs, input
byte counts, accepted/skipped wiki page counts, output bytes, output lines, and
a stable FNV-1a corpus hash.

## Visionary Wiki-Bard

The next corpus lane layers public-domain visionary/literary texts on top of
Wiki-Bard. The first additions are Project Gutenberg texts by Aleister Crowley
and William Blake:

| Source | File | URL |
| --- | --- | --- |
| Crowley, `Household Gods` | `data/raw/crowley-household-gods-pg14040.txt` | `https://www.gutenberg.org/cache/epub/14040/pg14040.txt` |
| Crowley, `Tannhäuser` | `data/raw/crowley-tannhauser-pg70261.txt` | `https://www.gutenberg.org/cache/epub/70261/pg70261.txt` |
| Blake, `Poems of William Blake` | `data/raw/blake-poems-pg574.txt` | `https://www.gutenberg.org/cache/epub/574/pg574.txt` |
| Blake, `The Marriage of Heaven and Hell` | `data/raw/blake-marriage-heaven-hell-pg45315.txt` | `https://www.gutenberg.org/cache/epub/45315/pg45315.txt` |

Download and clean:

```sh
curl -L https://www.gutenberg.org/cache/epub/14040/pg14040.txt \
  -o data/raw/crowley-household-gods-pg14040.txt
curl -L https://www.gutenberg.org/cache/epub/70261/pg70261.txt \
  -o data/raw/crowley-tannhauser-pg70261.txt
curl -L https://www.gutenberg.org/cache/epub/574/pg574.txt \
  -o data/raw/blake-poems-pg574.txt
curl -L https://www.gutenberg.org/cache/epub/45315/pg45315.txt \
  -o data/raw/blake-marriage-heaven-hell-pg45315.txt

cargo run -p nsrl-corpus -- clean-gutenberg \
  --corpus data/raw/crowley-household-gods-pg14040.txt \
  --out data/processed/crowley-household-gods.clean.txt
cargo run -p nsrl-corpus -- clean-gutenberg \
  --corpus data/raw/crowley-tannhauser-pg70261.txt \
  --out data/processed/crowley-tannhauser.clean.txt
cargo run -p nsrl-corpus -- clean-gutenberg \
  --corpus data/raw/blake-poems-pg574.txt \
  --out data/processed/blake-poems.clean.txt
cargo run -p nsrl-corpus -- clean-gutenberg \
  --corpus data/raw/blake-marriage-heaven-hell-pg45315.txt \
  --out data/processed/blake-marriage-heaven-hell.clean.txt
```

Build the combined corpus:

```sh
{
  printf '<|source:wiki-bard|>\n'
  cat data/processed/wiki-bard-corpus.txt
  printf '\n<|source:crowley-household-gods-pg14040|>\n'
  cat data/processed/crowley-household-gods.clean.txt
  printf '\n<|source:crowley-tannhauser-pg70261|>\n'
  cat data/processed/crowley-tannhauser.clean.txt
  printf '\n<|source:blake-poems-pg574|>\n'
  cat data/processed/blake-poems.clean.txt
  printf '\n<|source:blake-marriage-heaven-hell-pg45315|>\n'
  cat data/processed/blake-marriage-heaven-hell.clean.txt
  printf '\n'
} > data/processed/visionary-wikibard-corpus.txt
```

Current `visionary-wikibard-corpus.txt` SHA-256:

```text
4dd1b5632e649cc50fb265adf08f0b148d7f1e43b9111d1c23c203bf20c8f97b
```

Tokenize the first Blake/Crowley lane:

```sh
cargo run -p nsrl-corpus -- lexeme-tokenize \
  --corpus data/processed/visionary-wikibard-corpus.txt \
  --tokens-out data/processed/visionary-wikibard-lexeme-v4096.tokens.u16 \
  --vocab-out data/processed/visionary-wikibard-lexeme-v4096.vocab.tsv \
  --trace data/processed/visionary-wikibard-lexeme-v4096.tokens.trace.jsonl \
  --seq-len 32 \
  --stride 1 \
  --max-vocab 4096 \
  --lexeme-vocab-profile balanced \
  --lexeme-frequency-cap 4096 \
  --preview-tokens 32
```

Current Blake/Crowley full-corpus lexeme trace:

| corpus | bytes | lexemes | tokens | vocab | observed issue |
| --- | ---: | ---: | ---: | ---: | --- |
| `visionary-wikibard-corpus.txt` | 303,006,835 | 55,135,125 | 109,955,212 | 4,096 | SimpleWiki markup still dominates top vocab (`class`, `align`, `bgcolor`, `www`). |

Useful source terms are present in the vocab (`tiger`, `pan`), but the full
wiki stream is so large that Blake/Crowley are stylistic seasoning rather than
dominant training signal. The balanced vocab profile is now the preferred
tokenizer setting for mixed-source corpora: it uses a capped-sqrt frequency
score so repeated structural tokens keep their utility without drowning rare
concepts. For concept learning, pair that with trainer-side frequency caps and
`--quality-weight-profile cruft-aware`, which keeps document-history tokens
learnable but lowers their gradient power. A source-balanced sampler is still
worth adding before a full Wiki+Bard+Crowley+Blake hero run.

## Corpus Expansion Queue

The lesson from the lexeme runs is that clean, source-balanced text is more
valuable than raw byte volume. We should try to run out of clean data before
running out of compute, but avoid teaching the model markup.

Good next additions:

| Lane | Why it helps | Candidate source |
| --- | --- | --- |
| King James Bible | High-repetition prophetic syntax, useful for Blake/Crowley register. | Project Gutenberg `https://www.gutenberg.org/ebooks/10` |
| Milton | Long-period epic syntax and theological vocabulary. | Project Gutenberg author/texts |
| Romantic poetry pack | Byron, Shelley, Keats, Coleridge, Wordsworth; bridges Blake to dramatic verse. | Project Gutenberg author pages |
| Classical myth pack | Homer, Ovid, Virgil, Aeschylus, Sophocles, Euripides; character/action structure. | Project Gutenberg and Perseus-derived public-domain translations |
| Arthurian/medieval pack | Malory, Mabinogion, Celtic fairy material; mythic narrative glue. | Project Gutenberg |
| Victorian novel pack | Dickens, Austen, Eliot, Bronte, Hardy; dialogue and narrative continuity. | Standard Ebooks or Project Gutenberg |
| Occult/esoteric public-domain pack | Waite, Levi translations, Golden Dawn-adjacent public-domain works where licensing is clean. | Project Gutenberg/Internet Archive after manual license review |
| Clean encyclopedia lane | Replace raw SimpleWiki markup residue with paragraph-extracted articles and source-balanced sampling. | Existing SimpleWiki dump, but with stronger XML/wiki cleanup |

## Tokenize Corpus

The first tokenizer is byte identity:

```text
token_id = corpus_byte
vocab_size = 256
```

That is intentionally plain. It keeps the first Wiki-Bard trainer focused on
integer sequence learning instead of vocabulary training.

For a smoke token stream:

```sh
cargo run -p nsrl-corpus -- tokenize \
  --corpus data/processed/wiki-bard-corpus-smoke.txt \
  --tokens-out data/processed/wiki-bard-corpus-smoke.tokens.u8 \
  --trace data/processed/wiki-bard-corpus-smoke.tokens.trace.jsonl \
  --seq-len 128 \
  --stride 1 \
  --max-windows 4096
```

For the full first token stream:

```sh
cargo run -p nsrl-corpus -- tokenize \
  --corpus data/processed/wiki-bard-corpus.txt \
  --tokens-out data/processed/wiki-bard-corpus.tokens.u8 \
  --trace data/processed/wiki-bard-corpus.tokens.trace.jsonl \
  --seq-len 128 \
  --stride 1
```

The token trace uses `nsrl.token_trace.v1` and records the tokenizer ID, token
count, token hash, sliding next-token window count, and stable window hash.

For low-entropy text curriculum experiments, the ascii-lower tokenizer profile
keeps the same byte-vocabulary runtime but removes casing, Unicode, markup
punctuation, and repeated whitespace from the training stream:

```sh
cargo run -p nsrl-corpus -- tokenize \
  --corpus data/processed/wiki-bard-corpus.txt \
  --tokens-out data/processed/wiki-bard-corpus-ascii-lower.tokens.u8 \
  --trace data/processed/wiki-bard-corpus-ascii-lower.tokens.trace.jsonl \
  --seq-len 16 \
  --stride 36965 \
  --preview-tokens 32 \
  --text-profile ascii-lower
```

Current full-corpus ascii-lower token trace:

| tokenizer | input bytes | output tokens | stride | windows |
| --- | ---: | ---: | ---: | ---: |
| `byte_ascii_lower_text_u8_v1` | 302,813,212 | 285,913,078 | 36,965 | 7,735 |

## Lexeme Tokenize Corpus

The first concept-scaffold tokenizer emits stable `u16` lexical tokens instead
of asking the model to discover every word from characters. It reserves token
IDs `0..255` for byte fallback and assigns frequent ascii-lower lexemes from
`256` upward:

```sh
cargo run -p nsrl-corpus -- lexeme-tokenize \
  --corpus data/processed/wiki-bard-corpus.txt \
  --tokens-out data/processed/wiki-bard-corpus-lexeme.tokens.u16 \
  --vocab-out data/processed/wiki-bard-corpus-lexeme.vocab.tsv \
  --trace data/processed/wiki-bard-corpus-lexeme.tokens.trace.jsonl \
  --seq-len 32 \
  --stride 1 \
  --max-vocab 2048 \
  --preview-tokens 32
```

Current full-corpus lexeme token trace:

| tokenizer | input bytes | normalized bytes | lexemes | tokens | vocab | token bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `lexeme_ascii_lower_u16_v1` | 302,813,212 | 279,908,937 | 58,020,939 | 136,254,072 | 2,048 | 272,508,144 |

The lexeme lane skips NSRL corpus marker lines such as `<|source:...|>` and
`<|page:...|>` before normalization. The first preview starts with real corpus
text: `the complete works of william shakespeare...`.

This is the scaffold for semantic embedding pretraining, not the final
tokenizer. The top vocabulary is already word-like (`the`, `of`, `and`, `to`,
`be`, `not`), but the full corpus still leaks wiki residue such as `class`,
`align`, `bgcolor`, and `http`. That residue should be cleaned before treating
lexeme embeddings as semantic concepts.

## Lexeme Embedding Pretrain

The first concept-first training step learns an i16 embedding table over the
stable lexeme IDs. It is a deterministic integer skip-gram-style scaffold:
observed center/context pairs are pulled together while deterministic negative
pairs are pushed apart. This is not language-model training yet; it only gives
the later byte/grammar/spelling stages a lexical substrate to stand on.

```sh
cargo run -p nsrl-train -- \
  --mode lexeme-embedding \
  --tokens data/processed/wiki-bard-corpus-lexeme.tokens.u16 \
  --model-out data/processed/wiki-bard-lexeme-embedding-spread4096.nsrllex \
  --trace data/processed/wiki-bard-lexeme-embedding-spread4096.trace.jsonl \
  --vocab-size 2048 \
  --embedding-dim 16 \
  --context-radius 2 \
  --stride 33264 \
  --max-windows 4096 \
  --epochs 1 \
  --lr-shift 9
```

The trace uses `nsrl.training_lexeme_embedding_trace.v1` and records token and
window hashes, embedding hashes, positive-pair dot movement, negative-pair dot
movement, saturation counts, zero-delta counts, and L1 embedding movement.
The optimizer is a bounded integer hinge: positives are pulled only below a
dot margin of `1000000`, and deterministic negatives are pushed only above a
dot margin of `0`.

Current full-corpus spread smoke metrics:

| windows | stride | lr shift | positive dot delta | negative dot delta | saturation | zero deltas | embedding L1 | wall time | peak RSS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 33,264 | 9 | +4,529,776,335 | -17,635,550 | 0 | 292,739 | 292,445 | 14.06 s | 547 MB |

## Lexeme Softmax Head

The next concept-scaffold step freezes the learned lexeme embeddings and trains
a dynamic-vocab i8 output head from mean-pooled lexeme context features:

```text
[bias_q15, mean(lexeme_embedding(context_tokens))_q15]
```

The context length must be a power of two, so the mean remains an exact right
shift. This is intentionally still a shallow trainer: it proves stable
concept-token prediction before we spend more engineering on grammar and
spelling layers.

Full Wiki-Bard lexeme head smoke:

```sh
cargo run -p nsrl-train -- \
  --mode lexeme-softmax \
  --tokens data/processed/wiki-bard-corpus-lexeme.tokens.u16 \
  --model data/processed/wiki-bard-lexeme-embedding-spread4096.nsrllex \
  --model-out data/processed/wiki-bard-lexeme-softmax-spread4096.nsrllm \
  --trace data/processed/wiki-bard-lexeme-softmax-spread4096.trace.jsonl \
  --stride 33264 \
  --max-windows 4096 \
  --epochs 1 \
  --lr-shift 22 \
  --max-weight-delta 1
```

Current full-corpus lexeme head metrics:

| windows | stride | context | lr shift | accuracy per mille | prob error delta | saturation | zero deltas | head L1 | wall time | peak RSS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 33,264 | 1 | 22 | 77 | -345,159 | 3,554 | 142,453,041 | 59,131 | 23.05 s | 547 MB |

The full corpus proves the trainer works, but it also proves the tokenizer is
too dirty for a poetry demo: generation quickly falls into wiki residue such as
`www`, `class`, and color-code fragments.

For a cleaner language scaffold, the same lane can be built on Shakespeare
alone with a 2,048-token vocabulary:

```sh
cargo run -p nsrl-corpus -- lexeme-tokenize \
  --corpus data/raw/shakespeare-gutenberg-100.txt \
  --tokens-out data/processed/shakespeare-lexeme.tokens.u16 \
  --vocab-out data/processed/shakespeare-lexeme.vocab.tsv \
  --trace data/processed/shakespeare-lexeme.tokens.trace.jsonl \
  --seq-len 32 \
  --stride 1 \
  --max-vocab 2048 \
  --preview-tokens 32

cargo run -p nsrl-train -- \
  --mode lexeme-embedding \
  --tokens data/processed/shakespeare-lexeme.tokens.u16 \
  --model-out data/processed/shakespeare-lexeme-embedding-spread8192.nsrllex \
  --trace data/processed/shakespeare-lexeme-embedding-spread8192.trace.jsonl \
  --vocab-size 2048 \
  --embedding-dim 16 \
  --context-radius 2 \
  --stride 247 \
  --max-windows 8192 \
  --epochs 1 \
  --lr-shift 9 \
  --concept-frequency-cap 4096 \
  --frequency-weight-min-q15 4096 \
  --quality-weight-profile cruft-aware \
  --vocab data/processed/shakespeare-lexeme.vocab.tsv

cargo run -p nsrl-train -- \
  --mode lexeme-softmax \
  --tokens data/processed/shakespeare-lexeme.tokens.u16 \
  --model data/processed/shakespeare-lexeme-embedding-spread8192.nsrllex \
  --model-out data/processed/shakespeare-lexeme-softmax-seq8-spread8192.nsrllm \
  --trace data/processed/shakespeare-lexeme-softmax-seq8-spread8192.trace.jsonl \
  --seq-len 8 \
  --stride 247 \
  --max-windows 8192 \
  --epochs 1 \
  --lr-shift 22 \
  --target-frequency-cap 4096 \
  --frequency-weight-min-q15 4096 \
  --quality-weight-profile cruft-aware \
  --vocab data/processed/shakespeare-lexeme.vocab.tsv \
  --max-weight-delta 1
```

Current clean Shakespeare lexeme head metrics:

| windows | stride | context | lr shift | accuracy per mille | prob error delta | saturation | zero deltas | head L1 | wall time | peak RSS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 247 | 8 | 22 | 16 | -373,554 | 10,785 | 284,486,907 | 99,167 | 35.86 s | 9.9 MB |

## Lexeme Generation Scaffold

`lexeme-generate` renders generated `u16` lexemes back through the vocab TSV. It
also supports the same corpus-prior and strict-adjacency decode scaffolds as
the byte generator, but over lexeme bigrams:

```sh
cargo run -p nsrl-train -- \
  --mode lexeme-generate \
  --model data/processed/shakespeare-lexeme-softmax-seq8-spread8192.nsrllm \
  --vocab data/processed/shakespeare-lexeme.vocab.tsv \
  --tokens data/processed/shakespeare-lexeme.tokens.u16 \
  --prompt "to be or not to be" \
  --max-new-tokens 120 \
  --decode sample \
  --sample-seed 7 \
  --top-k 8 \
  --repeat-window 24 \
  --repeat-penalty-shift 1 \
  --max-repeat-run 3 \
  --corpus-prior \
  --strict-adjacency \
  --text-out data/processed/shakespeare-lexeme-softmax-seq8-prior-top8-s7.txt \
  --trace data/processed/shakespeare-lexeme-softmax-seq8-prior-top8-s7.trace.jsonl
```

Current best clean-lane sample:

```text
to be or not to be not what is a king s no such day? i ll go you? king john. he would he hath he was he is no such thing that which thou be it shall have a very well as you have not? no such day? what he would be thou? o thou dost love s a b b b j 0 2 4 2 keeper of this man s not what you are all but if it be as i have a good lord? why not? or what he shall have no more; but that which now. o thou dost know not what thou be my brother s well as you shall
```

This is a real improvement over character babble: it has lexeme-level phrases,
names, punctuation, and Shakespeare-like local transitions. It is still not
coherent prose. The remaining junk tokens (`j 0 2 4 2`) show that even the
Shakespeare-only Gutenberg text needs header/table cleanup or a better concept
vocab filter before the semantic scaffold can be trusted.

## Training Smoke

The first corpus-backed trainer consumes the raw `.tokens.u8` file and trains a
256-class output head over a tiny deterministic feature vector:

```text
[bias_q15, one_hot(last_context_byte)_q15]
```

This is not the final Transformer trainer. It is the first replayable proof
that Wiki-Bard token windows can drive integer base-2 softmax learning.

```sh
cargo run -p nsrl-train -- \
  --mode byte-softmax \
  --tokens data/processed/wiki-bard-corpus-smoke.tokens.u8 \
  --model-out data/processed/wiki-bard-byte-softmax-smoke.nsrlbm \
  --seq-len 128 \
  --stride 1 \
  --window-offset 0 \
  --batch-windows 1 \
  --max-windows 4096 \
  --epochs 1 \
  --lr-shift 25 \
  --trace data/processed/wiki-bard-byte-softmax-smoke.trace.jsonl
```

The training trace uses `nsrl.training_byte_softmax_trace.v1` and records the
token hash, window hash, initial/final byte prediction error, probability error,
weight hashes, saturation counts, and per-window update evidence.

## Generate From Baseline

The byte-softmax trainer can save a tiny model artifact with `--model-out`.
Generation loads that artifact and runs deterministic byte decoding:

```sh
cargo run -p nsrl-train -- \
  --mode byte-generate \
  --model data/processed/wiki-bard-byte-softmax-smoke.nsrlbm \
  --prompt "To be" \
  --max-new-tokens 64 \
  --decode sample \
  --sample-seed 1 \
  --top-k 16 \
  --text-out data/processed/wiki-bard-byte-generation-smoke.txt \
  --trace data/processed/wiki-bard-byte-generation-smoke.trace.jsonl
```

The generation trace uses `nsrl.byte_generation_trace.v1`. It records the model
hash, prompt bytes, generated bytes, and per-token logits/probabilities. The
text output file writes the prompt plus generated bytes as the visible demo
transcript.

## Learned Embedding Baseline

The next trainer learns a small byte embedding table in addition to the output
head. The context feature becomes:

```text
[bias_q15, mean(byte_embedding(context_tokens))_q15]
```

The context length must be a power of two, which keeps the mean as an exact
right shift. This is still not the final Transformer trainer, but it is the
first Wiki-Bard path with a learned hidden context state.

```sh
cargo run -p nsrl-train -- \
  --mode byte-embed-softmax \
  --tokens data/processed/wiki-bard-corpus-smoke.tokens.u8 \
  --model-out data/processed/wiki-bard-byte-embed-softmax-smoke.nsrlem \
  --seq-len 128 \
  --stride 1 \
  --window-offset 0 \
  --max-windows 4096 \
  --epochs 1 \
  --lr-shift 17 \
  --embed-lr-shift 0 \
  --trace data/processed/wiki-bard-byte-embed-softmax-smoke.trace.jsonl
```

The trace uses `nsrl.training_byte_embed_softmax_trace.v1` and records both
embedding-table and output-head hashes, saturation counts, zero-delta counts,
and L1 movement.

```sh
cargo run -p nsrl-train -- \
  --mode byte-embed-generate \
  --model data/processed/wiki-bard-byte-embed-softmax-smoke.nsrlem \
  --prompt "To be" \
  --max-new-tokens 64 \
  --decode sample \
  --sample-seed 1 \
  --top-k 16 \
  --text-out data/processed/wiki-bard-byte-embed-generation-smoke.txt \
  --trace data/processed/wiki-bard-byte-embed-generation-smoke.trace.jsonl
```

The generation trace uses `nsrl.byte_embed_generation_trace.v1`.

## Transformer Training Infrastructure

The final Wiki-Bard model still needs full Transformer backpropagation. The
current training ladder now has a checked gated-MLP weight update that mutates
`up`, `gate`, and `down` matrices from cached forward activations:

```sh
cargo run -p nsrl-train -- \
  --mode gated-mlp-backward \
  --lr-shift 20 \
  --trace data/processed/wiki-bard-gated-mlp-backward-smoke.trace.jsonl
```

The trace uses `nsrl.training_gated_mlp_backward_trace.v1`. This is not
corpus-scale training by itself; it is the chain-rule primitive needed before
the learned byte embeddings can feed a trainable Transformer block.

The first miniature Transformer-shaped corpus loop now wires byte embeddings
through causal attention, a trainable gated MLP, and a trainable byte output
head. The backward path updates the output head, the MLP `up`/`gate`/`down`
matrices, and the attention `Q`/`K`/`V`/`O` matrices:

```sh
cargo run -p nsrl-train -- \
  --mode mini-transformer-mlp \
  --tokens data/processed/wiki-bard-corpus-smoke.tokens.u8 \
  --seq-len 4 \
  --stride 1 \
  --window-offset 0 \
  --max-windows 4096 \
  --epochs 1 \
  --lr-shift 18 \
  --mlp-lr-shift 16 \
  --embed-lr-shift 14 \
  --attention-lr-shift 24 \
  --attention-qk-lr-shift 18 \
  --model-out data/processed/wiki-bard-mini-transformer-mlp-smoke.nsrlmt \
  --trace data/processed/wiki-bard-mini-transformer-mlp-smoke.trace.jsonl
```

For full-corpus experiments, avoid training only on the file header. Choose a
large `--stride` to spread the audited windows across the byte stream and use
`--window-offset` to rotate the slice deterministically between runs.

The current full-corpus spread run samples 4096 windows across the
302,813,212-byte corpus with a stride of 73,929. It also enables the
mini-Transformer rollback history in the trace writer, so a checked-invalid
attention state restores a recent deterministic checkpoint instead of aborting
the run. `--batch-windows` is an integer smoothing knob. The output head, gated
MLP, attention projections, and embeddings now accumulate raw gradients in
`i64` buffers and apply one averaged update per accepted batch.

```sh
cargo run -p nsrl-train -- \
  --mode mini-transformer-mlp \
  --tokens data/processed/wiki-bard-corpus.tokens.u8 \
  --seq-len 4 \
  --stride 73929 \
  --window-offset 0 \
  --batch-windows 2 \
  --max-windows 4096 \
  --epochs 1 \
  --lr-shift 18 \
  --mlp-lr-shift 17 \
  --embed-lr-shift 14 \
  --attention-lr-shift 24 \
  --attention-qk-lr-shift 18 \
  --model-out data/processed/wiki-bard-mini-transformer-actual-spread4096-fullacc-recommended.nsrlmt \
  --trace data/processed/wiki-bard-mini-transformer-actual-spread4096-fullacc-recommended.trace.jsonl
```

Latest recorded full-accumulator `--batch-windows 2 --lr-shift 18
--mlp-lr-shift 17 --embed-lr-shift 14 --attention-lr-shift 24
--attention-qk-lr-shift 18` metrics: 4096 examined windows, 4096 accepted
updates, 2048 accepted batches, 2048 applications for each accumulator family,
0 rollbacks/rejected windows, `final_invalid_forward_count = 0`,
probability-error delta `-23627423`, final accuracy `140` per mille, median
release CLI training time `542.50 ms`, and median deterministic sampled
generation time `4.21 ms` for 256 new bytes.

Accumulator tuning notes from the same corpus slice:

| batch windows | output | MLP | attention | QK | embedding | rollbacks | probability-error delta | final accuracy per mille | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 18 | 17 | 24 | 18 | 14 | 0 | -23627423 | 140 | Current full-accumulator default. |
| 2 | 18 | 17 | 25 | 19 | 15 | 0 | -10961548 | 33 | Attention mostly quantizes to zero; collapses to `r`. |
| 2 | 17 | 16 | 24 | 18 | 14 | 0 | -11143491 | 45 | Stronger head/MLP update, collapses to `e`. |
| 2 | 18 | 17 | 23 | 17 | 14 | 0 | -3720895 | 5 | More diverse sample, much worse objective. |
| 4 | 16 | 15 | 23 | 17 | 13 | 0 | -5821906 | 24 | Larger batch still over-smooths this tiny model. |

This remains a stability/optimization proof, not a language-quality claim. The
current sampled suffix uses 13 distinct characters with a longest same-character
run of 9. It improves measured accuracy over the MLP-only accumulator run, but
the raw sample is visibly space-biased, so decode hygiene and longer-context
training are still open work.

For the 8192-window spread sweep, use stride `36965`; the older `73929` stride
only yields 4097 windows across this corpus. The strongest zero-rollback
8192-window batch-2 setting found so far lowers the attention and embedding
shifts:

```sh
cargo run -p nsrl-train -- \
  --mode mini-transformer-mlp \
  --tokens data/processed/wiki-bard-corpus.tokens.u8 \
  --seq-len 4 \
  --stride 36965 \
  --window-offset 0 \
  --batch-windows 2 \
  --max-windows 8192 \
  --epochs 1 \
  --lr-shift 18 \
  --mlp-lr-shift 17 \
  --embed-lr-shift 13 \
  --attention-lr-shift 22 \
  --attention-qk-lr-shift 16 \
  --model-out data/processed/wiki-bard-mini-transformer-actual-spread8192s36965-fullacc-user_lowattn-batch2-out18-mlp17-attn22-qk16-emb13.nsrlmt \
  --trace data/processed/wiki-bard-mini-transformer-actual-spread8192s36965-fullacc-user_lowattn-batch2-out18-mlp17-attn22-qk16-emb13.trace.jsonl
```

8192-window full-accumulator notes:

| batch windows | output | MLP | attention | QK | embedding | rollbacks | probability-error delta | final accuracy per mille | sample behavior |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 18 | 17 | 22 | 16 | 13 | 0 | -34848734 | 88 | 12 distinct suffix chars, longest run 10. |
| 2 | 18 | 17 | 24 | 18 | 14 | 0 | -12196928 | 31 | Collapses toward `4/u`. |
| 4 | 18 | 17 | 22 | 16 | 13 | 0 | -7597 | 13 | Flatlines; deltas mostly zero. |
| 4 | 17 | 16 | 21 | 15 | 12 | 0 | -18072273 | 41 | Moves without rollback, but weaker than batch 2. |
| 4 | 16 | 15 | 20 | 14 | 12 | 0 | -40245733 | 79 | Better objective, but collapses toward `i`. |

Context-length probes show that this model is no longer blocked by the forward
runtime. At `seq_len=8`, the trainer can run but collapses into a repeated byte;
at `seq_len=16`, the same 8192-window lane keeps zero rollbacks and starts
forming repeated short fragments; at `seq_len=32`, the current optimizer falls
back into low-accuracy loops unless the attention/MLP shifts are cooled down.
The guarded `NSRLMT3` trainer serializes learned absolute i16 position
embeddings and can reject invalid batch-gradient candidates before saving a
model. On the repeated ascii-lower Shakespeare smoke corpus, learned positions
are a clear improvement over the fixed-position lane:

| seq_len | batch | output | MLP | attention V/O | QK | embedding | epochs | V/O error feedback | rollbacks | invalid forwards | accuracy per mille | probability-error delta | sample behavior |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 32 fixed position | 4 | 18 | 17 | 24 | 18 | 12 | 5 | off | 9788 | 585 | 47 | -44489176 | Invalid final eval windows; `iii/siii` loop. |
| 32 fixed position, cool | 4 | 18 | 18 | 25 | 19 | 13 | 5 | off | 0 | 0 | 95 | -50601075 | Stable `et/ee/uuu` loop. |
| 32 learned position | 4 | 18 | 17 | 24 | 18 | 12 | 5 | off | 0 | 0 | 357 | -163973010 | Varied fragments such as `tat`, `to`, `be`, `eita`; still not coherent. |
| 32 learned position, restored | 4 | 18 | 17 | 24 | 18 | 12 | 5 | off | 0 | 0 | 357 | -163973010 | Current-code reproduction in 35.0 s, 67 MB peak RSS; fragments such as `he`, `is`, `to`, `th`. |
| 32 learned position | 4 | 18 | 17 | 24 | 18 | 12 | 5 | on | 0 | 0 | 142 | -76478673 | V/O moves, but generation collapses toward `uoe`. |
| 32 learned position V/O continuation | 4 | 30 | 30 | 24 | 30 | 30 | 2 | on | 0 | 0 | 119 | +39878372 | Loaded the 357 per-mille model with `--model`; slow V/O residual feedback moves V/O but collapses greedy decode to `o`. |
| 32 learned position V/O guarded continuation | 4 | 30 | 30 | 24 | 30 | 30 | 2 | on + loss guard | 410 | 0 | 237 | +11161612 | `--reject-loss-regression` rejects 410 local regressions, but accepted local updates still worsen the slice and collapse toward spaces. |
| 32 learned position V/O oracle, 64 windows | 4 | 30 | 30 | oracle | 30 | 30 | 1 | oracle + strict loss guard | 15 | 0 | 437 | -198235 | Discrete V/O coordinate oracle accepts 1/16 batches, moves V by 2 L1, and improves configured probability error. |
| 32 learned position V/O oracle, 256 windows | 4 | 30 | 30 | oracle | 30 | 30 | 1 | oracle + strict loss guard | 61 | 0 | 308 | -53141 | Accepts 3/64 batches and moves V by 6 L1; loss improves, but argmax accuracy and generation do not. |
| 64 learned position probe | 4 | 18 | 17 | 24 | 18 | 12 | 3 | off | 0 | 0 | 143 | -18418212 | Stable longer context, but short probe repeats `trn/ttt`; context alone is not the bottleneck. |
| 32 learned position | 4 | 18 | 17 | 24 | 18 | 12 | 10 | off | 0 | 0 | 285 | -157304301 | Over-trains into `to/to/to` loops. |

Attention V/O updates have a sharp threshold with the current shared
`--attention-lr-shift`: shifts 22-24 leave V/O deltas at zero while preserving
the best score; shift 21 starts moving V/O but collapses accuracy to 23 per
mille, and shift 20 moves V/O aggressively but only reaches 143 per mille.
`--attention-vo-error-feedback` preserves sub-threshold V/O update residuals
across batches and proves the path can move at shift 24, but the measured smoke
run drops to 142 per mille and repeats `uoe`, so the flag is experimental rather
than the current default. Continuation training from the best `seq_len=32`
artifact confirms this is not just an early-training interaction: freezing the
other layers at shift 30 while applying V/O residual feedback at shift 24 moves
V/O by 32778 L1, worsens probability error by +39878372, and collapses greedy
decode to `o`. Adding `--reject-loss-regression` proves the objective guard can
reject bad local candidate batches, but the run still worsens the broader slice
and collapses toward spaces, so the V/O attention backward path should be
treated as an implementation risk rather than a hyperparameter problem. The
independent `--attention-vo-oracle` path removes that hand-written V/O gradient
from the update loop and accepts only exact discrete coordinate moves that
improve the configured probability objective. It can find small V updates that
improve loss, but the improvement is sparse and not yet generatively useful.
Longer context is therefore necessary but not sufficient; the next neural
bottleneck is better gradient scaling for the value/output attention path and
RMSNorm-aware gradient flow, not the runtime window.

The best current text-shaped `seq_len=16` setting is:

```sh
cargo run -p nsrl-train -- \
  --mode mini-transformer-mlp \
  --tokens data/processed/wiki-bard-corpus.tokens.u8 \
  --seq-len 16 \
  --stride 36965 \
  --window-offset 0 \
  --batch-windows 2 \
  --max-windows 8192 \
  --epochs 1 \
  --lr-shift 18 \
  --mlp-lr-shift 17 \
  --embed-lr-shift 12 \
  --attention-lr-shift 22 \
  --attention-qk-lr-shift 16 \
  --model-out data/processed/wiki-bard-mini-transformer-seq16-fullacc-emb12.nsrlmt \
  --trace data/processed/wiki-bard-mini-transformer-seq16-fullacc-emb12.trace.jsonl
```

| seq len | batch windows | output | MLP | attention | QK | embedding | rollbacks | probability-error delta | final accuracy per mille | sample behavior |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 2 | 18 | 17 | 22 | 16 | 13 | 0 | -16269795 | 47 | Collapses into a long `g` run. |
| 16 | 2 | 18 | 17 | 22 | 16 | 13 | 0 | -46717136 | 105 | Better objective, but drifts toward space/`e`. |
| 16 | 2 | 18 | 17 | 22 | 16 | 12 | 0 | -17662830 | 39 | More text-like; 21 suffix chars, longest run 4. |
| 32 | 2 | 18 | 17 | 22 | 16 | 13 | 0 | -10665541 | 27 | Over-contextualized for this trainer; loops around `d/c/p`. |

The trace uses `nsrl.training_mini_transformer_mlp_trace.v1`. Attention and
embeddings are no longer forward-only: byte embedding rows, `Q`, `K`, `V`, and
`O` are updated from cached probabilities, context rows, and the native base-2
softmax derivative.

The saved `.nsrlmt` artifact can then be reloaded for deterministic sampled
byte generation:

```sh
cargo run -p nsrl-train -- \
  --mode mini-transformer-generate \
  --model data/processed/wiki-bard-mini-transformer-actual-spread8192s36965-fullacc-seq16-retune-out18-mlp17-attn22-qk16-emb12.nsrlmt \
  --prompt "To be or not to be, " \
  --max-new-tokens 180 \
  --decode sample \
  --sample-seed 3 \
  --top-k 8 \
  --printable-only \
  --repeat-window 32 \
  --repeat-penalty-shift 1 \
  --max-repeat-run 3 \
  --text-out data/processed/wiki-bard-mini-transformer-seq16_emb12_epoch1-textsafe-top8-seed3-rw32-rps1-run3.txt \
  --trace data/processed/wiki-bard-mini-transformer-seq16_emb12_epoch1-textsafe-top8-seed3-rw32-rps1-run3.trace.jsonl
```

The generation trace uses `nsrl.mini_transformer_generation_trace.v1` and
records full-model, embedding, attention, MLP, and output-head hashes from the
loaded artifact. The text output file writes the prompt plus generated bytes as
a plain transcript; language quality is still a training target, not a trace
claim.

Text-safe raw-corpus neural decode probe:

```text
To be or not to be, haihh7rihsuieatersitRReaies 2tsah:triht e hW eatrarWl Ne haast7hrasa7 */ttrhhsrt/7h ha NJiasir etaWlaNe hsstar estrtrt ht h7saa hWet/rash haWl/s77trtrat er  st7hrsa :biie haiha2tah
```

The `--printable-only --repeat-window 32 --repeat-penalty-shift 1
--max-repeat-run 3` decode lane removes control bytes and hard repeat collapse.
Across a small seed sweep it holds the longest same-character run to 2-3 bytes
and keeps the generated suffix around 79% alphabetic bytes with 11-12% spaces.
This is a decode-quality improvement, not a language-quality claim: the model is
still producing learned byte fragments, not coherent prose.

The haiku experiments in `~/develop/crlplrimes` add one more useful decode
lesson: legal-action gates and corpus adjacency priors can make a weak neural
byte model stay inside the language lane without mutating model weights. The
Wiki-Bard generator now supports the same idea:

```sh
cargo run -p nsrl-train -- \
  --mode mini-transformer-generate \
  --model data/processed/wiki-bard-mini-transformer-wide16-poslearn-seq32-coherent-b4-hot.nsrlmt \
  --tokens data/processed/wiki-bard-corpus-ascii-lower.tokens.u8 \
  --tokenizer ascii-lower \
  --prompt "to be or not to be, " \
  --max-new-tokens 180 \
  --decode sample \
  --sample-seed 3 \
  --top-k 8 \
  --ascii-lower-only \
  --repeat-window 32 \
  --repeat-penalty-shift 1 \
  --max-repeat-run 3 \
  --corpus-prior \
  --strict-adjacency \
  --text-out data/processed/wiki-bard-mini-transformer-prior-decode.txt \
  --trace data/processed/wiki-bard-mini-transformer-prior-decode.trace.jsonl
```

`--corpus-prior` reranks candidates with an integer bigram prior derived from
`--tokens`; `--strict-adjacency` masks successors that never followed the
previous byte in that corpus. The generation trace records the prior corpus
hash plus per-step rejection counts, so the effect is auditable instead of
hidden inside the sampler.

Ascii-lower curriculum probes are also not coherent yet. The spread
8192-window run is stable with zero rollbacks but drifts into `du/dum` loops;
the contiguous Shakespeare curriculum can move weights with stronger shifts,
but still collapses into short cycles such as `...h...h` or `r r r`. This
narrows the problem: the data lane can be clean, and the integer optimizer can
move every subsystem, but the current one-block `NSRLMT3` neural trainer
(`d_model=16`, `hidden_dim=32`, learned absolute position embeddings) does not
yet have enough capacity/gradient quality to speak.

As a teacher baseline for what the normalized byte lane should eventually
produce, a deterministic n-gram generator over the first 2MB of
`byte_ascii_lower_text_u8_v1` emits coherent Shakespeare:

```text
shall i compare thee to a summer s day? thou art more lovely and more temperate: rough winds do shake the darling buds of may, and summer s lease hath all too short a date: sometime too hot the eye of heaven shines, and often is his gold complexion dimmed, and every fair from
```

That trace is `data/processed/wiki-bard-ngram-teacher-shall.trace.jsonl`. It is
explicitly marked as `not_neural`; it is a curriculum/debugging target, not a
replacement for the integer Transformer.

## Current Limits

This is the deterministic data lane, not the final tokenizer or trainer:

- SimpleWiki XML must be decompressed outside the crate.
- Wiki markup cleaning is intentionally minimal.
- Output is plain text with source/page markers.
- The byte tokenizers are intentionally not BPE or WordPiece.
- The lexeme tokenizer is deterministic and stable, but still corpus-local and
  still leaks source artifacts unless the input corpus is aggressively cleaned.
- The current byte trainers/generators are baseline output-head and learned
  embedding models, not the final Transformer.
- The current lexeme trainer freezes concept embeddings and trains a shallow
  output head; it is phrase scaffolding, not deep grammar or spelling.
- The mini Transformer training loop updates byte embeddings, the output head,
  gated MLP, and attention `Q`/`K`/`V`/`O`, but still has only one block and one
  attention head.
