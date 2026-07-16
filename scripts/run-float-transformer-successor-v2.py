#!/usr/bin/env python3

"""Train and evaluate the genuine float-transformer baseline for successor-v2.

The model is deliberately small enough for a repository replay. It is still a
real causal transformer: learned byte embeddings, RMS normalization, learned
Q/K/V/O projections, causal recurrent linear attention, a gated MLP, residual
connections, and a learned float32 output head are all trained by SGD. The
published comparison consumes the emitted Q8 logits through the same canonical
integer NLL evaluator as every other system.
"""

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np


FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
FNV_MASK = 0xFFFFFFFFFFFFFFFF
MAGIC = b"NSRLFT2\n"
TOKENIZER = "byte_identity_u8_v1"
VOCAB = 256


def fnv1a(data, value=FNV_OFFSET):
    for byte in data:
        value ^= int(byte)
        value = (value * FNV_PRIME) & FNV_MASK
    return value


def hex64(value):
    return f"0x{value:016x}"


def proof_dataset_hash(train, evaluation):
    return fnv1a(evaluation, fnv1a(b"\xff", fnv1a(train)))


def tensor_digests(model):
    fnv = FNV_OFFSET
    sha = hashlib.sha256()
    for name in sorted(model):
        name_bytes = name.encode("utf8") + b"\0"
        value = np.ascontiguousarray(model[name], dtype=np.float32).tobytes()
        fnv = fnv1a(name_bytes, fnv)
        fnv = fnv1a(value, fnv)
        sha.update(name_bytes)
        sha.update(value)
    return fnv, sha.hexdigest()


def initialize(seed, d_model, heads, hidden_dim):
    if d_model % heads:
        raise ValueError("d_model must be divisible by heads")
    rng = np.random.default_rng(seed)

    def normal(shape, scale=0.025):
        return rng.normal(0.0, scale, shape).astype(np.float32)

    return {
        "embeddings": normal((VOCAB, d_model), 0.04),
        "attention_rms": np.ones(d_model, dtype=np.float32),
        "q": normal((d_model, d_model), 0.015),
        "k": normal((d_model, d_model), 0.015),
        "v": normal((d_model, d_model)),
        "o": normal((d_model, d_model)),
        "mlp_rms": np.ones(d_model, dtype=np.float32),
        "up": normal((hidden_dim, d_model)),
        "gate": normal((hidden_dim, d_model)),
        "down": normal((d_model, hidden_dim)),
        "final_rms": np.ones(d_model, dtype=np.float32),
        "output": normal((VOCAB, d_model)),
        "bias": np.zeros(VOCAB, dtype=np.float32),
    }


def rms_forward(x, gamma):
    inv = np.reciprocal(
        np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + np.float32(2 ** -30))
    )
    return x * inv * gamma, (x, gamma, inv)


def rms_backward(dy, cache):
    x, gamma, inv = cache
    scaled = dy * gamma
    dot = np.sum(scaled * x, axis=-1, keepdims=True)
    dx = scaled * inv - x * (inv ** 3) * dot / x.shape[-1]
    axes = tuple(range(dy.ndim - 1))
    return dx, np.sum(dy * x * inv, axis=axes)


def hard_silu(x):
    ramp = np.clip((x + 2) / 4, 0, 1)
    derivative = np.where(x <= -2, 0, np.where(x >= 2, 1, (x + 1) / 2))
    return x * ramp, derivative.astype(np.float32)


def attention_forward(q, k, v, heads):
    seq_len, d_model = q.shape
    head_dim = d_model // heads
    qh = q.reshape(seq_len, heads, head_dim)
    kh = k.reshape(seq_len, heads, head_dim)
    vh = v.reshape(seq_len, heads, head_dim)
    positive_q = qh + np.float32(1.000030517578125)
    positive_k = kh + np.float32(1.000030517578125)
    output = np.empty_like(vh)
    state = np.zeros((heads, head_dim, head_dim), dtype=np.float32)
    key_sums = np.zeros((heads, head_dim), dtype=np.float32)
    for token in range(seq_len):
        state += np.einsum(
            "hi,hj->hij", positive_k[token], vh[token], optimize=True
        )
        key_sums += positive_k[token]
        numerator = np.einsum(
            "hi,hij->hj", positive_q[token], state, optimize=True
        )
        denominator = np.einsum(
            "hi,hi->h", positive_q[token], key_sums, optimize=True
        )
        output[token] = numerator / denominator[:, None]
    return output.reshape(seq_len, d_model), (
        positive_q,
        positive_k,
        vh,
        output,
        heads,
    )


def attention_backward(doutput, cache):
    positive_q, positive_k, values, output, heads = cache
    seq_len, _, head_dim = positive_q.shape
    dout = doutput.reshape(seq_len, heads, head_dim)
    dq = np.zeros_like(positive_q)
    dk = np.zeros_like(positive_k)
    dv = np.zeros_like(values)
    states = np.empty((seq_len, heads, head_dim, head_dim), dtype=np.float32)
    sums = np.empty((seq_len, heads, head_dim), dtype=np.float32)
    state = np.zeros((heads, head_dim, head_dim), dtype=np.float32)
    key_sum = np.zeros((heads, head_dim), dtype=np.float32)
    for token in range(seq_len):
        state += np.einsum(
            "hi,hj->hij", positive_k[token], values[token], optimize=True
        )
        key_sum += positive_k[token]
        states[token] = state
        sums[token] = key_sum
    dstate = np.zeros_like(state)
    dsum = np.zeros_like(key_sum)
    for token in reversed(range(seq_len)):
        denominator = np.einsum(
            "hi,hi->h", positive_q[token], sums[token], optimize=True
        )
        dnumerator = dout[token] / denominator[:, None]
        ddenominator = (
            -np.einsum("hi,hi->h", dout[token], output[token], optimize=True)
            / denominator
        )
        dq[token] += np.einsum(
            "hj,hij->hi", dnumerator, states[token], optimize=True
        )
        dq[token] += ddenominator[:, None] * sums[token]
        dstate += np.einsum(
            "hi,hj->hij", positive_q[token], dnumerator, optimize=True
        )
        dsum += ddenominator[:, None] * positive_q[token]
        dk[token] += np.einsum("hij,hj->hi", dstate, values[token], optimize=True)
        dk[token] += dsum
        dv[token] += np.einsum(
            "hi,hij->hj", positive_k[token], dstate, optimize=True
        )
    shape = (seq_len, heads * head_dim)
    return dq.reshape(shape), dk.reshape(shape), dv.reshape(shape)


def forward(model, tokens, heads):
    x = model["embeddings"][tokens]
    attention_input, attention_rms = rms_forward(x, model["attention_rms"])
    q = attention_input @ model["q"].T
    k = attention_input @ model["k"].T
    v = attention_input @ model["v"].T
    context, attention_cache = attention_forward(q, k, v, heads)
    attention_output = context @ model["o"].T
    x1 = x + attention_output
    mlp_input, mlp_rms = rms_forward(x1, model["mlp_rms"])
    up = mlp_input @ model["up"].T
    gate = mlp_input @ model["gate"].T
    activation, activation_derivative = hard_silu(gate)
    gated = up * activation
    final_sequence = x1 + gated @ model["down"].T
    features, final_rms = rms_forward(final_sequence[-1], model["final_rms"])
    logits = model["output"] @ features + model["bias"]
    cache = {
        "tokens": tokens,
        "attention_input": attention_input,
        "attention_rms": attention_rms,
        "context": context,
        "attention_cache": attention_cache,
        "x1": x1,
        "mlp_input": mlp_input,
        "mlp_rms": mlp_rms,
        "up": up,
        "activation": activation,
        "activation_derivative": activation_derivative,
        "gated": gated,
        "final_sequence": final_sequence,
        "features": features,
        "final_rms": final_rms,
    }
    return logits, cache


def loss_gradient(logits, target):
    shifted = logits - np.max(logits)
    probabilities = np.exp2(shifted)
    probabilities /= np.sum(probabilities)
    loss_bits = -np.log2(max(float(probabilities[target]), 2 ** -32))
    probabilities[target] -= 1
    return loss_bits, probabilities.astype(np.float32)


def backward(model, cache, dlogits, heads, gradients):
    for value in gradients.values():
        value.fill(0)
    gradients["output"] = np.outer(dlogits, cache["features"])
    gradients["bias"] = dlogits
    dfeatures = dlogits @ model["output"]
    dlast, gradients["final_rms"] = rms_backward(dfeatures, cache["final_rms"])
    dx = np.zeros_like(cache["final_sequence"])
    dx[-1] = dlast

    ddown = dx
    dx1 = dx.copy()
    gradients["down"] = ddown.T @ cache["gated"]
    dgated = ddown @ model["down"]
    dup = dgated * cache["activation"]
    dgate = dgated * cache["up"] * cache["activation_derivative"]
    gradients["up"] = dup.T @ cache["mlp_input"]
    gradients["gate"] = dgate.T @ cache["mlp_input"]
    dmlp = dup @ model["up"] + dgate @ model["gate"]
    dmlp_input, gradients["mlp_rms"] = rms_backward(dmlp, cache["mlp_rms"])
    dx1 += dmlp_input

    gradients["o"] = dx1.T @ cache["context"]
    dcontext = dx1 @ model["o"]
    dq, dk, dv = attention_backward(dcontext, cache["attention_cache"])
    gradients["q"] = dq.T @ cache["attention_input"]
    gradients["k"] = dk.T @ cache["attention_input"]
    gradients["v"] = dv.T @ cache["attention_input"]
    dattn = dq @ model["q"] + dk @ model["k"] + dv @ model["v"]
    drms, gradients["attention_rms"] = rms_backward(dattn, cache["attention_rms"])
    dx1 += drms
    np.add.at(gradients["embeddings"], cache["tokens"], dx1)


def starts(length, context, stride, maximum=None):
    values = list(range(0, length - context, stride))
    return values if maximum is None else values[:maximum]


def train(model, train_bytes, context, stride, maximum, epochs, batch, rate, heads):
    windows = starts(len(train_bytes), context, stride, maximum)
    gradients = {name: np.zeros_like(value) for name, value in model.items()}
    accumulated = {name: np.zeros_like(value) for name, value in model.items()}
    total_loss = 0.0
    updates = 0
    for _ in range(epochs):
        for batch_start in range(0, len(windows), batch):
            batch_starts = windows[batch_start : batch_start + batch]
            for value in accumulated.values():
                value.fill(0)
            for start in batch_starts:
                tokens = np.frombuffer(
                    train_bytes[start : start + context], dtype=np.uint8
                ).astype(np.int64)
                target = train_bytes[start + context]
                logits, cache = forward(model, tokens, heads)
                loss, dlogits = loss_gradient(logits, target)
                total_loss += loss
                backward(model, cache, dlogits, heads, gradients)
                for name in model:
                    accumulated[name] += gradients[name]
            scale = np.float32(rate / len(batch_starts))
            for name in model:
                update = np.clip(
                    accumulated[name] * scale,
                    np.float32(-0.01),
                    np.float32(0.01),
                )
                model[name] -= update
            updates += 1
    if not all(np.all(np.isfinite(value)) for value in model.values()):
        raise ValueError("float transformer produced a non-finite parameter")
    return {
        "windows": len(windows),
        "epochs": epochs,
        "optimizer_steps": updates,
        "mean_training_nll_bits": total_loss / (len(windows) * epochs),
    }


def evaluate(model, evaluation, context, heads):
    window_starts = starts(len(evaluation), context, 1)
    q8_logits = np.empty((len(window_starts), VOCAB), dtype="<i4")
    mistakes = 0
    total_float_nll = 0.0
    for row, start in enumerate(window_starts):
        tokens = np.frombuffer(
            evaluation[start : start + context], dtype=np.uint8
        ).astype(np.int64)
        target = evaluation[start + context]
        logits, _ = forward(model, tokens, heads)
        loss, _ = loss_gradient(logits, target)
        total_float_nll += loss
        mistakes += int(np.argmax(logits) != target)
        q8_logits[row] = np.rint(logits * np.float32(256.0)).astype(np.int32)
    return q8_logits, {
        "windows": len(window_starts),
        "mistakes": mistakes,
        "total_float_nll_millibits": round(total_float_nll * 1000),
        "mean_float_nll_millibits": round(total_float_nll * 1000 / len(window_starts)),
    }


def write_logits(path, q8_logits, context, dataset_hash, model_hash):
    header = MAGIC + struct.pack(
        "<IIIIQQ",
        1,
        context,
        q8_logits.shape[0],
        q8_logits.shape[1],
        dataset_hash,
        model_hash,
    )
    body = header + np.ascontiguousarray(q8_logits, dtype="<i4").tobytes()
    replay_hash = fnv1a(body)
    Path(path).write_bytes(body + struct.pack("<Q", replay_hash))
    return replay_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out-logits", required=True)
    parser.add_argument("--out-trace", required=True)
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--train-stride", type=int, default=19)
    parser.add_argument("--max-train-windows", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-windows", type=int, default=16)
    parser.add_argument("--learning-rate-millionths", type=int, default=3000)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    train_bytes = Path(args.train).read_bytes()
    eval_bytes = Path(args.eval).read_bytes()
    dataset_hash = proof_dataset_hash(train_bytes, eval_bytes)
    if len(eval_bytes) - args.context != 5_896:
        raise ValueError("float transformer evaluation must contain exactly 5,896 targets")

    model = initialize(args.seed, args.d_model, args.heads, args.hidden_dim)
    initial_model_hash, initial_sha256 = tensor_digests(model)
    training = train(
        model,
        train_bytes,
        args.context,
        args.train_stride,
        args.max_train_windows,
        args.epochs,
        args.batch_windows,
        args.learning_rate_millionths / 1_000_000,
        args.heads,
    )
    final_model_hash, final_sha256 = tensor_digests(model)
    logits, evaluation = evaluate(
        model, eval_bytes, args.context, args.heads
    )
    logits_path = Path(args.out_logits)
    trace_path = Path(args.out_trace)
    logits_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    logits_artifact_hash = write_logits(
        logits_path, logits, args.context, dataset_hash, final_model_hash
    )
    trace = {
        "schema": "nsrl.float_transformer_successor.v2",
        "contract": "integer-transformer-successor-v2",
        "bindings": {
            "dataset_hash": hex64(dataset_hash),
            "targets": len(eval_bytes) - args.context,
            "tokenizer": TOKENIZER,
            "tokenizer_hash": hex64(fnv1a(TOKENIZER.encode("utf8"))),
        },
        "architecture": {
            "kind": "genuine_float_transformer",
            "dtype": "float32",
            "vocab": VOCAB,
            "context": args.context,
            "d_model": args.d_model,
            "heads": args.heads,
            "layers": 1,
            "hidden_dim": args.hidden_dim,
            "attention": "causal_recurrent_linear_qkvo",
            "mlp": "gated_hard_silu",
            "residual_connections": 2,
        },
        "training": {
            **training,
            "optimizer": "float32_sgd",
            "batch_windows": args.batch_windows,
            "learning_rate_millionths": args.learning_rate_millionths,
            "train_stride": args.train_stride,
            "trained_parameter_groups": sorted(model),
        },
        "model": {
            "initial_hash": hex64(initial_model_hash),
            "final_hash": hex64(final_model_hash),
            "initial_sha256": initial_sha256,
            "final_sha256": final_sha256,
            "parameters": sum(value.size for value in model.values()),
        },
        "evaluation": {
            **evaluation,
            "partition": "eval",
            "stride": 1,
            "q8_logits_artifact_hash": hex64(logits_artifact_hash),
        },
        "assistance": {
            "suffix_memory": False,
            "retrieval_assistance": False,
            "routing_oracle": False,
        },
    }
    trace_path.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "trace": str(trace_path),
                "model_hash": hex64(final_model_hash),
                "targets": evaluation["windows"],
                "mistakes": evaluation["mistakes"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
