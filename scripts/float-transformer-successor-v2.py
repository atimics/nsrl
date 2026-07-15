#!/usr/bin/env python3
"""Deterministic NumPy float32 causal Transformer baseline for successor-v2."""

import argparse
import json
import math
import os
import struct
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np


SCHEMA = "nsrl.float_transformer_eval.v1"
CONTRACT = "integer-transformer-successor-v2"
DATASET_HASH = "0x8fe7b86378f81951"
MAGIC = b"NSRLFT1\n"
VOCAB = 256
CONTEXT = 64
D_MODEL = 32
FF_DIM = 64
SEED = 20260715
MAX_WINDOWS = 1024
EPOCHS = 12
BATCH_SIZE = 32
LEARNING_RATE = np.float32(0.003)
ADAM_BETA1 = np.float32(0.9)
ADAM_BETA2 = np.float32(0.999)
ADAM_EPSILON = np.float32(1e-8)
GRAD_CLIP = np.float32(1.0)
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
FNV_MASK = 0xFFFFFFFFFFFFFFFF
PARAMETER_ORDER = (
    "token_embedding",
    "position_embedding",
    "wq",
    "wk",
    "wv",
    "wo",
    "w1",
    "w2",
    "wout",
    "bout",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--logits-out", required=True)
    parser.add_argument("--trace-out", required=True)
    parser.add_argument("--runner-hash", required=True)
    return parser.parse_args()


def fnv64(data):
    value = FNV_OFFSET
    for byte in data:
        value = ((value ^ byte) * FNV_PRIME) & FNV_MASK
    return f"0x{value:016x}"


def initialize_model():
    rng = np.random.default_rng(SEED)

    def normal(shape, fan_in):
        scale = np.float32(1.0 / math.sqrt(fan_in))
        return (rng.standard_normal(shape).astype(np.float32) * scale).astype(np.float32)

    return {
        "token_embedding": normal((VOCAB, D_MODEL), D_MODEL),
        "position_embedding": normal((CONTEXT, D_MODEL), D_MODEL),
        "wq": normal((D_MODEL, D_MODEL), D_MODEL),
        "wk": normal((D_MODEL, D_MODEL), D_MODEL),
        "wv": normal((D_MODEL, D_MODEL), D_MODEL),
        "wo": normal((D_MODEL, D_MODEL), D_MODEL),
        "w1": normal((D_MODEL, FF_DIM), D_MODEL),
        "w2": normal((FF_DIM, D_MODEL), FF_DIM),
        "wout": normal((D_MODEL, VOCAB), D_MODEL),
        "bout": np.zeros((VOCAB,), dtype=np.float32),
    }


def softmax(values, axis=-1):
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exponentials = np.exp(shifted).astype(np.float32)
    return (exponentials / np.sum(exponentials, axis=axis, keepdims=True)).astype(np.float32)


def forward(model, contexts, retain_cache=False):
    x = (model["token_embedding"][contexts] + model["position_embedding"][None, :, :]).astype(
        np.float32
    )
    x_last = x[:, -1, :]
    q = x_last @ model["wq"]
    k = x @ model["wk"]
    v = x @ model["wv"]
    scores = np.einsum("bd,btd->bt", q, k, optimize=False).astype(np.float32)
    scores *= np.float32(1.0 / math.sqrt(D_MODEL))
    attention = softmax(scores)
    attended = np.einsum("bt,btd->bd", attention, v, optimize=False).astype(np.float32)
    y = (x_last + attended @ model["wo"]).astype(np.float32)
    ff_pre = y @ model["w1"]
    ff_hidden = np.tanh(ff_pre).astype(np.float32)
    z = (y + ff_hidden @ model["w2"]).astype(np.float32)
    logits = (z @ model["wout"] + model["bout"]).astype(np.float32)
    if not retain_cache:
        return logits
    return logits, {
        "contexts": contexts,
        "x": x,
        "x_last": x_last,
        "q": q,
        "k": k,
        "v": v,
        "attention": attention,
        "attended": attended,
        "y": y,
        "ff_hidden": ff_hidden,
        "z": z,
    }


def loss_and_gradients(model, contexts, targets):
    logits, cache = forward(model, contexts, retain_cache=True)
    probabilities = softmax(logits)
    batch = contexts.shape[0]
    loss = -np.log(np.maximum(probabilities[np.arange(batch), targets], np.float32(1e-30))).mean()
    dlogits = probabilities
    dlogits[np.arange(batch), targets] -= np.float32(1.0)
    dlogits *= np.float32(1.0 / batch)

    gradients = {}
    gradients["wout"] = cache["z"].T @ dlogits
    gradients["bout"] = np.sum(dlogits, axis=0)
    dz = dlogits @ model["wout"].T
    gradients["w2"] = cache["ff_hidden"].T @ dz
    dff_hidden = dz @ model["w2"].T
    dff_pre = dff_hidden * (np.float32(1.0) - cache["ff_hidden"] * cache["ff_hidden"])
    gradients["w1"] = cache["y"].T @ dff_pre
    dy = dz + dff_pre @ model["w1"].T
    gradients["wo"] = cache["attended"].T @ dy
    dattended = dy @ model["wo"].T
    dattention = np.einsum("bd,btd->bt", dattended, cache["v"], optimize=False)
    dv = cache["attention"][:, :, None] * dattended[:, None, :]
    dscores = cache["attention"] * (
        dattention - np.sum(dattention * cache["attention"], axis=1, keepdims=True)
    )
    scale = np.float32(1.0 / math.sqrt(D_MODEL))
    dq = np.einsum("bt,btd->bd", dscores, cache["k"], optimize=False) * scale
    dk = dscores[:, :, None] * cache["q"][:, None, :] * scale
    flat_x = cache["x"].reshape(-1, D_MODEL)
    gradients["wq"] = cache["x_last"].T @ dq
    gradients["wk"] = flat_x.T @ dk.reshape(-1, D_MODEL)
    gradients["wv"] = flat_x.T @ dv.reshape(-1, D_MODEL)
    dx = dk @ model["wk"].T + dv @ model["wv"].T
    dx[:, -1, :] += dq @ model["wq"].T + dy
    gradients["position_embedding"] = np.sum(dx, axis=0)
    gradients["token_embedding"] = np.zeros_like(model["token_embedding"])
    np.add.at(gradients["token_embedding"], contexts.reshape(-1), dx.reshape(-1, D_MODEL))

    return float(loss), {name: gradients[name].astype(np.float32) for name in PARAMETER_ORDER}


def train_model(train_bytes):
    if len(train_bytes) <= CONTEXT:
        raise ValueError("training corpus is shorter than the context")
    all_starts = np.arange(len(train_bytes) - CONTEXT, dtype=np.int64)
    selected = np.linspace(0, len(all_starts) - 1, MAX_WINDOWS, dtype=np.int64)
    starts = all_starts[selected]
    model = initialize_model()
    first = {name: value.copy() for name, value in model.items()}
    moments = {name: np.zeros_like(value) for name, value in model.items()}
    variances = {name: np.zeros_like(value) for name, value in model.items()}
    step = 0
    first_loss = None
    final_loss = None
    for epoch in range(EPOCHS):
        epoch_order = np.roll(starts, epoch * 17)
        for offset in range(0, len(epoch_order), BATCH_SIZE):
            batch_starts = epoch_order[offset : offset + BATCH_SIZE]
            contexts = np.stack(
                [train_bytes[start : start + CONTEXT] for start in batch_starts]
            ).astype(np.int64)
            targets = train_bytes[batch_starts + CONTEXT].astype(np.int64)
            loss, gradients = loss_and_gradients(model, contexts, targets)
            if first_loss is None:
                first_loss = loss
            final_loss = loss
            squared_norm = sum(
                float(np.sum(gradient.astype(np.float64) ** 2)) for gradient in gradients.values()
            )
            norm = math.sqrt(squared_norm)
            gradient_scale = np.float32(min(1.0, float(GRAD_CLIP) / max(norm, 1e-12)))
            step += 1
            correction1 = np.float32(1.0 - float(ADAM_BETA1) ** step)
            correction2 = np.float32(1.0 - float(ADAM_BETA2) ** step)
            for name in PARAMETER_ORDER:
                gradient = gradients[name] * gradient_scale
                moments[name] = ADAM_BETA1 * moments[name] + (np.float32(1.0) - ADAM_BETA1) * gradient
                variances[name] = ADAM_BETA2 * variances[name] + (
                    np.float32(1.0) - ADAM_BETA2
                ) * gradient * gradient
                mean_hat = moments[name] / correction1
                variance_hat = variances[name] / correction2
                model[name] -= LEARNING_RATE * mean_hat / (np.sqrt(variance_hat) + ADAM_EPSILON)
                model[name] = model[name].astype(np.float32)
    moved = {
        name: int(np.count_nonzero(model[name].view(np.uint32) != first[name].view(np.uint32)))
        for name in PARAMETER_ORDER
    }
    if any(count == 0 for count in moved.values()):
        raise RuntimeError(f"float transformer left a parameter tensor frozen: {moved}")
    return model, {
        "seed": SEED,
        "max_windows": MAX_WINDOWS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "adam",
        "learning_rate": float(LEARNING_RATE),
        "trained_parameters": "all",
        "steps": step,
        "first_batch_nll_nats": first_loss,
        "final_batch_nll_nats": final_loss,
        "moved_values": moved,
    }


def serialize_model(model):
    header = {
        "schema": "nsrl.float_transformer_model.v1",
        "dtype": "float32-le",
        "parameter_order": [
            {"name": name, "shape": list(model[name].shape)} for name in PARAMETER_ORDER
        ],
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf8")
    payload = b"".join(np.ascontiguousarray(model[name], dtype="<f4").tobytes() for name in PARAMETER_ORDER)
    return MAGIC + struct.pack("<Q", len(header_bytes)) + header_bytes + payload


def evaluate(model, eval_bytes):
    starts = np.arange(0, len(eval_bytes) - CONTEXT, dtype=np.int64)
    q8_parts = []
    q8_mistakes = 0
    float_mistakes = 0
    total_nll_nats = 0.0
    for offset in range(0, len(starts), 128):
        batch_starts = starts[offset : offset + 128]
        contexts = np.stack(
            [eval_bytes[start : start + CONTEXT] for start in batch_starts]
        ).astype(np.int64)
        targets = eval_bytes[batch_starts + CONTEXT].astype(np.int64)
        logits = forward(model, contexts)
        probabilities = softmax(logits)
        total_nll_nats += float(
            -np.log(np.maximum(probabilities[np.arange(len(targets)), targets], np.float32(1e-30))).sum()
        )
        float_mistakes += int(np.count_nonzero(np.argmax(logits, axis=1) != targets))
        q8 = np.rint(logits.astype(np.float64) * (256.0 / math.log(2.0)))
        q8 = np.clip(q8, np.iinfo(np.int32).min + 1, np.iinfo(np.int32).max).astype("<i4")
        q8_mistakes += int(np.count_nonzero(np.argmax(q8, axis=1) != targets))
        q8_parts.append(q8)
    return np.concatenate(q8_parts, axis=0), {
        "float_mistakes": float_mistakes,
        "q8_mistakes": q8_mistakes,
        "float_total_nll_nats": total_nll_nats,
        "float_mean_nll_bits": total_nll_nats / len(starts) / math.log(2.0),
    }


def main():
    args = parse_args()
    if not args.runner_hash.startswith("0x") or len(args.runner_hash) != 18:
        raise ValueError("--runner-hash must be a 64-bit hexadecimal hash")
    train_bytes = np.frombuffer(Path(args.train).read_bytes(), dtype=np.uint8)
    eval_bytes = np.frombuffer(Path(args.eval).read_bytes(), dtype=np.uint8)
    model, training = train_model(train_bytes)
    model_bytes = serialize_model(model)
    model_hash = fnv64(model_bytes)
    q8_logits, evaluation = evaluate(model, eval_bytes)
    targets = len(eval_bytes) - CONTEXT
    if targets != 5896 or q8_logits.shape != (targets, VOCAB):
        raise RuntimeError(f"unexpected evaluation geometry: {targets}, {q8_logits.shape}")
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.model_out).write_bytes(model_bytes)
    Path(args.logits_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.logits_out).write_bytes(np.ascontiguousarray(q8_logits, dtype="<i4").tobytes())
    trace = {
        "schema": SCHEMA,
        "contract": CONTRACT,
        "dataset_hash": DATASET_HASH,
        "targets": targets,
        "context": CONTEXT,
        "stride": 1,
        "model_hash": model_hash,
        "runner_hash": args.runner_hash,
        "architecture": {
            "kind": "causal-float-transformer",
            "dtype": "float32",
            "d_model": D_MODEL,
            "heads": 1,
            "ff_dim": FF_DIM,
            "attention": "scaled-dot-product-softmax",
            "evaluated_query": "last-causal-position",
            "residual_blocks": 2,
        },
        "training": training,
        "evaluation": evaluation,
    }
    Path(args.trace_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.trace_out).write_text(json.dumps(trace, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({"model_hash": model_hash, "targets": targets, **evaluation}, sort_keys=True))


if __name__ == "__main__":
    main()
