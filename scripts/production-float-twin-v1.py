#!/usr/bin/env python3

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np


FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
BOS = 256
EOS = 257


def fnv1a(data):
    value = FNV_OFFSET
    for byte in data:
        value ^= byte
        value = (value * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return value


def take(data, offset, dtype, count):
    dtype = np.dtype(dtype).newbyteorder("<")
    size = dtype.itemsize * count
    end = offset + size
    if end > len(data) - 8:
        raise ValueError("truncated NSRLPM1 tensor")
    return np.frombuffer(data[offset:end], dtype=dtype).copy(), end


def load_integer_model(model_path):
    data = Path(model_path).read_bytes()
    if len(data) < 66 or data[:8] != b"NSRLPM1\n":
        raise ValueError("bad NSRLPM1 header")
    expected = struct.unpack_from("<Q", data, len(data) - 8)[0]
    if fnv1a(data[:-8]) != expected:
        raise ValueError("bad NSRLPM1 checksum")
    version, vocab, d_model, heads, layers, hidden, context = struct.unpack_from("<7I", data, 8)
    if version != 1:
        raise ValueError("unsupported NSRLPM1 version")
    tokenizer_hash, seed = struct.unpack_from("<QQ", data, 36)
    shifts = struct.unpack_from("<6B", data, 52)
    offset = 58
    tensors = {}
    tensors["embeddings"], offset = take(data, offset, np.int16, vocab * d_model)
    tensors["attention_rms"], offset = take(data, offset, np.int16, layers * d_model)
    tensors["mlp_rms"], offset = take(data, offset, np.int16, layers * d_model)
    tensors["final_rms"], offset = take(data, offset, np.int16, d_model)
    for name in ["q", "k", "v", "o"]:
        tensors[name], offset = take(data, offset, np.int8, layers * d_model * d_model)
    tensors["up"], offset = take(data, offset, np.int8, layers * d_model * hidden)
    tensors["gate"], offset = take(data, offset, np.int8, layers * d_model * hidden)
    tensors["down"], offset = take(data, offset, np.int8, layers * hidden * d_model)
    tensors["output"], offset = take(data, offset, np.int16, vocab * d_model)
    tensors["bias"], offset = take(data, offset, np.int32, vocab)
    if offset != len(data) - 8:
        raise ValueError("wrong NSRLPM1 length")

    qkv_shift, o_shift, up_shift, gate_shift, down_shift, output_shift = shifts
    model = {
        "embeddings": (tensors["embeddings"].reshape(vocab, d_model).astype(np.float32) / 32768),
        "attention_rms": (tensors["attention_rms"].reshape(layers, d_model).astype(np.float32) / 32768),
        "mlp_rms": (tensors["mlp_rms"].reshape(layers, d_model).astype(np.float32) / 32768),
        "final_rms": (tensors["final_rms"].astype(np.float32) / 32768),
        "q": tensors["q"].reshape(layers, d_model, d_model).astype(np.float32) / (1 << qkv_shift),
        "k": tensors["k"].reshape(layers, d_model, d_model).astype(np.float32) / (1 << qkv_shift),
        "v": tensors["v"].reshape(layers, d_model, d_model).astype(np.float32) / (1 << qkv_shift),
        "o": tensors["o"].reshape(layers, d_model, d_model).astype(np.float32) / (1 << o_shift),
        "up": tensors["up"].reshape(layers, hidden, d_model).astype(np.float32) / (1 << up_shift),
        "gate": tensors["gate"].reshape(layers, hidden, d_model).astype(np.float32) / (1 << gate_shift),
        "down": tensors["down"].reshape(layers, d_model, hidden).astype(np.float32) / (1 << down_shift),
        "output": tensors["output"].reshape(vocab, d_model).astype(np.float32) * np.float32(2 ** (7 - output_shift)),
        "bias": tensors["bias"].astype(np.float32) / 256,
    }
    config = {
        "vocab_size": vocab,
        "d_model": d_model,
        "heads": heads,
        "layers": layers,
        "hidden_dim": hidden,
        "context_tokens": context,
        "tokenizer_hash": tokenizer_hash,
        "initialization_seed": seed,
        "integer_model_hash": expected,
        "artifact_sha256": hashlib.sha256(data).hexdigest(),
    }
    return model, config


def load_tokens(tokens_path, expected_tokenizer_hash, vocab_size):
    data = Path(tokens_path).read_bytes()
    if len(data) < 24 or data[:8] != b"NSRLTOK1":
        raise ValueError("bad NSRLTOK1 header")
    tokenizer_hash, count = struct.unpack_from("<QQ", data, 8)
    if tokenizer_hash != expected_tokenizer_hash:
        raise ValueError("float twin tokenizer binding mismatch")
    if len(data) != 24 + count * 4:
        raise ValueError("wrong NSRLTOK1 length")
    tokens = np.frombuffer(data, dtype="<u4", offset=24).copy()
    if np.any(tokens >= vocab_size):
        raise ValueError("token exceeds float twin vocabulary")
    return tokens, fnv1a(data[24:])


def document_windows(tokens, context_tokens, max_windows):
    windows = []
    document = []
    active = False
    for token in tokens.tolist():
        if token == BOS:
            document = []
            active = True
        elif token == EOS:
            if active and len(document) > context_tokens:
                for start in range(len(document) - context_tokens):
                    windows.append((np.asarray(document[start:start + context_tokens], dtype=np.int64), document[start + context_tokens]))
                    if len(windows) >= max_windows:
                        return windows
            document = []
            active = False
        elif active:
            document.append(token)
    return windows


def rms_forward(x, gamma):
    inv = np.reciprocal(np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + np.float32(1e-6)))
    return x * inv * gamma, (x, gamma, inv)


def rms_backward(dy, cache):
    x, gamma, inv = cache
    d_model = x.shape[-1]
    scaled = dy * gamma
    dot = np.sum(scaled * x, axis=-1, keepdims=True)
    dx = scaled * inv - x * (inv ** 3) * dot / d_model
    dgamma = np.sum(dy * x * inv, axis=tuple(range(dy.ndim - 1)))
    return dx, dgamma


def hard_silu(x):
    ramp = np.clip((x + 2) / 4, 0, 1)
    value = x * ramp
    derivative = np.where(x <= -2, 0, np.where(x >= 2, 1, (x + 1) / 2))
    return value, derivative.astype(np.float32)


def linear_attention_forward(q, k, v, heads):
    seq_len, d_model = q.shape
    head_dim = d_model // heads
    qh = q.reshape(seq_len, heads, head_dim)
    kh = k.reshape(seq_len, heads, head_dim)
    vh = v.reshape(seq_len, heads, head_dim)
    pq = qh + np.float32(1.000030517578125)
    pk = kh + np.float32(1.000030517578125)
    output = np.empty_like(vh)
    state = np.zeros((heads, head_dim, head_dim), dtype=np.float32)
    key_sums = np.zeros((heads, head_dim), dtype=np.float32)
    for token in range(seq_len):
        state += np.einsum("hi,hj->hij", pk[token], vh[token], optimize=True)
        key_sums += pk[token]
        numerator = np.einsum("hi,hij->hj", pq[token], state, optimize=True)
        denominator = np.einsum("hi,hi->h", pq[token], key_sums, optimize=True)
        output[token] = numerator / denominator[:, None]
    return output.reshape(seq_len, d_model), (pq, pk, vh, output, heads)


def linear_attention_backward(doutput, cache):
    pq, pk, vh, output, heads = cache
    seq_len, _, head_dim = pq.shape
    dout = doutput.reshape(seq_len, heads, head_dim)
    dpq = np.zeros_like(pq)
    dpk = np.zeros_like(pk)
    dv = np.zeros_like(vh)
    prefix_states = np.empty((seq_len, heads, head_dim, head_dim), dtype=np.float32)
    prefix_sums = np.empty((seq_len, heads, head_dim), dtype=np.float32)
    state = np.zeros((heads, head_dim, head_dim), dtype=np.float32)
    key_sums = np.zeros((heads, head_dim), dtype=np.float32)
    for token in range(seq_len):
        state += np.einsum("hi,hj->hij", pk[token], vh[token], optimize=True)
        key_sums += pk[token]
        prefix_states[token] = state
        prefix_sums[token] = key_sums
    dstate = np.zeros_like(state)
    dkey_sums = np.zeros_like(key_sums)
    for token in reversed(range(seq_len)):
        denominator = np.einsum(
            "hi,hi->h", pq[token], prefix_sums[token], optimize=True
        )
        dnumerator = dout[token] / denominator[:, None]
        ddenominator = (
            -np.einsum("hi,hi->h", dout[token], output[token], optimize=True)
            / denominator
        )
        dpq[token] += np.einsum(
            "hj,hij->hi", dnumerator, prefix_states[token], optimize=True
        )
        dpq[token] += ddenominator[:, None] * prefix_sums[token]
        dstate += np.einsum("hi,hj->hij", pq[token], dnumerator, optimize=True)
        dkey_sums += ddenominator[:, None] * pq[token]
        dpk[token] += np.einsum("hij,hj->hi", dstate, vh[token], optimize=True)
        dpk[token] += dkey_sums
        dv[token] += np.einsum("hi,hij->hj", pk[token], dstate, optimize=True)
    shape = (seq_len, heads * head_dim)
    return dpq.reshape(shape), dpk.reshape(shape), dv.reshape(shape)


def validate_recurrent_attention():
    rng = np.random.default_rng(7)
    q = rng.normal(0, 0.05, (3, 4)).astype(np.float32)
    k = rng.normal(0, 0.05, (3, 4)).astype(np.float32)
    v = rng.normal(0, 0.05, (3, 4)).astype(np.float32)
    dout = rng.normal(0, 0.05, (3, 4)).astype(np.float32)
    recurrent, cache = linear_attention_forward(q, k, v, 2)
    pq, pk, vh, _, heads = cache
    reference = np.empty_like(vh)
    for token in range(3):
        for head in range(heads):
            weights = pk[:token + 1, head] @ pq[token, head]
            reference[token, head] = (
                weights @ vh[:token + 1, head]
            ) / np.sum(weights)
    reference = reference.reshape(3, 4)
    if not np.allclose(recurrent, reference, rtol=1e-5, atol=1e-6):
        raise ValueError("recurrent attention forward parity failed")

    recurrent_grads = linear_attention_backward(dout, cache)
    explicit_grads = [np.zeros_like(pq), np.zeros_like(pk), np.zeros_like(vh)]
    dout_heads = dout.reshape(3, heads, 2)
    for token in range(3):
        for head in range(heads):
            keys = pk[:token + 1, head]
            values = vh[:token + 1, head]
            weights = keys @ pq[token, head]
            denominator = np.sum(weights)
            output = reference.reshape(3, heads, 2)[token, head]
            dnumerator = dout_heads[token, head] / denominator
            ddenominator = -np.dot(dout_heads[token, head], output) / denominator
            dweights = values @ dnumerator + ddenominator
            explicit_grads[2][:token + 1, head] += weights[:, None] * dnumerator
            explicit_grads[0][token, head] += dweights @ keys
            explicit_grads[1][:token + 1, head] += (
                dweights[:, None] * pq[token, head][None, :]
            )
    for recurrent_grad, explicit_grad in zip(recurrent_grads, explicit_grads):
        if not np.allclose(
            recurrent_grad,
            explicit_grad.reshape(3, 4),
            rtol=2e-5,
            atol=2e-6,
        ):
            raise ValueError("recurrent attention backward parity failed")


def forward(model, tokens, config):
    x = model["embeddings"][tokens]
    layers = []
    for layer in range(config["layers"]):
        attention_input, attention_rms = rms_forward(x, model["attention_rms"][layer])
        q = attention_input @ model["q"][layer].T
        k = attention_input @ model["k"][layer].T
        v = attention_input @ model["v"][layer].T
        context, attention_cache = linear_attention_forward(q, k, v, config["heads"])
        attention_output = context @ model["o"][layer].T
        x1 = x + attention_output
        mlp_input, mlp_rms = rms_forward(x1, model["mlp_rms"][layer])
        up = mlp_input @ model["up"][layer].T
        gate = mlp_input @ model["gate"][layer].T
        activation, activation_derivative = hard_silu(gate)
        gated = up * activation
        down = gated @ model["down"][layer].T
        layers.append({
            "x": x, "attention_input": attention_input, "attention_rms": attention_rms,
            "q": q, "k": k, "v": v, "context": context, "attention_cache": attention_cache,
            "x1": x1, "mlp_input": mlp_input, "mlp_rms": mlp_rms, "up": up,
            "gate": gate, "activation": activation, "activation_derivative": activation_derivative,
            "gated": gated,
        })
        x = x1 + down
    features, final_rms = rms_forward(x[-1], model["final_rms"])
    logits = model["output"] @ features + model["bias"]
    return logits, {"tokens": tokens, "layers": layers, "final_x": x, "features": features, "final_rms": final_rms}


def loss_and_gradient(logits, target):
    shifted = logits - np.max(logits)
    probabilities = np.exp(shifted)
    probabilities /= np.sum(probabilities)
    loss = -np.log(max(float(probabilities[target]), 1e-30))
    gradient = probabilities
    gradient[target] -= 1
    return loss, gradient.astype(np.float32)


def backward(model, cache, dlogits, config, gradients):
    for value in gradients.values():
        value.fill(0)
    gradients["output"] = np.outer(dlogits, cache["features"])
    gradients["bias"] = dlogits
    dfeatures = dlogits @ model["output"]
    dlast, gradients["final_rms"] = rms_backward(dfeatures, cache["final_rms"])
    dx = np.zeros_like(cache["final_x"])
    dx[-1] = dlast

    for layer in reversed(range(config["layers"])):
        item = cache["layers"][layer]
        ddown = dx
        dx1 = dx.copy()
        gradients["down"][layer] = ddown.T @ item["gated"]
        dgated = ddown @ model["down"][layer]
        dup = dgated * item["activation"]
        dgate = dgated * item["up"] * item["activation_derivative"]
        gradients["up"][layer] = dup.T @ item["mlp_input"]
        gradients["gate"][layer] = dgate.T @ item["mlp_input"]
        dmlp = dup @ model["up"][layer] + dgate @ model["gate"][layer]
        dmlp_input, gradients["mlp_rms"][layer] = rms_backward(dmlp, item["mlp_rms"])
        dx1 += dmlp_input

        dattn_output = dx1
        dx = dx1.copy()
        gradients["o"][layer] = dattn_output.T @ item["context"]
        dcontext = dattn_output @ model["o"][layer]
        dq, dk, dv = linear_attention_backward(dcontext, item["attention_cache"])
        gradients["q"][layer] = dq.T @ item["attention_input"]
        gradients["k"][layer] = dk.T @ item["attention_input"]
        gradients["v"][layer] = dv.T @ item["attention_input"]
        dattn_input = dq @ model["q"][layer] + dk @ model["k"][layer] + dv @ model["v"][layer]
        drms, gradients["attention_rms"][layer] = rms_backward(dattn_input, item["attention_rms"])
        dx += drms

    np.add.at(gradients["embeddings"], cache["tokens"], dx)
    return gradients


def tensor_hash(model):
    digest = hashlib.sha256()
    for name in sorted(model):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(model[name], dtype=np.float32).tobytes())
    return digest.hexdigest()


def train(model, windows, config, epochs, learning_rate, batch_windows):
    initial_hash = tensor_hash(model)
    movement = {name: 0 for name in model}
    losses = []
    initial_mistakes = 0
    for tokens, target in windows:
        logits, _ = forward(model, tokens, config)
        loss, _ = loss_and_gradient(logits, target)
        losses.append(loss)
        initial_mistakes += int(np.argmax(logits) != target)
    initial_loss = sum(losses) / len(losses)

    gradients = {name: np.zeros_like(value) for name, value in model.items()}
    accumulated = {name: np.zeros_like(value) for name, value in model.items()}
    for _ in range(epochs):
        for batch_start in range(0, len(windows), batch_windows):
            batch = windows[batch_start:batch_start + batch_windows]
            for value in accumulated.values():
                value.fill(0)
            for tokens, target in batch:
                logits, cache = forward(model, tokens, config)
                _, dlogits = loss_and_gradient(logits, target)
                backward(model, cache, dlogits, config, gradients)
                for name in model:
                    accumulated[name] += gradients[name]
            for name in model:
                update = np.float32(learning_rate / len(batch)) * accumulated[name]
                movement[name] += float(np.sum(np.abs(update), dtype=np.float64))
                model[name] -= update

    losses = []
    final_mistakes = 0
    for tokens, target in windows:
        logits, _ = forward(model, tokens, config)
        loss, _ = loss_and_gradient(logits, target)
        losses.append(loss)
        final_mistakes += int(np.argmax(logits) != target)
    finite = all(np.all(np.isfinite(value)) for value in model.values())
    return {
        "initial_tensor_hash": initial_hash,
        "final_tensor_hash": tensor_hash(model),
        "initial_loss_millionths": round(initial_loss * 1_000_000),
        "final_loss_millionths": round(sum(losses) / len(losses) * 1_000_000),
        "initial_mistakes": initial_mistakes,
        "final_mistakes": final_mistakes,
        "movement_trillionths": {name: round(value * 1_000_000_000_000) for name, value in movement.items()},
        "moved_parameter_groups": sorted(name for name, value in movement.items() if value > 0),
        "finite": finite,
    }


def parameter_count(model):
    return sum(value.size for value in model.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--context-tokens", type=int, default=4)
    parser.add_argument("--max-windows", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate-millionths", type=int, default=1000)
    parser.add_argument("--batch-windows", type=int, default=4)
    parser.add_argument("--allow-partial-gates", action="store_true")
    args = parser.parse_args()

    validate_recurrent_attention()

    model, config = load_integer_model(args.model)
    tokens, token_stream_hash = load_tokens(args.tokens, config["tokenizer_hash"], config["vocab_size"])
    windows = document_windows(tokens, args.context_tokens, args.max_windows)
    if not windows:
        raise ValueError("no float-twin training windows")
    if args.batch_windows <= 0:
        raise ValueError("batch windows must be positive")
    result = train(
        model,
        windows,
        config,
        args.epochs,
        args.learning_rate_millionths / 1_000_000,
        args.batch_windows,
    )
    out_path = Path(args.out)
    trace_path = Path(args.trace)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **model)
    moved_groups = result["moved_parameter_groups"]
    trace = {
        "schema": "nsrl.production_float_twin_smoke.v1",
        "profile": "p10m",
        "parameter_count": parameter_count(model),
        "bindings": {
            "integer_initial_model_hash": f"0x{config['integer_model_hash']:016x}",
            "integer_artifact_sha256": config["artifact_sha256"],
            "tokenizer_hash": f"0x{config['tokenizer_hash']:016x}",
            "token_stream_hash": f"0x{token_stream_hash:016x}",
            "initialization_seed": config["initialization_seed"],
        },
        "training": {
            "optimizer": "sgd",
            "attention_algorithm": "causal_recurrent_linear",
            "context_tokens": args.context_tokens,
            "windows": len(windows),
            "epochs": args.epochs,
            "learning_rate_millionths": args.learning_rate_millionths,
            "batch_windows": args.batch_windows,
            "initial_loss_millionths": result["initial_loss_millionths"],
            "final_loss_millionths": result["final_loss_millionths"],
            "initial_mistakes": result["initial_mistakes"],
            "final_mistakes": result["final_mistakes"],
        },
        "movement_trillionths": result["movement_trillionths"],
        "moved_parameter_groups": moved_groups,
        "tensor_hashes": {
            "initial": result["initial_tensor_hash"],
            "final": result["final_tensor_hash"],
        },
        "gates": {
            "same_shape": parameter_count(model) == 9_317_632,
            "integer_initialization_mapped": True,
            "all_parameter_groups_moved": len(moved_groups) == len(model),
            "all_parameters_finite": result["finite"],
            "loss_nonincreasing": result["final_loss_millionths"] <= result["initial_loss_millionths"],
            "tensor_hash_changed": result["initial_tensor_hash"] != result["final_tensor_hash"],
        },
        "known_non_claims": [
            "bounded_float_smoke_not_scaling_run",
            "numpy_reference_not_production_accelerator",
            "does_not_establish_integer_float_quality_parity",
        ],
    }
    trace_path.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"trace": args.trace, "gates": trace["gates"]}, sort_keys=True))
    if not args.allow_partial_gates and not all(trace["gates"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
