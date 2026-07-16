#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def tensor_hash(archive):
    digest = hashlib.sha256()
    for name in sorted(archive.files):
        value = np.asarray(archive[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(value.shape).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with np.load(args.left) as left, np.load(args.right) as right:
        same_names = sorted(left.files) == sorted(right.files)
        tensor_names = sorted(left.files) if same_names else []
        same_shapes = same_names and all(left[name].shape == right[name].shape for name in tensor_names)
        byte_identical_tensors = same_shapes and all(
            left[name].dtype == right[name].dtype
            and np.array_equal(left[name], right[name])
            for name in tensor_names
        )
        result = {
            "schema": "nsrl.production_float_artifact_equality.v1",
            "left": args.left,
            "right": args.right,
            "tensor_count": len(tensor_names),
            "same_tensor_names": same_names,
            "same_tensor_shapes": same_shapes,
            "left_tensor_sha256": tensor_hash(left) if same_names else None,
            "right_tensor_sha256": tensor_hash(right) if same_names else None,
            "byte_identical_tensors": byte_identical_tensors,
        }
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    if not byte_identical_tensors:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
