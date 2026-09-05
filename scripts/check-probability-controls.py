"""Check fixed-answer scores on sixteen known proof windows and existing counts."""

import argparse
import csv
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BASE = "benchmarks/integer-transformer-proof-v1/"
CANDIDATE = "data/experiments/integer-transformer-proof-v1/candidate-default/"
ONE = 32767
ARMS = ["native", "point_mass", "smoothed_point_mass"]
SOURCES = [
    "Cargo.toml", "Cargo.lock", "crates/nsrl-train/Cargo.toml",
    "crates/nsrl-train/src/bin/nsrl-mini-transformer-eval.rs",
    "crates/nsrl-train/src/bin/support/probability_controls.rs",
    "crates/nsrl-train/src/lib.rs", "crates/nsrl-train-core/src/lib.rs",
    "scripts/check-probability-controls.py",
    BASE + "promoted-candidate.json", BASE + "component-ablation.json",
    BASE + "eval.txt", CANDIDATE + "candidate.nsrlmt", CANDIDATE + "candidate.eval.json",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def score(probabilities, target):
    require(len(probabilities) == 256 and all(0 <= p <= ONE for p in probabilities),
            "invalid probability vector")
    mass = sum(probabilities)
    require(mass > 0, "zero probability mass")
    # Independent exact arithmetic over each class, including Q15 mass drift.
    distances = [Fraction(p, mass) - int(index == target)
                 for index, p in enumerate(probabilities)]
    return {
        "mass": mass,
        "l1": sum(abs(p - ONE * int(index == target))
                  for index, p in enumerate(probabilities)),
        "brier": sum(value * value for value in distances),
        "zero": int(probabilities[target] == 0),
    }


def check_rows(rows, tokens, trace):
    require(len(rows) == 16 * len(ARMS), "control row count differs")
    totals = {arm: {"mistakes": 0, "l1": 0, "brier": Fraction(0), "zeros": 0,
                    "minimum_mass": 256 * ONE, "maximum_mass": 0} for arm in ARMS}
    for window in range(16):
        start = (window * (len(tokens) - 65) + 7) // 15
        batch = rows[window * len(ARMS):(window + 1) * len(ARMS)]
        require([row["arm"] for row in batch] == ARMS, "control arm roster differs")
        require(len({row["predicted"] for row in batch}) == 1, "chosen answer changed")
        for row in batch:
            target, predicted = int(row["target"]), int(row["predicted"])
            require(int(row["start"]) == start and int(row["end"]) == 64 + start,
                    "window identity differs")
            require(target == tokens[64 + start] and 0 <= predicted <= 255,
                    "target or prediction differs")
            probabilities = list(map(int, row["probabilities_q15"].split(",")))
            measured = score(probabilities, target)
            require(int(row["probability_mass_q15"]) == measured["mass"], "mass differs")
            require(int(row["probability_error_q15"]) == measured["l1"], "L1 differs")
            require(Fraction(int(row["brier_numerator"]), int(row["brier_denominator"]))
                    == measured["brier"], "Brier differs")
            require(int(row["zero_target_probability"]) == measured["zero"], "zero count differs")
            if row["arm"] == "point_mass":
                require(probabilities == [ONE * int(index == predicted) for index in range(256)],
                        "point mass differs")
            if row["arm"] == "smoothed_point_mass":
                other_indices = [index for index in range(256) if index != predicted]
                expected = [0] * 256
                expected[predicted] = 29491
                for rank, index in enumerate(other_indices):
                    expected[index] = 12 + int(rank < 216)
                require(probabilities == expected, "fixed smoothing differs")
            total = totals[row["arm"]]
            total["mistakes"] += int(target != predicted)
            total["l1"] += measured["l1"]
            total["brier"] += measured["brier"]
            total["zeros"] += measured["zero"]
            total["minimum_mass"] = min(total["minimum_mass"], measured["mass"])
            total["maximum_mass"] = max(total["maximum_mass"], measured["mass"])
    native = totals["native"]
    require(trace["data"]["windows"] == 16 and trace["evaluation"]["invalid_forward_count"] == 0,
            "native evaluation coverage differs")
    require(native["l1"] == trace["evaluation"]["probability_error_q15"]
            and native["mistakes"] == trace["evaluation"]["mistakes"], "native totals differ")
    for total in totals.values():
        total["normalized_brier_mean"] = str(total.pop("brier") / 16)
    return totals


def check_failure_paths(rows, tokens, trace):
    for field, replacement in [
        ("start", "1"), ("predicted", "256"), ("arm", "changed"),
        ("probability_error_q15", "0"), ("brier_numerator", "0"),
    ]:
        changed = deepcopy(rows)
        changed[0][field] = replacement
        try:
            check_rows(changed, tokens, trace)
        except ValueError:
            continue
        raise ValueError("changed row accepted: " + field)
    try:
        check_rows(rows[:-1], tokens, trace)
    except ValueError:
        return
    raise ValueError("incomplete row roster accepted")


def aggregate_control():
    promoted = json.loads((ROOT / (BASE + "promoted-candidate.json")).read_bytes())
    components = json.loads((ROOT / (BASE + "component-ablation.json")).read_bytes())
    historical = json.loads((ROOT / (CANDIDATE + "candidate.eval.json")).read_bytes())
    for name in ["candidate.nsrlmt", "candidate.eval.json"]:
        data = (ROOT / (CANDIDATE + name)).read_bytes()
        require(sha(data) == promoted["files"][name]["sha256"]
                and len(data) == promoted["files"][name]["bytes"], "promoted file binding differs")
    require(promoted["model_hash"] == components["source_model_hash"] == historical["model"]["hash"],
            "source model differs")
    require(components["data"] == historical["data"], "component data differs")
    for name in ["mistakes", "probability_error_q15"]:
        require(promoted["metrics"][name] == components["metrics"]["combined"][name]
                == historical["evaluation"][name], "combined metric differs")
    targets = promoted["metrics"]["targets"]
    require(targets == components["data"]["windows"], "historical target count differs")
    results = {}
    for arm, metrics in components["metrics"].items():
        mistakes = metrics["mistakes"]
        require(type(mistakes) is int and 0 <= mistakes <= targets, "mistake count is invalid")
        control = 2 * ONE * mistakes
        results[arm] = {
            "targets": targets, "mistakes": mistakes, "answers_changed": 0,
            "historical_l1_q15": metrics["probability_error_q15"],
            "point_mass_l1_q15": control,
            "l1_reduction_fraction": str(Fraction(metrics["probability_error_q15"] - control,
                                                 metrics["probability_error_q15"])),
            "point_mass_normalized_brier_mean": str(Fraction(2 * mistakes, targets)),
            "point_mass_zero_target_probability_count": mistakes,
        }
    return results, promoted["model_hash"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/nsrl-mini-transformer-eval")
    parser.add_argument("--write-smoke", action="store_true")
    args = parser.parse_args()
    aggregate, model_hash = aggregate_control()
    directory = Path(tempfile.mkdtemp(prefix="nsrl-probability-controls-"))
    print(f"Native output directory: {directory}", flush=True)
    with (directory / "stdout.txt").open("wb") as stdout, (directory / "stderr.txt").open("wb") as stderr:
        process = subprocess.run([
            str(args.binary.resolve()), "--tokens", str(ROOT / (BASE + "eval.txt")),
            "--model", str(ROOT / (CANDIDATE + "candidate.nsrlmt")),
            "--max-windows", "16", "--out", str(directory / "eval.json"),
            "--probability-controls-out", str(directory / "controls.tsv"),
        ], stdout=stdout, stderr=stderr, timeout=60, check=False)
    require(process.returncode == 0, f"native evaluation exited {process.returncode}; outputs: {directory}")
    trace = json.loads((directory / "eval.json").read_bytes())
    require(trace["model"]["hash"] == model_hash, "smoke model differs")
    rows = list(csv.DictReader((directory / "controls.tsv").read_text().splitlines(), delimiter="\t"))
    tokens = (ROOT / (BASE + "eval.txt")).read_bytes()
    totals = check_rows(rows, tokens, trace)
    check_failure_paths(rows, tokens, trace)
    # Bind core arithmetic sources as well as the evaluator and audit code.
    sources = sorted(set(SOURCES + [str(path.relative_to(ROOT))
                                   for path in (ROOT / "crates/nsrl-core/src").rglob("*.rs")]))
    result = {
        "schema": "nsrl.probability_controls.v1", "promotion_evidence": False,
        "scope": "sixteen_known_windows_and_analytic_existing_count_control",
        "model_hash": model_hash,
        "sources": {name: sha((ROOT / name).read_bytes()) for name in sources},
        "native_smoke": {"windows": 16, "rows": 48, "scores": totals,
                         "trace_sha256": sha((directory / "eval.json").read_bytes()),
                         "controls_sha256": sha((directory / "controls.tsv").read_bytes())},
        "existing_count_counterfactual": aggregate,
    }
    output = ROOT / (BASE + "probability-controls-smoke.json")
    if args.write_smoke:
        output.write_bytes(encoded(result))
    else:
        require(output.read_bytes() == encoded(result), "frozen smoke differs")
    print(json.dumps({"native_smoke": totals, "existing_count_counterfactual": aggregate}, indent=2))


if __name__ == "__main__":
    main()
