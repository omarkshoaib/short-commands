import json
import argparse
from pathlib import Path

import numpy as np


def load_telemetry(path: Path):
	rows = []
	with path.open("r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			try:
				rows.append(json.loads(line))
			except Exception:
				pass
	return rows


def sweep_thresholds(rows, thresholds):
	# This is an illustrative ROC-like sweep; without ground truth labels
	# we approximate "positive" as (w2v2_mapped or whisper_mapped non-empty)
	metrics = []
	for thr in thresholds:
		tp = fp = tn = fn = 0
		for r in rows:
			kws_score = float(r.get("kws", {}).get("score", 0.0))
			pred = kws_score >= thr
			is_command = bool(r.get("w2v2", {}).get("mapped") or r.get("whisper", {}).get("mapped"))
			if pred and is_command:
				tp += 1
			elif pred and not is_command:
				fp += 1
			elif not pred and is_command:
				fn += 1
			else:
				tn += 1
		precision = tp / (tp + fp + 1e-6)
		recall = tp / (tp + fn + 1e-6)
		fpr = fp / (fp + tn + 1e-6)
		metrics.append({"thr": thr, "precision": precision, "recall": recall, "fpr": fpr, "tp": tp, "fp": fp, "fn": fn, "tn": tn})
	return metrics


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--telemetry", default="results/telemetry.jsonl")
	args = ap.parse_args()

	rows = load_telemetry(Path(args.telemetry))
	if not rows:
		print("No telemetry.")
		return
	thr = np.linspace(0.1, 0.9, 9)
	metrics = sweep_thresholds(rows, thr)
	print("thr\tprecision\trecall\tfpr\ttp\tfp\tfn\ttn")
	for m in metrics:
		print(f"{m['thr']:.2f}\t{m['precision']:.2f}\t{m['recall']:.2f}\t{m['fpr']:.2f}\t{m['tp']}\t{m['fp']}\t{m['fn']}\t{m['tn']}")


if __name__ == "__main__":
	main()
