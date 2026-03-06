"""
Paper B: Prepare HuggingFace dataset for TrajBias benchmark.
Creates structured dataset files ready for HuggingFace upload.
"""
import json
import os
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("data")
RESULTS_DIR = Path("results/w3_full")
STATS_DIR = Path("results/statistics_v2")
DEBIASING_DIR = Path("results/debiasing")
HF_DIR = Path("dataset/trajbias")
HF_DIR.mkdir(parents=True, exist_ok=True)

PROBES = ["outcome", "recovery", "recency", "peakend", "length"]
JUDGES = ["claude-sonnet-4-5", "deepseek-v3.2", "gemini-3-pro", "glm-5", "gpt-5.2", "qwen3-235b", "kimi-k2.5"]


def prepare_probe_pairs():
    """Export probe pair metadata (without full trajectories for size)."""
    for probe in PROBES:
        probe_file = DATA_DIR / f"bias_probe_{probe}.jsonl"
        if not probe_file.exists():
            continue

        out_file = HF_DIR / f"probes_{probe}.jsonl"
        count = 0
        with open(probe_file) as fin, open(out_file, "w") as fout:
            for line in fin:
                d = json.loads(line)
                # Include metadata only (trajectories are too large)
                meta = {
                    "pair_id": d["pair_id"],
                    "bias_type": d["bias_type"],
                    "original_id": d.get("original_id", ""),
                    "original_resolved": d.get("original_resolved"),
                    "perturbation_description": d.get("perturbation_description", ""),
                    "model": d.get("model", ""),
                    "instance_id": d.get("instance_id", ""),
                    "n_original_messages": len(d.get("original_messages", [])),
                    "n_perturbed_messages": len(d.get("perturbed_messages", [])),
                }
                fout.write(json.dumps(meta) + "\n")
                count += 1

        print(f"  {probe}: {count} pairs → {out_file.name}")


def prepare_eval_results():
    """Export all evaluation results."""
    for probe in PROBES:
        for judge in JUDGES:
            result_file = RESULTS_DIR / f"eval_{probe}_{judge}.jsonl"
            if not result_file.exists():
                continue

            out_file = HF_DIR / f"evals_{probe}_{judge}.jsonl"
            count = 0
            with open(result_file) as fin, open(out_file, "w") as fout:
                for line in fin:
                    d = json.loads(line)
                    record = {
                        "pair_id": d["pair_id"],
                        "bias_type": d.get("bias_type", probe),
                        "judge": d["judge"],
                        "version": d["version"],
                        "score": d.get("score"),
                        "scores": d.get("scores"),
                    }
                    fout.write(json.dumps(record) + "\n")
                    count += 1

            print(f"  {probe}×{judge}: {count} evals → {out_file.name}")


def prepare_statistics():
    """Copy statistical results."""
    for fname in os.listdir(STATS_DIR):
        if fname.endswith(".json"):
            src = STATS_DIR / fname
            dst = HF_DIR / f"stats_{fname}"
            with open(src) as f:
                data = json.load(f)
            with open(dst, "w") as f:
                json.dump(data, f, indent=2)
            print(f"  stats: {fname} → stats_{fname}")

    # Copy debiasing results
    debiasing_file = DEBIASING_DIR / "debiasing_results.json"
    if debiasing_file.exists():
        with open(debiasing_file) as f:
            data = json.load(f)
        with open(HF_DIR / "stats_debiasing.json", "w") as f:
            json.dump(data, f, indent=2)
        print(f"  stats: debiasing_results.json → stats_debiasing.json")


def create_dataset_card():
    """Create HuggingFace dataset card (README.md)."""
    card = """---
language:
  - en
license: mit
task_categories:
  - text-classification
tags:
  - llm-as-judge
  - agent-evaluation
  - bias
  - trajectory
  - benchmark
pretty_name: TrajBias
size_categories:
  - 10K<n<100K
---

# TrajBias: Structural Biases in LLM-as-Judge Evaluation of Agent Trajectories

## Dataset Description

TrajBias is a diagnostic benchmark for auditing biases in LLM-as-Judge evaluation of agent trajectories. It contains:

- **2,400 probe pairs**: Controlled perturbation experiments testing 5 bias types
- **20,000+ evaluations**: Scores from 7 judge models across 7 model families
- **Statistical results**: Complete analysis with Cliff's delta, Wilcoxon tests, and BH FDR correction

## Bias Types

| Bias | Probe Pairs | Finding |
|------|------------|---------|
| Error Recovery Penalty | 200 | Strongest effect (Cliff's δ=0.26-0.85, all p<0.001) |
| Recency Bias | 600 | Significant across all judges (δ=0.15-0.70) |
| Bidirectional Outcome Bias | 200 | Cancels in aggregate; significant per-direction |
| Position Sensitivity | 800 | Exploratory; 5/6 judges significant |
| Length Bias (Absent) | 600 | No effect, contradicting text-domain verbosity bias |

## Judge Models

7 models from 7 families: Claude Sonnet 4.5, GPT-5.2, Gemini 3 Pro, DeepSeek V3.2, Qwen3-235B, GLM-5, Kimi K2.5

## Files

### Probe Metadata
- `probes_{bias_type}.jsonl`: Pair IDs, perturbation descriptions, metadata

### Evaluation Results
- `evals_{bias_type}_{judge}.jsonl`: Per-pair scores with 5-dimension breakdown

### Statistical Analysis
- `stats_main_results.json`: All probe×judge effect sizes and p-values
- `stats_recovery_5dim.json`: Per-dimension analysis of Error Recovery
- `stats_recency_gradient.json`: Position gradient analysis
- `stats_outcome_bidirectional.json`: Resolved/unresolved subgroup analysis
- `stats_score_distributions.json`: Ceiling effect analysis
- `stats_attrition.json`: Parse failure rates
- `stats_debiasing.json`: Debiasing experiment results

## Usage

```python
from datasets import load_dataset
ds = load_dataset("TrajBias/trajbias-benchmark")
```

## Citation

```bibtex
@inproceedings{he2026trajbias,
  title={TrajBias: Structural Biases in LLM-as-Judge Evaluation of Agent Trajectories},
  author={He, Yufeng},
  booktitle={NeurIPS Datasets and Benchmarks},
  year={2026}
}
```
"""
    with open(HF_DIR / "README.md", "w") as f:
        f.write(card)
    print("  Dataset card: README.md")


def main():
    print("Preparing HuggingFace dataset for TrajBias")
    print("=" * 50)

    print("\n1. Probe pair metadata:")
    prepare_probe_pairs()

    print("\n2. Evaluation results:")
    prepare_eval_results()

    print("\n3. Statistical results:")
    prepare_statistics()

    print("\n4. Dataset card:")
    create_dataset_card()

    # Summary
    total_files = len(list(HF_DIR.glob("*")))
    total_size = sum(f.stat().st_size for f in HF_DIR.glob("*")) / 1024 / 1024
    print(f"\n{'='*50}")
    print(f"Total: {total_files} files, {total_size:.1f} MB in {HF_DIR}")
    print(f"\nTo upload: huggingface-cli upload TrajBias/trajbias-benchmark {HF_DIR}")


if __name__ == "__main__":
    main()
