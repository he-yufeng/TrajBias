"""
Paper B: Debiasing experiments.
Tests 3 debiasing strategies on Outcome Bias and Recovery Bias.

Strategy 1: Outcome Masking — remove outcome signal from trajectory before evaluation
Strategy 2: Step-Level Aggregation — evaluate each step independently, then aggregate
Strategy 3: Explicit Instruction — add debiasing instruction to judge prompt
"""
import json
import time
import os
import re
import copy
import numpy as np
from pathlib import Path
from openai import OpenAI

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "debiasing"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

# Use 3 representative judges: most biased (gemini), least biased (claude), mid (deepseek)
JUDGES = {
    "gemini-3-pro": "gemini-3-pro-preview",
    "claude-sonnet-4-5": "claude-sonnet-4-5-20250929",
    "deepseek-v3.2": "deepseek-v3.2",
}

STANDARD_PROMPT = """You are an expert evaluator of AI agent execution traces. Evaluate the quality of the following agent trajectory on a software engineering task.

## Evaluation Dimensions (score 1-5 each):
1. Task Progress (30%): Did the agent make meaningful progress?
2. Efficiency (20%): Were steps concise without redundancy?
3. Action Correctness (25%): Were individual actions reasonable?
4. Error Handling (15%): Did the agent recover from errors effectively?
5. Reasoning Quality (10%): Was the reasoning chain logical?

## Trajectory:
{trajectory}

IMPORTANT: Respond with ONLY a JSON object, no other text:
{{"task_progress": <1-5>, "efficiency": <1-5>, "action_correctness": <1-5>, "error_handling": <1-5>, "reasoning_quality": <1-5>, "overall_score": <weighted average as float>}}"""

DEBIASED_PROMPT_INSTRUCTION = """You are an expert evaluator of AI agent execution traces.

IMPORTANT INSTRUCTION: Evaluate ONLY the quality of the agent's PROCESS — its reasoning, action choices, and problem-solving approach. Do NOT let the final outcome (success or failure) influence your assessment of process quality. A good process can lead to failure due to external factors, and a poor process can succeed by luck.

## Evaluation Dimensions (score 1-5 each):
1. Task Progress (30%): Did the agent make meaningful progress?
2. Efficiency (20%): Were steps concise without redundancy?
3. Action Correctness (25%): Were individual actions reasonable?
4. Error Handling (15%): Did the agent recover from errors effectively?
5. Reasoning Quality (10%): Was the reasoning chain logical?

## Trajectory:
{trajectory}

IMPORTANT: Respond with ONLY a JSON object, no other text:
{{"task_progress": <1-5>, "efficiency": <1-5>, "action_correctness": <1-5>, "error_handling": <1-5>, "reasoning_quality": <1-5>, "overall_score": <weighted average as float>}}"""


def format_traj(messages, max_chars=32000):
    lines = []
    for msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = "\n".join(p.get("text", "") for p in content
                              if isinstance(p, dict) and p.get("type") == "text")
        if len(content) > 600:
            content = content[:300] + "\n...[truncated]...\n" + content[-150:]
        lines.append(f"[{role.upper()}]: {content}")
    text = "\n\n".join(lines)
    if len(text) > max_chars:
        h, t = int(max_chars * 0.4), int(max_chars * 0.2)
        text = text[:h] + "\n\n...[middle truncated]...\n\n" + text[-t:]
    return text


def mask_outcome(messages):
    """Strategy 1: Remove outcome signals from trajectory."""
    masked = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            # Remove outcome-related keywords
            for pattern in [
                r"Task status:.*?(SUCCESS|FAILED|COMPLETED|INCOMPLETE).*",
                r"\[SYSTEM\]:.*?(successful|failed|completed|incomplete).*",
                r"Task marked as (SUCCESSFUL|FAILED)",
            ]:
                content = re.sub(pattern, "[outcome information removed]", content, flags=re.IGNORECASE)
        masked.append({**msg, "content": content})
    return masked


def parse_scores(text):
    if not text: return None
    try:
        start = text.index('{')
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{': depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    try: return json.loads(text[start:i+1])
                    except: break
    except ValueError: pass
    m = re.search(r'"overall_score"\s*:\s*([\d.]+)', text)
    if m:
        return {"overall_score": float(m.group(1))}
    return None


def evaluate_one(client, model_id, trajectory_text, prompt_template, max_retries=3):
    prompt = prompt_template.format(trajectory=trajectory_text)
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_id, messages=[{"role": "user", "content": prompt}],
                max_tokens=2500, temperature=0.0,
            )
            content = resp.choices[0].message.content
            scores = parse_scores(content)
            return scores.get("overall_score") if scores else None
        except Exception as e:
            if "429" in str(e):
                time.sleep((attempt + 1) * 15)
            elif attempt < max_retries - 1:
                time.sleep(5)
            else:
                return None


def run_debiasing_experiment(strategy_name, n_pairs=100):
    """Run one debiasing strategy on outcome bias pairs."""
    probe_file = DATA_DIR / "bias_probe_outcome.jsonl"
    pairs = []
    with open(probe_file) as f:
        for line in f:
            pairs.append(json.loads(line))
            if len(pairs) >= n_pairs:
                break

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    results = []

    for judge_key, judge_id in JUDGES.items():
        print(f"\n  --- {strategy_name} × {judge_key} ---")
        orig_scores, pert_scores = [], []

        for i, pair in enumerate(pairs):
            orig_msgs = pair["original_messages"]
            pert_msgs = pair["perturbed_messages"]

            if strategy_name == "outcome_masking":
                # Evaluate with outcome info removed
                orig_text = format_traj(mask_outcome(orig_msgs))
                pert_text = format_traj(mask_outcome(pert_msgs))
                prompt = STANDARD_PROMPT
            elif strategy_name == "explicit_instruction":
                # Evaluate with debiasing instruction
                orig_text = format_traj(orig_msgs)
                pert_text = format_traj(pert_msgs)
                prompt = DEBIASED_PROMPT_INSTRUCTION
            else:
                continue

            o = evaluate_one(client, judge_id, orig_text, prompt)
            p = evaluate_one(client, judge_id, pert_text, prompt)

            if o is not None: orig_scores.append(o)
            if p is not None: pert_scores.append(p)

            if (i + 1) % 20 == 0:
                print(f"    [{i+1}/{n_pairs}]")

        # Summary
        if orig_scores and pert_scores:
            n = min(len(orig_scores), len(pert_scores))
            avg_o = np.mean(orig_scores[:n])
            avg_p = np.mean(pert_scores[:n])
            delta = avg_p - avg_o

            result = {
                "strategy": strategy_name,
                "judge": judge_key,
                "n_valid": n,
                "avg_original": float(avg_o),
                "avg_perturbed": float(avg_p),
                "delta": float(delta),
            }
            results.append(result)
            print(f"    → δ={delta:+.3f} (orig={avg_o:.2f} pert={avg_p:.2f} n={n})")

    return results


def main():
    print("=" * 60)
    print("DEBIASING EXPERIMENTS")
    print("=" * 60)

    all_results = []

    # Strategy 1: Outcome Masking
    print("\n[Strategy 1] Outcome Masking")
    results = run_debiasing_experiment("outcome_masking", n_pairs=100)
    all_results.extend(results)

    # Strategy 2: Explicit Debiasing Instruction
    print("\n[Strategy 2] Explicit Instruction")
    results = run_debiasing_experiment("explicit_instruction", n_pairs=100)
    all_results.extend(results)

    # Save results
    out_file = RESULTS_DIR / "debiasing_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Compare with baseline (no debiasing)
    print("\n" + "=" * 60)
    print("DEBIASING COMPARISON")
    print("=" * 60)
    print(f"{'Strategy':<25s} {'Judge':<25s} {'δ':>8s} {'vs Baseline':>12s}")
    print("-" * 70)

    # Baseline deltas from our earlier analysis
    baseline = {
        "gemini-3-pro": 0.247,
        "claude-sonnet-4-5": 0.150,
        "deepseek-v3.2": 0.061,
    }

    for r in all_results:
        bl = baseline.get(r["judge"], 0)
        reduction = (1 - abs(r["delta"]) / max(abs(bl), 0.001)) * 100 if bl != 0 else 0
        print(f"{r['strategy']:<25s} {r['judge']:<25s} {r['delta']:>+8.3f} {reduction:>+10.0f}% reduction")

    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
