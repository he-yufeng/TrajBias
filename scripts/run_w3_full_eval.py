"""
Paper B W3: Full-scale judge evaluation.
Target: 4 probes × 7 judges × 200 pairs = 5,600 evaluations (pointwise, original+perturbed).

7 Judge Models:
  1. gpt-5.2 (azure/gpt-5.2-2025-12-11)
  2. claude-sonnet-4-5 (claude-sonnet-4-5-20250929) - via Anthropic SDK
  3. gemini-3-pro (gemini-3-pro-preview)
  4. deepseek-v3.2
  5. qwen3-235b (qwen3-235b-a22b)
  6. glm-5 (openrouter/glm-5)
  7. kimi-k2.5
"""
import json, time, os, sys, re
from pathlib import Path
from openai import OpenAI
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "w3_full"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

JUDGES = {
    "gpt-5.2": "azure/gpt-5.2-2025-12-11",
    "claude-sonnet-4-5": "claude-sonnet-4-5-20250929",
    "gemini-3-pro": "gemini-3-pro-preview",
    "deepseek-v3.2": "deepseek-v3.2",
    "qwen3-235b": "qwen3-235b-a22b",
    "glm-5": "openrouter/glm-5",
    "kimi-k2.5": "kimi-k2.5",
}

PROBES = ["outcome", "length", "recovery", "recency"]

EVAL_PROMPT = """You are an expert evaluator of AI agent execution traces. Evaluate the quality of the following agent trajectory on a software engineering task.

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
        result = {"overall_score": float(m.group(1))}
        for dim in ["task_progress", "efficiency", "action_correctness",
                     "error_handling", "reasoning_quality"]:
            dm = re.search(rf'"{dim}"\s*:\s*(\d)', text)
            if dm: result[dim] = int(dm.group(1))
        return result
    return None


def evaluate_one(client, model_id, messages, max_retries=3):
    traj_text = format_traj(messages)
    prompt = EVAL_PROMPT.format(trajectory=traj_text)
    for attempt in range(max_retries):
        try:
            start = time.time()
            resp = client.chat.completions.create(
                model=model_id, messages=[{"role": "user", "content": prompt}],
                max_tokens=2500, temperature=0.0,
            )
            elapsed = time.time() - start
            content = resp.choices[0].message.content
            scores = parse_scores(content)
            return {
                "overall": scores.get("overall_score") if scores else None,
                "scores": scores, "elapsed": elapsed,
                "tokens_in": resp.usage.prompt_tokens if resp.usage else 0,
                "tokens_out": resp.usage.completion_tokens if resp.usage else 0,
            }
        except Exception as e:
            if "429" in str(e):
                time.sleep((attempt + 1) * 15)
            elif attempt < max_retries - 1:
                time.sleep(5)
            else:
                return {"overall": None, "scores": None, "elapsed": 0,
                        "tokens_in": 0, "tokens_out": 0, "error": str(e)[:100]}


def get_result_file(probe, judge_key):
    return RESULTS_DIR / f"eval_{probe}_{judge_key}.jsonl"


def count_completed(probe, judge_key):
    fpath = get_result_file(probe, judge_key)
    if not fpath.exists(): return 0
    with open(fpath) as f:
        return sum(1 for line in f if line.strip())


def run_probe_judge(probe, judge_key, judge_id, n_pairs=200):
    """Run evaluation for one probe × one judge. Appends results incrementally."""
    probe_file = DATA_DIR / f"bias_probe_{probe}.jsonl"
    if not probe_file.exists():
        print(f"  [SKIP] {probe_file} not found")
        return

    # Load pairs
    pairs = []
    with open(probe_file) as f:
        for line in f:
            pairs.append(json.loads(line))
            if len(pairs) >= n_pairs:
                break

    # Check how many already done
    done = count_completed(probe, judge_key)
    if done >= len(pairs) * 2:  # 2 evals per pair (orig + pert)
        print(f"  [SKIP] {probe}×{judge_key}: all {done} evals done")
        return

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    result_file = get_result_file(probe, judge_key)
    start_from = done // 2  # Resume from pair index

    orig_scores, pert_scores = [], []

    for i in range(start_from, len(pairs)):
        pair = pairs[i]
        r_orig = evaluate_one(client, judge_id, pair["original_messages"]) or {"overall": None, "scores": None, "tokens_in": 0, "tokens_out": 0}
        r_pert = evaluate_one(client, judge_id, pair["perturbed_messages"]) or {"overall": None, "scores": None, "tokens_in": 0, "tokens_out": 0}

        if r_orig.get("overall") is not None: orig_scores.append(r_orig["overall"])
        if r_pert.get("overall") is not None: pert_scores.append(r_pert["overall"])

        # Append results
        with open(result_file, "a") as f:
            f.write(json.dumps({
                "pair_id": pair["pair_id"], "bias_type": pair["bias_type"],
                "judge": judge_key, "version": "original",
                "score": r_orig["overall"], "scores": r_orig["scores"],
                "tokens": r_orig.get("tokens_in", 0) + r_orig.get("tokens_out", 0),
            }, default=str) + "\n")
            f.write(json.dumps({
                "pair_id": pair["pair_id"], "bias_type": pair["bias_type"],
                "judge": judge_key, "version": "perturbed",
                "score": r_pert["overall"], "scores": r_pert["scores"],
                "tokens": r_pert.get("tokens_in", 0) + r_pert.get("tokens_out", 0),
            }, default=str) + "\n")

        so = f"{r_orig['overall']:.1f}" if r_orig["overall"] else "FAIL"
        sp = f"{r_pert['overall']:.1f}" if r_pert["overall"] else "FAIL"
        if (i + 1) % 10 == 0 or i == len(pairs) - 1:
            print(f"  [{i+1}/{len(pairs)}] last: orig={so} pert={sp}")

    # Summary for this probe×judge
    if orig_scores and pert_scores:
        avg_o = sum(orig_scores) / len(orig_scores)
        avg_p = sum(pert_scores) / len(pert_scores)
        delta = avg_p - avg_o
        signal = "YES!" if abs(delta) > 0.3 else "weak" if abs(delta) > 0.15 else "no"
        print(f"  → {probe}×{judge_key}: orig={avg_o:.2f} pert={avg_p:.2f} "
              f"delta={delta:+.2f} signal={signal} (n={len(orig_scores)})")


def main():
    # Strategy: Run cheapest judges first, most probes first
    # Outcome probe is the strongest finding - prioritize it across all judges
    # Order judges by reliability: cheapest/most-stable first, expensive/flaky last
    stable_judges = ["deepseek-v3.2", "gemini-3-pro", "qwen3-235b", "glm-5",
                     "kimi-k2.5", "claude-sonnet-4-5", "gpt-5.2"]

    run_order = [
        # Phase 1: Outcome probe with all judges (strongest signal, highest priority)
        ("outcome", stable_judges),
        # Phase 2: Recovery probe with all judges (weak signal, needs more data)
        ("recovery", stable_judges),
        # Phase 3: Recency probe with all judges (untested at scale)
        ("recency", stable_judges),
        # Phase 4: Peak-End probe with all judges (5th probe, Kahneman Peak-End Rule)
        ("peakend", stable_judges),
        # Phase 5: Length probe with 3 judges (confirmed null)
        ("length", ["deepseek-v3.2", "claude-sonnet-4-5", "kimi-k2.5"]),
    ]

    total = sum(len(judges) for _, judges in run_order)
    print(f"{'='*60}")
    print(f"W3 FULL-SCALE JUDGE EVALUATION")
    print(f"Probe×Judge combinations: {total}")
    print(f"Pairs per combination: 200")
    print(f"{'='*60}")

    for probe, judge_keys in run_order:
        print(f"\n{'='*40}")
        print(f"PROBE: {probe}")
        print(f"{'='*40}")
        for jk in judge_keys:
            print(f"\n  --- {probe} × {jk} ---")
            run_probe_judge(probe, jk, JUDGES[jk], n_pairs=200)


if __name__ == "__main__":
    main()
