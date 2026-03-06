"""
Paper B: Comprehensive Statistical Analysis v2
- Cliff's delta + rank-biserial r (primary effect sizes for ordinal data)
- Cohen's d retained as secondary for comparison
- All 5 probes × all available judges
- Unified BH FDR correction across ALL tests
- Parse failure / attrition table
- Score distribution analysis (ceiling effect)
- Recovery 5-dimension per-judge analysis
- Recency position gradient analysis (p25 vs p50 vs p75)
- Bidirectional outcome bias analysis
"""
import json
import os
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from scipy import stats
from itertools import combinations

RESULTS_DIR = Path("results/w3_full")
PROBE_DIR = Path("data")
OUTPUT_DIR = Path("results/statistics_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_PROBES = ["outcome", "recovery", "recency", "peakend", "length"]
ALL_JUDGES = [
    "claude-sonnet-4-5", "deepseek-v3.2", "gemini-3-pro",
    "glm-5", "gpt-5.2", "qwen3-235b", "kimi-k2.5",
]
DIMENSIONS = ["task_progress", "efficiency", "action_correctness", "error_handling", "reasoning_quality"]


# ============================================================
# Effect Size Functions
# ============================================================

def cliffs_delta(x, y):
    """Compute Cliff's delta for two independent or paired samples.

    Cliff's delta = (#(x_i > y_j) - #(x_i < y_j)) / (n_x * n_y)
    For paired samples, we compare each x_i with each y_j.
    Range: [-1, 1]. Thresholds: |d| < 0.147 negligible, < 0.33 small, < 0.474 medium, else large.
    """
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.0

    more = 0
    less = 0
    for xi in x:
        for yj in y:
            if xi > yj:
                more += 1
            elif xi < yj:
                less += 1

    delta = (more - less) / (n_x * n_y)
    return float(delta)


def cliffs_delta_paired(orig, pert):
    """Cliff's delta for paired differences (perturbed - original).

    Compare each difference against 0.
    Equivalent to: proportion(d_i > 0) - proportion(d_i < 0).
    """
    diffs = pert - orig
    n_pos = np.sum(diffs > 0)
    n_neg = np.sum(diffs < 0)
    n = len(diffs)
    if n == 0:
        return 0.0
    return float((n_pos - n_neg) / n)


def rank_biserial_r(orig, pert):
    """Rank-biserial correlation from Wilcoxon signed-rank test.

    r_rb = (W+ - W-) / (W+ + W-)
    where W+ = sum of ranks of positive diffs, W- = sum of ranks of negative diffs.
    Range: [-1, 1]. Negative = pert < orig.
    """
    diffs = pert - orig
    non_zero = diffs[diffs != 0]
    n = len(non_zero)
    if n < 2:
        return 0.0

    try:
        # Compute ranks of absolute differences
        abs_diffs = np.abs(non_zero)
        ranks = stats.rankdata(abs_diffs)

        # Sum ranks by sign of difference
        w_plus = np.sum(ranks[non_zero > 0])   # sum of ranks where pert > orig
        w_minus = np.sum(ranks[non_zero < 0])   # sum of ranks where pert < orig

        total = w_plus + w_minus
        if total == 0:
            return 0.0
        r = (w_plus - w_minus) / total
        return float(r)
    except Exception:
        return 0.0


def cohens_d_paired(orig, pert):
    """Cohen's d for paired samples (retained as secondary metric)."""
    diffs = pert - orig
    sd = np.std(diffs, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(diffs) / sd)


# ============================================================
# Statistical Tests
# ============================================================

def wilcoxon_test(orig, pert):
    """Wilcoxon signed-rank test on paired scores."""
    diffs = pert - orig
    non_zero = diffs[diffs != 0]
    if len(non_zero) < 10:
        return {"statistic": None, "p_value": 1.0, "n_nonzero": int(len(non_zero)), "n_pairs": int(len(orig))}

    stat, p = stats.wilcoxon(non_zero, alternative="two-sided")
    z = stats.norm.ppf(1 - p / 2) if p > 0 else 10.0

    return {
        "statistic": float(stat),
        "p_value": float(p),
        "z_score": float(z),
        "n_pairs": int(len(orig)),
        "n_nonzero": int(len(non_zero)),
    }


def bootstrap_ci(orig, pert, n_bootstrap=10000, alpha=0.05):
    """Bootstrap 95% CI for mean difference."""
    diffs = pert - orig
    n = len(diffs)
    rng = np.random.default_rng(42)
    boot_means = np.array([np.mean(rng.choice(diffs, size=n, replace=True)) for _ in range(n_bootstrap)])
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lower, upper


def bh_fdr_correction(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR correction. Returns list of booleans."""
    n = len(p_values)
    if n == 0:
        return []
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    thresholds = np.arange(1, n + 1) / n * alpha
    significant = sorted_p <= thresholds

    if np.any(significant):
        max_k = np.max(np.where(significant)[0])
        adjusted = np.zeros(n, dtype=bool)
        adjusted[sorted_indices[:max_k + 1]] = True
        return adjusted.tolist()
    return [False] * n


# ============================================================
# Data Loading
# ============================================================

def load_probe_metadata(probe_name):
    """Load probe pair metadata."""
    probe_file = PROBE_DIR / f"bias_probe_{probe_name}.jsonl"
    if not probe_file.exists():
        return {}
    meta = {}
    with open(probe_file) as f:
        for line in f:
            d = json.loads(line)
            meta[d["pair_id"]] = d
    return meta


def load_eval_pairs(probe_name, judge_name):
    """Load evaluation results as paired dict. Returns {pair_id: {version: record}}."""
    result_file = RESULTS_DIR / f"eval_{probe_name}_{judge_name}.jsonl"
    if not result_file.exists():
        return None

    pairs = defaultdict(dict)
    with open(result_file) as f:
        for line in f:
            d = json.loads(line)
            pairs[d["pair_id"]][d["version"]] = d
    return dict(pairs)


def extract_paired_scores(pairs, score_key="score"):
    """Extract matched original/perturbed scores from pairs dict."""
    orig_scores, pert_scores, valid_ids = [], [], []
    for pid, versions in pairs.items():
        o = versions.get("original", {}).get(score_key) if "original" in versions else None
        p = versions.get("perturbed", {}).get(score_key) if "perturbed" in versions else None
        if o is not None and p is not None:
            orig_scores.append(o)
            pert_scores.append(p)
            valid_ids.append(pid)
    return np.array(orig_scores), np.array(pert_scores), valid_ids


def extract_dimension_scores(pairs, dimension):
    """Extract paired scores for a specific dimension from the scores dict."""
    orig_scores, pert_scores = [], []
    for pid, versions in pairs.items():
        o_scores = versions.get("original", {}).get("scores") or {}
        p_scores = versions.get("perturbed", {}).get("scores") or {}
        o_val = o_scores.get(dimension)
        p_val = p_scores.get(dimension)
        if o_val is not None and p_val is not None:
            orig_scores.append(o_val)
            pert_scores.append(p_val)
    return np.array(orig_scores), np.array(pert_scores)


# ============================================================
# Analysis Functions
# ============================================================

def analyze_probe_judge(probe, judge):
    """Full statistical analysis for one probe×judge combination."""
    pairs = load_eval_pairs(probe, judge)
    if pairs is None:
        return None

    orig, pert, valid_ids = extract_paired_scores(pairs)
    if len(orig) < 10:
        return None

    delta = float(np.mean(pert - orig))
    wtest = wilcoxon_test(orig, pert)
    ci_lo, ci_hi = bootstrap_ci(orig, pert)

    return {
        "probe": probe,
        "judge": judge,
        "n_pairs": int(len(orig)),
        "mean_delta": delta,
        "median_delta": float(np.median(pert - orig)),
        "cliffs_delta": cliffs_delta_paired(orig, pert),
        "rank_biserial_r": rank_biserial_r(orig, pert),
        "cohens_d": cohens_d_paired(orig, pert),
        "ci_95_lower": ci_lo,
        "ci_95_upper": ci_hi,
        **wtest,
    }


def analyze_attrition():
    """Parse failure / attrition table: per probe × judge."""
    print("\n" + "=" * 70)
    print("PARSE FAILURE / ATTRITION ANALYSIS")
    print("=" * 70)

    # Expected pairs per probe
    expected = {"outcome": 200, "recovery": 200, "recency": 200, "peakend": 200, "length": 200}

    attrition = {}
    for probe in ALL_PROBES:
        for judge in ALL_JUDGES:
            result_file = RESULTS_DIR / f"eval_{probe}_{judge}.jsonl"
            if not result_file.exists():
                attrition[(probe, judge)] = {"exists": False, "total_lines": 0, "valid_pairs": 0, "parse_fail": 0}
                continue

            pairs = defaultdict(dict)
            total_lines = 0
            null_scores = 0
            with open(result_file) as f:
                for line in f:
                    total_lines += 1
                    d = json.loads(line)
                    if d.get("score") is None:
                        null_scores += 1
                    pairs[d["pair_id"]][d["version"]] = d.get("score")

            # Count complete pairs (both original and perturbed have scores)
            valid_pairs = sum(1 for v in pairs.values()
                            if v.get("original") is not None and v.get("perturbed") is not None)

            attrition[(probe, judge)] = {
                "exists": True,
                "total_lines": total_lines,
                "total_pairs": len(pairs),
                "valid_pairs": valid_pairs,
                "null_scores": null_scores,
                "parse_fail_rate": null_scores / total_lines if total_lines > 0 else 0,
            }

    # Print table
    print(f"\n{'Probe':<12} {'Judge':<22} {'Lines':>6} {'Pairs':>6} {'Valid':>6} {'Null':>5} {'Fail%':>7}")
    print("-" * 70)
    for probe in ALL_PROBES:
        for judge in ALL_JUDGES:
            a = attrition[(probe, judge)]
            if not a["exists"]:
                print(f"{probe:<12} {judge:<22} {'—':>6} {'—':>6} {'—':>6} {'—':>5} {'N/A':>7}")
            else:
                print(f"{probe:<12} {judge:<22} {a['total_lines']:>6} {a['total_pairs']:>6} {a['valid_pairs']:>6} {a['null_scores']:>5} {a['parse_fail_rate']:>6.1%}")

    return attrition


def analyze_score_distribution():
    """Analyze score distributions and ceiling effects."""
    print("\n" + "=" * 70)
    print("SCORE DISTRIBUTION / CEILING EFFECT ANALYSIS")
    print("=" * 70)

    distributions = {}
    for probe in ALL_PROBES:
        for judge in ALL_JUDGES:
            pairs = load_eval_pairs(probe, judge)
            if pairs is None:
                continue

            all_scores = []
            for pid, versions in pairs.items():
                for v in ["original", "perturbed"]:
                    s = versions.get(v, {}).get("score")
                    if s is not None:
                        all_scores.append(s)

            if not all_scores:
                continue

            arr = np.array(all_scores)
            distributions[(probe, judge)] = {
                "n": len(arr),
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr, ddof=1)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "pct_above_4": float(np.mean(arr > 4) * 100),
                "pct_above_4.5": float(np.mean(arr > 4.5) * 100),
                "pct_at_5": float(np.mean(arr >= 4.99) * 100),
                "q25": float(np.percentile(arr, 25)),
                "q75": float(np.percentile(arr, 75)),
                "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
            }

    # Print summary
    print(f"\n{'Probe':<12} {'Judge':<22} {'Mean':>5} {'Med':>5} {'Std':>5} {'>4.0':>5} {'>4.5':>5} {'=5.0':>5}")
    print("-" * 70)
    for probe in ALL_PROBES:
        for judge in ALL_JUDGES:
            d = distributions.get((probe, judge))
            if d is None:
                continue
            print(f"{probe:<12} {judge:<22} {d['mean']:>5.2f} {d['median']:>5.2f} {d['std']:>5.2f} "
                  f"{d['pct_above_4']:>4.0f}% {d['pct_above_4.5']:>4.0f}% {d['pct_at_5']:>4.0f}%")

    # Overall ceiling effect summary
    all_scores_flat = []
    for (probe, judge), d in distributions.items():
        pairs = load_eval_pairs(probe, judge)
        if pairs:
            for pid, versions in pairs.items():
                for v in ["original", "perturbed"]:
                    s = versions.get(v, {}).get("score")
                    if s is not None:
                        all_scores_flat.append(s)

    if all_scores_flat:
        arr = np.array(all_scores_flat)
        print(f"\nOVERALL (n={len(arr)}): mean={np.mean(arr):.3f} median={np.median(arr):.3f} "
              f"std={np.std(arr, ddof=1):.3f} >4.0={np.mean(arr>4)*100:.1f}% >4.5={np.mean(arr>4.5)*100:.1f}%")

    return distributions


def analyze_recovery_5dim():
    """5-dimension analysis for Recovery Bias per judge."""
    print("\n" + "=" * 70)
    print("RECOVERY BIAS: 5-DIMENSION ANALYSIS")
    print("=" * 70)

    results = {}
    for judge in ALL_JUDGES:
        pairs = load_eval_pairs("recovery", judge)
        if pairs is None:
            continue

        judge_dims = {}
        for dim in DIMENSIONS:
            orig, pert = extract_dimension_scores(pairs, dim)
            if len(orig) < 10:
                continue

            delta = float(np.mean(pert - orig))
            wtest = wilcoxon_test(orig, pert)
            cd = cliffs_delta_paired(orig, pert)
            rbr = rank_biserial_r(orig, pert)

            judge_dims[dim] = {
                "n": int(len(orig)),
                "mean_delta": delta,
                "cliffs_delta": cd,
                "rank_biserial_r": rbr,
                "cohens_d": cohens_d_paired(orig, pert),
                "p_value": wtest["p_value"],
            }

        results[judge] = judge_dims

    # Print table
    for judge, dims in results.items():
        print(f"\n  {judge}:")
        print(f"    {'Dimension':<25} {'δ':>7} {'Cliff':>7} {'r_rb':>7} {'d':>7} {'p':>12}")
        for dim in DIMENSIONS:
            if dim in dims:
                d = dims[dim]
                sig = "***" if d["p_value"] < 0.001 else "**" if d["p_value"] < 0.01 else "*" if d["p_value"] < 0.05 else "ns"
                print(f"    {dim:<25} {d['mean_delta']:>+7.3f} {d['cliffs_delta']:>+7.3f} "
                      f"{d['rank_biserial_r']:>+7.3f} {d['cohens_d']:>+7.3f} {d['p_value']:>10.2e}{sig}")

    return results


def analyze_recency_gradient():
    """Recency position gradient analysis: compare p25 vs p50 vs p75."""
    print("\n" + "=" * 70)
    print("RECENCY BIAS: POSITION GRADIENT ANALYSIS")
    print("=" * 70)

    results = {}
    for judge in ALL_JUDGES:
        pairs = load_eval_pairs("recency", judge)
        if pairs is None:
            continue

        # Split by position
        position_data = {"p25": ([], []), "p50": ([], []), "p75": ([], [])}
        for pid, versions in pairs.items():
            o = versions.get("original", {}).get("score")
            p = versions.get("perturbed", {}).get("score")
            if o is None or p is None:
                continue

            for pos in ["p25", "p50", "p75"]:
                if pid.endswith(f"_{pos}"):
                    position_data[pos][0].append(o)
                    position_data[pos][1].append(p)
                    break

        judge_results = {}
        for pos in ["p25", "p50", "p75"]:
            orig_list, pert_list = position_data[pos]
            if len(orig_list) < 10:
                continue
            orig = np.array(orig_list)
            pert = np.array(pert_list)
            delta = float(np.mean(pert - orig))
            wtest = wilcoxon_test(orig, pert)

            judge_results[pos] = {
                "n": int(len(orig)),
                "mean_delta": delta,
                "cliffs_delta": cliffs_delta_paired(orig, pert),
                "rank_biserial_r": rank_biserial_r(orig, pert),
                "cohens_d": cohens_d_paired(orig, pert),
                "p_value": wtest["p_value"],
            }

        # Test gradient: is p75 significantly worse than p25?
        if "p25" in judge_results and "p75" in judge_results:
            p25_diffs = np.array(position_data["p25"][1]) - np.array(position_data["p25"][0])
            p75_diffs = np.array(position_data["p75"][1]) - np.array(position_data["p75"][0])
            # Use Mann-Whitney U to compare effect magnitudes
            if len(p25_diffs) >= 10 and len(p75_diffs) >= 10:
                u_stat, u_p = stats.mannwhitneyu(p75_diffs, p25_diffs, alternative="less")
                judge_results["gradient_test"] = {
                    "u_statistic": float(u_stat),
                    "p_value": float(u_p),
                    "p75_mean_delta": judge_results["p75"]["mean_delta"],
                    "p25_mean_delta": judge_results["p25"]["mean_delta"],
                    "gradient": judge_results["p75"]["mean_delta"] - judge_results["p25"]["mean_delta"],
                }

        results[judge] = judge_results

    # Print table
    for judge, positions in results.items():
        print(f"\n  {judge}:")
        print(f"    {'Position':<8} {'n':>4} {'δ':>7} {'Cliff':>7} {'r_rb':>7} {'d':>7} {'p':>12}")
        for pos in ["p25", "p50", "p75"]:
            if pos in positions:
                d = positions[pos]
                sig = "***" if d["p_value"] < 0.001 else "**" if d["p_value"] < 0.01 else "*" if d["p_value"] < 0.05 else "ns"
                print(f"    {pos:<8} {d['n']:>4} {d['mean_delta']:>+7.3f} {d['cliffs_delta']:>+7.3f} "
                      f"{d['rank_biserial_r']:>+7.3f} {d['cohens_d']:>+7.3f} {d['p_value']:>10.2e}{sig}")

        if "gradient_test" in positions:
            gt = positions["gradient_test"]
            sig = "***" if gt["p_value"] < 0.001 else "**" if gt["p_value"] < 0.01 else "*" if gt["p_value"] < 0.05 else "ns"
            print(f"    Gradient (p75-p25): {gt['gradient']:+.3f}, Mann-Whitney p={gt['p_value']:.4f}{sig}")

    return results


def analyze_outcome_bidirectional():
    """Bidirectional analysis for Outcome Bias: resolved vs unresolved subgroups."""
    print("\n" + "=" * 70)
    print("OUTCOME BIAS: BIDIRECTIONAL ANALYSIS")
    print("=" * 70)

    meta = load_probe_metadata("outcome")
    results = {}

    for judge in ALL_JUDGES:
        pairs = load_eval_pairs("outcome", judge)
        if pairs is None:
            continue

        resolved_orig, resolved_pert = [], []
        unresolved_orig, unresolved_pert = [], []

        for pid, versions in pairs.items():
            o = versions.get("original", {}).get("score")
            p = versions.get("perturbed", {}).get("score")
            if o is None or p is None:
                continue

            pair_meta = meta.get(pid, {})
            if pair_meta.get("original_resolved"):
                resolved_orig.append(o)
                resolved_pert.append(p)
            else:
                unresolved_orig.append(o)
                unresolved_pert.append(p)

        judge_result = {}
        for name, o_list, p_list in [
            ("resolved", resolved_orig, resolved_pert),
            ("unresolved", unresolved_orig, unresolved_pert),
        ]:
            if len(o_list) < 10:
                continue
            orig = np.array(o_list)
            pert = np.array(p_list)
            delta = float(np.mean(pert - orig))
            wtest = wilcoxon_test(orig, pert)

            judge_result[name] = {
                "n": int(len(orig)),
                "mean_delta": delta,
                "cliffs_delta": cliffs_delta_paired(orig, pert),
                "rank_biserial_r": rank_biserial_r(orig, pert),
                "cohens_d": cohens_d_paired(orig, pert),
                "p_value": wtest["p_value"],
            }

        if judge_result:
            results[judge] = judge_result

    # Print table
    print(f"\n{'Judge':<22} {'Direction':<12} {'n':>4} {'δ':>7} {'Cliff':>7} {'d':>7} {'p':>12}")
    print("-" * 75)
    for judge in ALL_JUDGES:
        if judge not in results:
            continue
        for direction in ["resolved", "unresolved"]:
            if direction not in results[judge]:
                continue
            d = results[judge][direction]
            sig = "***" if d["p_value"] < 0.001 else "**" if d["p_value"] < 0.01 else "*" if d["p_value"] < 0.05 else "ns"
            print(f"{judge:<22} {direction:<12} {d['n']:>4} {d['mean_delta']:>+7.3f} "
                  f"{d['cliffs_delta']:>+7.3f} {d['cohens_d']:>+7.3f} {d['p_value']:>10.2e}{sig}")

    return results


# ============================================================
# Main Analysis
# ============================================================

def main():
    print("=" * 70)
    print("PAPER B: COMPREHENSIVE STATISTICAL ANALYSIS v2")
    print("Primary effect sizes: Cliff's delta, rank-biserial r")
    print("Secondary: Cohen's d (for comparison)")
    print("=" * 70)

    # 1. Main analysis: all probe × judge
    all_results = []
    all_p_values = []

    for probe in ALL_PROBES:
        print(f"\n{'='*50}")
        print(f"PROBE: {probe}" + (" (Position Sensitivity)" if probe == "peakend" else ""))
        print(f"{'='*50}")

        for judge in ALL_JUDGES:
            result = analyze_probe_judge(probe, judge)
            if result is None:
                continue

            all_results.append(result)
            all_p_values.append(result["p_value"])

            sig = "***" if result["p_value"] < 0.001 else "**" if result["p_value"] < 0.01 else "*" if result["p_value"] < 0.05 else "ns"
            print(f"  {judge:<22}: δ={result['mean_delta']:+.3f} Cliff={result['cliffs_delta']:+.3f} "
                  f"r_rb={result['rank_biserial_r']:+.3f} d={result['cohens_d']:+.3f} "
                  f"p={result['p_value']:.2e}{sig} n={result['n_pairs']}")

    # 2. Unified BH FDR correction across ALL tests
    if all_p_values:
        significant_bh = bh_fdr_correction(all_p_values, alpha=0.05)
        print(f"\n{'='*50}")
        print(f"UNIFIED BH FDR CORRECTION (α=0.05, m={len(all_p_values)} tests)")
        print(f"{'='*50}")
        n_sig = sum(significant_bh)
        print(f"Significant after correction: {n_sig}/{len(all_p_values)}")
        for i, (result, sig) in enumerate(zip(all_results, significant_bh)):
            result["bh_significant"] = sig
            status = "SIG" if sig else " ns"
            print(f"  [{status}] {result['probe']:<12} × {result['judge']:<22}: "
                  f"p={result['p_value']:.2e} Cliff={result['cliffs_delta']:+.3f}")

    # 3. Attrition analysis
    attrition = analyze_attrition()

    # 4. Score distribution / ceiling effect
    distributions = analyze_score_distribution()

    # 5. Recovery 5-dimension analysis
    recovery_5dim = analyze_recovery_5dim()

    # 6. Recency gradient analysis
    recency_gradient = analyze_recency_gradient()

    # 7. Bidirectional outcome bias
    outcome_bidir = analyze_outcome_bidirectional()

    # ============================================================
    # Save all results
    # ============================================================

    # Main results
    with open(OUTPUT_DIR / "main_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Attrition
    attrition_json = {f"{k[0]}_{k[1]}": v for k, v in attrition.items()}
    with open(OUTPUT_DIR / "attrition.json", "w") as f:
        json.dump(attrition_json, f, indent=2)

    # Score distributions
    dist_json = {f"{k[0]}_{k[1]}": v for k, v in distributions.items()}
    with open(OUTPUT_DIR / "score_distributions.json", "w") as f:
        json.dump(dist_json, f, indent=2)

    # Recovery 5-dim
    with open(OUTPUT_DIR / "recovery_5dim.json", "w") as f:
        json.dump(recovery_5dim, f, indent=2)

    # Recency gradient
    with open(OUTPUT_DIR / "recency_gradient.json", "w") as f:
        json.dump(recency_gradient, f, indent=2)

    # Outcome bidirectional
    with open(OUTPUT_DIR / "outcome_bidirectional.json", "w") as f:
        json.dump(outcome_bidir, f, indent=2)

    print(f"\n{'='*50}")
    print(f"All results saved to {OUTPUT_DIR}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
