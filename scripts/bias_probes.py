"""
Paper B: Bias Probe Framework for Agent Trajectory Evaluation.
Implements 5 trajectory-specific bias probes.

Data source: SWE-smith-trajectories (public, HuggingFace)
"""
import json
import copy
import random
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TrajectoryPair:
    """A pair of (original, perturbed) trajectories for bias testing."""
    pair_id: str
    bias_type: str
    original_id: str
    original_messages: list
    perturbed_messages: list
    original_resolved: bool
    perturbation_description: str
    # metadata
    model: str = ""
    instance_id: str = ""


def load_swe_smith_trajectories(split="tool", max_n=500):
    """Load SWE-smith trajectories from HuggingFace (streaming)."""
    from datasets import load_dataset

    print(f"Loading SWE-smith-trajectories ({split} split)...")
    ds = load_dataset("SWE-bench/SWE-smith-trajectories", split=split, streaming=True)

    trajectories = []
    for ex in ds:
        messages = json.loads(ex["messages"]) if isinstance(ex["messages"], str) else ex["messages"]

        # Filter: need at least 5 steps and reasonable length
        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
        if len(assistant_msgs) < 3:
            continue

        # Estimate token count (rough: 4 chars per token)
        total_chars = sum(len(json.dumps(m)) for m in messages)
        est_tokens = total_chars // 4

        # Skip very long trajectories (>50K tokens for manageability)
        if est_tokens > 50000:
            continue

        trajectories.append({
            "traj_id": ex["traj_id"],
            "instance_id": ex["instance_id"],
            "messages": messages,
            "resolved": ex["resolved"],
            "model": ex["model"],
            "n_steps": len(assistant_msgs),
            "est_tokens": est_tokens,
        })

        if len(trajectories) >= max_n:
            break

    print(f"Loaded {len(trajectories)} trajectories "
          f"({sum(1 for t in trajectories if t['resolved'])} resolved)")
    return trajectories


# ── Probe 1: Outcome Bias ──────────────────────────────────────────
def create_outcome_bias_pairs(trajectories, n_pairs=200):
    """
    Outcome Bias: Does knowing the final outcome affect process quality judgment?

    Method: Take trajectories and flip the outcome signal.
    - Successful trajectories: append "Task failed" at the end
    - Failed trajectories: append "Task completed successfully" at the end
    Keep all intermediate steps identical.
    """
    pairs = []
    random.seed(42)

    # Split by outcome
    resolved = [t for t in trajectories if t["resolved"]]
    unresolved = [t for t in trajectories if not t["resolved"]]

    # Sample balanced
    n_each = min(n_pairs // 2, len(resolved), len(unresolved))
    sample_resolved = random.sample(resolved, n_each)
    sample_unresolved = random.sample(unresolved, n_each)

    for i, traj in enumerate(sample_resolved + sample_unresolved):
        original_msgs = copy.deepcopy(traj["messages"])
        perturbed_msgs = copy.deepcopy(traj["messages"])

        if traj["resolved"]:
            # Add false failure signal at end
            perturbed_msgs.append({
                "role": "system",
                "content": "Task status: FAILED. The agent did not complete the task successfully."
            })
            original_msgs.append({
                "role": "system",
                "content": "Task status: SUCCESS. The agent completed the task successfully."
            })
            desc = "resolved->false_failure_appended"
        else:
            # Add false success signal at end
            perturbed_msgs.append({
                "role": "system",
                "content": "Task status: SUCCESS. The agent completed the task successfully."
            })
            original_msgs.append({
                "role": "system",
                "content": "Task status: FAILED. The agent did not complete the task successfully."
            })
            desc = "unresolved->false_success_appended"

        pairs.append(TrajectoryPair(
            pair_id=f"outcome_{i:04d}",
            bias_type="outcome_bias",
            original_id=traj["traj_id"],
            original_messages=original_msgs,
            perturbed_messages=perturbed_msgs,
            original_resolved=traj["resolved"],
            perturbation_description=desc,
            model=traj["model"],
            instance_id=traj["instance_id"],
        ))

    print(f"Created {len(pairs)} outcome bias pairs")
    return pairs


# ── Probe 2: Trajectory Length Bias ────────────────────────────────
def create_length_bias_pairs(trajectories, n_pairs=200, pad_counts=(2, 4, 6)):
    """
    Length Bias: Are longer trajectories scored differently than shorter ones?

    Method: Insert K redundant-but-correct steps (e.g., "checking status",
    "reviewing current state") that don't change the outcome.
    """
    pairs = []
    random.seed(43)
    sample = random.sample(trajectories, min(n_pairs, len(trajectories)))

    FILLER_STEPS = [
        {"role": "assistant", "content": "Let me verify the current state of the repository before proceeding."},
        {"role": "user", "content": "[File listing shows no changes from the expected state]"},
        {"role": "assistant", "content": "Good, the state looks correct. Let me continue with the next step."},
        {"role": "user", "content": "[Status confirmed - no unexpected changes detected]"},
    ]

    for i, traj in enumerate(sample):
        for k in pad_counts:
            perturbed_msgs = copy.deepcopy(traj["messages"])

            # Find midpoint to insert filler
            assistant_indices = [j for j, m in enumerate(perturbed_msgs)
                               if m.get("role") == "assistant"]
            if len(assistant_indices) < 2:
                continue

            # Insert filler pairs at midpoint
            mid_idx = assistant_indices[len(assistant_indices) // 2]
            filler = FILLER_STEPS[:k]  # Take k steps
            for fi, filler_step in enumerate(reversed(filler)):
                perturbed_msgs.insert(mid_idx + 1, copy.deepcopy(filler_step))

            pairs.append(TrajectoryPair(
                pair_id=f"length_{i:04d}_k{k}",
                bias_type=f"length_bias_k{k}",
                original_id=traj["traj_id"],
                original_messages=traj["messages"],
                perturbed_messages=perturbed_msgs,
                original_resolved=traj["resolved"],
                perturbation_description=f"inserted_{k}_filler_steps_at_midpoint",
                model=traj["model"],
                instance_id=traj["instance_id"],
            ))

    print(f"Created {len(pairs)} length bias pairs")
    return pairs


# ── Probe 3: Error Recovery Bias ──────────────────────────────────
def create_error_recovery_pairs(trajectories, n_pairs=200):
    """
    Error Recovery Bias: Is an error-then-fix sequence penalized vs clean execution?

    Method: Insert an error+recovery subsequence into clean trajectories.
    """
    pairs = []
    random.seed(44)

    # Only use resolved trajectories (clean execution)
    resolved = [t for t in trajectories if t["resolved"]]
    sample = random.sample(resolved, min(n_pairs, len(resolved)))

    ERROR_RECOVERY_SEQ = [
        {"role": "assistant", "content": "Let me try applying this fix.\n\n```bash\ncd /nonexistent_directory\n```"},
        {"role": "user", "content": "bash: cd: /nonexistent_directory: No such file or directory"},
        {"role": "assistant", "content": "I see, that directory doesn't exist. Let me correct the path and navigate to the right location."},
        {"role": "user", "content": "[Navigation successful - now in the correct directory]"},
    ]

    for i, traj in enumerate(sample):
        perturbed_msgs = copy.deepcopy(traj["messages"])

        # Find a good insertion point (after first few steps)
        assistant_indices = [j for j, m in enumerate(perturbed_msgs)
                           if m.get("role") == "assistant"]
        if len(assistant_indices) < 4:
            continue

        # Insert error-recovery after ~25% of the trajectory
        insert_idx = assistant_indices[max(1, len(assistant_indices) // 4)]
        for fi, step in enumerate(reversed(ERROR_RECOVERY_SEQ)):
            perturbed_msgs.insert(insert_idx + 1, copy.deepcopy(step))

        pairs.append(TrajectoryPair(
            pair_id=f"recovery_{i:04d}",
            bias_type="error_recovery_bias",
            original_id=traj["traj_id"],
            original_messages=traj["messages"],
            perturbed_messages=perturbed_msgs,
            original_resolved=traj["resolved"],
            perturbation_description="error_recovery_inserted_at_25pct",
            model=traj["model"],
            instance_id=traj["instance_id"],
        ))

    print(f"Created {len(pairs)} error recovery bias pairs")
    return pairs


# ── Probe 4: Recency Bias (Serial Position Effect) ────────────────
def create_recency_bias_pairs(trajectories, n_pairs=200):
    """
    Recency Bias: Does the position of an error affect its perceived severity?

    Method: Insert the same error at 25%, 50%, and 75% positions.
    """
    pairs = []
    random.seed(45)

    resolved = [t for t in trajectories if t["resolved"] and t["n_steps"] >= 8]
    sample = random.sample(resolved, min(n_pairs, len(resolved)))

    ERROR_STEP = [
        {"role": "assistant", "content": "I'll check the test results.\n\n```bash\npython -m pytest tests/ -x\n```"},
        {"role": "user", "content": "FAILED tests/test_core.py::test_main - AssertionError: Expected output does not match"},
        {"role": "assistant", "content": "I see there's a test failure. Let me investigate and continue with the approach."},
    ]

    for i, traj in enumerate(sample):
        assistant_indices = [j for j, m in enumerate(traj["messages"])
                           if m.get("role") == "assistant"]
        n_steps = len(assistant_indices)

        for position_pct in [25, 50, 75]:
            perturbed_msgs = copy.deepcopy(traj["messages"])
            insert_idx = assistant_indices[int(n_steps * position_pct / 100)]

            for fi, step in enumerate(reversed(ERROR_STEP)):
                perturbed_msgs.insert(insert_idx + 1, copy.deepcopy(step))

            pairs.append(TrajectoryPair(
                pair_id=f"recency_{i:04d}_p{position_pct}",
                bias_type=f"recency_bias_p{position_pct}",
                original_id=traj["traj_id"],
                original_messages=traj["messages"],
                perturbed_messages=perturbed_msgs,
                original_resolved=traj["resolved"],
                perturbation_description=f"same_error_at_{position_pct}pct",
                model=traj["model"],
                instance_id=traj["instance_id"],
            ))

    print(f"Created {len(pairs)} recency bias pairs")
    return pairs


# ── Probe 5: Peak-End Bias ─────────────────────────────────────────
def create_peak_end_bias_pairs(trajectories, n_pairs=200):
    """
    Peak-End Bias: Do judges disproportionately weight the quality
    of the peak (most critical) step and the final step?

    Based on Kahneman's Peak-End Rule (meta-analysis r=0.581, Alaybek 2022).
    APEMO (arXiv 2602.17910) hypothesizes this exists but lacks empirical validation.

    Method: For each trajectory, introduce identical quality degradation
    at 4 different positions: peak (~50%), end (last), middle (~50%-1), early (~25%).
    If peak-end bias exists, degrading peak or end causes disproportionate score drops.
    """
    pairs = []
    random.seed(46)

    resolved = [t for t in trajectories if t["resolved"] and t["n_steps"] >= 8]
    sample = random.sample(resolved, min(n_pairs, len(resolved)))

    # Degradation: replace an action step with an off-topic/confused action
    DEGRADED_STEP = {
        "role": "assistant",
        "content": "I'm going to apply the fix now.\n\n```bash\ngrep -r 'TODO' . | head -5\n```\n\nLet me search for related TODO items in the codebase first."
    }
    DEGRADED_RESPONSE = {
        "role": "user",
        "content": "[Search returned 5 results, none directly relevant to the current issue]"
    }

    for i, traj in enumerate(sample):
        assistant_indices = [j for j, m in enumerate(traj["messages"])
                           if m.get("role") == "assistant"]
        n = len(assistant_indices)
        if n < 8:
            continue

        # Define 4 positions
        positions = {
            "early": assistant_indices[n // 4],        # 25% position
            "middle": assistant_indices[n // 2 - 1],   # Just before midpoint
            "peak": assistant_indices[n // 2],          # Midpoint (often most complex)
            "end": assistant_indices[-2],               # Second-to-last (last is often wrap-up)
        }

        for pos_name, pos_idx in positions.items():
            perturbed_msgs = copy.deepcopy(traj["messages"])
            # Replace the step at pos_idx with degraded version
            perturbed_msgs[pos_idx] = copy.deepcopy(DEGRADED_STEP)
            if pos_idx + 1 < len(perturbed_msgs):
                perturbed_msgs[pos_idx + 1] = copy.deepcopy(DEGRADED_RESPONSE)

            pairs.append(TrajectoryPair(
                pair_id=f"peakend_{i:04d}_{pos_name}",
                bias_type=f"peak_end_bias_{pos_name}",
                original_id=traj["traj_id"],
                original_messages=traj["messages"],
                perturbed_messages=perturbed_msgs,
                original_resolved=traj["resolved"],
                perturbation_description=f"degraded_step_at_{pos_name}_position",
                model=traj["model"],
                instance_id=traj["instance_id"],
            ))

    print(f"Created {len(pairs)} peak-end bias pairs")
    return pairs


# ── Main: Generate all probes ─────────────────────────────────────
def generate_all_probes(max_trajectories=500):
    """Generate all bias probe pairs and save to disk."""
    trajectories = load_swe_smith_trajectories(max_n=max_trajectories)

    probes = {}
    probes["outcome"] = create_outcome_bias_pairs(trajectories, n_pairs=200)
    probes["length"] = create_length_bias_pairs(trajectories, n_pairs=200)
    probes["recovery"] = create_error_recovery_pairs(trajectories, n_pairs=200)
    probes["recency"] = create_recency_bias_pairs(trajectories, n_pairs=200)
    probes["peakend"] = create_peak_end_bias_pairs(trajectories, n_pairs=200)

    # Save each probe type
    for probe_name, pairs in probes.items():
        output_file = DATA_DIR / f"bias_probe_{probe_name}.jsonl"
        with open(output_file, "w") as f:
            for pair in pairs:
                f.write(json.dumps(asdict(pair), default=str) + "\n")
        print(f"Saved {len(pairs)} pairs to {output_file}")

    # Summary
    print(f"\n{'='*60}")
    print("PROBE GENERATION SUMMARY")
    print(f"{'='*60}")
    total = sum(len(p) for p in probes.values())
    print(f"Total pairs generated: {total}")
    for name, pairs in probes.items():
        print(f"  {name}: {len(pairs)}")
    print(f"Data saved to: {DATA_DIR}")

    return probes


if __name__ == "__main__":
    generate_all_probes(max_trajectories=500)
