"""Compare random and scripted policies; useful before RL training is available."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from incident_commander_env.demo_agents import run_random_response, run_scripted_response


def main() -> None:
    out_dir = Path("outputs/evals")
    plot_dir = Path("outputs/plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in range(20):
        random_result = run_random_response(seed=seed, difficulty="mixed")
        scripted_result = run_scripted_response(seed=seed, difficulty="mixed")
        rows.append(
            {
                "seed": seed,
                "policy": "random",
                "scenario_id": random_result["scenario_id"],
                "score": random_result["final_score"],
                "resolved": random_result["resolved"],
            }
        )
        rows.append(
            {
                "seed": seed,
                "policy": "scripted",
                "scenario_id": scripted_result["scenario_id"],
                "score": scripted_result["final_score"],
                "resolved": scripted_result["resolved"],
            }
        )

    csv_path = out_dir / "baseline_vs_scripted.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["seed", "policy", "scenario_id", "score", "resolved"],
        )
        writer.writeheader()
        writer.writerows(rows)

    try:
        import matplotlib.pyplot as plt

        policies = ["random", "scripted"]
        values = [
            mean(row["score"] for row in rows if row["policy"] == policy)
            for policy in policies
        ]
        plt.figure(figsize=(6, 4))
        plt.bar(policies, values, color=["#6b7280", "#0f766e"])
        plt.ylabel("Mean episode reward")
        plt.xlabel("Policy")
        plt.title("Incident Commander smoke evaluation")
        plt.ylim(-0.4, 1.1)
        plt.tight_layout()
        plt.savefig(plot_dir / "baseline_vs_scripted.png", dpi=160)
    except Exception as exc:
        print(f"Plot generation skipped: {exc}")

    by_policy = {
        policy: mean(row["score"] for row in rows if row["policy"] == policy)
        for policy in ("random", "scripted")
    }
    print(f"Wrote {csv_path}")
    print(by_policy)


if __name__ == "__main__":
    main()
