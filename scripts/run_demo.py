"""Run one Incident Commander episode with the scripted policy."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from incident_commander_env.demo_agents import run_scripted_response


def main() -> None:
    result = run_scripted_response(seed=7, difficulty="mixed")
    compact = {
        "scenario_id": result["scenario_id"],
        "final_score": result["final_score"],
        "resolved": result["resolved"],
        "steps": len(result["transcript"]) - 1,
        "final_message": result["transcript"][-1]["message"],
        "rubric": result["transcript"][-1]["rubric_scores"],
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
