"""Convenience entry point for the Incident Commander environment."""

from __future__ import annotations

import json

from incident_commander_env.demo_agents import run_scripted_response


def main() -> None:
    result = run_scripted_response(seed=7, difficulty="mixed")
    print(
        json.dumps(
            {
                "scenario_id": result["scenario_id"],
                "final_score": result["final_score"],
                "resolved": result["resolved"],
                "steps": len(result["transcript"]) - 1,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
