"""Model configuration for Incident Commander agents."""

from __future__ import annotations

import os


DEFAULT_AGENT_MODEL_ID = os.getenv("INCIDENT_AGENT_MODEL_ID", "Qwen/Qwen3.5-9B")
DEFAULT_JUDGE_MODEL_ID = os.getenv("INCIDENT_JUDGE_MODEL_ID", "google/gemma-4-31B-it")
DEFAULT_JUDGE_ENSEMBLE_SIZE = int(os.getenv("INCIDENT_JUDGE_ENSEMBLE_SIZE", "10"))

AGENT_MODEL_BY_ROLE = {
    "monitor": DEFAULT_AGENT_MODEL_ID,
    "investigator": DEFAULT_AGENT_MODEL_ID,
    "remediator": DEFAULT_AGENT_MODEL_ID,
    "communicator": DEFAULT_AGENT_MODEL_ID,
    "judge": DEFAULT_JUDGE_MODEL_ID,
}

