"""Minimal TRL/Unsloth GRPO training scaffold for the hackathon.

This is intentionally small and inspectable. It demonstrates how completions
can be rolled through the environment and scored with the same composable
rubrics used by the server. Run `python training/train_grpo.py --dry-run`
locally; run without `--dry-run` in a Colab/HF environment with TRL and Unsloth.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from incident_commander_env.models import AgentRole, IncidentAction
from incident_commander_env.server.incident_environment import IncidentCommanderEnvironment


SYSTEM_PROMPT = """You are an incident-response policy.
Return a JSON list of tool calls. Each call must contain tool_name, agent_role,
and arguments. Use only evidence from tool outputs and shared notes.
"""


def parse_actions(completion: str) -> list[IncidentAction]:
    match = re.search(r"\[[\s\S]*\]", completion)
    if not match:
        return []
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    actions = []
    for item in payload:
        try:
            actions.append(IncidentAction.model_validate(item))
        except Exception:
            continue
    return actions[:8]


def rollout_completion(
    completion: str,
    seed: int = 0,
    difficulty: str = "mixed",
    scenario_id: str | None = None,
) -> float:
    env = IncidentCommanderEnvironment()
    obs = env.reset(seed=seed, difficulty=difficulty, scenario_id=scenario_id)
    reward = 0.0
    for action in parse_actions(completion):
        obs = env.step(action)
        reward += float(obs.reward or 0.0)
        if obs.done:
            break
    reward += obs.rubric_scores.total
    return reward


def incident_reward_func(completions: list[str], **kwargs: Any) -> list[float]:
    seeds = kwargs.get("seed") or list(range(len(completions)))
    return [
        rollout_completion(completion, seed=int(seeds[i % len(seeds)]))
        for i, completion in enumerate(completions)
    ]


def dry_run() -> None:
    example = json.dumps(
        [
            {
                "tool_name": "check_metrics",
                "agent_role": AgentRole.MONITOR.value,
                "arguments": {"service": "checkout-api"},
            },
            {
                "tool_name": "query_logs",
                "agent_role": AgentRole.INVESTIGATOR.value,
                "arguments": {"service": "checkout-api"},
            },
            {
                "tool_name": "submit_root_cause",
                "agent_role": AgentRole.INVESTIGATOR.value,
                "arguments": {
                    "root_cause": "bad deploy memory leak",
                    "evidence": "checkout-api:v42 heap_growth_mb gc_pause_ms",
                },
            },
            {
                "tool_name": "deploy_fix",
                "agent_role": AgentRole.REMEDIATOR.value,
                "arguments": {"fix_id": "rollback_checkout_api_v42"},
            },
        ]
    )
    print(
        {
            "dry_run_reward": rollout_completion(
                example,
                scenario_id="checkout_bad_deploy_memory_leak",
            )
        }
    )


def train() -> None:
    try:
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
        from unsloth import FastLanguageModel
    except Exception as exc:
        raise SystemExit(
            "Install TRL, Unsloth, datasets, and transformers in the training "
            f"runtime before running full GRPO. Import error: {exc}"
        )

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=4096,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        use_gradient_checkpointing="unsloth",
    )
    prompts = [
        {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "An incident has started. Produce tool calls for one "
                        "episode. Available roles: monitor, investigator, "
                        "remediator, communicator."
                    ),
                },
            ],
            "seed": i,
        }
        for i in range(64)
    ]
    dataset = Dataset.from_list(prompts)
    config = GRPOConfig(
        output_dir="outputs/grpo_incident_commander",
        num_generations=4,
        max_prompt_length=1024,
        max_completion_length=1024,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        logging_steps=1,
        max_steps=50,
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=dataset,
        reward_funcs=incident_reward_func,
    )
    trainer.train()
    trainer.save_model("outputs/grpo_incident_commander/final")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    else:
        train()


if __name__ == "__main__":
    main()
