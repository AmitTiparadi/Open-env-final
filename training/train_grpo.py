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
import os

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from incident_commander_env.models import AgentRole, IncidentAction
from incident_commander_env.server.incident_environment import IncidentCommanderEnvironment

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OUTPUT = ROOT / "outputs" / "grpo_incident_commander"

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


def configure_special_tokens(model: Any, tokenizer: Any) -> str:
    def token_id(token: str | None) -> int | None:
        if not token:
            return None
        value = tokenizer.convert_tokens_to_ids(token)
        if value is None:
            return None
        if getattr(tokenizer, "unk_token_id", None) is not None and value == tokenizer.unk_token_id:
            return None
        return int(value)

    eos_token = None
    eos_token_id = None
    for candidate in ("<|im_end|>", tokenizer.eos_token, "<|endoftext|>"):
        candidate_id = token_id(candidate)
        if candidate_id is not None:
            eos_token = candidate
            eos_token_id = candidate_id
            break
    if eos_token is None or eos_token_id is None:
        raise ValueError("Could not find a valid EOS token in the tokenizer vocabulary.")

    tokenizer.eos_token = eos_token
    tokenizer.eos_token_id = eos_token_id
    if tokenizer.pad_token is None or token_id(tokenizer.pad_token) is None:
        tokenizer.pad_token = eos_token
        tokenizer.pad_token_id = eos_token_id

    if getattr(model, "config", None) is not None:
        model.config.eos_token_id = eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.eos_token_id = eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    return eos_token


def ensure_peft_model(model: Any, FastLanguageModel: Any, args: argparse.Namespace) -> Any:
    if getattr(model, "peft_config", None) is not None:
        return model
    return FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        use_gradient_checkpointing="unsloth",
    )


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


def train(args: argparse.Namespace) -> None:
    try:
        from datasets import Dataset
        from unsloth import FastLanguageModel
        from trl import GRPOConfig, GRPOTrainer
    except Exception as exc:
        raise SystemExit(
            "Install TRL, Unsloth, datasets, and transformers in the training "
            f"runtime before running full GRPO. Import error: {exc}"
        )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )
    configure_special_tokens(model, tokenizer)
    model = ensure_peft_model(model, FastLanguageModel, args)
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
        output_dir=str(args.output_dir),
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=1,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to=args.report_to,
        run_name=args.run_name,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_strategy="every_save" if args.push_to_hub else "end",
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=dataset,
        reward_funcs=incident_reward_func,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir / "final"))
    if args.push_to_hub:
        trainer.push_to_hub()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model-name", default=os.getenv("RL_BASE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hub-model-id", default=os.getenv("RL_HUB_MODEL_ID"))
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--report-to", default=os.getenv("REPORT_TO", "none"))
    parser.add_argument("--run-name", default="incident-commander-grpo")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    if args.push_to_hub and not args.hub_model_id:
        raise SystemExit("--hub-model-id is required when --push-to-hub is set")
    if args.dry_run:
        dry_run()
    else:
        train(args)


if __name__ == "__main__":
    main()
