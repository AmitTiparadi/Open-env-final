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
from incident_commander_env.scenarios import IncidentScenario, generate_scenario
from incident_commander_env.server.incident_environment import IncidentCommanderEnvironment

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OUTPUT = ROOT / "outputs" / "grpo_incident_commander"
VALID_TOOL_NAMES = {
    "list_tools",
    "check_metrics",
    "query_logs",
    "share_note",
    "submit_root_cause",
    "deploy_fix",
    "send_update",
    "finish_incident",
}
TOOL_ALIASES = {
    "check_metric": "check_metrics",
    "metrics": "check_metrics",
    "logs": "query_logs",
    "query_log": "query_logs",
    "note": "share_note",
    "share": "share_note",
    "root_cause": "submit_root_cause",
    "submit": "submit_root_cause",
    "fix": "deploy_fix",
    "deploy": "deploy_fix",
    "update": "send_update",
    "communicate": "send_update",
    "finish": "finish_incident",
    "close": "finish_incident",
}
ROLE_ALIASES = {
    "monitoring": AgentRole.MONITOR.value,
    "monitor": AgentRole.MONITOR.value,
    "investigate": AgentRole.INVESTIGATOR.value,
    "investigator": AgentRole.INVESTIGATOR.value,
    "remediate": AgentRole.REMEDIATOR.value,
    "remediator": AgentRole.REMEDIATOR.value,
    "communicate": AgentRole.COMMUNICATOR.value,
    "communicator": AgentRole.COMMUNICATOR.value,
}
_DEBUG_REWARD_COUNT = 0

SYSTEM_PROMPT = """You are an incident-response policy.
Return a JSON list of tool calls. Each call must contain tool_name, agent_role,
and arguments. Use only evidence from tool outputs and shared notes.
"""


def render_prompt(scenario: IncidentScenario) -> list[dict[str, str]]:
    alerts = "\n".join(f"- {alert}" for alert in scenario.alerts)
    user_prompt = f"""Incident started.
Scenario id: {scenario.scenario_id}
Difficulty: {scenario.difficulty}
Affected service: {scenario.affected_service}
Visible alerts:
{alerts}

Produce a complete incident-response JSON list. Prefer this order:
check_metrics/query_logs, share_note, submit_root_cause, deploy_fix,
send_update, finish_incident. Use only valid tool names and roles.
"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", completion))
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(completion)


def normalize_name(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def normalize_tool_name(value: Any) -> str:
    name = normalize_name(value)
    return TOOL_ALIASES.get(name, name)


def normalize_role(value: Any) -> str:
    role = normalize_name(value)
    return ROLE_ALIASES.get(role, role)


def json_values_from_text(text: str) -> list[Any]:
    decoder = json.JSONDecoder()
    values = []
    index = 0
    while index < len(text):
        char = text[index]
        if char not in "[{":
            index += 1
            continue
        try:
            value, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            index += 1
            continue
        values.append(value)
        index += max(end, 1)
    return values


def candidate_action_items(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in ("actions", "tool_calls", "calls", "steps"):
        if isinstance(payload.get(key), list):
            return payload[key]
    if any(key in payload for key in ("tool_name", "tool", "name", "agent_role", "role")):
        return [payload]
    return []


def normalize_action_item(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    if isinstance(item.get("action"), dict):
        nested = normalize_action_item(item["action"])
        if nested:
            return nested

    arguments = (
        item.get("arguments")
        or item.get("args")
        or item.get("parameters")
        or item.get("input")
        or {}
    )
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"text": arguments}
    if not isinstance(arguments, dict):
        arguments = {"value": arguments}

    tool_name = item.get("tool_name") or item.get("tool") or item.get("name")
    agent_role = (
        item.get("agent_role")
        or item.get("role")
        or item.get("agent")
        or item.get("actor")
        or arguments.pop("agent_role", None)
        or arguments.pop("role", None)
    )
    normalized = {
        "tool_name": normalize_tool_name(tool_name),
        "agent_role": normalize_role(agent_role),
        "arguments": arguments,
    }
    if not normalized["tool_name"] or not normalized["agent_role"]:
        return None
    return normalized


def parse_actions(completion: Any) -> list[IncidentAction]:
    completion_text = completion_to_text(completion)
    actions = []
    for payload in json_values_from_text(completion_text):
        for item in candidate_action_items(payload):
            normalized = normalize_action_item(item)
            if not normalized:
                continue
            try:
                actions.append(IncidentAction.model_validate(normalized))
            except Exception:
                continue
            if len(actions) >= 8:
                return actions
    return actions[:8]


def format_fallback_reward(completion: Any) -> float:
    text = completion_to_text(completion)
    text_lower = text.lower()
    reward = -0.05
    if json_values_from_text(text):
        reward += 0.015
    mentioned_tools = {
        tool_name for tool_name in VALID_TOOL_NAMES if tool_name in text_lower
    }
    reward += min(0.035, 0.005 * len(mentioned_tools))
    if "tool_name" in text_lower or "agent_role" in text_lower:
        reward += 0.01
    return round(reward, 4)


def rollout_completion(
    completion: Any,
    seed: int = 0,
    difficulty: str = "mixed",
    scenario_id: str | None = None,
) -> float:
    env = IncidentCommanderEnvironment()
    obs = env.reset(seed=seed, difficulty=difficulty, scenario_id=scenario_id)
    actions = parse_actions(completion)
    if not actions:
        return format_fallback_reward(completion)

    tool_names = [action.tool_name for action in actions]
    valid_tool_count = sum(tool_name in VALID_TOOL_NAMES for tool_name in tool_names)
    reward = min(0.05, 0.006 * valid_tool_count)
    if "submit_root_cause" in tool_names:
        root_cause_index = tool_names.index("submit_root_cause")
        prior_tools = set(tool_names[:root_cause_index])
        if prior_tools & {"check_metrics", "query_logs", "share_note"}:
            reward += 0.04
    if {"deploy_fix", "send_update", "finish_incident"}.issubset(set(tool_names)):
        reward += 0.03

    for action in actions:
        obs = env.step(action)
        reward += float(obs.reward or 0.0)
        if obs.done:
            break
    reward += obs.rubric_scores.total
    return reward


def maybe_debug_reward(
    completion: Any,
    actions: list[IncidentAction],
    reward: float,
    scenario_id: str | None,
) -> None:
    global _DEBUG_REWARD_COUNT
    limit = int(os.getenv("GRPO_DEBUG_REWARD_SAMPLES", "0"))
    if _DEBUG_REWARD_COUNT >= limit:
        return
    _DEBUG_REWARD_COUNT += 1
    payload = {
        "debug_reward_sample": _DEBUG_REWARD_COUNT,
        "scenario_id": scenario_id,
        "reward": round(float(reward), 4),
        "parsed_actions": [action.model_dump(mode="json") for action in actions],
        "completion_preview": completion_to_text(completion)[:1200],
    }
    print(json.dumps(payload, ensure_ascii=True), flush=True)


def incident_reward_func(completions: list[str], **kwargs: Any) -> list[float]:
    seeds = kwargs.get("seed")
    scenario_ids = kwargs.get("scenario_id")
    difficulties = kwargs.get("difficulty")

    def value_at(values: Any, index: int, default: Any) -> Any:
        if values is None:
            return default
        if isinstance(values, list):
            if not values:
                return default
            return values[index % len(values)]
        return values

    rewards = []
    for i, completion in enumerate(completions):
        scenario_id = value_at(scenario_ids, i, None)
        reward = rollout_completion(
            completion,
            seed=int(value_at(seeds, i, i)),
            difficulty=str(value_at(difficulties, i, "mixed")),
            scenario_id=scenario_id,
        )
        maybe_debug_reward(completion, parse_actions(completion), reward, scenario_id)
        rewards.append(reward)
    return rewards


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


def reward_probe() -> None:
    examples = {
        "ideal": json.dumps(
            [
                {
                    "tool_name": "check_metrics",
                    "agent_role": "monitor",
                    "arguments": {"service": "checkout-api"},
                },
                {
                    "tool_name": "query_logs",
                    "agent_role": "investigator",
                    "arguments": {"service": "checkout-api"},
                },
                {
                    "tool_name": "submit_root_cause",
                    "agent_role": "investigator",
                    "arguments": {
                        "root_cause": "bad_deploy_memory_leak",
                        "evidence": "checkout-api:v42 heap_growth_mb memory gc_pause_ms",
                    },
                },
                {
                    "tool_name": "deploy_fix",
                    "agent_role": "remediator",
                    "arguments": {"fix_id": "rollback_checkout_api_v42"},
                },
                {
                    "tool_name": "send_update",
                    "agent_role": "communicator",
                    "arguments": {
                        "message": (
                            "checkout-api impact is mitigated. Root cause was "
                            "bad_deploy_memory_leak. Rollback deployed."
                        )
                    },
                },
                {
                    "tool_name": "finish_incident",
                    "agent_role": "communicator",
                    "arguments": {"summary": "checkout-api incident mitigated"},
                },
            ]
        ),
        "fenced_aliases": """```json
{"actions": [
  {"tool": "metrics", "role": "monitor", "args": {"service": "checkout-api"}},
  {"tool": "logs", "role": "investigator", "args": {"service": "checkout-api"}},
  {"tool": "root_cause", "role": "investigator", "args": {"root_cause": "bad deploy memory leak", "evidence": "checkout-api:v42 heap_growth_mb gc_pause_ms"}},
  {"tool": "fix", "role": "remediator", "args": {"fix_id": "rollback_checkout_api_v42"}}
]}
```""",
        "openenv_style": json.dumps(
            {
                "actions": [
                    {
                        "type": "call_tool",
                        "tool_name": "submit_root_cause",
                        "arguments": {
                            "agent_role": "investigator",
                            "root_cause": "bad_deploy_memory_leak",
                            "evidence": "checkout-api:v42 heap_growth_mb gc_pause_ms",
                        },
                    }
                ]
            }
        ),
        "tool_words_no_json": (
            "I will check_metrics, query_logs, submit_root_cause, deploy_fix, "
            "send_update, and finish_incident."
        ),
    }
    rows = []
    for name, completion in examples.items():
        actions = parse_actions(completion)
        reward = rollout_completion(
            completion,
            scenario_id="checkout_bad_deploy_memory_leak",
        )
        rows.append(
            {
                "name": name,
                "parsed_actions": len(actions),
                "tools": [action.tool_name for action in actions],
                "reward": round(float(reward), 4),
            }
        )
    print(json.dumps(rows, indent=2))


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
    if args.debug_reward_samples:
        os.environ["GRPO_DEBUG_REWARD_SAMPLES"] = str(args.debug_reward_samples)
    prompts = []
    for i in range(args.num_prompts):
        scenario = generate_scenario(seed=i, difficulty=args.difficulty)
        prompts.append(
            {
                "prompt": render_prompt(scenario),
                "seed": i,
                "scenario_id": scenario.scenario_id,
                "difficulty": scenario.difficulty,
            }
        )
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
    parser.add_argument("--reward-probe", action="store_true")
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
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--difficulty", default="mixed")
    parser.add_argument("--debug-reward-samples", type=int, default=0)
    args = parser.parse_args()
    if args.push_to_hub and not args.hub_model_id:
        raise SystemExit("--hub-model-id is required when --push-to-hub is set")
    if args.dry_run:
        dry_run()
    elif args.reward_probe:
        reward_probe()
    else:
        train(args)


if __name__ == "__main__":
    main()
