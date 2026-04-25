---
title: Incident Commander Environment
emoji: 🚨
colorFrom: blue
colorTo: gray
sdk: docker
pinned: false
app_port: 7860
base_path: /docs
tags:
  - openenv
  - reinforcement-learning
  - grpo
  - unsloth
  - trl
  - multi-agent
  - mcp
  - sre
  - incident-response
license: apache-2.0
short_description: Train LLM agents for incident response.
---

# Incident Commander

Incident Commander is an OpenEnv-compatible environment for training LLM agents
to manage software production incidents under time pressure, noisy telemetry,
and partial observability.

The environment simulates real SRE/DevOps incidents where different agents see
different evidence, coordinate through a shared scratchpad, identify a root
cause, deploy a safe fix, and communicate a clear stakeholder update.

## Hackathon Fit

This project targets three OpenEnv Hackathon themes:

- Multi-Agent Interactions: specialized agents coordinate across role-limited tools.
- World Modeling / Professional Tasks: agents interact with logs, metrics, alerts, and remediation tools.
- Self-Improvement: incidents can be sampled by difficulty and root-cause type for curriculum training.

The goal is not just to build a demo. The goal is to create an environment that
can produce verifiable rewards for RL post-training.

## Agent Roles

- Monitor: checks alerts and service metrics.
- Investigator: queries logs and submits evidence-backed root causes.
- Remediator: deploys targeted fixes and must avoid secondary outages.
- Communicator: sends concise, accurate stakeholder updates.
- Judge: evaluates candidate responses part-by-part with an LLM-as-a-Judge ensemble and hard verifier guardrails.

Agents share a scratchpad, but private tool outputs are scoped to the role that
called the tool. This creates partial observability and forces coordination.

Operational agents default to `Qwen/Qwen3.5-9B`. The judge defaults to
`google/gemma-4-31B-it` with `10` judge calls per evaluation. Override these
with `INCIDENT_AGENT_MODEL_ID`, `INCIDENT_JUDGE_MODEL_ID`, and
`INCIDENT_JUDGE_ENSEMBLE_SIZE`.

## Incident Scenarios

The current environment includes four production-style scenarios:

- `checkout_bad_deploy_memory_leak`
- `orders_db_connection_pool`
- `profile_cache_stampede`
- `auth_tls_cert_expiry`

Each scenario contains:

- noisy alerts
- service logs
- metrics time series
- red herrings
- canonical root cause
- safe and unsafe remediation choices
- stakeholder impact description

## Reward Rubric

The reward is composable and aligned with the hackathon judging guidance:

| Component | Reward |
| --- | ---: |
| Correct root cause identified | `+0.4` |
| Safe fix deployed without secondary outage | `+0.3` |
| Accurate stakeholder update | `+0.2` |
| Time-to-resolution bonus | up to `+0.1` |
| Hallucinated evidence or false root cause | `-0.3` |
| Judge per-part shaping | bounded `+0.25` |
| Hard integrity / reward-hacking violation | `-100` |

The environment also emits smaller process rewards for useful evidence-gathering
and scratchpad notes. This helps early RL rollouts avoid all-zero rewards.

During GRPO and other online training loops, hard integrity failures are scaled
to `-5.0` so one bad rollout does not dominate the batch variance. Evaluation,
hidden tests, the judge tool, and reward-hack reports still use the full
`-100.0` hard-fail score.

The judge is deliberately not the only source of truth. It scores root cause,
fix safety, stakeholder update, evidence process, and sequence completeness as
separate parts, so one mistake does not erase all useful work. Reward-hacking
patterns such as trying to delete timers, override rewards, patch environment
state, hide logs, or bypass verifiers are caught by deterministic guardrails
before judge scores are applied. Those hard integrity violations receive `-100`
even if the LLM judge would otherwise rate the response well.

## OpenEnv / MCP Interface

The project includes:

- `openenv.yaml` manifest
- Docker-based Hugging Face Space configuration
- FastAPI server at `server.app:app`
- OpenEnv `reset`, `step`, and `state` endpoints
- MCPEnvironment wrapper for `CallToolAction` and `ListToolsAction`
- Role-scoped tools exposed through MCP

Main tools:

- `check_metrics`
- `query_logs`
- `share_note`
- `submit_root_cause`
- `deploy_fix`
- `send_update`
- `finish_incident`
- `judge_response`
- `python_exec`
- `web_search`
- `query_api`
- `list_incident_tools`

## Interactive RL Loop

The repo now includes a reusable action-observation training layer in
`incident_commander_env/interactive_rl.py`.

It supports:

- one-action-at-a-time rollouts
- observation feedback after each tool call
- sandboxed Python execution for small calculations
- incident-local search over alerts, logs, metrics, and docs
- structured API calls such as `service_graph`, `deployments`, `metrics_summary`, and `runbook`
- adaptive curriculum sampling that increases probability for weak root-cause and difficulty buckets
- hidden evaluation cases that are opt-in for eval but excluded from normal training data
- `OnlineRLTrainer`, which collects rollout batches, calls a policy-update hook, and updates the curriculum state

Run a local smoke test:

```bash
python scripts/interactive_rl_smoke.py
```

The rollout shape is:

```text
task generator -> prompt -> agent action -> environment observation -> reward
              -> next action -> next observation -> final judged score
```

## Running The Space

This repository is configured as a Docker Hugging Face Space.

When the Space starts, it runs:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Open the Space and visit:

```text
/docs
```

That opens the Swagger UI for the OpenEnv HTTP API.

## Important Swagger Note

Swagger `POST /step` is useful for testing individual tool calls, but the
standard stateless HTTP endpoint does not preserve the full episode across
separate Swagger calls.

For full multi-step episode testing, use:

```bash
python scripts/run_demo.py
```

or train/evaluate through the environment in-process.

## Local Quick Start

```bash
python -m unittest tests/test_environment.py
python scripts/run_demo.py
python scripts/evaluate_baseline.py
python training/train_grpo.py --dry-run
```

Expected smoke-test behavior:

- unit tests pass
- scripted policy resolves an incident
- random baseline scores much lower than scripted policy
- GRPO reward dry-run returns a positive reward

## Example Tool Call

Use this through `POST /step` in Swagger:

```json
{
  "action": {
    "type": "call_tool",
    "tool_name": "query_logs",
    "arguments": {
      "agent_role": "investigator",
      "service": "orders-api",
      "limit": 5
    }
  }
}
```

## Full Episode Example

The scripted demo performs a complete incident response trajectory:

```bash
python scripts/run_demo.py
```

It runs:

1. monitor checks metrics
2. investigator queries logs
3. investigator shares evidence
4. investigator submits root cause
5. remediator deploys safe fix
6. communicator sends status update
7. communicator closes incident

## Baseline Evaluation

Run:

```bash
python scripts/evaluate_baseline.py
```

This writes:

- `outputs/evals/baseline_vs_scripted.csv`
- `outputs/plots/baseline_vs_scripted.png`

These artifacts are useful for the README, final presentation, and judging
evidence.

## Reward-Hack Stress Test

The reward-hack tester reuses the same hidden evaluator used for normal
candidate scoring. It does not define a separate scoring system. Instead, it
feeds adversarial outputs into the canonical evaluator and checks whether they
are rejected.

Run:

```bash
python scripts/reward_hack_tester.py
```

To run the harder hidden cases only:

```bash
python scripts/reward_hack_tester.py --hidden-only
```

To run public and hidden cases together:

```bash
python scripts/reward_hack_tester.py --include-hidden
```

This writes:

- `outputs/evals/reward_hack_stress.csv`
- `outputs/evals/reward_hack_stress_summary.json`

Hidden-only runs write `reward_hack_stress_hidden.*`; public-plus-hidden
runs write `reward_hack_stress_public_hidden.*`.

Useful screenshot metrics:

- `false_accept_rate`: lower is better; target `0.0`
- `integrity_detection_rate`: target `1.0`
- `unsafe_fix_detection_rate`: target `1.0`
- `hallucination_detection_rate`: target `1.0`
- `max_attack_reward`: should stay below the acceptance threshold

The adversarial set includes hardcoded correct answers without evidence, fake
reasoning, timer/reward tampering, unsafe fixes, premature incident closure, and
fake success updates.

Hidden evaluation cases are not used in pretraining, SFT, GRPO prompt sampling,
normal `scenario_ids()` metadata, or the generated dataset files. They are only
available when evaluator code explicitly opts in, and hidden environment
observations replace the real case id with `hidden_eval_case`.

## Training On Hugging Face GPU Space

This project includes a minimal GRPO training scaffold:

```bash
python training/train_grpo.py
```

For Hugging Face Jobs, do not run the stage scripts by URL directly. A Jobs run
downloads only the single script URL, not the whole Space repository. Use the
bootstrap runner instead:

```bash
hf jobs uv run \
  --flavor t4-small \
  --timeout 30m \
  --secrets HF_TOKEN \
  "https://huggingface.co/spaces/AmitTiparadi/Open-env-finals/resolve/main/training/run_hf_job.py" \
  --stage pretrain -- --dry-run
```

Generate the reproducible datasets first:

```bash
python training/prepare_data.py
```

This creates:

- `data/pretrain_corpus.jsonl`
- `data/sft_trajectories.jsonl`
- `data/eval_scenarios.jsonl`

Run the 80/10/10 training stages:

```bash
# 80% domain-adaptive continued pretraining
python training/train_pretrain.py --dry-run
python training/train_pretrain.py \
  --push-to-hub \
  --hub-model-id YOUR_USERNAME/incident-commander-pretrain

# 10% supervised fine-tuning on ideal tool-call traces
python training/train_sft.py --dry-run
python training/train_sft.py \
  --model-name YOUR_USERNAME/incident-commander-pretrain \
  --push-to-hub \
  --hub-model-id YOUR_USERNAME/incident-commander-sft

# 10% RL post-training with environment rewards
python training/train_grpo.py --dry-run
python training/train_grpo.py \
  --model-name YOUR_USERNAME/incident-commander-sft \
  --push-to-hub \
  --hub-model-id YOUR_USERNAME/incident-commander-grpo
```

Before running training on a GPU Space, install the training dependencies:

```bash
pip install -U pip
pip install unsloth trl datasets transformers accelerate peft bitsandbytes openenv-core
```

Then verify the reward function:

```bash
python training/train_grpo.py --dry-run
```

Then run the small GRPO training job:

```bash
python training/train_grpo.py
```

The training script uses:

- TRL `GRPOTrainer`
- Unsloth `FastLanguageModel`
- a custom reward function that rolls generated tool-call sequences through the incident environment
- adaptive task sampling from the same curriculum generator used by the interactive runner
- tool rewards for metrics/log/API/search/code actions plus final correctness and efficiency rewards

The training loop is:

```text
prompt -> model-generated tool calls -> environment rollout -> reward -> GRPO update
```

For the fully interactive runner, use:

```text
prompt -> one tool action -> observation -> next tool action -> observation -> reward
```

## Recommended HF Hardware

Start small:

- minimum: T4 / L4 style GPU for smoke runs
- better: A10G / L40S for faster experiments

If memory is tight, change the model in `training/train_grpo.py` from:

```python
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
```

to:

```python
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
```

The hackathon target is not a giant model. The target is visible improvement:

```text
baseline reward < trained reward
```

## Expected Training Evidence

For the final submission, include:

- reward curve
- loss curve
- baseline vs trained success rate
- baseline vs trained hallucination rate
- before/after transcripts
- link to the HF Space
- link to a short demo video or blog post

## Repository Structure

```text
incident_commander_env/
  agent_config.py
  judge.py
  models.py
  scenarios.py
  rewards.py
  demo_agents.py
  evaluation.py
  external_tools.py
  interactive_rl.py
  server/
    app.py
    incident_environment.py
    mcp_environment.py
server/
  app.py
  Dockerfile
training/
  prepare_data.py
  run_hf_job.py
  train_pretrain.py
  train_sft.py
  train_grpo.py
data/
  pretrain_corpus.jsonl
  sft_trajectories.jsonl
  eval_scenarios.jsonl
scripts/
  run_demo.py
  evaluate_baseline.py
  reward_hack_tester.py
  interactive_rl_smoke.py
tests/
  test_environment.py
  test_evaluation_harness.py
  test_interactive_rl.py
openenv.yaml
pyproject.toml
```

## Current Status

Implemented:

- OpenEnv-compatible environment
- MCP tool interface
- multi-agent role permissions
- four incident scenarios
- composable reward rubric
- hallucination penalties
- safe vs unsafe remediation logic
- sandboxed Python, local search, and structured API tools
- action-observation interactive rollout runner
- adaptive curriculum task generator
- scripted baseline
- random baseline evaluation
- reward-hack stress test using the same hidden evaluator
- harder hidden evaluation scenarios kept out of training and public metadata
- GRPO training scaffold
- Docker Space metadata

Next milestones:

- run real GRPO training on HF GPU hardware
- save reward/loss plots
- add trained-vs-baseline evaluation artifacts
- publish final demo video or blog post
