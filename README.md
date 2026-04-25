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

Agents share a scratchpad, but private tool outputs are scoped to the role that
called the tool. This creates partial observability and forces coordination.

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

The environment also emits smaller process rewards for useful evidence-gathering
and scratchpad notes. This helps early RL rollouts avoid all-zero rewards.

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
- `list_incident_tools`

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

## Training On Hugging Face GPU Space

This project includes a minimal GRPO training scaffold:

```bash
python training/train_grpo.py
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

The training loop is:

```text
prompt -> model-generated tool calls -> environment rollout -> reward -> GRPO update
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
  models.py
  scenarios.py
  rewards.py
  demo_agents.py
  server/
    app.py
    incident_environment.py
    mcp_environment.py
server/
  app.py
  Dockerfile
training/
  train_grpo.py
scripts/
  run_demo.py
  evaluate_baseline.py
tests/
  test_environment.py
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
- scripted baseline
- random baseline evaluation
- GRPO training scaffold
- Docker Space metadata

Next milestones:

- run real GRPO training on HF GPU hardware
- save reward/loss plots
- add trained-vs-baseline evaluation artifacts
- publish final demo video or blog post
