"""Supervised fine-tuning for Incident Commander tool-call trajectories.

This is the 10% SFT stage in the proposed 80/10/10 split. It trains on
`data/sft_trajectories.jsonl`, which contains ideal incident-response tool-call
sequences generated from the simulator.

Run locally without GPU dependencies:
    python training/train_sft.py --dry-run

Run after pretraining on a HF GPU Space:
    python training/train_sft.py --model-name USER/incident-commander-pretrain --push-to-hub --hub-model-id USER/incident-commander-sft
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_DATA = ROOT / "data" / "sft_trajectories.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs" / "sft_incident_commander"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_no}: {exc}") from exc
    return rows


def validate_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Generate it with: python training/prepare_data.py"
        )
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"{path} is empty")
    bad_rows = []
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 3:
            bad_rows.append(row.get("id", "<missing id>"))
            continue
        if messages[-1].get("role") != "assistant":
            bad_rows.append(row.get("id", "<missing id>"))
    if bad_rows:
        raise ValueError(f"Malformed SFT rows: {bad_rows[:5]}")
    return rows


def dry_run(args: argparse.Namespace) -> None:
    rows = validate_rows(args.data_path)
    assistant_chars = sum(len(row["messages"][-1]["content"]) for row in rows)
    print(
        json.dumps(
            {
                "stage": "sft",
                "split_budget": "10%",
                "model_name": args.model_name,
                "data_path": str(args.data_path),
                "rows": len(rows),
                "assistant_chars": assistant_chars,
                "max_steps": args.max_steps,
                "output_dir": str(args.output_dir),
                "sample": rows[0],
            },
            indent=2,
        )
    )


def load_training_stack() -> tuple[Any, Any, Any, Any]:
    try:
        from datasets import load_dataset
        from unsloth import FastLanguageModel
        from trl import SFTConfig, SFTTrainer
    except Exception as exc:
        raise SystemExit(
            "Install training dependencies before running SFT: "
            "pip install unsloth trl datasets transformers accelerate peft bitsandbytes trackio\n"
            f"Import error: {exc}"
        )
    return load_dataset, SFTConfig, SFTTrainer, FastLanguageModel


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
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )


def maybe_apply_chat_template(dataset: Any, tokenizer: Any) -> Any:
    def render(example: dict[str, Any]) -> dict[str, str]:
        messages = example["messages"]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            text = "\n".join(
                f"{message['role'].upper()}: {message['content']}"
                for message in messages
            )
        return {"text": text}

    return dataset.map(render, remove_columns=dataset.column_names)


def train(args: argparse.Namespace) -> None:
    validate_rows(args.data_path)
    load_dataset, SFTConfig, SFTTrainer, FastLanguageModel = load_training_stack()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )
    eos_token = configure_special_tokens(model, tokenizer)
    model = ensure_peft_model(model, FastLanguageModel, args)

    dataset = load_dataset("json", data_files=str(args.data_path), split="train")
    dataset = maybe_apply_chat_template(dataset, tokenizer)
    split = dataset.train_test_split(test_size=args.eval_ratio, seed=args.seed)

    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        dataset_text_field="text",
        eos_token=eos_token,
        max_length=args.max_seq_length,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=max(1, args.max_steps // 5),
        save_strategy="steps",
        save_steps=max(1, args.max_steps // 2),
        report_to=args.report_to,
        run_name=args.run_name,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_strategy="every_save" if args.push_to_hub else "end",
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        args=training_args,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir / "final"))
    if args.push_to_hub:
        trainer.push_to_hub()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model-name", default=os.getenv("SFT_BASE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hub-model-id", default=os.getenv("SFT_HUB_MODEL_ID"))
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--eval-ratio", type=float, default=0.25)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--report-to", default=os.getenv("REPORT_TO", "none"))
    parser.add_argument("--run-name", default="incident-commander-sft")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.push_to_hub and not args.hub_model_id:
        raise SystemExit("--hub-model-id is required when --push-to-hub is set")
    if args.dry_run:
        dry_run(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
