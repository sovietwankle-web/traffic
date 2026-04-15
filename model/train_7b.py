"""
LoRA微调脚本：基于Qwen2.5-7B-Instruct + QLoRA (4-bit量化)
适配NVIDIA RTX 4090 (24GB VRAM)
当1.5B模型未达到90%准确率时使用此脚本。
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


# ============================================================
# 配置
# ============================================================

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "model/checkpoints_7b"
FINAL_DIR = "model/final_adapter_7b"

DATA_DIR = "data/dataset"
TRAIN_FILE = os.path.join(DATA_DIR, "train.jsonl")
VAL_FILE = os.path.join(DATA_DIR, "val.jsonl")

# QLoRA参数（4-bit量化 + LoRA）
LORA_R = 32                # 7B模型用更高rank
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# 训练参数
NUM_EPOCHS = 2
BATCH_SIZE = 2              # 7B模型需要更小batch
GRAD_ACCUM = 8              # 有效batch_size = 2 * 8 = 16
LEARNING_RATE = 1e-4        # QLoRA通常用稍低lr
MAX_SEQ_LENGTH = 1024
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01


def find_local_model(model_id):
    """查找本地缓存的模型"""
    from pathlib import Path
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_name = model_id.replace("/", "--")
    model_cache = cache_dir / f"models--{model_name}"
    if model_cache.exists():
        snapshots = model_cache / "snapshots"
        if snapshots.exists():
            dirs = list(snapshots.iterdir())
            if dirs:
                return str(dirs[0])
    return model_id


def train():
    """执行QLoRA微调（7B模型）"""
    if not os.path.exists(TRAIN_FILE):
        print(f"Training data not found: {TRAIN_FILE}")
        sys.exit(1)

    model_source = find_local_model(MODEL_ID)
    print(f"Loading model: {MODEL_ID} (from {model_source})")

    # 4-bit量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # 打印VRAM使用
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM after loading: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved / {total:.1f}GB total")

    print(f"Model loaded. Device: {model.device}")

    # 加载数据
    data_files = {"train": TRAIN_FILE}
    if os.path.exists(VAL_FILE):
        data_files["validation"] = VAL_FILE
    dataset = load_dataset("json", data_files=data_files)
    print(f"Train: {len(dataset['train'])}, Val: {len(dataset.get('validation', []))}")

    # LoRA配置
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=LORA_TARGET_MODULES,
    )

    # 训练配置
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        logging_steps=50,
        eval_strategy="steps" if "validation" in dataset else "no",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True if "validation" in dataset else False,
        metric_for_best_model="eval_loss",
        max_length=MAX_SEQ_LENGTH,
        bf16=True,
        report_to="none",
        seed=42,
        packing=False,
        optim="paged_adamw_8bit",  # 7B模型用8bit优化器节省VRAM
    )

    print(f"\nTraining config:")
    print(f"  Model: {MODEL_ID}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  LoRA rank: {LORA_R}, alpha: {LORA_ALPHA}")
    print(f"  Quantization: 4-bit NF4 + double quantization")
    print(f"  Optimizer: paged_adamw_8bit")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print("\n" + "=" * 60)
    print("Starting QLoRA fine-tuning (7B)...")
    print("=" * 60)

    trainer.train()

    print(f"\nSaving model to {FINAL_DIR}")
    trainer.save_model(FINAL_DIR)
    tokenizer.save_pretrained(FINAL_DIR)

    print("Training complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for 7B model")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--lora-r", type=int, default=LORA_R)
    args = parser.parse_args()

    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    LORA_R = args.lora_r
    LORA_ALPHA = LORA_R * 2

    train()
