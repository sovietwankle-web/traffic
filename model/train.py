"""
LoRA微调脚本：基于Qwen2.5-1.5B-Instruct微调车辆行为异常检测模型。
适配NVIDIA RTX 4090 (24GB VRAM) - 全量数据，bf16精度。
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

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "model/checkpoints"
FINAL_DIR = "model/final_adapter"

DATA_DIR = "data/dataset"
TRAIN_FILE = os.path.join(DATA_DIR, "train.jsonl")
VAL_FILE = os.path.join(DATA_DIR, "val.jsonl")

# LoRA参数
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# 训练参数 (4090优化)
NUM_EPOCHS = 3
BATCH_SIZE = 8              # 4090 24GB可以用更大batch
GRAD_ACCUM = 4              # 有效batch_size = 8 * 4 = 32
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 1024
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01

USE_4BIT = False


def find_local_model(model_id):
    """查找本地缓存的模型路径"""
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


def load_model_and_tokenizer():
    """加载基座模型和分词器"""
    model_source = find_local_model(MODEL_ID)
    local_only = model_source != MODEL_ID
    print(f"Loading model: {MODEL_ID} (source: {model_source})")

    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=True,
        padding_side="right",
        local_files_only=local_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()

    if USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=local_only,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=local_only,
        )

    print(f"Model loaded. Device: {model.device}, dtype: {model.dtype}")

    if use_cuda:
        allocated = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM: {allocated:.2f}GB allocated / {total:.1f}GB total")

    return model, tokenizer


def get_lora_config():
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=LORA_TARGET_MODULES,
    )


def train():
    if not os.path.exists(TRAIN_FILE):
        print(f"Training data not found: {TRAIN_FILE}")
        sys.exit(1)

    model, tokenizer = load_model_and_tokenizer()

    print(f"Loading training data from {TRAIN_FILE}")
    data_files = {"train": TRAIN_FILE}
    if os.path.exists(VAL_FILE):
        data_files["validation"] = VAL_FILE
    dataset = load_dataset("json", data_files=data_files)
    print(f"Train samples: {len(dataset['train'])}")
    if "validation" in dataset:
        print(f"Val samples: {len(dataset['validation'])}")

    lora_config = get_lora_config()
    print(f"LoRA config: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()

    # CPU回退
    train_dataset = dataset["train"]
    val_dataset = dataset.get("validation")
    if not use_cuda:
        train_dataset = train_dataset.shuffle(seed=42).select(range(min(1500, len(train_dataset))))
        if val_dataset:
            val_dataset = val_dataset.shuffle(seed=42).select(range(min(300, len(val_dataset))))
        print(f"CPU mode: reduced train to {len(train_dataset)}, val to {len(val_dataset) if val_dataset else 0}")

    bs = BATCH_SIZE if use_cuda else 1
    ga = GRAD_ACCUM if use_cuda else 16

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        gradient_accumulation_steps=ga,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        logging_steps=50,
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=bool(val_dataset),
        metric_for_best_model="eval_loss",
        max_length=MAX_SEQ_LENGTH,
        bf16=use_bf16,
        fp16=use_cuda and not use_bf16,
        report_to="none",
        seed=42,
        packing=False,
    )

    print(f"\nTraining config:")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Effective batch size: {bs * ga}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Max seq length: {MAX_SEQ_LENGTH}")
    print(f"  Precision: bf16={use_bf16}, fp16={use_cuda and not use_bf16}")
    print(f"  Device: {'CUDA' if use_cuda else 'CPU'}")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print("\n" + "=" * 60)
    print("Starting LoRA fine-tuning (1.5B)...")
    print("=" * 60)

    trainer.train()

    print(f"\nSaving model to {FINAL_DIR}")
    trainer.save_model(FINAL_DIR)
    tokenizer.save_pretrained(FINAL_DIR)
    print("Training complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for vehicle behavior detection")
    parser.add_argument("--4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--lora-r", type=int, default=LORA_R)
    args = parser.parse_args()

    USE_4BIT = args.__dict__["4bit"]
    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    LORA_R = args.lora_r
    LORA_ALPHA = LORA_R * 2

    train()
