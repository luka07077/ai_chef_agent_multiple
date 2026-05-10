"""
QLoRA Fine-tuning Script for AI Chef Agent
========================================
Fine-tunes Qwen2.5-7B-Instruct with QLoRA (4-bit NF4) to give it a chef persona.
Note: requires a CUDA GPU (Colab T4 16GB works), does not support MPS/CPU.

How to run:
    cd /Users/luka/Code/python/ai_chef_agent
    python lora_tuning/train_lora.py

Dependencies (install in a venv):
    pip install transformers>=4.43.0 peft>=0.12.0 trl>=0.10.0 datasets accelerate bitsandbytes tqdm

Training data format (data/lora_dataset/train.jsonl and val.jsonl):
    One JSON object per line, with a messages field containing conversation turns:
    {"messages": [
        {"role": "system",    "content": "You are a professional AI chef..."},
        {"role": "user",      "content": "..."},
        {"role": "assistant", "content": "..."}
    ], "category": "fridge"}

Output:
    data/lora_adapter/              — final LoRA adapter weights
    data/lora_adapter/checkpoints/  — per-epoch checkpoints
"""

import os
import sys
import time
import logging
import json

# must be set before importing torch on Apple Silicon
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback,
)
from trl import SFTTrainer, SFTConfig
from tqdm import tqdm

# -- project root path --
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

# -- logging setup --
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("lora_train")

# -- constants --
BASE_MODEL_ID   = "Qwen/Qwen2.5-7B-Instruct"
TRAIN_JSONL     = os.path.join(PROJECT_ROOT, "data", "lora_dataset", "train.jsonl")
VAL_JSONL       = os.path.join(PROJECT_ROOT, "data", "lora_dataset", "val.jsonl")
ADAPTER_DIR     = os.path.join(PROJECT_ROOT, "data", "lora_adapter")
CHECKPOINT_DIR  = os.path.join(ADAPTER_DIR, "checkpoints")
MAX_SEQ_LEN     = 1024


# -- device detection --
def detect_device() -> tuple[str, bool]:
    """
    QLoRA only works on CUDA (bitsandbytes depends on CUDA kernels).
    Returns ("cuda", use_bf16), raises an error if no CUDA is found.
    """
    if torch.cuda.is_available():
        device = "cuda"
        use_bf16 = torch.cuda.is_bf16_supported()
        logger.info(f"CUDA device found: {torch.cuda.get_device_name(0)}, bf16={use_bf16}")
        return device, use_bf16

    raise RuntimeError(
        "QLoRA (bitsandbytes 4-bit) requires a CUDA GPU.\n"
        "Please run on Google Colab (T4/A100) or a machine with an NVIDIA GPU."
    )


# -- QLoRA 4-bit quantization config --
def build_bnb_config() -> BitsAndBytesConfig:
    """NF4 4-bit quantization with double quant to save even more VRAM."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


# -- data loading --
def load_jsonl(path: str) -> list[dict]:
    """Read a JSONL file line by line and return a list of dicts."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Training data not found: {path}\n"
            f"Please run lora_tuning/dataset_builder.py first to generate the dataset, "
            f"or manually create the data/lora_dataset/ directory and put train.jsonl/val.jsonl in it."
        )
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} records from: {path}")
    return records


# -- dataset formatting --
def format_dataset(records: list[dict], tokenizer) -> Dataset:
    """
    Convert messages-format records into a HuggingFace Dataset.
    The text field contains the full conversation formatted by apply_chat_template.
    """
    texts = []
    logger.info("Formatting training data...")
    for item in tqdm(records, desc="apply_chat_template", leave=False):
        messages = item.get("messages", [])
        if not messages:
            continue
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append({"text": text})
        except Exception as e:
            logger.warning(f"Formatting failed, skipping: {e}")
    logger.info(f"Formatting done, valid samples: {len(texts)}")
    return Dataset.from_list(texts)


# -- LoRA config --
def build_lora_config() -> LoraConfig:
    return LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


# -- training arguments --
def build_training_args(use_bf16: bool) -> SFTConfig:
    """
    Training args optimized for Colab T4 16GB (QLoRA 7B).
    SFTConfig extends TrainingArguments with max_length / dataset_text_field.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # T4 doesn't support bf16, use fp16; A100/H100 can use bf16
    use_fp16 = not use_bf16

    return SFTConfig(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,          # keep at most 2 checkpoints to save disk space
        report_to="none",            # disable wandb/tensorboard reporting
        dataloader_num_workers=0,    # >0 can cause deadlocks on Colab
        optim="paged_adamw_32bit",   # recommended optimizer for QLoRA, saves VRAM
        # SFTConfig specific
        max_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        packing=False,               # keep it simple, no sample packing
    )


# -- main training flow --
def main():
    start_time = time.time()
    device, use_bf16 = detect_device()

    # 1. load tokenizer
    logger.info(f"Loading tokenizer from HuggingFace: {BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"   # right padding is more stable for SFT training

    # 2. load raw data & format it
    train_records = load_jsonl(TRAIN_JSONL)
    val_records   = load_jsonl(VAL_JSONL)
    train_dataset = format_dataset(train_records, tokenizer)
    val_dataset   = format_dataset(val_records,   tokenizer)

    # 3. load base model with 4-bit QLoRA
    logger.info(f"Loading base model with QLoRA 4-bit quantization: {BASE_MODEL_ID}")
    bnb_config = build_bnb_config()

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # required for QLoRA: marks quantized layers as trainable and enables gradient checkpointing
    model = prepare_model_for_kbit_training(model)

    # 4. inject LoRA adapters
    lora_config = build_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. training args
    training_args = build_training_args(use_bf16)

    # 6. SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # 7. start training
    logger.info("=" * 60)
    logger.info("  Starting QLoRA fine-tuning for AI Chef persona (7B)...")
    logger.info(f"  Train set: {len(train_dataset)} samples  |  Val set: {len(val_dataset)} samples")
    logger.info("=" * 60)

    train_result = trainer.train()

    # 8. save LoRA adapter only (not the full model weights)
    os.makedirs(ADAPTER_DIR, exist_ok=True)
    trainer.model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)

    # 9. print training summary
    elapsed = time.time() - start_time
    final_loss = train_result.training_loss
    minutes, seconds = divmod(int(elapsed), 60)

    logger.info("\n" + "=" * 60)
    logger.info("  LoRA fine-tuning complete!")
    logger.info(f"  Time elapsed  : {minutes}m {seconds}s")
    logger.info(f"  Final loss    : {final_loss:.4f}")
    logger.info(f"  Adapter path  : {ADAPTER_DIR}")
    logger.info("=" * 60)

    # save training summary as JSON
    summary = {
        "base_model":   BASE_MODEL_ID,
        "adapter_path": ADAPTER_DIR,
        "train_samples": len(train_dataset),
        "val_samples":   len(val_dataset),
        "elapsed_seconds": round(elapsed, 1),
        "final_train_loss": round(final_loss, 4),
        "device": device,
        "bf16": use_bf16,
    }
    summary_path = os.path.join(ADAPTER_DIR, "train_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"Training summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
