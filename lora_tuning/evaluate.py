"""
LoRA Fine-tuning Evaluation Script for AI Chef Agent
========================================
Metric: chef suggestion accuracy
  - For each question in test.jsonl, both models generate a response
  - qwen-max-latest acts as judge, making a binary yes/no call on whether the answer
    meets professional chef standards
  - Accuracy = number of "yes" judgements / total samples * 100%
  - Compares base Qwen2.5-7B vs QLoRA fine-tuned model, and shows the improvement

How to run:
    python evaluate.py            # full eval (base + fine-tuned)
    python evaluate.py --skip-base  # only eval fine-tuned model (saves time)

Environment variables:
    DASHSCOPE_API_KEY  — required for calling qwen-max-latest as judge
"""

import os
import sys
import json
import time
import logging
import argparse

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from tqdm import tqdm
from openai import OpenAI
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# -- project root path --
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("lora_eval")

# -- constants --
BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_DIR   = os.path.join(PROJECT_ROOT, "data", "lora_adapter")
TEST_JSONL    = os.path.join(PROJECT_ROOT, "data", "lora_dataset", "test.jsonl")
REPORT_PATH   = os.path.join(PROJECT_ROOT, "data", "lora_eval_report.json")

GEN_KWARGS = dict(
    max_new_tokens=200,
    temperature=0.7,
    do_sample=True,
    pad_token_id=None,
)

# judge prompt for chef suggestion accuracy
JUDGE_PROMPT_TMPL = """You are a Michelin restaurant inspector. Please judge whether the following AI chef assistant's answer meets the standard of a high-end private chef service.

Evaluation criteria (consider all four, doesn't need to be perfect on all, but most should pass):
1. Accurate information: cooking/nutrition/food safety knowledge is basically correct, no obvious errors
2. Specific suggestions: gives practical, actionable guidance rather than vague general advice
3. Professional depth: shows culinary knowledge beyond what a typical home cook would know
4. Chef persona: appropriate tone that reflects professional service (different from a generic neutral AI tone)

User question: {question}
AI answer: {answer}

Only output: yes or no, no explanation."""

# judge prompt for food safety accuracy
SAFETY_JUDGE_TMPL = """You are a food safety expert. Please judge whether the following AI assistant's answer to a food safety question is correct and helpful.

Evaluation criteria (consider all together):
1. Correctly identified the food safety risk (spoilage, toxins, harmful ingredients, etc.)
2. Gave appropriate handling advice (e.g. stop eating, see a doctor, rinse mouth, avoid consuming, refrigerate, etc. — depends on the situation)
3. The answer does not lead the user to take any dangerous action

User question: {question}
AI answer: {answer}

Only output: yes or no, no explanation."""


# -- device detection --
def detect_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        return "cuda", torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Using Apple Silicon MPS")
        return "mps", torch.bfloat16
    logger.warning("Using CPU (inference will be slow)")
    return "cpu", torch.float32


# -- data loading --
def load_test_data(path: str) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Test set not found: {path}\n"
            f"Please run dataset_builder.py first to generate the dataset."
        )
    # fix overly strict correct_action keywords from older versions
    KEYWORD_FIXES = {"不要冲洗": "不要", "吐掉": "吐"}

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                if record.get("correct_action") in KEYWORD_FIXES:
                    record["correct_action"] = KEYWORD_FIXES[record["correct_action"]]
                records.append(record)
    logger.info(f"Loaded test set: {len(records)} samples")
    return records


# -- QLoRA 4-bit quantization config (reused for inference to save VRAM) --
def _bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


# -- model loading --
def load_base_model(device: str, torch_dtype: torch.dtype):
    logger.info(f"Loading base model (4-bit QLoRA): {BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def load_finetuned_model(device: str, torch_dtype: torch.dtype):
    if not os.path.exists(ADAPTER_DIR):
        raise FileNotFoundError(
            f"LoRA adapter not found: {ADAPTER_DIR}\n"
            f"Please run train_lora.py first to complete training."
        )
    logger.info(f"Loading fine-tuned model (4-bit QLoRA + adapter): {BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()
    return model, tokenizer


# -- inference --
@torch.inference_mode()
def generate_response(model, tokenizer, messages: list[dict], device: str) -> str:
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    gen_kwargs = {**GEN_KWARGS, "pad_token_id": tokenizer.eos_token_id}
    output_ids = model.generate(**inputs, **gen_kwargs)
    new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# -- judge: qwen-max-latest binary evaluation --
def build_judge_client() -> OpenAI | None:
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        logger.warning("DASHSCOPE_API_KEY not found, chef suggestion accuracy will not be calculated.")
        return None
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


# -- generic judge API call --
def _call_judge(prompt: str, client: OpenAI) -> bool | None:
    """Call qwen-max-latest for a binary judgment, returns True/False/None on failure."""
    try:
        response = client.chat.completions.create(
            model="qwen-max-latest",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        return "yes" in raw.lower()
    except Exception as e:
        logger.debug(f"Judge API call failed: {e}")
        return None


# -- metric 1: chef suggestion accuracy (LLM-as-Judge) --
def judge_chef_accuracy(question: str, answer: str, client: OpenAI) -> bool | None:
    """Use qwen-max-latest to judge chef response quality, returns True/False/None."""
    prompt = JUDGE_PROMPT_TMPL.format(question=question, answer=answer)
    return _call_judge(prompt, client)


# -- metric 2: food safety accuracy (LLM-as-Judge, replaces fragile keyword matching) --
def judge_food_safety(question: str, answer: str, client: OpenAI) -> bool | None:
    """Use qwen-max-latest to judge whether the food safety answer is correct and helpful."""
    prompt = SAFETY_JUDGE_TMPL.format(question=question, answer=answer)
    return _call_judge(prompt, client)


def compute_food_safety_accuracy(results: list[dict]) -> float | None:
    """
    Compute LLM judge pass rate for category='safety' items.
    Reads the safety_correct field already stored by evaluate_model.
    """
    safety = [r for r in results if r.get("category") == "safety"
              and r.get("safety_correct") is not None]
    if not safety:
        return None
    correct = sum(1 for r in safety if r["safety_correct"])
    return correct / len(safety) * 100.0


# -- core evaluation --
def evaluate_model(
    model,
    tokenizer,
    test_data: list[dict],
    model_name: str,
    device: str,
    judge_client: OpenAI | None,
) -> dict:
    """
    Generate responses and compute two metrics:
    - Chef suggestion accuracy (qwen-max-latest judge, all 100 samples)
    - Food safety accuracy (LLM judge, only safety category ~25 samples)
    """
    results = []
    logger.info(f"\nStarting evaluation: {model_name} ({len(test_data)} samples)")

    for item in tqdm(test_data, desc=f"Evaluating {model_name}", unit="sample"):
        messages       = item.get("messages", [])
        category       = item.get("category", "general")
        correct_action = item.get("correct_action")   # only present for safety category

        user_msgs = [m for m in messages if m.get("role") == "user"]
        question  = user_msgs[-1]["content"] if user_msgs else ""

        prompt_messages = [m for m in messages if m.get("role") != "assistant"]
        answer = generate_response(model, tokenizer, prompt_messages, device)

        # metric 1: chef suggestion accuracy (LLM judge, all samples)
        judge_correct = judge_chef_accuracy(question, answer, judge_client) if judge_client else None

        # metric 2: food safety accuracy (LLM judge, safety category only)
        safety_correct = None
        if category == "safety" and judge_client:
            safety_correct = judge_food_safety(question, answer, judge_client)

        results.append({
            "model":          model_name,
            "category":       category,
            "question":       question,
            "answer":         answer,
            "judge_correct":  judge_correct,
            "safety_correct": safety_correct,
            "correct_action": correct_action,
        })

    # metric 1: chef suggestion accuracy
    judged = [r for r in results if r["judge_correct"] is not None]
    chef_accuracy = (
        sum(1 for r in judged if r["judge_correct"]) / len(judged) * 100.0
        if judged else None
    )

    # metric 2: food safety accuracy
    food_safety_accuracy = compute_food_safety_accuracy(results)

    return {
        "model_name":               model_name,
        "sample_count":             len(results),
        "chef_accuracy_pct":        chef_accuracy,
        "food_safety_accuracy_pct": food_safety_accuracy,
        "detail":                   results,
    }


# -- report printing & saving --
def print_report(base_metrics: dict | None, ft_metrics: dict, elapsed: float):
    def fmt(val):
        return f"{val:.1f}%" if val is not None else "N/A"

    def delta_str(base_val, ft_val):
        if base_val is None or ft_val is None:
            return "N/A"
        d = ft_val - base_val
        return f"{'↑' if d >= 0 else '↓'}{abs(d):.1f}pp"

    print()
    print("╔════════════════════════════════════════════════════════╗")
    print("║         AI Chef LoRA Fine-tuning Evaluation Report     ║")
    print("╚════════════════════════════════════════════════════════╝")
    print()

    ft_chef  = ft_metrics.get("chef_accuracy_pct")
    ft_safe  = ft_metrics.get("food_safety_accuracy_pct")

    if base_metrics:
        base_chef = base_metrics.get("chef_accuracy_pct")
        base_safe = base_metrics.get("food_safety_accuracy_pct")

        print("┌──────────────────────────────┬────────────┬────────────┬──────────┐")
        print("│ Metric                       │ Base Model │ Fine-tuned │  Delta   │")
        print("├──────────────────────────────┼────────────┼────────────┼──────────┤")
        print(f"│ Chef suggestion accuracy      │ {fmt(base_chef):^10} │ {fmt(ft_chef):^10} │ {delta_str(base_chef, ft_chef):^8} │")
        print(f"│ Food safety accuracy          │ {fmt(base_safe):^10} │ {fmt(ft_safe):^10} │ {delta_str(base_safe, ft_safe):^8} │")
        print("└──────────────────────────────┴────────────┴────────────┴──────────┘")
    else:
        print("┌──────────────────────────────┬────────────┐")
        print("│ Metric                       │ Fine-tuned │")
        print("├──────────────────────────────┼────────────┤")
        print(f"│ Chef suggestion accuracy      │ {fmt(ft_chef):^10} │")
        print(f"│ Food safety accuracy          │ {fmt(ft_safe):^10} │")
        print("└──────────────────────────────┴────────────┘")

    print()
    minutes, seconds = divmod(int(elapsed), 60)
    print(f"Samples: {ft_metrics['sample_count']}  |  Judge model: qwen-max-latest  |  Time: {minutes}m {seconds}s")
    print(f"Report saved to: {REPORT_PATH}")
    print()


def save_report(base_metrics: dict | None, ft_metrics: dict, elapsed: float):
    report = {
        "judge_model":         "qwen-max-latest",
        "elapsed_seconds":     round(elapsed, 1),
        "base_model":          base_metrics,
        "finetuned_model":     ft_metrics,
        "report_generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"Full report written to: {REPORT_PATH}")


# -- entry point --
def parse_args():
    parser = argparse.ArgumentParser(description="AI Chef LoRA Fine-tuning Evaluation")
    parser.add_argument(
        "--skip-base", action="store_true", default=False,
        help="Skip base model evaluation, only evaluate the fine-tuned model (saves time)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()
    device, torch_dtype = detect_device()

    test_data = load_test_data(TEST_JSONL)
    judge_client = build_judge_client()

    base_metrics = None

    if not args.skip_base:
        logger.info("Phase 1: evaluating base model...")
        base_model, base_tokenizer = load_base_model(device, torch_dtype)
        base_metrics = evaluate_model(
            base_model, base_tokenizer, test_data,
            model_name="Qwen2.5-7B-Instruct (base)",
            device=device,
            judge_client=judge_client,
        )
        del base_model
        torch.cuda.empty_cache()
        logger.info("Base model evaluation done, VRAM released.")
    else:
        logger.info("Skipped base model evaluation (--skip-base).")

    logger.info("Phase 2: evaluating fine-tuned model (base + LoRA adapter)...")
    ft_model, ft_tokenizer = load_finetuned_model(device, torch_dtype)
    ft_metrics = evaluate_model(
        ft_model, ft_tokenizer, test_data,
        model_name="Qwen2.5-7B-Instruct + QLoRA (AI Chef)",
        device=device,
        judge_client=judge_client,
    )

    elapsed = time.time() - start_time
    print_report(base_metrics, ft_metrics, elapsed)
    save_report(base_metrics, ft_metrics, elapsed)


if __name__ == "__main__":
    main()
