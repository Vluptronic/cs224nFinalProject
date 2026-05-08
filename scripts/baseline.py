import os
import re
import string
import argparse
from collections import Counter

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from huggingface_hub import login


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline LLaMA evaluation on BioASQ with generated-answer EM/F1.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="rag-datasets/rag-mini-bioasq")
    parser.add_argument("--dataset_config", type=str, default="question-answer-passages")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--num_eval_examples", type=int, default=-1, help="-1 means evaluate all held-out examples.")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


def maybe_login():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        login(token=token)


def get_torch_dtype(dtype_name):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    return "auto"


def normalize_answer_field(answer):
    if isinstance(answer, list):
        return answer[0] if answer else ""
    return str(answer)


def normalize_text(s):
    """SQuAD-style normalization for exact match and token F1."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = " ".join(s.split())
    return s


def exact_match(prediction, gold):
    return int(normalize_text(prediction) == normalize_text(gold))


def token_f1(prediction, gold):
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold).split()
    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def build_prompt(question):
    return f"""You are a biology question-answering assistant. Answer concisely.

### Question:
{question}

### Answer:
"""


def generate_answer(model, tokenizer, prompt, device, max_prompt_length, max_new_tokens):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_length,
        padding=False,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def evaluate_generation(model, tokenizer, eval_ds, device, args):
    if args.num_eval_examples is not None and args.num_eval_examples > 0:
        eval_ds = eval_ds.select(range(min(args.num_eval_examples, len(eval_ds))))

    em_scores = []
    f1_scores = []
    examples_for_print = []

    for i, ex in enumerate(eval_ds):
        question = ex["question"]
        gold = normalize_answer_field(ex["answer"])
        prompt = build_prompt(question)
        pred = generate_answer(model, tokenizer, prompt, device, args.max_prompt_length, args.max_new_tokens)

        em_scores.append(exact_match(pred, gold))
        f1_scores.append(token_f1(pred, gold))

        if len(examples_for_print) < 1:
            examples_for_print.append((question, gold, pred))

    metrics = {
        "exact_match": sum(em_scores) / len(em_scores),
        "token_f1": sum(f1_scores) / len(f1_scores),
        "num_eval_examples": len(em_scores),
    }
    return metrics, examples_for_print


def main():
    args = parse_args()
    set_seed(args.seed)
    maybe_login()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = get_torch_dtype(args.dtype)

    print(f"Using device: {device}")
    print(f"Loading dataset: {args.dataset_name}, config={args.dataset_config}")
    ds = load_dataset(args.dataset_name, args.dataset_config)
    full_ds = ds[args.split].shuffle(seed=args.seed)
    split_ds = full_ds.train_test_split(test_size=args.test_size, seed=args.seed)
    eval_ds = split_ds["test"]
    print(f"Held-out eval size: {len(eval_ds)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading base model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()

    metrics, examples = evaluate_generation(model, tokenizer, eval_ds, device, args)

    print("=" * 60)
    print("Baseline generated-answer evaluation")
    print(f"Exact Match: {metrics['exact_match']:.4f}")
    print(f"Token F1:    {metrics['token_f1']:.4f}")
    print(f"N examples:   {metrics['num_eval_examples']}")
    print("=" * 60)

    for q, gold, pred in examples:
        print("\nQualitative sample")
        print("-" * 60)
        print(f"Question: {q}")
        print(f"Gold:     {gold}")
        print(f"Pred:     {pred}")
        print("-" * 60)


if __name__ == "__main__":
    main()
