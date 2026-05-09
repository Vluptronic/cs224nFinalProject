import os
import re
import string
import argparse
from collections import Counter

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from huggingface_hub import login
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = lambda x, **kwargs: x


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline LLaMA BioASQ: answer-only loss + small generated EM/F1 eval.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="rag-datasets/rag-mini-bioasq")
    parser.add_argument("--dataset_config", type=str, default="question-answer-passages")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    # loss evaluation uses teacher forcing over the full held-out set
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)

    # generated-answer evaluation is intentionally small and slower
    parser.add_argument("--num_generation_eval_examples", type=int, default=50)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=64)

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
    s = str(s).lower()
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


def tokenize_for_loss(example, tokenizer, max_length):
    prompt = build_prompt(example["question"])
    target = normalize_answer_field(example["answer"]) + tokenizer.eos_token
    full_text = prompt + target

    full = tokenizer(full_text, truncation=True, padding="max_length", max_length=max_length)
    prompt_ids = tokenizer(prompt, truncation=True, max_length=max_length, padding=False)["input_ids"]

    labels = full["input_ids"].copy()
    prompt_len = len(prompt_ids)
    labels[:prompt_len] = [-100] * prompt_len
    labels = [-100 if tok == tokenizer.pad_token_id else lab for tok, lab in zip(full["input_ids"], labels)]
    full["labels"] = labels
    return full


def collate_fn(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
    }


def evaluate_loss(model, tokenizer, eval_ds, device, args):
    tokenized_eval = eval_ds.map(
        lambda ex: tokenize_for_loss(ex, tokenizer, args.max_length),
        remove_columns=eval_ds.column_names,
    )
    tokenized_eval.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    loader = DataLoader(tokenized_eval, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    total_loss = 0.0
    num_batches = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Loss eval"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += float(outputs.loss.item())
            num_batches += 1
    return total_loss / max(num_batches, 1)


def generate_answer(model, tokenizer, prompt, device, max_prompt_length, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_prompt_length, padding=False).to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()


def evaluate_generation(model, tokenizer, eval_ds, device, args):
    n = min(args.num_generation_eval_examples, len(eval_ds)) if args.num_generation_eval_examples > 0 else len(eval_ds)
    small_eval = eval_ds.select(range(n))

    em_scores, f1_scores = [], []
    sample = None
    model.eval()
    for ex in tqdm(small_eval, desc="Generation eval"):
        question = ex["question"]
        gold = normalize_answer_field(ex["answer"])
        prompt = build_prompt(question)
        pred = generate_answer(model, tokenizer, prompt, device, args.max_prompt_length, args.max_new_tokens)
        em_scores.append(exact_match(pred, gold))
        f1_scores.append(token_f1(pred, gold))
        if sample is None:
            sample = {"question": question, "gold": gold, "pred": pred}

    return {
        "exact_match": sum(em_scores) / max(len(em_scores), 1),
        "token_f1": sum(f1_scores) / max(len(f1_scores), 1),
        "num_generation_eval_examples": len(em_scores),
    }, sample


def print_sample(sample):
    if sample is None:
        return
    print("\nQualitative sample")
    print("-" * 80)
    print(f"Question: {sample['question']}")
    print(f"Gold:     {sample['gold']}")
    print(f"Pred:     {sample['pred']}")
    print("-" * 80)


def main():
    args = parse_args()
    set_seed(args.seed)
    maybe_login()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = get_torch_dtype(args.dtype)
    print(f"Using device: {device}")

    ds = load_dataset(args.dataset_name, args.dataset_config)
    full_ds = ds[args.split].shuffle(seed=args.seed)
    split_ds = full_ds.train_test_split(test_size=args.test_size, seed=args.seed)
    eval_ds = split_ds["test"]
    print(f"Held-out eval size: {len(eval_ds)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()

    eval_loss = evaluate_loss(model, tokenizer, eval_ds, device, args)
    metrics, sample = evaluate_generation(model, tokenizer, eval_ds, device, args)

    print("=" * 80)
    print("Baseline evaluation")
    print(f"Answer-only eval loss: {eval_loss:.4f}")
    print(f"Generated Exact Match: {metrics['exact_match']:.4f}")
    print(f"Generated Token F1:    {metrics['token_f1']:.4f}")
    print(f"Generation eval N:     {metrics['num_generation_eval_examples']}")
    print("=" * 80)
    print_sample(sample)


if __name__ == "__main__":
    main()
