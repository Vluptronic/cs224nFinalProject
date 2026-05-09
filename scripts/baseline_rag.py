import os
import re
import string
import argparse
from collections import Counter

import torch
import lancedb
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = lambda x, **kwargs: x


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline + LanceDB RAG BioASQ: answer-only loss + small generated EM/F1 eval.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="rag-datasets/rag-mini-bioasq")
    parser.add_argument("--qa_config", type=str, default="question-answer-passages")
    parser.add_argument("--corpus_config", type=str, default="text-corpus")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--corpus_split", type=str, default="passages")
    parser.add_argument("--test_size", type=float, default=0.2)

    parser.add_argument("--db_path", type=str, default="/root/zhanhe/CS115BFP/lancedb_bioasq")
    parser.add_argument("--table_name", type=str, default="bioasq_passages")
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--rebuild_index", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--num_generation_eval_examples", type=int, default=50)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
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


def get_text_field(example):
    for key in ["passage", "text", "contents", "content"]:
        if key in example:
            return key
    raise ValueError(f"Could not find passage text field. Available fields: {list(example.keys())}")


def get_id_field(example):
    for key in ["id", "passage_id", "doc_id"]:
        if key in example:
            return key
    raise ValueError(f"Could not find passage id field. Available fields: {list(example.keys())}")


def build_or_load_lancedb(args, embedder):
    db = lancedb.connect(args.db_path)
    existing_tables = db.list_tables()
    if args.table_name in existing_tables and not args.rebuild_index:
        print(f"Opening existing LanceDB table: {args.table_name}")
        return db.open_table(args.table_name)

    print("Building LanceDB index from BioASQ text corpus...")
    corpus_ds = load_dataset(args.dataset_name, args.corpus_config)
    corpus = corpus_ds[args.corpus_split]
    sample = corpus[0]
    text_field = get_text_field(sample)
    id_field = get_id_field(sample)

    # Chunking strategy: use dataset-provided full passage/sentence rows; no custom splitting/overlap.
    texts = [str(ex[text_field]) for ex in corpus]
    ids = [str(ex[id_field]) for ex in corpus]
    embeddings = embedder.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)

    rows = [{"passage_id": pid, "text": txt, "vector": vec.tolist()} for pid, txt, vec in zip(ids, texts, embeddings)]
    table = db.create_table(args.table_name, data=rows, mode="overwrite")
    print(f"Created LanceDB table with {len(rows)} full-passage chunks.")
    return table


def retrieve_passages(question, table, embedder, top_k):
    qvec = embedder.encode([question], normalize_embeddings=True)[0].tolist()
    results = table.search(qvec).limit(top_k).to_list()
    passages = [r["text"] for r in results]
    passage_ids = [str(r["passage_id"]) for r in results]
    return passages, passage_ids


def build_rag_prompt(question, retrieved_passages):
    context = "\n".join([f"[{i + 1}] {p}" for i, p in enumerate(retrieved_passages)])
    return f"""You are a biology question-answering assistant. Answer concisely using the retrieved context.

### Question:
{question}

### Retrieved Context:
{context}

### Answer:
"""


def tokenize_for_loss(example, tokenizer, max_length, table, embedder, top_k):
    retrieved_passages, retrieved_ids = retrieve_passages(example["question"], table, embedder, top_k)
    prompt = build_rag_prompt(example["question"], retrieved_passages)
    target = normalize_answer_field(example["answer"]) + tokenizer.eos_token
    full_text = prompt + target

    full = tokenizer(full_text, truncation=True, padding="max_length", max_length=max_length)
    prompt_ids = tokenizer(prompt, truncation=True, max_length=max_length, padding=False)["input_ids"]

    labels = full["input_ids"].copy()
    prompt_len = len(prompt_ids)
    labels[:prompt_len] = [-100] * prompt_len
    labels = [-100 if tok == tokenizer.pad_token_id else lab for tok, lab in zip(full["input_ids"], labels)]
    full["labels"] = labels
    full["retrieved_passage_ids"] = retrieved_ids
    return full


def collate_fn(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
    }


def evaluate_loss(model, tokenizer, eval_ds, table, embedder, device, args):
    tokenized_eval = eval_ds.map(
        lambda ex: tokenize_for_loss(ex, tokenizer, args.max_length, table, embedder, args.top_k),
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


def recall_hit(example, retrieved_ids):
    if "relevant_passage_ids" not in example:
        return None
    gold = set(str(x) for x in example["relevant_passage_ids"])
    if len(gold) == 0:
        return None
    retrieved = set(str(x) for x in retrieved_ids)
    return int(len(gold & retrieved) > 0)


def evaluate_generation(model, tokenizer, eval_ds, table, embedder, device, args):
    n = min(args.num_generation_eval_examples, len(eval_ds)) if args.num_generation_eval_examples > 0 else len(eval_ds)
    small_eval = eval_ds.select(range(n))

    em_scores, f1_scores, recall_hits = [], [], []
    sample = None
    model.eval()
    for ex in tqdm(small_eval, desc="Generation eval"):
        question = ex["question"]
        gold = normalize_answer_field(ex["answer"])
        retrieved_passages, retrieved_ids = retrieve_passages(question, table, embedder, args.top_k)
        prompt = build_rag_prompt(question, retrieved_passages)
        pred = generate_answer(model, tokenizer, prompt, device, args.max_prompt_length, args.max_new_tokens)
        em_scores.append(exact_match(pred, gold))
        f1_scores.append(token_f1(pred, gold))
        hit = recall_hit(ex, retrieved_ids)
        if hit is not None:
            recall_hits.append(hit)
        if sample is None:
            sample = {"question": question, "gold": gold, "pred": pred, "retrieved_ids": retrieved_ids, "retrieved_passages": retrieved_passages}

    metrics = {
        "exact_match": sum(em_scores) / max(len(em_scores), 1),
        "token_f1": sum(f1_scores) / max(len(f1_scores), 1),
        "num_generation_eval_examples": len(em_scores),
    }
    if recall_hits:
        metrics[f"recall_at_{args.top_k}"] = sum(recall_hits) / len(recall_hits)
    return metrics, sample


def print_sample(sample):
    if sample is None:
        return
    print("\nQualitative sample")
    print("-" * 80)
    print(f"Question: {sample['question']}")
    print(f"Gold:     {sample['gold']}")
    print(f"Pred:     {sample['pred']}")
    print(f"Retrieved IDs: {sample['retrieved_ids']}")
    print("Retrieved passages:")
    for i, p in enumerate(sample["retrieved_passages"], 1):
        print(f"[{i}] {p[:700]}")
    print("-" * 80)


def main():
    args = parse_args()
    set_seed(args.seed)
    maybe_login()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = get_torch_dtype(args.dtype)
    print(f"Using device: {device}")

    qa_ds = load_dataset(args.dataset_name, args.qa_config)
    full_ds = qa_ds[args.split].shuffle(seed=args.seed)
    split_ds = full_ds.train_test_split(test_size=args.test_size, seed=args.seed)
    eval_ds = split_ds["test"]
    print(f"Held-out eval size: {len(eval_ds)}")

    print(f"Loading embedder: {args.embedding_model}")
    embedder = SentenceTransformer(args.embedding_model)
    table = build_or_load_lancedb(args, embedder)

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

    eval_loss = evaluate_loss(model, tokenizer, eval_ds, table, embedder, device, args)
    metrics, sample = evaluate_generation(model, tokenizer, eval_ds, table, embedder, device, args)

    print("=" * 80)
    print("Baseline + LanceDB RAG evaluation")
    print(f"Top-k:                 {args.top_k}")
    print(f"Answer-only eval loss: {eval_loss:.4f}")
    print(f"Generated Exact Match: {metrics['exact_match']:.4f}")
    print(f"Generated Token F1:    {metrics['token_f1']:.4f}")
    if f"recall_at_{args.top_k}" in metrics:
        print(f"Recall@{args.top_k}:             {metrics[f'recall_at_{args.top_k}']:.4f}")
    print(f"Generation eval N:     {metrics['num_generation_eval_examples']}")
    print("=" * 80)
    print_sample(sample)


if __name__ == "__main__":
    main()
