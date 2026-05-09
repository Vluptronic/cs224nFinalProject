# Four BioASQ Pipelines

Scripts:

1. `baseline.py` — base LLaMA, question-only, answer-only loss + small generated EM/F1.
2. `lora.py` — trains LoRA, question-only, answer-only loss + small generated EM/F1.
3. `baseline_rag.py` — base LLaMA + LanceDB RAG, answer-only loss + small generated EM/F1.
4. `lora_rag.py` — saved LoRA adapter + LanceDB RAG, answer-only loss + small generated EM/F1.

Before running:

```bash
export HF_TOKEN="your_new_hf_token"
pip install torch datasets transformers peft accelerate sentence-transformers lancedb tqdm
```

Run examples:

```bash
python baseline.py --dtype float16
python lora.py --dtype float16
python baseline_rag.py --dtype float16 --rebuild_index
python lora_rag.py --dtype float16
```

Generated EM/F1 is intentionally computed on a small subset by default:

```bash
--num_generation_eval_examples 50
```

Loss evaluation still runs on the full held-out evaluation split using teacher forcing and answer-only label masking.
