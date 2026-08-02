# Llama-3.2-3B QLoRA Fine-Tuning for Python Code Repair

## Overview

This repository contains evaluation results of a **Llama-3.2-3B-Instruct** model fine-tuned with **QLoRA** (4-bit NF4 quantization, LoRA rank=16) on a hybrid Python debugging dataset for automated code repair.

**Model:** `llama-3.2-3b-instruct` (3B parameters)  
**Task:** Given buggy Python code, generate the corrected version.  
**Training:** 3 epochs, effective batch size 16, learning rate 2e-4, max sequence length 768  

---

## Dataset

The model was trained on a cleaned hybrid Python debugging dataset constructed from three sources:

| Source | Description |
|---|---|
| **PDB** | Program Defect Benchmark — real-world Python bugs with test cases and ground-truth fixes |
| **DebugBench** | Curated debugging examples with bug type labels (syntax, logic, runtime) and execution metadata |
| **TSSB-3M** | Large-scale synthetic bug-fix pairs; filtered for validity and deduplicated |

### Dataset Statistics (Post-Cleaning)

| Split | Samples |
|---|---|
| Train | 20,077 |
| Validation | 2,510 |
| Test | 2,510 |
| **Total** | **25,097** |

### Cleaning Pipeline

1. **Deduplication** — SHA256 hashes removed exact duplicate buggy-fixed pairs
2. **Validity filtering** — Removed syntactically invalid Python, empty examples, and examples where `buggy_code == fixed_code`
3. **Bug-type labeling** — TSSB examples without native labels were categorized using regex-based heuristics (syntax, logic, runtime)
4. **Cross-split integrity** — Verified zero buggy_code overlap between train/val/test splits; 412 fixed_code overlaps between train and test retained as they represent different bugs with convergent fixes

### Data Format

Each row contains:

| Column | Description |
|---|---|
| `buggy_code` | Python code containing the bug |
| `fixed_code` | Ground-truth corrected Python code |
| `bug_type` | Bug category (syntax, logic, runtime, or unlabeled) |
| `source` | Origin dataset (PDB, DebugBench, TSSB) |
| `difficulty` | Estimated repair difficulty |
| `edit_distance` | Levenshtein distance between buggy and fixed |

Additional metadata columns (where available): `public_tests`, `oracle_tests`, `patch_size`, `location_overlap`, `repair_agreement`.

---

## Results

| Metric | Validation (2,510) | Test (2,510) |
|---|---|---|
| Exact Match | 0.1876 | 0.1813 |
| SacreBLEU | 91.0 | 89.8 |
| ROUGE-1 (F1) | 0.8827 | 0.8810 |
| ROUGE-2 (F1) | 0.8189 | 0.8132 |
| ROUGE-L (F1) | 0.8780 | 0.8747 |
| Edit Ratio | 0.8955 | 0.8932 |

**Key observations:**
- Minimal val-test gap (~0.6% EM difference) indicates no overfitting
- High ROUGE/BLEU with moderate EM is characteristic of code repair — the model often produces semantically correct fixes with minor syntactic differences from the ground truth
- These metrics serve as a baseline for the CaliDebug project's multi-model comparison

---

## Files

| File | Description |
|---|---|
| `val_results_full.csv` | Validation predictions (reference + generated) |
| `test_results_full.csv` | Test predictions (reference + generated) |
| `val_metrics.csv` | Validation metric summary |
| `test_metrics.csv` | Test metric summary |

---

## Setup

### Load the model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

BASE_MODEL = "path/to/llama-3-2-3b-instruct-model"
ADAPTER = "path/to/best-checkpoint"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb_config,
    device_map="auto", torch_dtype=torch.float16,
)

model = PeftModel.from_pretrained(base_model, ADAPTER)
model.eval()
```

### Generate a repair

```python
def repair(buggy_code):
    messages = [
        {"role": "system", "content": "You are an expert Python debugger. Repair the supplied code while making only necessary changes."},
        {"role": "user", "content": f"Buggy code:\n```python\n{buggy_code}\n```\n\nReturn only the corrected Python code."},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, pad_token_id=tokenizer.pad_token_id, do_sample=False)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
```

---

## Hardware

- **Training:** Single NVIDIA T4 (15GB VRAM), Kaggle
- **Inference:** T4 or CPU (supports 4-bit quantization)

---

## Related Project

This model is part of **CaliDebug** — a confidence-guided adaptive repair framework for reliable LLM-based code debugging. CaliDebug uses behavioral signals (test score improvement, patch consistency, location overlap, patch size, repair agreement) with a calibrated confidence model to decide whether to accept, refine, or reject generated patches.

---
