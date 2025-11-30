# 🏷️ eBay University Machine Learning Challenge 2025  
**Team:** USM_AI_Renaissance  
**Task:** Named Entity Recognition (NER) on eBay.de Motors Listings  

---

## 📘 Overview
This repository implements a full NLP pipeline for the **2025 eBay University ML Challenge**.  
The goal is to automatically identify and label **named entities** (aspects) from German eBay product titles related to vehicle parts — such as _manufacturer_, _compatible model_, _quantity_, and _included components_.  

The challenge metric is the **Averaged Fβ-Score (↑)** on hidden leaderboard data, evaluated through EvalAI.  
Our system achieves **Rank 23** with **0.876912** on the public leaderboard. The link to the leaderboard: https://eval.ai/web/challenges/challenge-page/2508/leaderboard/6263

---

## 🧠 Architecture Summary
### 1. **Domain-Adaptive Pretraining**
We perform **Masked Language Model (MLM)** pretraining using 2 M unlabeled eBay.de titles to adapt the base model (XLM-RoBERTa) to e-commerce language.

- Model: `xlm-roberta-base`
- Max sequence length: 64
- Masking probability: 0.15  
- Dataset: `Listing_Titles.tsv` (2 M samples)
- Output: `xlmr_ebay_mlm/` (checkpoint directory)

### 2. **Fine-Tuned NER Model**
After MLM adaptation, we fine-tune the model on the 5 K labeled titles in `Tagged_Titles_Train.tsv` to predict BIO tags.

- Input features: tokenized titles + category id token (`[CAT_1]`, `[CAT_2]`)
- Model: `xlmr_ebay_mlm` → `AutoModelForTokenClassification`
- Loss: Cross-Entropy
- Batch size: 16  
- Learning rate: 5e-5  
- Epochs: 3  
- Max sequence length: 128  

The resulting checkpoint (e.g., `ebay_ner_model/checkpoint-471`) is used for inference.

### 3. **Inference and Submission Generation**
The `roberta_infer.py` script tokenizes each quiz listing, predicts token-level BIO tags, converts them to aspect spans, and writes a **UTF-8 tab-separated submission file**:
```
record_id category_id aspect_name aspect_value
5001 1 Kompatible_Fahrzeug_Marke OPEL
5001 1 Kompatibles_Fahrzeug_Modell ASTRA H 1.7
5001 1 Hersteller CDTI-SET
```
- Output file: `submission_quiz.tsv`  
- Format validated against EvalAI submission specs.

---

## 🗂️ Repository Structure
```
eBay_ML_Challenge_2025/
├── Annexure_updated.pdf # eBay dataset and aspect descriptions
├── Listing_Titles.tsv # 2M unlabeled titles for MLM
├── Tagged_Titles_Train.tsv # 5K labeled titles for NER
├── mlm_pretrain.py # Domain-adaptive MLM training
├── roberta_baseline.py # Fine-tuning NER model
├── roberta_infer.py # Generate EvalAI submission file
├── submission_quiz.tsv # Generated output (UTF-8 TSV)
├── ebay_ner_model/ # NER checkpoints
├── xlmr_ebay_mlm/ # MLM-pretrained checkpoint
└── env/ # Python environment (venv)
```


---

## ⚙️ Setup and Requirements
### 1. Environment
```bash
python3 -m venv env
source env/bin/activate
pip install -U pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets pandas scikit-learn tqdm
```

## 🖥️ 2. GPU Environment

- CUDA ≥ 12.0
- VRAM ≥ 12 GB (e.g., Tesla P100 or better)
- Multi-GPU automatically supported via torch.nn.DataParallel

---

## 🚀 Training Workflow

Step 1 — MLM Pretraining
```bash
python mlm_pretrain.py
```
Creates the folder: xlmr_ebay_mlm/ with adapted domain-specific weights.

Step 2 — NER Fine-tuning
```bash
python roberta_baseline.py
```
Saves trained model checkpoints under: ebay_ner_model/

Step 3 — Inference and Submission
```bash
python roberta_infer.py
```
Produces: submission_quiz.tsv (EvalAI-ready)

Step 4 — EvalAI Submission
Upload the generated TSV file to the competition portal.

Expected columns (tab-separated):
record_id    category_id    aspect_name    aspect_value

Submission requirements:
- UTF-8 encoding
- No header line
- Tab-separated
- .tsv or .tsv.gz accepted

---

## ⚡ Tips for Faster or Better Training

Setting                         | Effect
--------------------------------|----------------------------------------
max_length=64                   | 2× faster training on short titles
fp16=True                       | Enables mixed precision for speed + VRAM savings
mlm_probability=0.10            | Faster MLM at slight quality tradeoff
gradient_accumulation_steps     | Simulates larger batch sizes
Re-use xlmr_ebay_mlm            | Improves NER Fβ by 1–2 points vs raw model

---

## 📈 Leaderboard Performance

Model                         | Pretraining       | Fβ-Score
------------------------------|------------------|---------
XLM-RoBERTa-Base (baseline)   | None              | 0.867 ± 0.002
Domain-adapted MLM + NER      | MLM (2M titles)   | 0.876912 ↑

---

## 🧾 License and Compliance

This repository abides by the official eBay 2025 ML Challenge rules:

- Uses permissive open-source components (MIT/BSD/Apache 2.0)
- No eBay data sent to third-party AI platforms
- No manual annotations performed
- Models are self-contained and commercially deployable

---

## 👩‍💻 Authors

USM_AI_Renaissance (University of Southern Mississippi)  
- Suramya Angdembay 
- Gunjan Sah

### For reproducibility and research inquiries, contact: Suramya Angdembay
### For Data, contact eBay Ml challenge team


