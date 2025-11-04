# mlm_pretrain.py

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

# ---------------------------
# 0. Setup
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

base_model_name = "xlm-roberta-base"
output_dir = "xlmr_ebay_mlm"   # this dir will be used later in NER training

# ---------------------------
# 1. Load titles
# ---------------------------
df = pd.read_csv(
    "Listing_Titles.tsv",
    sep="\t",
    keep_default_na=False,
    na_values=None,
    dtype=str,
)

# Column is "Title" in the raw file
titles = df["Title"].astype(str).tolist()
print("Total titles:", len(titles))

# Optional: subsample for speed if needed
# e.g. use at most 1,000,000 titles
MAX_TITLES = 200_000  # try 500k first; you can lower to 200k if needed
if len(titles) > MAX_TITLES:
    titles = titles[:MAX_TITLES]
    print("Subsampled to:", len(titles))


# ---------------------------
# 2. Dataset
# ---------------------------
class TitleDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=64):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        # remove batch dim
        return {k: v.squeeze(0) for k, v in enc.items()}


tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForMaskedLM.from_pretrained(base_model_name).to(device)

dataset = TitleDataset(titles, tokenizer, max_length=64)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

# ---------------------------
# 3. Training arguments
# ---------------------------
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    per_device_train_batch_size=16,    # was 32 -> much less peak memory
    gradient_accumulation_steps=4,  # adjust if OOM
    num_train_epochs=1.0,            # keep it small for time
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps=500,
    save_strategy="epoch",
    prediction_loss_only=True,
    fp16=True,
    dataloader_num_workers=4, 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

print("Starting MLM pretraining...")
trainer.train()

print(f"Saving MLM-adapted model to {output_dir} ...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print("Done.")

