import pandas as pd

# Listing titles (2 million rows)
df_listing = pd.read_csv(
    "Listing_Titles.tsv",
    sep="\t",
    keep_default_na=False,  # don't turn "" into NaN
    na_values=None,
    dtype=str
)

# Tagged train data (≈ 56k rows)
df_train = pd.read_csv(
    "Tagged_Titles_Train.tsv",
    sep="\t",
    keep_default_na=False,
    na_values=None,
    dtype=str
)
df_listing = df_listing.rename(columns={
    "Record Number": "record_id",
    "Category": "category_id",
    "Title": "title"
})

df_train = df_train.rename(columns={
    "Record Number": "record_id",
    "Category": "category_id",
    "Title": "title",
    "Token": "token",
    "Tag": "raw_tag"
})
print(df_listing.shape, df_listing.columns)
print(df_train.shape, df_train.columns)

def record_to_bio(group):
    tokens = group['token'].tolist()
    raw_tags = group['raw_tag'].tolist()
    bios = []

    prev_tag_nonempty = None  # last seen non-empty, non-"O" tag
    prev_raw_tag = None       # raw tag of previous token (can be same string, or "")

    for i, rt in enumerate(raw_tags):
        if rt == "O":
            bios.append("O")
            prev_tag_nonempty = None
            prev_raw_tag = "O"
        elif rt == "":
            # continuation of previous non-empty tag
            if prev_tag_nonempty is None:
                # edge case: blank but we have nothing to continue -> call it "O"
                bios.append("O")
            else:
                bios.append("I-" + prev_tag_nonempty)
            prev_raw_tag = ""
        else:
            # new explicit tag
            # decide if B- or new B- (always B, because spec says even same tag twice can be two entities)
            bios.append("B-" + rt)
            if rt != "O":
                prev_tag_nonempty = rt
            else:
                prev_tag_nonempty = None
            prev_raw_tag = rt

    return {
        "tokens": tokens,
        "bio_tags": bios,
        "category_id": group['category_id'].iloc[0],
        "record_id": group['record_id'].iloc[0],
        "title": group['title'].iloc[0],
    }

grouped = df_train.groupby('record_id', sort=True).apply(record_to_bio)
train_sequences = list(grouped)

all_labels = sorted({tag for seq in train_sequences for tag in seq["bio_tags"]})
# Example: ["B-Hersteller","B-Kompatibles_Fahrzeug_Modell",...,"I-...","O"]

label2id = {lbl:i for i,lbl in enumerate(all_labels)}
id2label = {i:lbl for lbl,i in label2id.items()}

from transformers import AutoTokenizer

model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def encode_sequence(tokens, bio_tags, category_id):
    # prepend synthetic category token
    cat_token = f"[CAT_{category_id}]"
    input_tokens = [cat_token] + tokens
    input_tags   = ["O"] + bio_tags  # CAT token gets "O"

    enc = tokenizer(
        input_tokens,
        is_split_into_words=True,
        return_offsets_mapping=False,
        truncation=True,
        max_length=128,   # adjust if titles can be longer; most won't
        return_tensors=None,
    )

    # enc.word_ids() gives index of which original input_tokens word each subword came from
    word_ids = enc.word_ids()

    labels = []
    for wi in word_ids:
        if wi is None:
            labels.append(-100)  # special tokens like <s>, </s>
        else:
            # only label the first subword of each original token
            # rule: if this word_id is same as previous word_id, use -100
            if len(labels)>0 and word_ids[labels.index(labels[-1]) if labels[-1]!=-100 else len(labels)-1] == wi:
                # Actually easier: track prev_wi
                pass

def encode_sequence(tokens, bio_tags, category_id):
    cat_token = f"[CAT_{category_id}]"
    input_tokens = [cat_token] + tokens
    input_tags   = ["O"] + bio_tags

    enc = tokenizer(
        input_tokens,
        is_split_into_words=True,
        return_offsets_mapping=False,
        truncation=True,
        max_length=128,
    )

    word_ids = enc.word_ids()

    labels = []
    prev_wi = None
    for wi in word_ids:
        if wi is None:
            labels.append(-100)
        else:
            if wi != prev_wi:
                # first subword of this original token → real label id
                lbl = input_tags[wi]
                labels.append(label2id[lbl])
                prev_wi = wi
            else:
                # continuation subword → ignore in loss
                labels.append(-100)

    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": labels,
    }

encoded_train = [encode_sequence(seq["tokens"], seq["bio_tags"], seq["category_id"])
                 for seq in train_sequences]


import torch
from torch.utils.data import Dataset, DataLoader

class NERDataset(Dataset):
    def __init__(self, enc_list):
        self.data = enc_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
        }

train_dataset = NERDataset(encoded_train)

from transformers import DataCollatorForTokenClassification
data_collator = DataCollatorForTokenClassification(tokenizer)

from transformers import AutoModelForTokenClassification

num_labels = len(label2id)

model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./ebay_ner_model",
    per_device_train_batch_size=16,
    num_train_epochs=3,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps=50,
    save_strategy="epoch",
    # evaluation_strategy="no",  # baseline: no val split yet
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=None,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
