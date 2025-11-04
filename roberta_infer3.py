import csv
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification

# ---------------------------
# 0. Setup: device + checkpoint
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

checkpoint_dir = "ebay_ner_model/checkpoint-471"  # your best checkpoint

tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
model = AutoModelForTokenClassification.from_pretrained(checkpoint_dir).to(device)
model.eval()

# Normalize config mappings
raw_id2label = model.config.id2label
raw_label2id = model.config.label2id

# Ensure id2label has int keys
if isinstance(raw_id2label, dict):
    id2label = {int(k) if str(k).isdigit() else k: v for k, v in raw_id2label.items()}
elif isinstance(raw_id2label, list):
    id2label = {i: lbl for i, lbl in enumerate(raw_id2label)}
else:
    id2label = raw_id2label  # fallback

# Ensure label2id is label:string -> id:int
if isinstance(raw_label2id, dict):
    label2id = {lbl: int(v) if str(v).isdigit() else v for lbl, v in raw_label2id.items()}
elif isinstance(raw_label2id, list):
    label2id = {lbl: i for i, lbl in enumerate(raw_label2id)}
else:
    label2id = raw_label2id  # fallback


# ---------------------------
# 0.5 Build allowed_aspects from training tags
# ---------------------------
df_train_tags = pd.read_csv(
    "Tagged_Titles_Train.tsv",
    sep="\t",
    keep_default_na=False,
    na_values=None,
    dtype=str,
).rename(
    columns={
        "Record Number": "record_id",
        "Category": "category_id",
        "Title": "title",
        "Token": "token",
        "Tag": "raw_tag",
    }
)

allowed_aspects = {}
for _, row in df_train_tags.iterrows():
    cat = row["category_id"]
    tag = row["raw_tag"]
    if not tag:  # empty string = continuation, not an aspect name
        continue
    # just in case, strip whitespace
    tag = tag.strip()
    if tag == "":
        continue
    try:
        cid = int(cat)
    except ValueError:
        continue
    allowed_aspects.setdefault(cid, set()).add(tag)

print("Allowed aspects per category (from train):")
for cid, tags in allowed_aspects.items():
    print(f"  Category {cid}: {sorted(tags)}")


def is_valid_aspect(cat_id_str, aspect_name):
    """
    cat_id_str: '1' or '2'
    aspect_name: e.g. 'Hersteller'
    """
    try:
        cid = int(cat_id_str)
    except ValueError:
        return False
    return aspect_name in allowed_aspects.get(cid, set())


# ---------------------------
# 1. Load listing titles
# ---------------------------
df_listing = pd.read_csv(
    "Listing_Titles.tsv",
    sep="\t",
    keep_default_na=False,
    na_values=None,
    dtype=str,
).rename(
    columns={
        "Record Number": "record_id",
        "Category": "category_id",
        "Title": "title",
    }
)

# restrict to quiz superset: 5001..30000
df_quiz = df_listing.copy()
df_quiz["record_id_int"] = df_quiz["record_id"].astype(int)
df_quiz = df_quiz[
    (df_quiz["record_id_int"] >= 5001)
    & (df_quiz["record_id_int"] <= 30000)
]

print("Quiz subset size:", len(df_quiz))


# ---------------------------
# 2. Encoding for inference
# ---------------------------
def encode_sequence_infer(tokens, category_id):
    """
    tokens: list of whitespace tokens from title
    category_id: "1" or "2"
    """
    cat_token = f"[CAT_{category_id}]"
    input_tokens = [cat_token] + tokens

    enc = tokenizer(
        input_tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    # word_ids ties subwords back to the input_tokens index
    word_ids = enc.word_ids()
    return enc, word_ids, input_tokens


# ---------------------------
# 3. Predict BIO tags per whitespace token
# ---------------------------
@torch.no_grad()
def predict_tags_for_listing(tokens, category_id):
    enc, word_ids, input_tokens = encode_sequence_infer(tokens, category_id)

    enc = {k: v.to(device) for k, v in enc.items()}

    outputs = model(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
    )
    logits = outputs.logits[0]  # (seq_len, num_labels)
    preds = torch.argmax(logits, dim=-1).tolist()

    # Map back to one label per ORIGINAL whitespace token
    pred_tags = []
    seen_wi = set()

    for idx, wi in enumerate(word_ids):
        if wi is None:
            continue
        if wi in seen_wi:
            # continuation subword of same original token
            continue
        seen_wi.add(wi)

        # wi == 0 is our category pseudo-token -> skip it
        if wi == 0:
            continue

        tag_id = preds[idx]
        tag_label = id2label[tag_id]
        pred_tags.append(tag_label)

    # If sequence was truncated, pred_tags can be shorter than tokens.
    if len(pred_tags) != len(tokens):
        min_len = min(len(pred_tags), len(tokens))
        print(
            f"Warning: clipping to min length for this title: "
            f"pred_tags={len(pred_tags)}, tokens={len(tokens)} -> {min_len}"
        )
        pred_tags = pred_tags[:min_len]

    return pred_tags


# ---------------------------
# 4. BIO → spans (aspect_name, aspect_value)
# ---------------------------
def bio_to_spans(tokens, bio_tags):
    spans = []
    current_aspect = None
    current_tokens = []

    def flush():
        nonlocal current_aspect, current_tokens, spans
        if current_aspect is not None and current_aspect != "O":
            spans.append((current_aspect, " ".join(current_tokens)))
        current_aspect = None
        current_tokens = []

    for tok, tag in zip(tokens, bio_tags):
        if tag == "O":
            flush()
            continue

        if tag.startswith("B-"):
            flush()
            current_aspect = tag[2:]
            current_tokens = [tok]
        elif tag.startswith("I-"):
            aspect = tag[2:]
            if current_aspect == aspect:
                current_tokens.append(tok)
            else:
                # malformed I- without preceding B-: treat as new B-
                flush()
                current_aspect = aspect
                current_tokens = [tok]
        else:
            # unexpected label format
            flush()

    flush()
    return spans  # list of (aspect_name, aspect_value)


# ---------------------------
# 5. Loop over quiz listings and build submission rows
# ---------------------------
rows_for_submission = []

for idx, row in df_quiz.iterrows():
    rid = row["record_id"]
    cid = row["category_id"]
    title = row["title"]

    # Whitespace-only tokenization exactly as in the Annexure
    tokens = [t for t in title.split() if t.strip() != ""]

    if not tokens:
        continue

    bio_tags = predict_tags_for_listing(tokens, cid)
    spans = bio_to_spans(tokens, bio_tags)

    for aspect_name, aspect_val in spans:
        rows_for_submission.append(
            {
                "record_id": rid,
                "category_id": cid,
                "aspect_name": aspect_name,
                "aspect_value": aspect_val,
            }
        )

print("Total extracted spans (before filtering):", len(rows_for_submission))

submission_df = pd.DataFrame(
    rows_for_submission,
    columns=["record_id", "category_id", "aspect_name", "aspect_value"],
)

# ---------------------------
# 6. Filter invalid aspect/category combos (based on train usage)
# ---------------------------
before = len(submission_df)
submission_df = submission_df[
    submission_df.apply(
        lambda r: is_valid_aspect(r["category_id"], r["aspect_name"]),
        axis=1,
    )
].reset_index(drop=True)
after = len(submission_df)

print(f"Filtered out {before - after} invalid aspect/category rows")
print("Final spans to submit:", after)

# ---------------------------
# 7. Write EvalAI submission file
# ---------------------------
submission_df.to_csv(
    "submission_quiz.tsv",
    sep="\t",
    index=False,
    header=False,  # EvalAI allows optional header; we keep it off
    quoting=csv.QUOTE_NONE,
)

print("Wrote submission_quiz.tsv")

