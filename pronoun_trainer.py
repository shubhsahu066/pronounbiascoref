import json
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

MODEL_NAME = "roberta-base"
LABELS = ["O", "B-PRONOUN"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

def load_dataset(path):
    texts, spans = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            texts.append(ex["text"])
            spans.append(ex["spans"])
    return Dataset.from_dict({"text": texts, "spans": spans})

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)

def tokenize_and_align(example):
    encoding = tokenizer(
        example["text"],
        truncation=True,
        return_offsets_mapping=True
    )

    labels = ["O"] * len(encoding["offset_mapping"])

    for span in example["spans"]:
        for i, (start, end) in enumerate(encoding["offset_mapping"]):
            if start == span["start"]:
                labels[i] = "B-PRONOUN"

    encoding["labels"] = [LABEL2ID[l] for l in labels]
    encoding.pop("offset_mapping")
    return encoding

def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)

    true_labels, true_preds = [], []

    for p, l in zip(preds, labels):
        for pi, li in zip(p, l):
            if li != -100:
                true_labels.append(li)
                true_preds.append(pi)

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels,
        true_preds,
        average="binary",
        pos_label=LABEL2ID["B-PRONOUN"],
        zero_division=0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def main():
    dataset = load_dataset("dataset.jsonl")
    dataset = dataset.train_test_split(test_size=0.15, seed=42)

    tokenized = dataset.map(
        tokenize_and_align,
        batched=False
    )

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    args = TrainingArguments(
        output_dir="./pronoun_model_roberta",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=4,
        weight_decay=0.01,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model("./pronoun_model_roberta")

if __name__ == "__main__":
    main()
