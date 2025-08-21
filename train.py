import os
import argparse
import random
from typing import List, Tuple

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_conll_2col(path: str) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Reads a 2-column CoNLL file:
        TOKEN TAG
    Blank line separates sentences.
    Returns (tokens_per_sentence, tags_per_sentence).
    """
    toks, labs = [], []
    all_toks, all_labs = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if toks:
                    all_toks.append(toks)
                    all_labs.append(labs)
                    toks, labs = [], []
                continue
            parts = line.split()
            if len(parts) < 2:
                # tolerate malformed lines
                continue
            tok, tag = parts[0], parts[-1]
            toks.append(tok)
            labs.append(tag)
    if toks:
        all_toks.append(toks)
        all_labs.append(labs)
    return all_toks, all_labs

def build_label_maps(tags: List[List[str]]):
    """
    Build label list & maps from training tags only.
    Ensures 'O' is index 0 for convenience.
    """
    uniq = set()
    for seq in tags:
        uniq.update(seq)
    labels = ["O"] + sorted([x for x in uniq if x != "O"])
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    return labels, label2id, id2label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/train.conll")
    parser.add_argument("--output_dir", type=str, default="outputs/bert-base-cased-timeNER")
    parser.add_argument("--model_name", type=str, default="bert-base-cased",
                        help="Backbone (paper used BERT base cased).")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256,
                        help="Increase to 512 for longer sentences if needed.")
    parser.add_argument("--label_all_tokens", action="store_true",
                        help="If set, label all wordpieces; default labels only first subword.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    # 1) Load training data
    tokens, tags = read_conll_2col(args.train_path)
    labels, label2id, id2label = build_label_maps(tags)
    print(f"Labels: {labels}")

    # 2) Hugging Face tokenizer/model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    # 3) Build Dataset
    ds_train = Dataset.from_dict({"tokens": tokens, "ner_tags": tags})

    def encode_batch(batch):
        tokenized = tokenizer(
            batch["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=args.max_length,
        )
        aligned_labels = []
        for i, word_labels in enumerate(batch["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            prev_word = None
            label_ids = []
            for wid in word_ids:
                if wid is None:
                    label_ids.append(-100)  # ignore in loss
                elif wid != prev_word:
                    label_ids.append(label2id[word_labels[wid]])
                else:
                    # subword piece
                    if args.label_all_tokens:
                        # Optional: convert B- to I- for subsequent pieces if using BIO
                        lab = word_labels[wid]
                        if lab.startswith("B-"):
                            lab = "I-" + lab[2:]
                        label_ids.append(label2id.get(lab, label2id[word_labels[wid]]))
                    else:
                        label_ids.append(-100)
                prev_word = wid
            aligned_labels.append(label_ids)
        tokenized["labels"] = aligned_labels
        return tokenized

    ds_train = ds_train.map(encode_batch, batched=True, remove_columns=["tokens", "ner_tags"])

    # 4) TrainingArguments (no eval)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_ratio=0.06,
        weight_decay=0.01,
        logging_steps=50,
        save_steps=1000,                 # adjust to your dataset size
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
        gradient_accumulation_steps=1,
        seed=args.seed,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    # 5) Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    # 6) Save final model (for reuse in attacks)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    

if __name__ == "__main__":
    main()