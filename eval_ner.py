import argparse
import os
from typing import List, Tuple

import torch
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report


def read_conll_2col(path: str) -> Tuple[List[List[str]], List[List[str]]]:
    """Reads 2-column CoNLL (TOKEN TAG) with blank lines between sentences."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="outputs/bert-base-cased-timeNER",
                        help="Path to the fine-tuned model directory (with config.json, tokenizer files, weights).")
    parser.add_argument("--test_path", type=str, default="data/test.conll",
                        help="Path to 2-column CoNLL test file.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()

    assert os.path.exists(args.model_dir), f"Model dir not found: {args.model_dir}"
    assert os.path.exists(args.test_path), f"Test file not found: {args.test_path}"

    # Load tokenizer & model
    print(f"Loading model from: {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.eval()

    # Labels
    id2label = model.config.id2label
    label2id = model.config.label2id
    labels_sorted = [id2label[i] for i in range(len(id2label))]
    print(f"Model labels: {labels_sorted}")

    # Read test set
    print(f"Reading test set: {args.test_path}")
    tokens_list, tags_list = read_conll_2col(args.test_path)
    num_sents = len(tokens_list)
    num_tokens = sum(len(s) for s in tokens_list)
    print(f"Loaded {num_sents} sentences / {num_tokens} tokens")

    # Sanity check: test labels should be subset of model labels
    uniq_test_labels = sorted({t for seq in tags_list for t in seq})
    missing = [t for t in uniq_test_labels if t not in label2id]
    if missing:
        print(f"⚠️  Warning: test labels not in model: {missing}")

    # Build a simple dataset for batching convenience
    ds = Dataset.from_dict({"tokens": tokens_list, "ner_tags": tags_list})

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    

    # Evaluation loop (word-piece alignment; label only first subword)
    all_preds: List[List[str]] = []
    all_refs: List[List[str]] = []

    # iterate in batches
    for start in range(0, len(ds), args.batch_size):
        batch = ds[start : start + args.batch_size]
        batch_tokens = batch["tokens"]  # List[List[str]]
        batch_refs = batch["ner_tags"]  # List[List[str]]

        # Tokenize as split words so we can map back using fast tokenizer encodings
        encodings = tokenizer(
            batch_tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
            padding=True,
        )

        with torch.no_grad():
            logits = model(
                input_ids=encodings["input_ids"].to(device),
                attention_mask=encodings["attention_mask"].to(device),
                token_type_ids=encodings.get("token_type_ids", None).to(device) if "token_type_ids" in encodings else None,
            ).logits  # (bsz, seq_len, num_labels)

        pred_ids = logits.argmax(dim=-1).cpu().numpy()  # (bsz, seq_len)

        # Recover word-level predictions using encodings.word_ids()
        for i, word_labels in enumerate(batch_refs):
            encoding = encodings.encodings[i]  # fast tokenizer encoding
            word_ids = encoding.word_ids  # list[Optional[int]] aligned to tokens
            seq_pred_ids = pred_ids[i]

            word_level_preds: List[str] = []
            seen_word = None
            for tok_idx, wid in enumerate(word_ids):
                if wid is None:
                    continue
                if wid != seen_word:
                    # first subword of this word → take prediction
                    label_id = int(seq_pred_ids[tok_idx])
                    word_level_preds.append(id2label[label_id])
                    seen_word = wid
                else:
                    # subsequent subwords → skip (standard NER eval)
                    continue

            # Trim to same length as references (in case of truncation)
            L = min(len(word_labels), len(word_level_preds))
            all_refs.append(word_labels[:L])
            all_preds.append(word_level_preds[:L])

    # Metrics
    p = precision_score(all_refs, all_preds)
    r = recall_score(all_refs, all_preds)
    f1 = f1_score(all_refs, all_preds)

    print("\n Results on test set")
    print(f"Precision: {p:.4f}")
    print(f"Recall   : {r:.4f}")
    print(f"F1       : {f1:.4f}")

    print("\nSeqeval classification report")
    print(classification_report(all_refs, all_preds, digits=4))


if __name__ == "__main__":
    main()