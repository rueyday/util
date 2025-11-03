"""
Train a small decoder-only LM using Hugging Face Trainer + optional DeepSpeed.
This script expects:
  --data_files  : one or more text files (UTF-8). Each file may contain long text; we'll split into lines.
  --tokenizer_dir: directory containing tokenizer.json produced by train_tokenizer.py
  --config: model json config (example_config.json)
  --deepspeed: optional deepspeed config json path
"""
import argparse
import json
import os
from pathlib import Path
import math
import random

import numpy as np
import torch
from datasets import load_dataset, Dataset
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)

def build_tokenizer(tokenizer_dir):
    # Expect tokenizer.json created by tokenizers lib
    tokenizer_file = Path(tokenizer_dir) / "tokenizer.json"
    if tokenizer_file.exists():
        tok = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file))
        # Ensure pad token exists
        if tok.pad_token is None:
            tok.add_special_tokens({"pad_token": "<|pad|>"})
        return tok
    else:
        # fallback to gpt2 base
        from transformers import GPT2TokenizerFast
        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        if tok.pad_token is None:
            tok.add_special_tokens({"pad_token": "<|pad|>"})
        return tok

def read_and_dedupe(files):
    seen = set()
    out = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if not s:
                    continue
                h = s.encode("utf-8")
                if h in seen:
                    continue
                seen.add(h)
                out.append(s)
    return out

def tokenize_and_group(tokenizer, texts, seq_length, stride=0):
    """
    Tokenize list of texts (strings), then concatenate and chunk into blocks of seq_length.
    Returns a Dataset with input_ids and labels.
    """
    # Tokenize (fast)
    encodings = tokenizer(texts, return_attention_mask=False, truncation=False)
    # Concatenate all input_ids
    all_ids = []
    for ids in encodings["input_ids"]:
        all_ids.extend(ids)
    # Optionally add eos between documents (already may be included by tokenizer post_processor)
    total_len = len(all_ids)
    # Drop tail that is smaller than seq_length
    if total_len < seq_length:
        # pad shorter sequences later in data collator if needed; for now return one chunk padded
        chunks = []
        if total_len > 0:
            chunk = all_ids + [tokenizer.pad_token_id] * (seq_length - total_len)
            chunks.append(chunk)
    else:
        usable_len = (total_len // seq_length) * seq_length
        chunks = [all_ids[i : i + seq_length] for i in range(0, usable_len, seq_length)]
    # Build labels = same as input_ids for causal LM
    ds = Dataset.from_dict({"input_ids": chunks, "labels": chunks})
    return ds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_files", nargs="+", required=True)
    parser.add_argument("--tokenizer_dir", default=None)
    parser.add_argument("--output_dir", default="runs/exp")
    parser.add_argument("--config", default="example_config.json")
    parser.add_argument("--deepspeed", default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    if args.tokenizer_dir and Path(args.tokenizer_dir).exists():
        tokenizer = build_tokenizer(args.tokenizer_dir)
        print("Loaded tokenizer from", args.tokenizer_dir)
    else:
        print("No tokenizer found; using GPT-2 tokenizer fallback.")
        tokenizer = build_tokenizer(None)

    # Read + dedupe text lines
    print("Reading input files:", args.data_files)
    lines = read_and_dedupe(args.data_files)
    print(f"Read {len(lines)} unique non-empty lines/documents.")

    # Quick sanity: if very large dataset, you may wish to sample for local runs
    if len(lines) > 5_000_000:
        print("Very large dataset detected; sampling 2M lines for local preprocessing.")
        lines = random.sample(lines, 2_000_000)

    # Tokenize + group into blocks
    print("Tokenizing and grouping into blocks of length", args.seq_length)
    lm_ds = tokenize_and_group(tokenizer, lines, seq_length=args.seq_length)
    print("Built lm dataset with", len(lm_ds), "blocks.")

    # Train / val split
    split = lm_ds.train_test_split(test_size=0.02, seed=args.seed)
    train_ds = split["train"]
    eval_ds = split["test"]
    print("Train blocks:", len(train_ds), "Eval blocks:", len(eval_ds))

    # Build model from config
    cfg = json.load(open(args.config))
    model_config = GPT2Config(
        vocab_size=cfg.get("vocab_size", tokenizer.vocab_size),
        n_positions=cfg.get("n_positions", args.seq_length),
        n_ctx=cfg.get("n_ctx", args.seq_length),
        n_embd=cfg.get("n_embd", 768),
        n_layer=cfg.get("n_layer", 12),
        n_head=cfg.get("n_head", 12),
        resid_pdrop=cfg.get("dropout", 0.0),
        embd_pdrop=cfg.get("dropout", 0.0),
    )
    model = GPT2LMHeadModel(model_config)
    # Resize embeddings if tokenizer vocab differs
    model.resize_token_embeddings(model_config.vocab_size)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training args (Trainer will wrap DeepSpeed if deepspeed arg provided)
    training_args = TrainingArguments(
        output_dir="checkpoints",
        do_eval=True,
        eval_steps=100,
        logging_steps=50,
        save_steps=500,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=1,
        learning_rate=5e-4,
        weight_decay=0.01,
        warmup_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    # Kick off training
    trainer.train()
    trainer.save_model(args.output_dir)
    print("Training finished. Model saved to", args.output_dir)

if __name__ == "__main__":
    main()
