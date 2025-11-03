"""
preprocess.py
Simple script to build a HF `datasets` dataset from raw text files and do basic dedup/cleanup.
Produces a `.jsonl` dataset or HF dataset folder.
"""
import argparse
from datasets import load_dataset, Dataset
import os
import hashlib

def dedupe_lines(lines):
    seen = set()
    out = []
    for l in lines:
        h = hashlib.sha1(l.encode('utf-8')).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        out.append(l)
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', nargs='+', required=True, help='input txt files (one or many)')
    parser.add_argument('--out_dir', default='processed', help='output folder')
    args = parser.parse_args()


    texts = []
    for f in args.input:
        with open(f, 'r', encoding='utf-8') as fh:
            texts.extend([ln.strip() for ln in fh if ln.strip()])


    texts = dedupe_lines(texts)
    os.makedirs(args.out_dir, exist_ok=True)


    ds = Dataset.from_dict({ 'text': texts })
    ds.save_to_disk(args.out_dir)
    print(f"Saved processed dataset ({len(texts)} lines) to {args.out_dir}")


if __name__ == '__main__':
    main()