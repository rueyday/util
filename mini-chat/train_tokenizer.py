"""
Train a sentencepiece BPE tokenizer using the `tokenizers` and `sentencepiece` libraries.
"""
import argparse
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
from tokenizers import normalizers
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

def train_tokenizer(input_files, out_dir, vocab_size=32000):
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
    tokenizer.pre_tokenizer = Whitespace()


    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>"])
    tokenizer.train(input_files, trainer)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<|bos|> $A <|eos|>",
        pair="<|bos|> $A <|eos|> $B:1 <|eos|>:1",
        special_tokens=[("<|bos|>", tokenizer.token_to_id("<|bos|>")), ("<|eos|>", tokenizer.token_to_id("<|eos|>"))]
    )
    tokenizer.save(f"{out_dir}/tokenizer.json")
    print('Saved tokenizer to', out_dir)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', nargs='+', required=True)
    parser.add_argument('--out_dir', default='tokenizer')
    parser.add_argument('--vocab_size', type=int, default=32000)
    args = parser.parse_args()
    train_tokenizer(args.input, args.out_dir, args.vocab_size)


if __name__ == '__main__':
    main()