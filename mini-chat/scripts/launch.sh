python train_tokenizer.py --input data/clean_corpus.txt --vocab_size 32000 --out_dir tokenizer

torchrun --nproc_per_node=2 train.py \
--data_files data/clean_corpus.txt \
--tokenizer_dir tokenizer \
--output_dir runs/exp1 \
--config example_config.json \
--deepspeed deepspeed_config.json
