#!/bin/bash

train_test_data_dir='/data/jhli/mt-exp/goodquality/data'
tune_set='nist06'
data_dir='./workspace/data'
if [ ! -d "$data_dir" ]; then mkdir -p "$data_dir"; fi
data_prefix="$data_dir/gq"

python3.6 ./preprocess.py -train_src $train_test_data_dir/train.bpe.cn \
                       -train_tgt $train_test_data_dir/train.bpe.en \
                       -valid_src $train_test_data_dir/$tune_set.bpe.cn  \
                       -valid_tgt $train_test_data_dir/$tune_set.bpe.en0 \
                       -save_data $data_prefix \
                       -src_vocab_size 30000  \
                       -tgt_vocab_size 30000 \
                       -src_seq_length 150 \
                       -tgt_seq_length 150
