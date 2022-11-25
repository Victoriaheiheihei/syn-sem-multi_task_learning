#!/bin/bash

train_test_data_dir='/data/jhli/mt-exp/goodquality/data'
model_file='/data/jhli/mt-exp/goodquality/t2t/workspace-opennmt/model/model_step_150000.pt'
output_dir='/data/jhli/mt-exp/goodquality/t2t/workspace-opennmt/model'
tests=(nist02 nist03 nist04 nist05 nist06 nist08)

for ((i=0;i<=0;i++));
do
output_file=$output_dir/${tests[i]}.tran

CUDA_VISIBLE_DEVICES=2  python3 ../translate.py \
                        -model $model_file \ 
                        -src $train_test_data_dir/${tests[i]}'.bpe.cn' \
                        -output $output_file 
                        -beam_size 5

done
