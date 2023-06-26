#!/usr/bin/env bash

echo "Evaluate model: $1"

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=8182

user_dir=../../ofa_module
bpe_dir=../../utils/BPE

# val or test
split=$2

data=$DATADIR/vqa_data/vqa_${split}.tsv
ans2label_file=$DATADIR/vqa_data/trainval_ans2label.pkl

path=$1


result_path=../../results/vqa_${split}_beam-large-anno-ref
selected_cols=0,5,2,3,4
valid_batch_size=20

export PYTHONPATH=../../fairseq/:$PYTHONPATH

export NCCL_HOME=~/anaconda3/pkgs/nccl-2.14.3.1-h0800d71_0
#export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ens32
export NCCL_IB_DISABLE=1

description=tep



python3 -m torch.distributed.launch --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --bpe-dir=${bpe_dir} \
    --selected-cols=${selected_cols} \
    --ans2label-file=${ans2label_file} \
    --valid-batch-size=${valid_batch_size} \
    --task=vqa_gen \
    --batch-size=32 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --fp16 \
    --beam-search-vqa-eval \
    --beam=5 \
    --unnormalized \
    --temperature=1.0 \
    --num-workers=0 \
    --max-len-b=640 \
    --model-overrides="{\"description\":\"${description}\",\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\",\"ans2label_file\":\"${ans2label_file}\",\"valid_batch_size\":\"${valid_batch_size}\"}"
