#!/usr/bin/env bash
echo "Evaluate model: $1"

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=7091
export PYTHONPATH=../../fairseq/:$PYTHONPATH
export NCCL_HOME=~/anaconda3/pkgs/nccl-2.14.3.1-h0800d71_0
#export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ens32
export NCCL_IB_DISABLE=1
user_dir=../../ofa_module
bpe_dir=../../utils/BPE

# dev or test
split=$2

data=$DATADIR/snli_ve_data/snli_ve_${split}.tsv


path=$1


result_path=../../results/snli_ve
description=tep




selected_cols=0,2,3,4,5

python3 -m torch.distributed.launch --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --bpe-dir=${bpe_dir} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=snli_ve \
    --batch-size=16 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --fp16 \
    --num-workers=0 \
    --selected-cols=${selected_cols} \
    --model-overrides="{\"description\":\"${description}\",\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"
