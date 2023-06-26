#!/usr/bin/env bash
echo "Evaluate model: $1"

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=16088
export PYTHONPATH=../../fairseq/:$PYTHONPATH


export NCCL_HOME=~/anaconda3/pkgs/nccl-2.14.3.1-h0800d71_0
#export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ens32
export NCCL_IB_DISABLE=1

########################## Evaluate Refcoco ##########################
user_dir=../../ofa_module
bpe_dir=../../utils/BPE
selected_cols=0,4,2,3

data=$DATADIR/refcoco_data/refcocog_test.tsv


path=$1


result_path=../../results/refcoco-joint-fewshot-1ref-plus
description=tep

mkdir -r ${result_path}

split='refcocog_test'
python3 -m torch.distributed.launch --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --bpe-dir=${bpe_dir} \
    --task=refcoco \
    --batch-size=16 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --beam=5 \
    --min-len=4 \
    --max-len-a=0 \
    --max-len-b=4 \
    --no-repeat-ngram-size=3 \
    --fp16 \
    --num-workers=0 \
    --selected-cols=${selected_cols} \
    --model-overrides="{\"description\":\"${description}\",\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}" | tee -a ${result_path}/log_val.txt

data=$DATADIR/refcoco_data/refcocog_val.tsv
split='refcocog_val'
python3 -m torch.distributed.launch --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
        --bpe-dir=${bpe_dir} \
    --task=refcoco \
    --batch-size=16 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --beam=5 \
    --min-len=4 \
    --max-len-a=0 \
    --max-len-b=4 \
    --no-repeat-ngram-size=3 \
    --fp16 \
    --num-workers=0 \
    --model-overrides="{\"description\":\"${description}\",\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}" | tee -a ${result_path}/log_A.txt

