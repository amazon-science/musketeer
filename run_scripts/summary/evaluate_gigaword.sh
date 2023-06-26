#!/usr/bin/env bash
echo "Evaluate model: $1"

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=2081

user_dir=../../ofa_module
bpe_dir=../../utils/BPE

data=$DATADIR/gigaword/gigaword_test.tsv
path=$1

result_path=../../results/gigaword
selected_cols=0,1
split='test'

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
    --task=gigaword \
    --batch-size=32 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --beam=6 \
    --lenpen=0.7 \
    --max-len-b=32 \
    --no-repeat-ngram-size=3 \
    --fp16 \
    --num-workers=0 \
    --model-overrides="{\"description\":\"${description}\",\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"

python3 eval_rouge.py ${result_path}/test_predict.json
