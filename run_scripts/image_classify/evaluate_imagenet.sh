#!/usr/bin/env bash
echo "Evaluate model: $1"

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=8087

user_dir=../../ofa_module
bpe_dir=../../utils/BPE



data=$DATADIR/imagenet_1k_data/imagenet_1k_val.tsv
# Note: If you have shuffled the data in advance, please uncomment the line below.
# data=${data_dir}/imagenet_1k_train_1.tsv,${data_dir}/imagenet_1k_train_2.tsv,${data_dir}/imagenet_1k_train_3.tsv,${data_dir}/imagenet_1k_train_4.tsv,${data_dir}/imagenet_1k_train_5.tsv,${data_dir}/imagenet_1k_train_6.tsv,${data_dir}/imagenet_1k_train_7.tsv,${data_dir}/imagenet_1k_train_8.tsv,${data_dir}/imagenet_1k_train_9.tsv,${data_dir}/imagenet_1k_train_10.tsv,${data_dir}/imagenet_1k_val_subset.tsv
ans2label_file=$DATADIR/imagenet_1k_data/class2label_new.pkl

### Path to CKPT
path=$1


result_path=../../results/imagenet_1k_val
selected_cols=0,2


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
    --task=image_classify \
    --batch-size=64 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=val \
    --results-path=${result_path} \
    --fp16 \
    --num-workers=0 \
    --model-overrides="{\"description\":\"${description}\",\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\",\"ans2label_file\":\"${ans2label_file}\"}"
