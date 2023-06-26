#!/usr/bin/env bash
echo "Evaluate model: $1"

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.


export MASTER_PORT=10233


export PYTHONPATH=../../fairseq/:$PYTHONPATH

user_dir=../../ofa_module
bpe_dir=../../utils/BPE

export NCCL_HOME=~/anaconda3/pkgs/nccl-2.14.3.1-h0800d71_0
#export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ens32
echo $NCCL_SOCKET_IFNAME
#export NCCL_IB_DISABLE=1

### Path to Data
data=$DATADIR/caption_data/caption_test.tsv


### Path to CKPT
path=$1


result_path=../../results/caption_coco_zero_rich_anno_seven-base-cap
mkdir ${result_path}
description=tep


selected_cols=1,4,2
split='test'

python3 -m torch.distributed.launch --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --bpe-dir=${bpe_dir} \
    --task=caption \
    --batch-size=16 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --beam=5 \
    --max-len-b=16 \
    --no-repeat-ngram-size=3 \
    --fp16 \
    --num-workers=0 \
    --ddp-backend=no_c10d \
    --selected-cols=${selected_cols} \
    --model-overrides="{\"description\":\"${description}\",\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"eval_cider\":False,\"selected_cols\":\"${selected_cols}\"}" | tee -a ${result_path}/log_val.txt

python coco_eval.py ${result_path}/test_predict.json $DATADIR/caption_data/test_caption_coco_format.json
