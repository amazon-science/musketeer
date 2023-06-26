#!/usr/bin/env bash
echo "Evaluate model: $1"

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=8081

user_dir=../../ofa_module
bpe_dir=../../utils/BPE

# val or test
split=$2

data=$DATADIR/vqa_data/vqa_${split}.tsv
ans2label_file=$DATADIR/vqa_data/trainval_ans2label.pkl
path=$1
result_path=../../results/vqa_${split}_allcand
selected_cols=0,5,2,3,4

python3 -m torch.distributed.launch --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=vqa_gen \
    --batch-size=8 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --fp16 \
    --ema-eval \
    --num-workers=0 \
    --model-overrides="{\"description\":\"${description}\",\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\",\"ans2label_file\":\"${ans2label_file}\"}"