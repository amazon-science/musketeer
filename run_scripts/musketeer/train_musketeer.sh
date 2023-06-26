#!/usr/bin/env

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=7273
export PYTHONPATH=../../fairseq/:$PYTHONPATH

log_dir=./joint_logs/eq/six/no-det-base-tep-test
save_dir=./joint_checkpoints/eq/six/no-det-base-tep-test

mkdir -p $log_dir $save_dir

bpe_dir=../../utils/BPE
user_dir=../../ofa_module

cap_data_dir=$DATADIR/caption_data
cap_data=${cap_data_dir}/caption_stage1_train.tsv,${cap_data_dir}/caption_val.tsv

restore_file="../../checkpoints/ofa_base.pt"

cap_selected_cols=0,4,2

sn_data_dir=$DATADIR/snli_ve_data
sn_data=${sn_data_dir}/snli_ve_train.tsv,${sn_data_dir}/snli_ve_dev.tsv
sn_selected_cols=0,2,3,4,5

ref_data_dir=$DATADIR/refcoco_data
ref_data=${ref_data_dir}/refcoco_train.tsv,${ref_data_dir}/refcoco_val.tsv
ref_selected_cols=0,4,2,3

vqa_data_dir=$DATADIR/vqa_data
vqa_data=${vqa_data_dir}/vqa_train.tsv,${vqa_data_dir}/vqa_val.tsv

# Note: If you have shuffled the data in advance, please uncomment the line below.
# data=${data_dir}/vqa_train_1.tsv,${data_dir}/vqa_train_2.tsv,${data_dir}/vqa_train_3.tsv,${data_dir}/vqa_train_4.tsv,${data_dir}/vqa_train_5.tsv,${data_dir}/vqa_train_6.tsv,${data_dir}/vqa_train_7.tsv,${data_dir}/vqa_train_8.tsv,${data_dir}/vqa_train_9.tsv,${data_dir}/vqa_train_10.tsv,${data_dir}/vqa_val.tsv
vqa_ans2label_file=${vqa_data_dir}/trainval_ans2label.pkl
vqa_selected_cols=0,5,2,3,4

det_data=$DATADIR/coco/coco_detection_train.tsv
det_selected_cols=0,1,2

img_data_dir=$DATADIR/imagenet_1k_data
img_data=${img_data_dir}/imagenet_1k_train_80k.tsv,${img_data_dir}/imagenet_1k_val_subset.tsv
img_ans2label_file=${img_data_dir}/class2label_new.pkl
img_selected_cols=0,2

nlg_data_dir=$DATADIR/gigaword
nlg_data=${nlg_data_dir}/gigaword_train.tsv,${nlg_data_dir}/gigaword_dev.tsv
nlg_selected_cols=0,1

task=musketeer
arch=ofa_base
criterion=adjust_label_smoothed_cross_entropy
label_smoothing=0.1
lr=3e-5
max_epoch=300
warmup_ratio=0.06
batch_size=2
update_freq=16
resnet_drop_path_rate=0.0
encoder_drop_path_rate=0.1
decoder_drop_path_rate=0.1
dropout=0.1
attention_dropout=0.0
max_src_length=512
max_tgt_length=500
num_bins=1000
eq_sampling=7000 #for each task sampling, can be override by subset_sampling
subset_sampling=vg #

sample_patch_num=196
max_image_size=1024
code_image_size=320

noise_ratio=0.2
warmup_updates=1000

drop_worst_ratio=0.2
prompt_type="prev_output"

export NCCL_HOME=~/anaconda3/pkgs/nccl-2.14.3.1-h0800d71_0
#export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ens32 # find port name in ifconfig, not lo
export NCCL_IB_DISABLE=1

for max_epoch in {10,}; do
  echo "max_epoch "${max_epoch}
  for lr in {5e-5,1e-4,5e-4}; do
    echo "lr "${lr}
    for patch_image_size in {480,}; do
      echo "patch_image_size "${patch_image_size}
      log_file=${log_dir}/${max_epoch}"_"${lr}"_"${patch_image_size}".log"
      save_path=${save_dir}/${max_epoch}"_"${lr}"_"${patch_image_size}
      mkdir -p $save_path
      python3 -m torch.distributed.launch --master_port=${MASTER_PORT} ../../train.py \
        $sn_data \
        --img-data=${img_data} \
        --sn-data=${sn_data} \
        --ref-data=${ref_data} \
        --cap-data=${cap_data} \
        --vqa-data=${vqa_data} \
        --nlg-data=${nlg_data} \
        --detection-data=${det_data} \
        --img-selected-cols=$img_selected_cols \
        --nlg-selected-cols=$nlg_selected_cols \
        --detection-selected-cols=${det_selected_cols} \
        --vqa-selected-cols=${vqa_selected_cols} \
        --vqa-ans2label-file=${vqa_ans2label_file} \
        --ans2label-file=${img_ans2label_file} \
        --sn-selected-cols=${sn_selected_cols} \
        --ref-selected-cols=${ref_selected_cols} \
        --cap-selected-cols=${cap_selected_cols} \
        --bpe-dir=${bpe_dir} \
        --user-dir=${user_dir} \
        --restore-file=${restore_file} \
        --reset-optimizer --reset-dataloader --reset-meters \
        --save-dir=${save_path} \
        --task=${task} \
        --arch=${arch} \
        --criterion=${criterion} \
        --label-smoothing=${label_smoothing} \
        --batch-size=${batch_size} \
        --update-freq=${update_freq} \
        --encoder-normalize-before \
        --decoder-normalize-before \
        --share-decoder-input-output-embed \
        --share-all-embeddings \
        --layernorm-embedding \
        --patch-layernorm-embedding \
        --code-layernorm-embedding \
        --resnet-drop-path-rate=${resnet_drop_path_rate} \
        --encoder-drop-path-rate=${encoder_drop_path_rate} \
        --decoder-drop-path-rate=${decoder_drop_path_rate} \
        --dropout=${dropout} \
        --attention-dropout=${attention_dropout} \
        --weight-decay=0.01 --optimizer=adam --adam-betas="(0.9,0.999)" --adam-eps=1e-08 --clip-norm=0.1 \
        --lr-scheduler=polynomial_decay --lr=${lr} \
        --max-epoch=${max_epoch} --warmup-ratio=${warmup_ratio} \
        --warmup-updates=${warmup_updates} \
        --log-format=simple --log-interval=10 \
        --fixed-validation-seed=7 \
        --no-epoch-checkpoints --keep-best-checkpoints=1 \
        --save-interval=2 --validate-interval=1 \
        --save-interval-updates=1000 --validate-interval-updates=500 \
        --eval-acc \
        --eval-args='{"beam":5,"min_len":4,"max_len_a":0,"max_len_b":4}' \
        --best-checkpoint-metric=score --maximize-best-checkpoint-metric \
        --max-src-length=${max_src_length} \
        --max-tgt-length=${max_tgt_length} \
        --find-unused-parameters \
        --add-type-embedding \
        --scale-attn \
        --scale-fc \
        --scale-heads \
        --use-rdrop \
        --disable-entangle \
        --num-bins=${num_bins} \
        --code-image-size=${code_image_size} \
        --sample-patch-num=${sample_patch_num} \
        --max-image-size=${max_image_size} \
        --patch-image-size=${patch_image_size} \
        --detection-patch-image-size=1024 \
        --drop-worst-ratio=${drop_worst_ratio} \
        --drop-worst-after=6000 \
        --prompt-type=${prompt_type} \
        --vqa-prompt-type=${prompt_type} \
        --add-caption \
        --fp16-scale-window=512 \
        --add-object \
        --description=tep \
        --disable-validation \
        --ddp-backend=no_c10d \
        --eq-sampling=${eq_sampling} \
        --subset-sampling=${subset_sampling} \
        --fp16 \
        --noise-ratio=${noise_ratio} \
        --num-workers=0 | tee -a ${log_file}
    done
  done
done
