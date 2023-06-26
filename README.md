# Introduction
This codebase is an official implementation for [Musketeer](https://arxiv.org/abs/2305.07019),
which aims at building a sequence-to-sequence vision-language model whose parameters are jointly trained 
on all tasks (all for one) and fully shared among multiple tasks (one for all).


# Requirements
* python 3.7.4
* pytorch 1.8.1
* torchvision 0.9.1
* JAVA 1.8 (for COCO evaluation)


# Installation
This implementation based on [OFA](https://github.com/OFA-Sys/OFA) and [fairseq](https://github.com/facebookresearch/fairseq).

```bash
pip install -r requirements.txt
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```
# Datasets and Pretrained Checkpoints
Please prepare data and OFA-pretrained checkpoints according to [datasets.md](datasets.md) and [checkpoints.md](checkpoints.md).

After download and unzip these datasets in your data directory, make sure that the directory structure is arranged as follows,

    ├── your_data_directory
    │   ├── caption_data
    │   ├── snli_ve_data
    │   ├── refcoco_data
    │   ├── vqa_data
    │   ├── coco
    │   ├── imagenet_1k_data
    │   ├── gigaword




then run
```bash
export DATADIR=your_data_directory
```

For preparing pretrained checkpoints, please run
```bash
mkdir checkpoints
wget https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/ofa_base.pt
wget https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/ofa_large.pt
mv ofa_base.pt ofa_large.pt checkpoints
```

# Training
To train Musketeer with Task Explanation Prompt (TEP), please run
```bash
cd run_scripts/musketeer
bash train_musketeer.sh 
```
To train Musketeer with Base Prompt (baseP), please run
```bash
cd run_scripts/musketeer
bash train_musketeer_baseP.sh 
```

# Evaluation on Visual Grounding

```bash
cd run_scripts/vg
bash evaluate_refcoco_base.sh your_checkpoint_file
```
replace `your_checkpoint_file` with your trained model file dir.

# Related Codebase
We thanks following (but not limited to) researchers for sharing their code,
* [OFA](https://github.com/OFA-Sys/OFA)
* [Fairseq](https://github.com/pytorch/fairseq)

# Acknowledgement
This code was developed by [Zhaoyang Zhang](https://zzyfd.github.io/#/) while he was interning at the AWS Rekognition Team.


# Citation
If this code helps your research or project, please cite

```
@article{zhang2023musketeer,
  title={Musketeer (All for One, and One for All): A Generalist Vision-Language Model with Task Explanation Prompts},
  author={Zhang, Zhaoyang and Shen, Yantao and Shi, Kunyu and Cai, Zhaowei and Fang, Jun and Deng, Siqi and Yang, Hao and Modolo, Davide and Tu, Zhuowen and Soatto, Stefano},
  journal={arXiv preprint arXiv:2305.07019},
  year={2023}
}
```


# Contact Info
If you have any question, feel free to contact [Zhaoyang Zhang](https://zzyfd.github.io/#/) or his mentor at AWS, [Yantao Shen](https://yantaoshen.github.io/)

```
Zhaoyang Zhang: zhaoyangzhang@link.cuhk.edu.hk
Yantao Shen: yantaos@amazon.com
```







