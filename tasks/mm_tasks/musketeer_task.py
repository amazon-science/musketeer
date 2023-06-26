# Illustration Title: Implement Task-Joint Training for OFA model
# Copyright (c) 2022 Zhaoyang Zhang
# Licensed under the Apache License, Version 2.0.
# found in the LICENSE file in the root directory.

from dataclasses import dataclass, field
import json
import logging
from typing import Optional
from argparse import Namespace

import torch
from fairseq import metrics, tasks
from fairseq.tasks import register_task
from fairseq.optim.amp_optimizer import AMPOptimizer
from fairseq.data import FairseqDataset, iterators
import math

from tasks.ofa_task import OFATask, OFAConfig
from data.mm_data.refcoco_dataset import RefcocoDataset
from data.file_dataset import FileDataset

from tasks.mm_tasks.snli_ve import SnliVeTask
from tasks.mm_tasks.refcoco import RefcocoTask
from tasks.mm_tasks.caption import CaptionTask
from tasks.mm_tasks.vqa_gen import VqaGenTask
from tasks.cv_tasks.detection_task import DetectionTask
from tasks.nlg_tasks.gigaword import GigawordTask
from tasks.cv_tasks.image_classify import ImageClassifyTask
from tasks.mm_tasks.image_gen import ImageGenTask
from tasks.nlu_tasks.cola import COLATask
from fairseq.dataclass import ChoiceEnum

from data.mm_data.musketeer_data import *

logger = logging.getLogger(__name__)

EVAL_CLIP_METHOD = ChoiceEnum(["ii_sim", "ti_sim"])


@dataclass
class MusketeerConfig(OFAConfig):
    eq_sampling: int = field(
        default=0, metadata={"help": "Samping size"}
    )
    subset_sampling: Optional[str] = field(
        default=None, metadata={"help": "Subset Samping size"}
    )
    max_image_size: int = field(
        default=512, metadata={"help": ""}
    )
    sn_data: Optional[str] = field(
        default=None,
        metadata={"help": "sn data"},
    )

    ref_data: Optional[str] = field(
        default=None,
        metadata={"help": "ref data"},
    )
    cap_data: Optional[str] = field(
        default=None,
        metadata={"help": "cap data"},
    )
    sn_selected_cols: Optional[str] = field(
        default=None,
        metadata={"help": "sn data selected cols"},
    )
    ref_selected_cols: Optional[str] = field(
        default=None,
        metadata={"help": "ref data selected cols"},
    )
    cap_selected_cols: Optional[str] = field(
        default=None,
        metadata={"help": "cap data selected cols"},
    )
    code_image_size: int = field(
        default=128, metadata={"help": "the resolution of the generated image in the image infilling task"}
    )

    pretrain_seed: int = field(
        default=7,
        metadata={"help": "pretrain seed"},
    )

    mask_ratio: float = field(
        default=0.3,
        metadata={"help": "fraction of words/subwords that will be masked"},
    )
    random_ratio: float = field(
        default=0.0,
        metadata={"help": "instead of using [MASK], use random token this often"},
    )
    keep_ratio: float = field(
        default=0.0,
        metadata={"help": "instead of using [MASK], keep original token this often"},
    )
    mask_length: str = field(
        default="span-poisson",
        metadata={"help": "mask length to choose ['subword', 'word', 'span-poisson']"},
    )
    poisson_lambda: float = field(
        default=3.0,
        metadata={"help": "randomly shuffle sentences for this proportion of inputs"},
    )
    replace_length: int = field(
        default=1,
        metadata={"help": "when masking N tokens, replace with 0, 1, or N tokens (use -1 for N)"},
    )

    eval_acc: bool = field(
        default=False, metadata={"help": "evaluation with accuracy"}
    )
    eval_args: Optional[str] = field(
        default='{}',
        metadata={
            "help": 'generation args, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string'
        },
    )
    eval_print_samples: bool = field(
        default=False, metadata={"help": "print sample generations during validation"}
    )

    scst: bool = field(
        default=False, metadata={"help": "Self-critical sequence training"}
    )

    scst_args: str = field(
        default='{}',
        metadata={
            "help": 'generation args for Self-critical sequence training, as JSON string'
        },
    )
    ans2label_dict: Optional[str] = field(
        default='{"no": 0, "yes":1, "maybe": 2}',
        metadata={"help": 'answer to label dict'},
    )

    add_caption: bool = field(
        default=False,
        metadata={"help": "add caption to encoder"},
    )
    valid_batch_size: int = field(
        default=20,
        metadata={"help": "valid batch size per step"},
    )
    prompt_type: Optional[str] = field(
        default=None,
        metadata={"help": "prompt_type"},
    )
    eval_bleu: bool = field(
        default=False, metadata={"help": "evaluation with BLEU scores"}
    )
    eval_cider: bool = field(
        default=False, metadata={"help": "evaluation with CIDEr scores"}
    )
    eval_cider_cached_tokens: Optional[str] = field(
        default=None,
        metadata={"help": "path to cached cPickle file used to calculate CIDEr scores"},
    )

    ## VQA config

    vqa_data: Optional[str] = field(
        default=None,
        metadata={"help": "vqa data"},
    )

    vqa_selected_cols: Optional[str] = field(
        default=None,
        metadata={"help": "vqa data selected cols"},
    )

    max_object_length: int = field(
        default=30, metadata={"help": "the maximum object sequence length"}
    )

    ans2label_file: Optional[str] = field(
        default=None,
        metadata={"help": "path to load ans2label file"},
    )

    vqa_ans2label_file: Optional[str] = field(
        default=None,
        metadata={"help": "path to load ans2label file"},
    )
    add_object: bool = field(
        default=False,
        metadata={"help": "add object to encoder"},
    )
    uses_ema: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to use ema"},
    )
    val_inference_type: Optional[str] = field(
        default='allcand',
        metadata={"help": "inference type in validation (allcand or beamsearch), default to allcand"},
    )

    vqa_ans2label_dict: Optional[str] = field(
        default='{"no": 0, "yes":1}',
        metadata={"help": 'answer to label dict'},
    )

    vqa_eval_args: Optional[str] = field(
        default='{"beam":5,"unnormalized":true,"temperature":1.0}',
        metadata={
            "help": 'generation args as JSON string for inference, only activated when --val-inference-type=beamsearch'
        }, )
    vqa_prompt_type: Optional[str] = field(
        default=None,
        metadata={"help": "prompt_type"},
    )

    ### Det config
    detection_data: Optional[str] = field(
        default=None,
        metadata={"help": "detection data"},
    )

    detection_selected_cols: Optional[str] = field(
        default=None,
        metadata={"help": "detection data selected cols"},
    )
    detection_patch_image_size: int = field(
        default=1024, metadata={"help": "image size for detection data"}
    )
    add_blank_id_to_dict: Optional[bool] = field(
        default=False,
        metadata={"help": "Add blank prediction to the dictionary for NAT models"},
    )

    ### Description config
    description: Optional[str] = field(
        default='base',
        metadata={"help": "description type"},
    )

    ### Imgnet config
    img_ans2label_dict: Optional[str] = field(
        default='{"no": 0, "yes":1}',
        metadata={"help": 'answer to label dict'},
    )
    img_ans2label_file: Optional[str] = field(
        default=None,
        metadata={"help": "path to load ans2label file"},
    )
    img_data: Optional[str] = field(
        default=None,
        metadata={"help": "image classification data"},
    )
    img_selected_cols: Optional[str] = field(
        default=None,
        metadata={"help": "image classify data selected cols"},
    )

    ### gigaword config
    nlg_data: Optional[str] = field(
        default=None,
        metadata={"help": "image classification data"},
    )
    eval_rouge: bool = field(
        default=False, metadata={"help": "evaluation with rouge scores"}
    )

    eval_detok: str = field(
        default="space",
        metadata={
            "help": "detokenize before computing BLEU or CIDEr (e.g., 'moses'); "
                    "required if using --eval-bleu or --eval-cider; "
                    "use 'space' to disable detokenization; see fairseq.data.encoders for other options"
        },
    )
    eval_detok_args: Optional[str] = field(
        default="{}",
        metadata={"help": "args for building the tokenizer, if needed, as JSON string"},
    )

    noise_ratio: float = field(
        default=0.0, metadata={"help": "noise ratio for prev output"}
    )
    nlg_selected_cols: Optional[str] = field(
        default=None,
        metadata={"help": "gigaword  data selected cols"},
    )

    ### image gen configs
    img_gen_data: Optional[str] = field(
        default=None,
        metadata={"help": "image gen data"},
    )

    img_gen_selected_cols: Optional[str] = field(
        default=None,
        metadata={"help": "img gen  data selected cols"},
    )

    sampling_times: int = field(
        default=1, metadata={"help": "sample times"}
    )

    # options for reporting CLIP score during validation
    eval_clip_method: EVAL_CLIP_METHOD = field(
        default='ti_sim',
        metadata={
            "help": "evaluation with CLIP scores. ii_sim means Similarity between generated Images and ref Images, ti_sim means Similarity between generated Images and input Text"}
    )

    vqgan_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "path of vqgan model"}
    )
    vqgan_config_path: Optional[str] = field(
        default=None,
        metadata={"help": "path of vqgan config"}
    )
    clip_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "clip model path"}
    )
    gen_images_path: str = field(
        default='', metadata={"help": "where to store generated images during evalution. Don't dump images if None. "}
    )

    ###  cola configs
    cola_data: Optional[str] = field(
        default=None,
        metadata={"help": "cola  data"},
    )
    cola_selected_cols: Optional[str] = field(
        default=None,
        metadata={"help": "cola  data selected cols"},
    )
    cola_ans2label_dict: Optional[str] = field(
        default='{"no": 0, "yes": 1}',
        metadata={"help": 'answer to label dict'},
    )
    cola_prompt_type: ChoiceEnum(["none", "src", "prev_output"]) = field(
        default="none",
        metadata={"help": "decoder prompt"},
    )


@register_task("musketeer", dataclass=MusketeerConfig)
class Musketeers(OFATask):
    def __init__(self, cfg: MusketeerConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        self.sn_task = SnliVeTask(cfg, src_dict, tgt_dict, data=cfg.sn_data)
        self.cap_task = CaptionTask(cfg, src_dict, tgt_dict, data=cfg.cap_data)
        self.ref_task = RefcocoTask(cfg, src_dict, tgt_dict, data=cfg.ref_data)
        self.vqa_task = VqaGenTask(cfg, src_dict, tgt_dict, data=cfg.vqa_data)
        self.det_task = DetectionTask(cfg, src_dict, tgt_dict, data=cfg.detection_data)
        self.img_task = ImageClassifyTask(cfg, src_dict, tgt_dict, data=cfg.img_data)
        self.nlg_task = GigawordTask(cfg, src_dict, tgt_dict, data=cfg.nlg_data)
        self.img_gen_task = ImageGenTask(cfg, src_dict, tgt_dict, data=cfg.img_gen_data)
        self.cola_task = COLATask(cfg, src_dict, tgt_dict, data=cfg.cola_data)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        if self.cfg.sn_data is not None:
            sn_paths = self.cfg.sn_data.split(',')
            assert len(sn_paths) > 0
            sn_paths = sn_paths[(epoch - 1) % (len(sn_paths) - 1)] if split == 'train' else sn_paths[-1]
            print(self.cfg.sn_selected_cols)
            sn_dataset = FileDataset(sn_paths, self.cfg.sn_selected_cols)
        else:
            sn_dataset = []

        if self.cfg.cap_data is not None:
            cap_paths = self.cfg.cap_data.split(',')
            assert len(cap_paths) > 0
            cap_paths = cap_paths[(epoch - 1) % (len(cap_paths) - 1)] if split == 'train' else cap_paths[-1]
            print(self.cfg.cap_selected_cols)
            cap_dataset = FileDataset(cap_paths,
                                      self.cfg.cap_selected_cols) if split == 'train' else []  ### Avoid loading redundant dataset if not validation
        else:
            cap_dataset = []

        if self.cfg.ref_data is not None:

            ref_paths = self.cfg.ref_data.split(',')
            assert len(ref_paths) > 0
            ref_paths = ref_paths[(epoch - 1) % (len(ref_paths) - 1)] if split == 'train' else ref_paths[-1]
            print(self.cfg.ref_selected_cols)
            ref_dataset = FileDataset(ref_paths,
                                      self.cfg.ref_selected_cols) if split == 'train' else []  ### Avoid loading redundant dataset if not validation
        else:
            ref_dataset = []

        if self.cfg.vqa_data is not None:

            vqa_paths = self.cfg.vqa_data.split(',')
            assert len(vqa_paths) > 0
            vqa_paths = vqa_paths[(epoch - 1) % (len(vqa_paths) - 1)] if split == 'train' else vqa_paths[-1]

            print(self.cfg.vqa_selected_cols)
            vqa_dataset = FileDataset(vqa_paths,
                                      self.cfg.vqa_selected_cols) if split == 'train' else []  ### Avoid loading redundant dataset if not validation
        else:
            vqa_dataset = []

        if self.cfg.detection_data is not None:

            det_paths = self.cfg.detection_data.split(',')
            assert len(det_paths) > 0
            det_paths = det_paths[0] if split == 'train' else det_paths[-1]

            print(self.cfg.detection_selected_cols)
            det_dataset = FileDataset(det_paths,
                                      self.cfg.detection_selected_cols) if split == 'train' else []  ### Avoid loading redundant dataset if not validation
        else:
            det_dataset = []

        if self.cfg.img_data is not None:

            img_paths = self.cfg.img_data.split(',')
            assert len(img_paths) > 0
            img_paths = img_paths[0] if split == 'train' else img_paths[-1]

            print(self.cfg.img_selected_cols)
            img_dataset = FileDataset(img_paths,
                                      self.cfg.img_selected_cols) if split == 'train' else []  ### Avoid loading redundant dataset if not validation
        else:
            img_dataset = []

        if self.cfg.nlg_data is not None:

            nlg_paths = self.cfg.nlg_data.split(',')
            assert len(nlg_paths) > 0
            nlg_paths = nlg_paths[0] if split == 'train' else nlg_paths[-1]

            print(self.cfg.nlg_selected_cols)
            nlg_dataset = FileDataset(nlg_paths,
                                      self.cfg.nlg_selected_cols) if split == 'train' else []  ### Avoid loading redundant dataset if not validation
        else:
            nlg_dataset = []

        if self.cfg.img_gen_data is not None:

            img_gen_paths = self.cfg.img_gen_data.split(',')
            assert len(img_gen_paths) > 0
            img_gen_paths = img_gen_paths[0] if split == 'train' else img_gen_paths[-1]

            print(img_gen_paths, self.cfg.img_gen_selected_cols)
            img_gen_dataset = FileDataset(img_gen_paths,
                                          self.cfg.img_gen_selected_cols) if split == 'train' else []
            print(split, len(img_gen_dataset))  ### Avoid loading redundant dataset if not validation
        else:
            img_gen_dataset = []

        if self.cfg.cola_data is not None:

            cola_paths = self.cfg.cola_data.split(',')
            assert len(cola_paths) > 0
            cola_paths = cola_paths[0] if split == 'train' else cola_paths[-1]

            print(self.cfg.cola_selected_cols)
            cola_dataset = FileDataset(cola_paths,
                                       self.cfg.cola_selected_cols) if split == 'train' else []  ### Avoid loading redundant dataset if not validation
        else:
            cola_dataset = []

        self.datasets[split] = MusketeerDataset(
            self.cfg,
            split,
            sn_dataset,
            self.bpe,
            self.src_dict,
            tgt_dict=self.tgt_dict,
            max_src_length=self.cfg.max_src_length,
            max_tgt_length=self.cfg.max_tgt_length,
            patch_image_size=self.cfg.patch_image_size,
            num_bins=self.cfg.num_bins,
            cap_dataset=cap_dataset,
            ref_dataset=ref_dataset,
            vqa_dataset=vqa_dataset,
            vqa_trie=self.vqa_task.constraint_trie,
            sn_trie=self.sn_task.constraint_trie,
            img_trie=self.img_task.constraint_trie,
            cola_trie=self.cola_task.constraint_trie,
            det_dataset=det_dataset,
            img_dataset=img_dataset,
            nlg_dataset=nlg_dataset,
            img_gen_dataset=img_gen_dataset,
            cola_dataset=cola_dataset,
            eq_sampling=self.cfg.eq_sampling,
            subset_sampling=self.cfg.subset_sampling

        )
        self.time_avg = [0, 0]

    def build_model(self, cfg):
        model = super().build_model(cfg)
        self.sn_task.bpe = self.bpe
        self.cap_task.bpe = self.bpe
        self.ref_task.bpe = self.bpe
        self.vqa_task.bpe = self.bpe
        self.det_task.bpe = self.bpe
        self.img_task.bpe = self.bpe
        self.nlg_task.bpe = self.bpe
        self.img_gen_task.bpe = self.bpe
        self.cola_task.bpe = self.bpe

        model = self.sn_task.build_shared_model(cfg, model)
        model = self.cap_task.build_shared_model(cfg, model)
        model = self.ref_task.build_shared_model(cfg, model)
        model = self.vqa_task.build_shared_model(cfg, model)
        model = self.det_task.build_shared_model(cfg, model)
        model = self.img_task.build_shared_model(cfg, model)
        model = self.nlg_task.build_shared_model(cfg, model)
        model = self.cola_task.build_shared_model(cfg, model)

        if self.cfg.img_gen_data is not None:
            model = self.img_gen_task.build_shared_model(cfg, model)

        return model

    def train_step(
            self, sample, model, criterion, optimizer, update_num, ignore_grad=False, **extra_kwargs
    ):
        model.train()
        model.set_num_updates(update_num)

        sample = [s for s in sample if s is not None]

        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                loss, sample_size, logging_output = criterion(model, sample, update_num=update_num)
                end.record()

                torch.cuda.synchronize()
                self.time_avg[1] += 1
                self.time_avg[0] += start.elapsed_time(end)

        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)

        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):

        ### validate sn only
        sn_sample = sample[0]

        loss_sn = 0
        loss_ref = 0
        if sn_sample is not None:
            loss_sn, sample_size_sn, logging_output_sn = self.sn_task.valid_step(sn_sample, model, criterion)
            loss_sn /= sample_size_sn

        loss = loss_sn + loss_ref
        sample_size = 1

        return loss, sample_size, logging_output_sn

    def build_generator(
            self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None, prefix_allowed_tokens_fn=None,
    ):
        seq_generator = super().build_generator(models, args, seq_gen_cls, extra_gen_cls_kwargs,
                                                prefix_allowed_tokens_fn)
        seq_generator.constraint_trie = self.constraint_trie

        return seq_generator

    def get_batch_iterator(
            self,
            dataset,
            max_tokens=None,
            max_sentences=None,
            max_positions=None,
            ignore_invalid_inputs=False,
            required_batch_size_multiple=1,
            seed=1,
            num_shards=1,
            shard_id=0,
            num_workers=0,
            epoch=1,
            data_buffer_size=0,
            disable_iterator_cache=False,
    ):
        assert isinstance(dataset, FairseqDataset)

        setattr(dataset, 'dataset', dataset.get_main_set().dataset)
        dataset.set_epoch(epoch)

        batch_sampler = [
            [j for j in range(i, min(i + max_sentences, len(dataset)))]
            for i in range(0, len(dataset), max_sentences)
        ]
        total_row_count = dataset.dataset.get_total_row_count()
        num_batches = math.ceil(math.ceil(total_row_count / num_shards) / max_sentences)
        if len(batch_sampler) < num_batches:
            batch_sampler.append([1])

        # return a reusable, sharded iterator
        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=1,
            shard_id=0,
            num_workers=num_workers,
            epoch=epoch,
            buffer_size=data_buffer_size
        )

        return epoch_iter

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        def sum_logs(key):
            import torch
            result = sum(log.get(key, 0) for log in logging_outputs)
            if torch.is_tensor(result):
                result = result.cpu()
            return result

        def compute_score(meters):
            score = meters["_snli_score_sum"].sum / meters["_snli_cnt"].sum
            score = score if isinstance(score, float) else score.item()
            return round(score, 4)

        if sum_logs("_snli_cnt") > 0:
            metrics.log_scalar("_snli_score_sum", sum_logs("_snli_score_sum"))
            metrics.log_scalar("_snli_cnt", sum_logs("_snli_cnt"))
            metrics.log_derived("snli_score", compute_score)
            metrics.log_derived("score", compute_score)
