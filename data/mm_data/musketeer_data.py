# Illustration Title: Implement Task-Joint Training for OFA model
# Copyright (c) 2022 Zhaoyang Zhang
# Licensed under the Apache License, Version 2.0.
# found in the LICENSE file in the root directory.

import logging
import warnings

from PIL import Image, ImageFile
from data import data_utils
from data.ofa_dataset import OFADataset
from data.mm_data.refcoco_dataset import RefcocoDataset
from data.mm_data.snli_ve_dataset import SnliVeDataset
from data.mm_data.caption_dataset import CaptionDataset
from data.mm_data.vqa_gen_dataset import VqaGenDataset
from data.cv_data.detection_dataset import DetectionDataset
from data.cv_data.image_classify_dataset import ImageClassifyDataset
from data.nlg_data.summary_dataset import SummaryDataset
from data.mm_data.image_gen_dataset import ImageGenDataset
from data.nlu_data.cola_dataset import COLADataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


class MusketeerDataset(OFADataset):

    def __init__(
            self,
            cfg,
            split,
            dataset,
            bpe,
            src_dict,
            tgt_dict=None,
            max_src_length=128,
            max_tgt_length=30,
            seed=7,
            num_bins=1000,
            patch_image_size=384,
            cap_dataset=None,
            ref_dataset=None,
            vqa_dataset=None,
            vqa_trie=None,
            sn_trie=None,
            img_trie=None,
            cola_trie=None,
            det_dataset=None,
            img_dataset=None,
            nlg_dataset=None,
            cola_dataset=None,
            img_gen_dataset=None,
            eq_sampling=0,
            subset_sampling=None

    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)
        self.cfg = cfg
        self.seed = seed
        self.subsample = 1 if split == 'train' else 1

        self.sn_dataset = SnliVeDataset(split, dataset, bpe,
                                        src_dict,
                                        tgt_dict, max_src_length, max_tgt_length, patch_image_size,
                                        add_caption=self.cfg.add_caption,
                                        constraint_trie=sn_trie,
                                        imagenet_default_mean_and_std=self.cfg.imagenet_default_mean_and_std,
                                        prompt_type=self.cfg.prompt_type,
                                        description=self.cfg.description
                                        )

        self.cap_dataset = CaptionDataset(split, cap_dataset, bpe,
                                          src_dict,
                                          tgt_dict, max_src_length, max_tgt_length, patch_image_size,
                                          imagenet_default_mean_and_std=self.cfg.imagenet_default_mean_and_std,
                                          scst=getattr(self.cfg, 'scst', False),
                                          description=self.cfg.description
                                          )

        self.ref_dataset = RefcocoDataset(split, ref_dataset, bpe,
                                          src_dict,
                                          tgt_dict, max_src_length, max_tgt_length, patch_image_size, num_bins=num_bins,
                                          imagenet_default_mean_and_std=cfg.imagenet_default_mean_and_std,
                                          max_image_size=cfg.max_image_size,
                                          description=self.cfg.description
                                          )

        self.vqa_dataset = VqaGenDataset(split,
                                         vqa_dataset,
                                         bpe,
                                         src_dict,
                                         tgt_dict,
                                         max_src_length=cfg.max_src_length,
                                         max_object_length=cfg.max_object_length,
                                         max_tgt_length=cfg.max_tgt_length,
                                         patch_image_size=cfg.patch_image_size,
                                         add_object=cfg.add_object,
                                         constraint_trie=vqa_trie,
                                         imagenet_default_mean_and_std=cfg.imagenet_default_mean_and_std,
                                         prompt_type=cfg.vqa_prompt_type,
                                         description=self.cfg.description)

        self.det_dataset = DetectionDataset(
            split,
            det_dataset,
            bpe,
            src_dict,
            tgt_dict,
            patch_image_size=self.cfg.detection_patch_image_size,
            max_src_length=self.cfg.max_src_length,
            max_tgt_length=self.cfg.max_tgt_length,
            seed=self.cfg.pretrain_seed,
            code_dict_size=self.cfg.code_dict_size,
            num_bins=self.cfg.num_bins,
            code_image_size=self.cfg.code_image_size,
            detection_dataset=det_dataset,
            max_image_size=self.cfg.max_image_size,
            mask_ratio=self.cfg.mask_ratio,
            random_ratio=self.cfg.random_ratio,
            keep_ratio=self.cfg.keep_ratio,
            mask_length=self.cfg.mask_length,
            poisson_lambda=self.cfg.poisson_lambda,
            replace_length=self.cfg.replace_length,
            description=self.cfg.description)

        self.img_dataset = ImageClassifyDataset(
            split,
            img_dataset,
            self.bpe,
            self.src_dict,
            self.tgt_dict,
            max_src_length=self.cfg.max_src_length,
            max_tgt_length=self.cfg.max_tgt_length,
            patch_image_size=self.cfg.patch_image_size,
            constraint_trie=img_trie,
            imagenet_default_mean_and_std=self.cfg.imagenet_default_mean_and_std,
            description=self.cfg.description
        )

        self.nlg_dataset = SummaryDataset(
            split,
            nlg_dataset,
            self.bpe,
            self.src_dict,
            self.tgt_dict,
            code_dict_size=self.cfg.code_dict_size,
            num_bins=self.cfg.num_bins,
            max_src_length=self.cfg.max_src_length,
            max_tgt_length=self.cfg.max_tgt_length,
            noise_ratio=self.cfg.noise_ratio,
            description=self.cfg.description
        )

        self.img_gen_dataset = ImageGenDataset(
            split,
            img_gen_dataset,
            self.bpe,
            self.src_dict,
            self.tgt_dict,
            max_src_length=self.cfg.max_src_length,
            code_dict_size=self.cfg.code_dict_size,
            code_image_size=self.cfg.code_image_size,
            description=self.cfg.description
        )

        self.cola_dataset = COLADataset(
            split,
            cola_dataset,
            self.bpe,
            self.src_dict,
            self.tgt_dict,
            max_src_length=self.cfg.max_src_length,
            max_tgt_length=self.cfg.max_tgt_length,
            constraint_trie=cola_trie,
            prompt_type=self.cfg.cola_prompt_type,
            description=self.cfg.description
        )
        print(img_gen_dataset, cola_dataset)

        #### Equal Sampling
        if subset_sampling == 'vg':
            sample_size = len(self.ref_dataset)
        elif subset_sampling == 'caption':
            sample_size = len(self.cap_dataset)
        else:
            sample_size = eq_sampling if eq_sampling  > 0 else int('inf')

        if len(self.sn_dataset) > 0:
            self.sn_dataset.dataset.row_count = min(len(self.sn_dataset), sample_size)

        if len(self.ref_dataset) > 0:
            self.ref_dataset.dataset.row_count = min(len(self.ref_dataset), sample_size)

        if len(self.cap_dataset) > 0:
            self.cap_dataset.dataset.row_count = min(len(self.cap_dataset), sample_size)

        if len(self.vqa_dataset) > 0:
            self.vqa_dataset.dataset.row_count = min(len(self.vqa_dataset), sample_size)

        if len(self.det_dataset) > 0:
            self.det_dataset.dataset.row_count = min(len(self.det_dataset), sample_size)

        if len(self.img_dataset) > 0:
            self.img_dataset.dataset.row_count = min(len(self.img_dataset), sample_size)

        if len(self.nlg_dataset) > 0:
            self.nlg_dataset.dataset.row_count = min(len(self.nlg_dataset), sample_size)

        if len(self.img_gen_dataset) > 0:
            self.img_gen_dataset.dataset.row_count = min(len(self.img_gen_dataset), sample_size)

        if len(self.cola_dataset) > 0:
            self.cola_dataset.dataset.row_count = min(len(self.cola_dataset), sample_size)

        print(len(self.sn_dataset), len(self.cap_dataset), len(ref_dataset), len(vqa_dataset), len(self.det_dataset),
              len(img_dataset), len(nlg_dataset), len(img_gen_dataset), len(cola_dataset))

        self.main_set = None
        self.main_len = 0
        self.datalist = [self.sn_dataset, self.cap_dataset, self.ref_dataset, self.vqa_dataset, self.det_dataset,
                         self.img_dataset, self.nlg_dataset, self.img_gen_dataset, self.cola_dataset]

        for curr_set in self.datalist:
            self.main_set = curr_set if len(curr_set) > self.main_len else self.main_set
            self.main_len = max(self.main_len, len(curr_set))

    def set_epoch(self, epoch, **unused):
        self.epoch = epoch

    def get_main_set(self):

        return self.main_set

    def __len__(self):

        return self.main_len

    def __getitem__(self, index):
        with data_utils.numpy_seed(self.seed, self.epoch):
            sn_samples = self.sn_dataset[index % (len(self.sn_dataset))] if len(self.sn_dataset) > 0 else None
            cap_samples = self.cap_dataset[index % (len(self.cap_dataset))] if len(self.cap_dataset) > 0 else None
            ref_samples = self.ref_dataset[index % (len(self.ref_dataset))] if len(self.ref_dataset) > 0 else None
            vqa_samples = self.vqa_dataset[index % (len(self.vqa_dataset))] if len(self.vqa_dataset) > 0 else None
            det_samples = self.det_dataset[index % (len(self.det_dataset))] if len(self.det_dataset) > 0 else None
            img_samples = self.img_dataset[index % (len(self.img_dataset))] if len(self.img_dataset) > 0 else None
            nlg_samples = self.nlg_dataset[index % len(self.nlg_dataset)] if len(self.nlg_dataset) > 0 else None
            img_gen_samples = self.img_gen_dataset[index % len(self.img_gen_dataset)] if len(
                self.img_gen_dataset) > 0 else None
            cola_samples = self.cola_dataset[index % len(self.cola_dataset)] if len(self.cola_dataset) > 0 else None

        return sn_samples, ref_samples, cap_samples, vqa_samples, det_samples, img_samples, nlg_samples, img_gen_samples, cola_samples

    def collater(self, samples, pad_to_length=None):

        # samples_coll = ([],) * len(self.datalist)
        # for sample_tuple in samples:
        #     for i, dataset in enumerate(self.datalist):
        #         if sample_tuple[i] is not None:
        #             samples_coll[i].append(sample_tuple[i])
        #
        # res_coll= ()
        # for i, sample_each_coll in enumerate(samples_coll):
        #     res_coll += (self.datalist[i].collater(sample_each_coll, pad_to_length=pad_to_length))
        #
        # return res_coll

        samples_v1 = []
        samples_v2 = []
        samples_v3 = []
        samples_v4 = []
        samples_v5 = []
        samples_v6 = []
        samples_v7 = []
        samples_v8 = []
        samples_v9 = []

        for sample_tuple in samples:
            if sample_tuple[0] is not None:
                samples_v1.append(sample_tuple[0])
            if sample_tuple[1] is not None:
                samples_v2.append(sample_tuple[1])
            if sample_tuple[2] is not None:
                samples_v3.append(sample_tuple[2])
            if sample_tuple[3] is not None:
                samples_v4.append(sample_tuple[3])
            if sample_tuple[4] is not None:
                samples_v5.append(sample_tuple[4])
            if sample_tuple[5] is not None:
                samples_v6.append(sample_tuple[5])
            if sample_tuple[6] is not None:
                samples_v7.append(sample_tuple[6])
            if sample_tuple[7] is not None:
                samples_v8.append(sample_tuple[7])
            if sample_tuple[8] is not None:
                samples_v9.append(sample_tuple[8])

        res_v1 = self.sn_dataset.collater(samples_v1, pad_to_length=pad_to_length) if samples_v1 != [] else None

        res_v2 = self.ref_dataset.collater(samples_v2, pad_to_length=pad_to_length) if samples_v2 != [] else None

        res_v3 = self.cap_dataset.collater(samples_v3, pad_to_length=pad_to_length) if samples_v3 != [] else None

        res_v4 = self.vqa_dataset.collater(samples_v4, pad_to_length=pad_to_length) if samples_v4 != [] else None

        res_v5 = self.det_dataset.collater(samples_v5, pad_to_length=pad_to_length) if samples_v5 != [] else None

        res_v6 = self.img_dataset.collater(samples_v6, pad_to_length=pad_to_length) if samples_v6 != [] else None

        res_v7 = self.nlg_dataset.collater(samples_v7, pad_to_length=pad_to_length) if samples_v7 != [] else None

        res_v8 = self.img_gen_dataset.collater(samples_v8, pad_to_length=pad_to_length) if samples_v8 != [] else None

        res_v9 = self.cola_dataset.collater(samples_v9, pad_to_length=pad_to_length) if samples_v9 != [] else None

        return res_v1, res_v2, res_v3, res_v4, res_v5, res_v6, res_v7, res_v8, res_v9
