# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.
# -----------------------------------------------------------------------------
# Modifications made by Zhaoyang Zhang on 2022.12.25:
#
# - Add support for dataloader with rich task descriptions.
# -----------------------------------------------------------------------------
from io import BytesIO

import logging
import warnings

import numpy as np
import torch
import base64
import utils.transforms as T

from PIL import Image, ImageFile

from data import data_utils
from data.ofa_dataset import OFADataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class RefcocoDataset(OFADataset):
    def __init__(
            self,
            split,
            dataset,
            bpe,
            src_dict,
            tgt_dict=None,
            max_src_length=80,
            max_tgt_length=30,
            patch_image_size=512,
            imagenet_default_mean_and_std=False,
            num_bins=1000,
            max_image_size=512,
            get_type=False,
            description='base'
    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.patch_image_size = patch_image_size
        self.num_bins = num_bins
        self.get_type = get_type
        self.description = description

        if imagenet_default_mean_and_std:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        # for positioning
        self.positioning_transform = T.Compose([
            T.RandomResize([patch_image_size], max_size=patch_image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std, max_image_size=max_image_size)
        ])

        print(self.description + '!!!!')

        if type(bpe).__name__ == 'GPT2BPE':

            if self.description == 'description_gpt':

                self.prompt = 'Task: Referring expression comprehension. Referring expression comprehension is a task in natural language processing that involves determining the meaning of a phrase or sentence in context. This involves taking a piece of text as input and outputting a prediction of what the text is referring to. ' \
                              'For example, given the sentence "I saw the cat sitting on the couch," the model would need to understand that the phrase "the cat" is referring to a specific cat, rather than all cats in general.' \
                              'To perform this task, a model typically needs to have a deep understanding of language and be able to analyze the context in which a phrase or sentence is used. This can be done using techniques such as natural language processing, contextual analysis, and knowledge representation. ' \
                              'Referring expression comprehension is important for many natural language processing applications, such as machine translation, summarization, and question answering. It is also a key component of many conversational AI systems, as it allows the system to understand the meaning of phrases and sentences in the context of a conversation.' \
                              'Dataset: RefCOCO. RefCOCO (Referring Expression Comprehension in Context) is a dataset used in the field of natural language processing for the task of reference understanding. The dataset consists of images with associated text captions, where the captions contain referring expressions that describe specific objects in the images.' \
                              'The goal of the RefCOCO dataset is to provide a large and diverse set of examples for training and evaluating models for the task of reference understanding. The dataset includes a total of 50,000 images and 5.4 million referring expressions, each of which has been manually annotated to indicate the specific object that the expression is referring to.' \
                              'Prompt: which region does the text " {} " describe?'


            elif self.description == 'wiki':
                self.prompt = 'Visual Grounding (VG) aims to locate the most relevant object or region in an image, based on a natural language query. The query can be a phrase, a sentence, or even a multi-round dialogue.' \
                              'Prompt: which region does the text " {} " describe?'


            elif self.description == 'wiki-tep':
                self.prompt = 'Visual Grounding (VG) aims to locate the most relevant object or region in an image, based on a natural language query. The query can be a phrase, a sentence, or even a multi-round dialogue.' \
                              'Dataset Description: RefCOCO is a dataset for referring expressions in images, which is built on top of the COCO dataset. Referring expressions are natural language phrases that refer to specific objects or regions in an image. For example, a referring expression might be "the dog in the center of the picture" or "the red car on the right side of the image".' \
                              'Annotating a dataset like RefCOCO involves manually labeling the objects in each image with bounding boxes and class labels, as well as creating referring expressions that refer to specific objects or regions in the image. This is typically done by trained annotators who use specialized software tools to draw the bounding boxes and assign the class labels, as well as to generate the referring expressions.' \
                              'Input format: A Task Prompt, a Text describe the target region and a Image containing the target region' \
                              'Output format: x0 + y0 + x1 + y1' \
                              'Output description: horizonal coordinates of leftupper points of target region +  vertical coordinates of leftupper points of target region  + horizonal coordinates of rightlower points of target region +  vertical coordinates of rightlower points of target region ' \
                              'Prompt: which region does the text " {} " describe?'


            elif self.description == 'tep':
                self.prompt = 'Dataset Description: RefCOCO is a dataset for referring expressions in images, which is built on top of the COCO dataset. Referring expressions are natural language phrases that refer to specific objects or regions in an image. For example, a referring expression might be "the dog in the center of the picture" or "the red car on the right side of the image".' \
                              'Annotating a dataset like RefCOCO involves manually labeling the objects in each image with bounding boxes and class labels, as well as creating referring expressions that refer to specific objects or regions in the image. This is typically done by trained annotators who use specialized software tools to draw the bounding boxes and assign the class labels, as well as to generate the referring expressions.' \
                              'Input format: A Task Prompt, a Text describing the target region and a Image containing the target region' \
                              'Output format: x0 + y0 + x1 + y1' \
                              'Output description: horizonal coordinates of leftupper points of target region +  vertical coordinates of leftupper points of target region  + horizonal coordinates of rightlower points of target region +  vertical coordinates of rightlower points of target region ' \
                              'Prompt: which region does the text " {} " describe?'


            # elif self.description == 'tep-old':
            #     self.prompt = 'Dataset Description: RefCOCO is a dataset for referring expressions in images, which is built on top of the COCO dataset. Referring expressions are natural language phrases that refer to specific objects or regions in an image. For example, a referring expression might be "the dog in the center of the picture" or "the red car on the right side of the image".' \
            #                   'Annotating a dataset like RefCOCO involves manually labeling the objects in each image with bounding boxes and class labels, as well as creating referring expressions that refer to specific objects or regions in the image. This is typically done by trained annotators who use specialized software tools to draw the bounding boxes and assign the class labels, as well as to generate the referring expressions.' \
            #                   'Input format: A Task Prompt, a Text describe the target region and a Image containing the target region' \
            #                   'Output format: x0 + y0 + x1 + y1' \
            #                   'Output description: horizonal coordinates of leftupper points of target region +  vertical coordinates of leftupper points of target region  + horizonal coordinates of rightlower points of target region +  vertical coordinates of rightlower points of target region ' \
            #                   'Prompt: which region does the text " {} " describe?'



            elif self.description == 'onehot':
                self.prompt = '0000100 {}'

            elif self.description == 'base':
                self.prompt = 'which region does the text " {} " describe?'




        elif type(bpe).__name__ == 'BertBPE':
            self.prompt = '这段文字" {} "描述的是哪个区域？'

    def __getitem__(self, index):
        uniq_id, base64_str, text, region_coord = self.dataset[index]

        image = Image.open(BytesIO(base64.urlsafe_b64decode(base64_str))).convert("RGB")
        w, h = image.size
        boxes_target = {"boxes": [], "labels": [], "area": [], "size": torch.tensor([h, w])}
        x0, y0, x1, y1 = region_coord.strip().split(',')
        region = torch.tensor([float(x0), float(y0), float(x1), float(y1)])
        boxes_target["boxes"] = torch.tensor([[float(x0), float(y0), float(x1), float(y1)]])
        boxes_target["labels"] = np.array([0])
        boxes_target["area"] = torch.tensor([(float(x1) - float(x0)) * (float(y1) - float(y0))])

        patch_image, patch_boxes = self.positioning_transform(image, boxes_target)
        resize_h, resize_w = patch_boxes["size"][0], patch_boxes["size"][1]
        patch_mask = torch.tensor([True])
        quant_x0 = "<bin_{}>".format(int((patch_boxes["boxes"][0][0] * (self.num_bins - 1)).round()))
        quant_y0 = "<bin_{}>".format(int((patch_boxes["boxes"][0][1] * (self.num_bins - 1)).round()))
        quant_x1 = "<bin_{}>".format(int((patch_boxes["boxes"][0][2] * (self.num_bins - 1)).round()))
        quant_y1 = "<bin_{}>".format(int((patch_boxes["boxes"][0][3] * (self.num_bins - 1)).round()))
        region_coord = "{} {} {} {}".format(quant_x0, quant_y0, quant_x1, quant_y1)
        src_caption = self.pre_caption(text, self.max_src_length)

        src_item = self.encode_text(self.prompt.format(src_caption))

        tgt_item = self.encode_text(region_coord, use_bpe=False)

        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, tgt_item])

        example = {
            "id": uniq_id,
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item,
            "w_resize_ratio": resize_w / w,
            "h_resize_ratio": resize_h / h,
            "region_coord": region
        }
        return example

    def collate(self, samples, pad_idx, eos_idx):
        if len(samples) == 0:
            return {}

        def merge(key):
            return data_utils.collate_tokens(
                [s[key] for s in samples],
                pad_idx,
                eos_idx=eos_idx,
            )

        id = np.array([s["id"] for s in samples])
        src_tokens = merge("source")
        src_lengths = torch.LongTensor([s["source"].ne(pad_idx).long().sum() for s in samples])

        patch_images = torch.stack([sample['patch_image'] for sample in samples], dim=0)
        patch_masks = torch.cat([sample['patch_mask'] for sample in samples])

        w_resize_ratios = torch.stack([s["w_resize_ratio"] for s in samples], dim=0)
        h_resize_ratios = torch.stack([s["h_resize_ratio"] for s in samples], dim=0)
        region_coords = torch.stack([s['region_coord'] for s in samples], dim=0)

        prev_output_tokens = None
        target = None
        if samples[0].get("target", None) is not None:
            target = merge("target")
            tgt_lengths = torch.LongTensor([s["target"].ne(pad_idx).long().sum() for s in samples])
            ntokens = tgt_lengths.sum().item()

            if samples[0].get("prev_output_tokens", None) is not None:
                prev_output_tokens = merge("prev_output_tokens")
        else:
            ntokens = src_lengths.sum().item()

        batch = {
            "id": id,
            "nsentences": len(samples),
            "ntokens": ntokens,
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
                "patch_images": patch_images,
                "patch_masks": patch_masks,
                "prev_output_tokens": prev_output_tokens
            },
            "target": target,
            "w_resize_ratios": w_resize_ratios,
            "h_resize_ratios": h_resize_ratios,
            "region_coords": region_coords
        }
        if self.get_type:
            batch["net_input"]["task_name"] = 'visual grounding'
        return batch

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch containing the data of the task
        """
        return self.collate(samples, pad_idx=self.pad, eos_idx=self.eos)
