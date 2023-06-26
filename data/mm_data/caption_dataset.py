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
import string

import numpy as np
import torch
import base64
from torchvision import transforms

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


class CaptionDataset(OFADataset):
    def __init__(
            self,
            split,
            dataset,
            bpe,
            src_dict,
            tgt_dict=None,
            max_src_length=128,
            max_tgt_length=30,
            patch_image_size=224,
            imagenet_default_mean_and_std=False,
            scst=False,
            get_type=False,
            description='base'

    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.patch_image_size = patch_image_size
        self.scst = scst
        self.get_type = get_type
        self.transtab = str.maketrans({key: None for key in string.punctuation})
        self.description = description
        if imagenet_default_mean_and_std:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((patch_image_size, patch_image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        if type(bpe).__name__ == 'GPT2BPE':
            # base

            self.prompt = " what does the image describe?"
            # rich
            if self.description == 'description_gpt':
                self.prompt = "Task: Image caption. Image captioning is a task in the field of computer vision and natural language processing where a model is trained to generate a natural language description of an image. This involves taking an image as input and outputting a sentence or paragraph that accurately describes the contents of the image." \
                              "To perform this task, a model typically needs to be able to understand the contents of an image, as well as the relationships between objects within the image and the actions that can be performed on those objects. This can be done using techniques such as object detection, scene graph generation, and natural language processing." \
                              "Image captioning has many potential applications, such as providing alternative text descriptions for images for visually impaired users, automatically generating captions for social media posts, or assisting in the development of more advanced AI systems that can reason about the world from visual input." \
                              "Dataset: COCO. COCO (Common Objects in Context) is a dataset used in the field of computer vision for the task of image captioning. The dataset consists of a large collection of images, each of which has been annotated with a series of captions that describe the contents of the image." \
                              "The goal of the COCO dataset is to provide a large and diverse set of examples for training and evaluating models for the task of image captioning. The dataset includes a total of 123,000 images and more than 5 million captions, covering a wide range of visual concepts and relationship types." \
                              "Models trained on the COCO dataset can be used for a variety of natural language processing and computer vision tasks, such as visual question answering and image retrieval. The dataset has been widely used in research and has contributed to significant advances in the field of image captioning." \
                              "Prompt: what does the image describe?"

            elif self.description == 'wiki':
                self.prompt = 'Image Captioning is the task of describing the content of an image in words. This task lies at the intersection of computer vision and natural language processing. ' \
                              'Prompt: what does the image describe?'



            elif self.description == 'wiki-tep':
                self.prompt = 'Image Captioning is the task of describing the content of an image in words. This task lies at the intersection of computer vision and natural language processing. ' \
                              'Dataset Description: Dataset Description: RIn addition to object detection, the COCO dataset also includes annotations for image captioning. Image captioning involves generating a natural language description of the objects and scenes depicted in an image.' \
                              'To annotate a dataset for image captioning, annotators must assign a series of text descriptions to each image in the dataset. These descriptions should capture the key objects and scene elements present in the image, as well as their relationships and interactions.' \
                              'Input format: A Task Prompt  and an Image ' \
                              'Output format: Text describe this image ' \
                              'Output description: Text that describe the input image' \
                              'Prompt: what does the image describe?'


            elif self.description == 'tep':
                self.prompt = 'Dataset Description: Dataset Description: RIn addition to object detection, the COCO dataset also includes annotations for image captioning. Image captioning involves generating a natural language description of the objects and scenes depicted in an image.' \
                              'To annotate a dataset for image captioning, annotators must assign a series of text descriptions to each image in the dataset. These descriptions should capture the key objects and scene elements present in the image, as well as their relationships and interactions.' \
                              'Input format: A Task Prompt  and an Image ' \
                              'Output format: Text describe this image ' \
                              'Output description: Text that describe the input image' \
                              'Prompt: what does the image describe?'





            elif self.description == 'onehot':
                self.prompt = "0001000"
                # print('Ohot Caption')

            elif self.description == 'base':
                self.prompt = " what does the image describe?"


        elif type(bpe).__name__ == 'BertBPE':
            self.prompt = "图片描述了什么内容?"

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
        }
        if self.get_type:
            batch["net_input"]["task_name"] = 'image caption'
        return batch

    def __getitem__(self, index):
        uniq_id, image, caption = self.dataset[index]

        image = Image.open(BytesIO(base64.urlsafe_b64decode(image)))
        patch_image = self.patch_resize_transform(image)
        patch_mask = torch.tensor([True])

        if self.split == 'train' and not self.scst:
            caption = caption.translate(self.transtab).strip()
            caption_token_list = caption.strip().split()
            tgt_caption = ' '.join(caption_token_list[:self.max_tgt_length])
        else:
            caption = ' '.join(caption.strip().split())
            caption_list = [cap.translate(self.transtab).strip() for cap in caption.strip().split('&&')]
            tgt_caption = '&&'.join(caption_list)
        src_item = self.encode_text(self.prompt)
        tgt_item = self.encode_text(" {}".format(tgt_caption))

        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, tgt_item])

        example = {
            "id": uniq_id,
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item
        }
        return example

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch containing the data of the task
        """
        return self.collate(samples, pad_idx=self.pad, eos_idx=self.eos)
