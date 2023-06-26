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


class SnliVeDataset(OFADataset):
    def __init__(
            self,
            split,
            dataset,
            bpe,
            src_dict,
            tgt_dict=None,
            max_src_length=80,
            max_tgt_length=30,
            patch_image_size=224,
            add_caption=False,
            constraint_trie=None,
            imagenet_default_mean_and_std=False,
            prompt_type="none",
            get_type=False,
            description='base'
    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.patch_image_size = patch_image_size
        self.get_type = get_type
        self.add_caption = add_caption
        self.constraint_trie = constraint_trie
        self.prompt_type = prompt_type
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

        ref_dict = None

        if samples[0].get("ref_dict", None) is not None:
            ref_dict = np.array([s['ref_dict'] for s in samples])

        constraint_masks = None

        if samples[0].get("constraint_mask", None) is not None:
            constraint_masks = merge("constraint_mask")

        decoder_prompts = None

        if samples[0].get("decoder_prompt", None) is not None:
            decoder_prompts = np.array([s['decoder_prompt'].tolist() for s in samples])

        prev_output_tokens = None
        target = None
        if samples[0].get("target", None) is not None:
            target = merge("target")
            tgt_lengths = torch.LongTensor(
                [s["target"].ne(pad_idx).long().sum() for s in samples]
            )
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
            "ref_dict": ref_dict,
            "constraint_masks": constraint_masks,
            "decoder_prompts": decoder_prompts,
            "target": target
        }
        if self.get_type:
            batch["net_input"]["task_name"] = 'visual entailment'
        return batch

    def __getitem__(self, index):
        # print(self.dataset[index])
        uniq_id, image, hypothesis, caption, label = self.dataset[index]

        if label == 'contradiction':
            label = 'no'
        elif label == 'entailment':
            label = 'yes'
        elif label == 'neutral':
            label = 'maybe'
        else:
            raise NotImplementedError

        image = Image.open(BytesIO(base64.urlsafe_b64decode(image)))
        patch_image = self.patch_resize_transform(image)
        patch_mask = torch.tensor([True])

        hypothesis = self.pre_caption(hypothesis, self.max_src_length)

        src_item = self.encode_text(' does the image describe " {} "?'.format(hypothesis))

        tgt_item = self.encode_text(" {}".format(label))
        ref_dict = {label: 1.0}

        if self.add_caption:
            caption = self.pre_caption(caption, self.max_src_length)
            if self.description == 'description_gpy':
                src_item = self.encode_text(
                    'Task: Visual Entailment. Visual entailment  is a task in the field of computer vision and natural language processing where a model is trained to determine whether one image can be derived from another based on associated text. This involves taking a pair of sentences (a premise and a hypothesis) and an image as input and outputting a binary prediction indicating whether the hypothesis logically follows from the premise and the visual evidence provided by the image.'
                    'To perform this task, a model typically needs to be able to understand the contents of an image, as well as the relationships between objects within the image and the actions that can be performed on those objects. It also needs to be able to analyze the relationship between the premise and the hypothesis and use that information to make a prediction. This can be done using techniques such as visual reasoning, scene graph generation, and natural language processing.'
                    'Visual entailment  has many potential applications, such as providing a more intuitive and flexible way for people to search for images, allowing a computer to automatically generate a caption for an image, or assisting in the development of more advanced AI systems that can reason about the world from visual input. '
                    'DataSet: SNLI-VE(Syntactic-Natural Language Inference with Visual Evidence). SNLI-VE (Syntactic-Natural Language Inference with Visual Evidence) is a dataset used in the field of natural language processing for the task of visual entailment. The dataset consists of pairs of sentences, where the first sentence is a premise and the second is a hypothesis, along with an associated image that provides visual evidence for the relationship between the two sentences.'
                    'The goal of the SNLI-VE dataset is to provide a large and diverse set of examples for training and evaluating models for the task of visual entailment. The dataset includes a total of 100,000 sentence pairs and images, each of which has been annotated with one of three labels: "Entailment," "Contradiction," or "Neutral."'
                    'Models trained on the SNLI-VE dataset can be used for a variety of natural language processing tasks, such as visual question answering and dialog systems. The dataset has been widely used in research and has contributed to significant advances in the field of visual entailment.'
                    ' Prompt: can image and text1 " {} " imply text2 " {} "?'.format(caption, hypothesis))

            elif self.description == 'wiki':
                src_item = self.encode_text(
                    'Visual Entailment (VE) - is a task consisting of image-sentence pairs whereby a premise is defined by an image, rather than a natural language sentence as in traditional Textual Entailment tasks. The goal is to predict whether the image semantically entails the text.'
                    ' Prompt: can image and text1 " {} " imply text2 " {} "?'.format(caption, hypothesis))

            elif self.description == 'wiki-tep':
                src_item = self.encode_text(
                    'Visual Entailment (VE) - is a task consisting of image-sentence pairs whereby a premise is defined by an image, rather than a natural language sentence as in traditional Textual Entailment tasks. The goal is to predict whether the image semantically entails the text.'
                    'Dataset Description: SNLI-VE is a dataset for visual entailment, which is the task of determining whether a given natural language sentence is entailed by a given image. The SNLI-VE dataset is a large-scale dataset that includes over 200,000 images and more than 1.2 million sentence pairs.'
                    'Annotating a dataset like SNLI-VE involves manually labeling the images with sentence pairs and labels indicating whether the sentences are entailed by the image. This is typically done by trained annotators who use specialized software tools to create the sentence pairs and assign the labels. The sentences should be natural language sentences that are related to the content of the images, and the labels should indicate whether one sentence logically follows from the other given the information in the image.'
                    'Input format: A Task Prompt,  a condition Text 1 , a implied result Text 2 and a  Image'
                    'Output format: yes or no or maybe'
                    'Output description:  can imply or can not imply or maybe imply'
                    'Prompt: can image and text1 " {} " imply text2 " {} "?'.format(caption, hypothesis))

            elif self.description == 'tep':
                src_item = self.encode_text(
                    'Dataset Description: SNLI-VE is a dataset for visual entailment, which is the task of determining whether a given natural language sentence is entailed by a given image. The SNLI-VE dataset is a large-scale dataset that includes over 200,000 images and more than 1.2 million sentence pairs.' \
                    'Annotating a dataset like SNLI-VE involves manually labeling the images with sentence pairs and labels indicating whether the sentences are entailed by the image. This is typically done by trained annotators who use specialized software tools to create the sentence pairs and assign the labels. The sentences should be natural language sentences that are related to the content of the images, and the labels should indicate whether one sentence logically follows from the other given the information in the image.' \
                    'Input format: A Task Prompt,  a condition Text 1 , a implied result Text 2 and an  Image' \
                    'Output format: yes or no or maybe' \
                    'Output description:  can imply or can not imply or maybe imply' \
                    ' Prompt: can image and text1 " {} " imply text2 " {} "?'.format(caption, hypothesis))


            elif self.description == 'onehot':
                src_item = self.encode_text(' 0010000 " {} "?'.format(caption, hypothesis))

            elif self.description == 'base':
                src_item = self.encode_text(
                    ' can image and text1 " {} " imply text2 " {} "?'.format(caption, hypothesis))

        src_item = torch.cat([self.bos_item, src_item, self.eos_item])

        if self.prompt_type == 'none':
            prev_output_item = torch.cat([self.bos_item, tgt_item])
            target_item = torch.cat([prev_output_item[1:], self.eos_item])
            decoder_prompt = self.bos_item

        elif self.prompt_type == 'src':
            prev_output_item = torch.cat([src_item, tgt_item])
            target_item = torch.cat([prev_output_item[1:], self.eos_item])
            decoder_prompt = src_item

        elif self.prompt_type == 'prev_output':
            prev_output_item = torch.cat([src_item[:-1], tgt_item])
            target_item = torch.cat([prev_output_item[1:], self.eos_item])
            decoder_prompt = src_item[:-1]

        else:

            raise NotImplementedError

        target_item[:-len(tgt_item) - 1] = self.tgt_dict.pad()

        example = {
            "id": uniq_id,
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item,
            "decoder_prompt": decoder_prompt,
            "ref_dict": ref_dict,
        }

        if self.constraint_trie is not None:
            constraint_mask = torch.zeros((len(target_item), len(self.tgt_dict))).bool()
            start_idx = len(target_item) - len(tgt_item) - 1
            for i in range(len(target_item) - len(tgt_item) - 1, len(target_item)):
                constraint_prefix_token = [self.tgt_dict.bos()] + target_item[start_idx:i].tolist()
                constraint_nodes = self.constraint_trie.get_next_layer(constraint_prefix_token)
                constraint_mask[i][constraint_nodes] = True
            example["constraint_mask"] = constraint_mask
        return example

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch containing the data of the task
        """
        return self.collate(samples, pad_idx=self.pad, eos_idx=self.eos)
