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


class VqaGenDataset(OFADataset):
    def __init__(
            self,
            split,
            dataset,
            bpe,
            src_dict,
            tgt_dict=None,
            max_src_length=128,
            max_object_length=30,
            max_tgt_length=30,
            patch_image_size=224,
            add_object=False,
            constraint_trie=None,
            imagenet_default_mean_and_std=False,
            prompt_type="none",
            get_type=False,
            description='tep'
    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)
        self.max_src_length = max_src_length
        self.max_object_length = max_object_length
        self.max_tgt_length = max_tgt_length
        self.patch_image_size = patch_image_size
        self.get_type = get_type
        self.add_object = add_object
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

    def __getitem__(self, index):
        item = self.dataset[index]
        if len(item) == 5:
            uniq_id, image, question, ref, predict_objects = item
        else:
            uniq_id, image, question, ref, predict_objects, caption = item

        image = Image.open(BytesIO(base64.urlsafe_b64decode(image)))
        patch_image = self.patch_resize_transform(image)
        patch_mask = torch.tensor([True])

        question = self.pre_question(question, self.max_src_length)
        question = question + '?' if not question.endswith('?') else question

        if self.description == 'description_gpt':
            src_item = self.encode_text(
                'Task:  Visual Question Answering(VQA). Visual Question Answering (VQA) is a task in the field of computer vision and natural language processing where a model is trained to answer questions about an image. This involves taking an image and a natural language question as input and outputting a natural language answer.'
                'To perform this task, a model typically needs to be able to understand the contents of an image, as well as the relationships between objects within the image and the actions that can be performed on those objects. This can be done using techniques such as object detection, scene graph generation, and natural language processing.'
                'VQA has many potential applications, such as providing a more intuitive and flexible way for people to search for information within images, allowing a computer to automatically generate a caption for an image, or assisting in the development of more advanced AI systems that can reason about the world from visual input. '
                'Dataset: VQAv2. VQAv2 (Visual Question Answering version 2) is a dataset used in the field of computer vision for the task of visual question answering. The dataset consists of a large collection of images, along with a set of questions and answers about the images.'
                'The goal of the VQAv2 dataset is to provide a large and diverse set of examples for training and evaluating models for the task of visual question answering. The dataset includes a total of 153,000 images, 444,000 questions, and 1.4 million answers, covering a wide range of visual concepts and relationship types.'
                'Models trained on the VQAv2 dataset can be used for a variety of natural language processing and computer vision tasks, such as image captioning and visual entailment. The dataset has been widely used in research and has contributed to significant advances in the field of visual question answering.'
                'Prompt: {}'.format(question))


        elif self.description == 'wiki':
            src_item = self.encode_text(
                'We propose the task of free-form and open-ended Visual Question Answering (VQA). Given an image and a natural language question about the image, the task is to provide an accurate natural language answer. '
                'Prompt: {}'.format(question))



        elif self.description == 'wiki-tep':

            src_item = self.encode_text(
                'We propose the task of free-form and open-ended Visual Question Answering (VQA). Given an image and a natural language question about the image, the task is to provide an accurate natural language answer. '
                'Dataset Description: VQAv2 is a dataset for visual question answering (VQA), which is a task that involves generating natural language answers to questions about images. The VQAv2 dataset is a large-scale dataset that includes over 200,000 images and more than 1.2 million questions and answers.' \
                'Annotating a dataset like VQAv2 involves manually labeling the images with questions and answers. This is typically done by trained annotators who use specialized software tools to create the questions and answers. The questions should be natural language questions that are related to the content of the images, and the answers should be natural language responses that provide accurate and relevant information about the images.' \
                'Input format: A Task Prompt ,  a question description text  and  a description image'
                'Output format: Text'
                'Output description:  Answers '
                'Prompt: {}'.format(question))




        elif self.description == 'tep':
            src_item = self.encode_text(
                'Dataset Description: VQAv2 is a dataset for visual question answering (VQA), which is a task that involves generating natural language answers to questions about images. The VQAv2 dataset is a large-scale dataset that includes over 200,000 images and more than 1.2 million questions and answers.' \
                'Annotating a dataset like VQAv2 involves manually labeling the images with questions and answers. This is typically done by trained annotators who use specialized software tools to create the questions and answers. The questions should be natural language questions that are related to the content of the images, and the answers should be natural language responses that provide accurate and relevant information about the images.' \
                'Input format: A Task Prompt ,  a question description text  and  a description image'
                'Output format: Text'
                'Output description:  Answers '
                'Prompt: {}'.format(question))


        elif self.description == 'onehot':
            src_item = self.encode_text(' 0100000 {}'.format(question))


        elif self.description == 'base':
            src_item = self.encode_text(' {}'.format(question))

        ref_dict = {item.split('|!+')[1]: float(item.split('|!+')[0]) for item in ref.split('&&')}
        answer = max(ref_dict, key=ref_dict.get)
        conf = torch.tensor([ref_dict[answer]])
        tgt_item = self.encode_text(" {}".format(answer))

        if self.add_object and predict_objects is not None:
            predict_object_seq = ' '.join(predict_objects.strip().split('&&')[:self.max_object_length])
            predict_object_item = self.encode_text(" object: {}".format(predict_object_seq))
            src_item = torch.cat([src_item, predict_object_item])

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
            "conf": conf,
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

        conf = None
        if samples[0].get("conf", None) is not None:
            conf = torch.cat([s['conf'] for s in samples], dim=0)

        ref_dict = None
        if samples[0].get("ref_dict", None) is not None:
            ref_dict = np.array([s['ref_dict'] for s in samples])

        constraint_masks = None
        if samples[0].get("constraint_mask", None) is not None:
            constraint_masks = merge("constraint_mask")

        decoder_prompts = None
        if samples[0].get("decoder_prompt", None) is not None:
            decoder_prompts = np.array([s['decoder_prompt'].tolist() for s in samples])

        prefix_tokens = None
        if samples[0].get("decoder_prompt", None) is not None:
            prefix_tokens = merge("decoder_prompt")
            prefix_tokens = prefix_tokens[:, 1:]

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
            "conf": conf,
            "ref_dict": ref_dict,
            "constraint_masks": constraint_masks,
            "decoder_prompts": decoder_prompts,
            "target": target,
            "prefix_tokens": prefix_tokens
        }
        if self.get_type:
            batch["net_input"]["task_name"] = 'visual question & answering'
        return batch

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch containing the data of the task
        """
        return self.collate(samples, pad_idx=self.pad, eos_idx=self.eos)
