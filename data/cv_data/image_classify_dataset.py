from io import BytesIO

import logging
import warnings
import functools

import numpy as np
import torch
import base64
from torchvision import transforms
from timm.data import create_transform
from utils.vision_helper import RandomAugment

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


class ImageClassifyDataset(OFADataset):
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
            constraint_trie=None,
            imagenet_default_mean_and_std=False,
            description='base'
    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.patch_image_size = patch_image_size
        self.constraint_trie = constraint_trie
        self.description = description

        if imagenet_default_mean_and_std:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        if self.split != 'train':
            self.patch_resize_transform = transforms.Compose([
                lambda image: image.convert("RGB"),
                transforms.Resize([patch_image_size, patch_image_size], interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
            logger.info("val split, do not use random augmentation.")
        else:
            self.patch_resize_transform = create_transform(
                input_size=patch_image_size,
                is_training=True,
                color_jitter=0.4,
                auto_augment='rand-m9-mstd0.5-inc1',
                interpolation='bicubic',
                re_prob=0.25,
                re_mode='pixel',
                re_count=1,
                mean=mean,
                std=std,
            )
            self.patch_resize_transform = transforms.Compose(functools.reduce(lambda x, y: x + y, [
                [lambda image: image.convert("RGB"), ],
                self.patch_resize_transform.transforms[:2],
                [self.patch_resize_transform.transforms[2]],
                [RandomAugment(2, 7, isPIL=True,
                               augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness', 'ShearX',
                                     'ShearY', 'TranslateX', 'TranslateY', 'Rotate']), ],
                self.patch_resize_transform.transforms[3:],
            ]))
            logger.info("train split, use random augmentation.")

    def __getitem__(self, index):
        # print(self.dataset[index])
        image, label_name = self.dataset[index]

        image = Image.open(BytesIO(base64.urlsafe_b64decode(image)))
        patch_image = self.patch_resize_transform(image)
        patch_mask = torch.tensor([True])

        if self.description == 'onehot':
            src_item = self.encode_text('0000010')

        elif self.description == 'description_gpt':
            src_item = self.encode_text(
                'Image classification is a computer vision task where the goal is to automatically classify images into one of several pre-defined categories. The ImageNet dataset is a widely used benchmark dataset for image classification tasks, containing over 1 million images across 1,000 different categories.'
                'The task of image classification on ImageNet typically involves training a deep neural network, such as a convolutional neural network (CNN), on a subset of the dataset known as the training set. During training, the network learns to recognize patterns and features in the images that are useful for distinguishing between different categories.'
                'Prompt:  what does the image describe?')

        elif self.description == 'base':

            src_item = self.encode_text(' what does the image describe?')

        elif self.description == 'tep':
            src_item = self.encode_text(
                'Dataset Description:  ImageNet is a large-scale dataset for image classification, object detection, and object segmentation. It contains over 14 million images, each labeled with the name of one of 1000 object categories. The images in ImageNet are annotated by human labelers, who have assigned a label to each image indicating the main object or concept depicted in it.'
                'The annotation process for ImageNet involves two steps: (1) determining the set of object categories to be used for labeling the images and (2) labeling the images with these categories.'
                'Determining the set of object categories: The object categories used for ImageNet were determined through a process called "WordNet hierarchy expansion." WordNet is a large database of English words and their relationships to one another. The ImageNet organizers used WordNet to expand the set of object categories to include all the nouns in WordNet, resulting in a list of over 200,000 categories. They then selected a subset of these categories to use for ImageNet, based on their relevance to image classification and their difficulty level. The final set of categories used in ImageNet consists of 1000 object categories.'
                'Labeling the images: Once the set of object categories has been determined, the images in ImageNet are labeled by human annotators. The annotators are shown an image and asked to select the object category that best describes the main object or concept depicted in the image. In some cases, multiple object categories may be applicable to a single image. In these cases, the annotators are asked to select all the relevant categories.'
                'Input format: Task prompt and an input Image'
                'Output format: Text '
                'Output description: A class name this image describe'
                'Prompt:  what does the image describe?')


        elif self.description == 'wiki-tep':
            src_item = self.encode_text(
                'Image Classification is a fundamental task that attempts to comprehend an entire image as a whole. The goal is to classify the image by assigning it to a specific label. '
                'Dataset Description:  ImageNet is a large-scale dataset for image classification, object detection, and object segmentation. It contains over 14 million images, each labeled with the name of one of 1000 object categories. The images in ImageNet are annotated by human labelers, who have assigned a label to each image indicating the main object or concept depicted in it.'
                'The annotation process for ImageNet involves two steps: (1) determining the set of object categories to be used for labeling the images and (2) labeling the images with these categories.'
                'Determining the set of object categories: The object categories used for ImageNet were determined through a process called "WordNet hierarchy expansion." WordNet is a large database of English words and their relationships to one another. The ImageNet organizers used WordNet to expand the set of object categories to include all the nouns in WordNet, resulting in a list of over 200,000 categories. They then selected a subset of these categories to use for ImageNet, based on their relevance to image classification and their difficulty level. The final set of categories used in ImageNet consists of 1000 object categories.'
                'Labeling the images: Once the set of object categories has been determined, the images in ImageNet are labeled by human annotators. The annotators are shown an image and asked to select the object category that best describes the main object or concept depicted in the image. In some cases, multiple object categories may be applicable to a single image. In these cases, the annotators are asked to select all the relevant categories.'
                'Input format: Task prompt and an input Image'
                'Output format: Text '
                'Output description: A class name this image describe'
                'Prompt:  what does the image describe?')

        elif self.description == 'wiki':
            src_item = self.encode_text(
                'Image Classification is a fundamental task that attempts to comprehend an entire image as a whole. The goal is to classify the image by assigning it to a specific label. '
                'Prompt:  what does the image describe?')
        elif self.description == 'annotation':
            src_item = self.encode_text(''
                                        'Dataset Description:  ImageNet is a large-scale dataset for image classification, object detection, and object segmentation. It contains over 14 million images, each labeled with the name of one of 1000 object categories. The images in ImageNet are annotated by human labelers, who have assigned a label to each image indicating the main object or concept depicted in it.'
                                        'The annotation process for ImageNet involves two steps: (1) determining the set of object categories to be used for labeling the images and (2) labeling the images with these categories.'
                                        'Determining the set of object categories: The object categories used for ImageNet were determined through a process called "WordNet hierarchy expansion." WordNet is a large database of English words and their relationships to one another. The ImageNet organizers used WordNet to expand the set of object categories to include all the nouns in WordNet, resulting in a list of over 200,000 categories. They then selected a subset of these categories to use for ImageNet, based on their relevance to image classification and their difficulty level. The final set of categories used in ImageNet consists of 1000 object categories.'
                                        'Labeling the images: Once the set of object categories has been determined, the images in ImageNet are labeled by human annotators. The annotators are shown an image and asked to select the object category that best describes the main object or concept depicted in the image. In some cases, multiple object categories may be applicable to a single image. In these cases, the annotators are asked to select all the relevant categories.'
                                        'Input format: Task prompt + image'
                                        'Output format: image class '
                                        'Output description: A class name this image describe'
                                        'Prompt:  what does the image describe?')

        tgt_item = self.encode_text(" {}".format(label_name))
        ref_dict = {label_name: 1.0}

        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, tgt_item])

        example = {
            "id": index,
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item,
            "ref_dict": ref_dict,
        }

        if self.constraint_trie is not None:
            constraint_mask = torch.zeros((len(prev_output_item), len(self.tgt_dict))).bool()
            for i in range(len(prev_output_item)):
                constraint_prefix_token = prev_output_item[:i + 1].tolist()
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
            "target": target,
        }

        return batch

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch containing the data of the task
        """
        return self.collate(samples, pad_idx=self.pad, eos_idx=self.eos)
