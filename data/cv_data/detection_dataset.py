# Illustration Title: Add support for COCO detection
# Copyright (c) 2022 Zhaoyang Zhang
# Licensed under the Apache License, Version 2.0.
# found in the LICENSE file in the root directory.
from io import BytesIO

import math
import logging
import random
import warnings

import numpy as np
import torch
import base64
from torchvision import transforms

from PIL import Image, ImageFile

from data import data_utils
from data.ofa_dataset import OFADataset
from utils.vision_helper import RandomAugment
import utils.transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


def get_whole_word_mask(bpe, dictionary):
    if bpe is not None:

        def is_beginning_of_word(i):
            if i < dictionary.nspecial:
                return True
            tok = dictionary[i]
            if tok.startswith("madeupword"):
                return True
            try:
                return bpe.is_beginning_of_word(tok)
            except ValueError:
                return True

        mask_whole_words = torch.ByteTensor(
            list(map(is_beginning_of_word, range(len(dictionary))))
        )
        return mask_whole_words
    return None


class DetectionDataset(OFADataset):
    def __init__(
            self,
            split,
            dataset,
            bpe,
            src_dict,
            tgt_dict=None,
            max_src_length=128,
            max_tgt_length=30,
            seed=7,
            code_dict_size=8192,
            num_bins=1000,
            patch_image_size=384,
            code_image_size=128,
            pure_text_dataset=None,
            pure_image_dataset=None,
            detection_dataset=None,
            all_object_list=None,
            all_caption_list=None,
            type2ans_dict=None,
            ans2type_dict=None,
            max_image_size=512,
            mask_ratio=0.3,
            random_ratio=0.0,
            keep_ratio=0.0,
            mask_length="span-poisson",
            poisson_lambda=3.0,
            replace_length=1,
            get_type=False,
            description='annotation_n'
    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.seed = seed
        self.code_dict_size = code_dict_size
        self.num_bins = num_bins
        self.patch_image_size = patch_image_size
        self.code_image_size = code_image_size
        self.get_type = get_type
        self.pure_text_dataset = pure_text_dataset
        self.pure_image_dataset = pure_image_dataset
        self.detection_dataset = detection_dataset
        self.epoch = 0
        self.description = description
        self.all_object_list = all_object_list
        self.all_caption_list = all_caption_list
        self.type2ans_dict = type2ans_dict
        self.ans2type_dict = ans2type_dict

        self.mask_ratio = mask_ratio
        self.random_ratio = random_ratio
        self.keep_ratio = keep_ratio
        self.mask_length = mask_length
        self.poisson_lambda = poisson_lambda
        self.replace_length = replace_length
        if self.replace_length not in [-1, 0, 1]:
            raise ValueError(f"invalid arg: replace_length={self.replace_length}")
        if self.mask_length not in ["subword", "word", "span-poisson"]:
            raise ValueError(f"invalid arg: mask-length={self.mask_length}")
        if self.mask_length == "subword" and self.replace_length not in [0, 1]:
            raise ValueError(f"if using subwords, use replace-length=1 or 0")

        self.mask_idx = src_dict.index("<mask>")
        self.mask_whole_word = (
            get_whole_word_mask(self.bpe, self.src_dict)
            if self.mask_length != "subword"
            else None
        )
        self.mask_span_distribution = None
        if self.mask_length == "span-poisson":
            _lambda = self.poisson_lambda
            lambda_to_the_k = 1
            e_to_the_minus_lambda = math.exp(-_lambda)
            k_factorial = 1
            ps = []
            for k in range(0, 128):
                ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
                lambda_to_the_k *= _lambda
                k_factorial *= k + 1
                if ps[-1] < 0.0000001:
                    break
            ps = torch.FloatTensor(ps)
            self.mask_span_distribution = torch.distributions.Categorical(ps)

        self.pos_tgt_item = self.encode_text(" yes")
        self.neg_tgt_item = self.encode_text(" no")

        self.mask_left = self.mask_top = int(0.5 * self.code_image_size)
        self.mask_right = self.mask_bottom = int(1.5 * self.code_image_size)
        self.mask_ids = [
            i * self.code_image_size * 2 + j
            for i in range(self.code_image_size * 2) for j in range(self.code_image_size * 2)
            if not (self.mask_left <= i < self.mask_right and self.mask_top <= j < self.mask_bottom)
        ]

        scales = np.arange(patch_image_size, 481).tolist()

        # for image-text pair
        self.patch_resize_transform = transforms.Compose([
            T.RandomResize(scales, max_size=672),
            transforms.CenterCrop(patch_image_size),
            RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        # for pure image
        self.patch_crop_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        # for detection
        self.detection_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResize([self.code_image_size * 2], max_size=self.code_image_size * 2),
            # T.LargeScaleJitter(output_size=self.code_image_size*2, aug_scale_min=1.0, aug_scale_max=1.5),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_image_size=max_image_size)
        ]) if split == 'train' else T.Compose([
            T.RandomResize([self.code_image_size * 2], max_size=self.code_image_size * 2),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_image_size=max_image_size)
        ])

        ### Pixel2Seq Aug
        # normalize = T.Compose([
        #     T.ToTensor(),
        #     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])
        # if split == 'train':
        #
        #
        #     scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        #
        #     self.detection_transform = T.Compose([
        #         T.RandomHorizontalFlip(),
        #         T.RandomSelect(
        #             T.RandomResize(scales, max_size=512),
        #             T.Compose([
        #                 T.RandomResize([400, 500, 600]),
        #                 T.RandomSizeCrop(384, 600),
        #                 T.RandomResize(scales, max_size=512),
        #             ])
        #         ),
        #         normalize,
        #     ])
        # else:
        #     self.detection_transform =  = T.Compose([
        #         T.RandomResize([800], max_size=512),
        #         normalize,
        #     ])

        # for visual grounding
        self.visual_grounding_transform = T.Compose([
            T.RandomResize(scales, max_size=672),
            T.ObjectCenterCrop((patch_image_size, patch_image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_image_size=max_image_size)
        ])

    def set_epoch(self, epoch, **unused):
        self.epoch = epoch

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
        orig_img_size = np.array([s["orig_img_size"] for s in samples])
        src_tokens = merge("source")
        src_lengths = torch.LongTensor([s["source"].ne(pad_idx).long().sum() for s in samples])

        patch_images = torch.stack([sample['patch_image'] for sample in samples], dim=0)
        patch_masks = torch.cat([sample['patch_mask'] for sample in samples])

        code_masks = None
        if samples[0].get("code_mask", None) is not None:
            code_masks = torch.cat([sample['code_mask'] for sample in samples])

        conf = torch.cat([s['conf'] for s in samples], dim=0)

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
            tgt_lengths = None

        batch = {
            "id": id,
            "orig_img_size": orig_img_size,
            "nsentences": len(samples),
            "ntokens": ntokens,
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
                "patch_images": patch_images,
                "patch_masks": patch_masks,
                "code_masks": code_masks,
                "prev_output_tokens": prev_output_tokens
            },
            "target": target,
            "tgt_lengths": tgt_lengths,
            "conf": conf
        }
        if self.get_type:
            batch["net_input"]["task_name"] = 'object detection'
        return batch

    def get_negative_caption(self, caption, gt_objects):
        prob = random.random()
        if gt_objects is not None and gt_objects != '' and prob > 0.6:
            gt_object = random.choice(gt_objects.strip().split('&&'))
            negative_object = random.choice(self.all_object_list[:-1])
            negative_object = self.all_object_list[-1] if negative_object == gt_object else negative_object
            negative_caption = caption.replace(gt_object, negative_object)
        else:
            negative_caption = random.choice(self.all_caption_list)
        return negative_caption

    def get_negative_answer(self, answer, conf):
        prob = random.random()
        if conf > (prob + 0.1) and answer in self.ans2type_dict:
            negative_answer_type = self.ans2type_dict[answer]
            if negative_answer_type == 'how many' and answer.isdigit() and prob > 0.5:
                negative_answer = int(answer) + random.choice([-1, 1]) if answer != 0 else 1
            else:
                negative_answer_list = self.type2ans_dict[negative_answer_type]
                negative_answer = random.choice(negative_answer_list[:-1])
                negative_answer = negative_answer_list[-1] if negative_answer == answer else negative_answer
            return negative_answer

        negative_answer_list = self.type2ans_dict['other']
        negative_answer = random.choice(negative_answer_list[:-1])
        negative_answer = negative_answer_list[-1] if negative_answer == answer else negative_answer
        return negative_answer

    def process_detection(self, index):
        image_id, image, label = self.dataset[index]
        image = Image.open(BytesIO(base64.urlsafe_b64decode(image))).convert("RGB")

        w, h = image.size
        boxes_target = {"boxes": [], "labels": [], "area": [], "size": torch.tensor([h, w])}
        label_list = label.strip().split('&&')
        for label in label_list:
            x0, y0, x1, y1, cat_id, cat = label.strip().split(',', 5)
            boxes_target["boxes"].append([float(x0), float(y0), float(x1), float(y1)])
            boxes_target["labels"].append(cat)
            boxes_target["area"].append((float(x1) - float(x0)) * (float(y1) - float(y0)))
        boxes_target["boxes"] = torch.tensor(boxes_target["boxes"])
        boxes_target["labels"] = np.array(boxes_target["labels"])
        boxes_target["area"] = torch.tensor(boxes_target["area"])

        # Randomly shuffle objects
        indices = torch.randperm(boxes_target["boxes"].size()[0])
        if len(indices) > 1:
            boxes_target["boxes"] = boxes_target["boxes"][indices]
            boxes_target["labels"] = boxes_target["labels"][indices]
            boxes_target["area"] = boxes_target["area"][indices]

        patch_image, boxes_target = self.detection_transform(image, boxes_target)
        patch_mask = torch.tensor([True])
        code_mask = torch.tensor([False])
        conf = torch.tensor([2.0])

        quant_boxes = []
        for i, box in enumerate(boxes_target["boxes"]):
            quant_boxes.extend(["<bin_{}>".format(int((pos * (self.num_bins - 1)).round())) for pos in box[:4]])
            quant_boxes.append(self.bpe.encode(' {}'.format(boxes_target["labels"][i])))

        # src_item = self.encode_text(' what are the objects in the image?')

        # rich 1
        if self.description == 'description_gpt':
            src_item = self.encode_text("Task: Object Detection. "
                                        "Object detection is a task in computer vision where a model is trained to identify and "
                                        "locate objects within an image. This involves both recognizing that an object is present in the image and accurately "
                                        "locating where in the image the object is. To do this, a model is typically trained on a large dataset of images that have been "
                                        "labeled with the objects they contain. "
                                        "This allows the model to learn the visual characteristics of the objects it needs to detect."
                                        "During training, the model is presented with an image and a set of bounding boxes that indicate the location of the objects in the image. "
                                        "The model then uses this information to learn how to identify and locate objects in images. "
                                        "Object detection has many practical applications, such as self-driving cars, which use object detection to identify and locate other vehicles, pedestrians, and obstacles on the road. It can also be used in security systems to automatically detect objects of interest, such as weapons or suspicious packages. In medical imaging, object detection can be used to automatically identify and locate objects such as tumors or other abnormalities in images of the human body."
                                        "Dataset: COCO. "
                                        "COCO (Common Objects in Context) is a dataset used in the field of computer vision for the task of object detection. The dataset consists of a large collection of images, each of which has been annotated with bounding boxes and labels indicating the presence and location of objects within the image."
                                        "The goal of the COCO dataset is to provide a large and diverse set of examples for training and evaluating models for the task of object detection. The dataset includes a total of 328,000 images and more than 2.5 million labeled objects, covering 90 object categories."
                                        "Models trained on the COCO dataset can be used for a variety of object detection tasks, such as self-driving cars and security systems. The dataset has been widely used in research and has contributed to significant advances in the field of object detection."
                                        "Prompt: what are the objects in the image?")


        elif self.description == 'wiki':
            src_item = self.encode_text(
                'Object detection is the task of detecting instances of objects of a certain class within an image.'
                "Prompt: what are the objects in the image?")



        elif self.description == 'wiki-tep':
            src_item = self.encode_text(
                'Object detection is the task of detecting instances of objects of a certain class within an image.'
                'Dataset Description: COCO, or the Common Objects in Context dataset, is a large-scale dataset for object detection, segmentation, and captioning. The dataset is commonly used to train and evaluate object detection algorithms.'
                'Annotating a dataset like COCO involves manually labeling the objects in each image with bounding boxes and class labels. This is typically done by trained annotators who use specialized software tools to draw the bounding boxes and assign the class labels to the objects in the images.' \
                'Input format: A Task Prompt  and a Image containing target objects' \
                'Output format: mutiple {x0 + y0 + x1 + y1} ' \
                'Output description: mutiple bounding boxes (each consists of horizonal coordinates of leftupper points of target region +  vertical coordinates of leftupper points of target region  + horizonal coordinates of rightlower points of target region +  vertical coordinates of rightlower points of target region )' \
                "Prompt: what are the objects in the image?")




        elif self.description == 'tep':
            src_item = self.encode_text(
                'Dataset Description: COCO, or the Common Objects in Context dataset, is a large-scale dataset for object detection, segmentation, and captioning. The dataset is commonly used to train and evaluate object detection algorithms.'
                'Annotating a dataset like COCO involves manually labeling the objects in each image with bounding boxes and class labels. This is typically done by trained annotators who use specialized software tools to draw the bounding boxes and assign the class labels to the objects in the images.' \
                'Input format: A Task Prompt  and a Image containing target objects' \
                'Output format: mutiple {x0 + y0 + x1 + y1} ' \
                'Output description: mutiple bounding boxes (each consists of horizonal coordinates of leftupper points of target region +  vertical coordinates of leftupper points of target region  + horizonal coordinates of rightlower points of target region +  vertical coordinates of rightlower points of target region )' \
                "Prompt: what are the objects in the image?")







        elif self.description == 'onehot':
            src_item = self.encode_text("0000001")
        elif self.description == 'base':
            src_item = self.encode_text("what are the objects in the image? ")

        tgt_item = self.encode_text(' '.join(quant_boxes), use_bpe=False)

        src_item = torch.cat([self.bos_item, src_item, self.eos_item])

        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, tgt_item])

        example = {
            "id": image_id,
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "code_mask": code_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item,
            "conf": conf,
            'orig_img_size': [h, w],
        }
        return [example]

    def __getitem__(self, index):
        with data_utils.numpy_seed(self.seed, self.epoch):
            detection_samples = self.process_detection(index)

    def word_starts(self, source):
        if self.mask_whole_word is not None:
            is_word_start = self.mask_whole_word.gather(0, source)
        else:
            is_word_start = torch.ones(source.size())
        is_word_start[0] = 0
        is_word_start[-1] = 0
        return is_word_start

    def add_whole_word_mask(self, source, p):
        is_word_start = self.word_starts(source)
        num_to_mask = int(math.ceil(is_word_start.float().sum() * p))
        num_inserts = 0
        if num_to_mask == 0:
            return source

        if self.mask_span_distribution is not None:
            lengths = self.mask_span_distribution.sample(sample_shape=(num_to_mask,))

            # Make sure we have enough to mask
            cum_length = torch.cumsum(lengths, 0)
            while cum_length[-1] < num_to_mask:
                lengths = torch.cat(
                    [
                        lengths,
                        self.mask_span_distribution.sample(sample_shape=(num_to_mask,)),
                    ],
                    dim=0,
                )
                cum_length = torch.cumsum(lengths, 0)

            # Trim to masking budget
            i = 0
            while cum_length[i] < num_to_mask:
                i += 1
            lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i - 1])
            num_to_mask = i + 1
            lengths = lengths[:num_to_mask]

            # Handle 0-length mask (inserts) separately
            lengths = lengths[lengths > 0]
            num_inserts = num_to_mask - lengths.size(0)
            num_to_mask -= num_inserts
            if num_to_mask == 0:
                return self.add_insertion_noise(source, num_inserts / source.size(0))

            assert (lengths > 0).all()
        else:
            lengths = torch.ones((num_to_mask,)).long()
        assert is_word_start[-1] == 0
        word_starts = is_word_start.nonzero(as_tuple=False)
        indices = word_starts[
            torch.randperm(word_starts.size(0))[:num_to_mask]
        ].squeeze(1)
        mask_random = torch.FloatTensor(num_to_mask).uniform_() < self.random_ratio

        source_length = source.size(0)
        assert source_length - 1 not in indices
        to_keep = torch.ones(source_length, dtype=torch.bool)
        is_word_start[
            -1
        ] = 255  # acts as a long length, so spans don't go over the end of doc
        if self.replace_length == 0:
            to_keep[indices] = 0
        else:
            # keep index, but replace it with [MASK]
            source[indices] = self.mask_idx
            source[indices[mask_random]] = torch.randint(
                4, len(self.tgt_dict) - self.code_dict_size - self.num_bins, size=(mask_random.sum(),)
            )

        if self.mask_span_distribution is not None:
            assert len(lengths.size()) == 1
            assert lengths.size() == indices.size()
            lengths -= 1
            while indices.size(0) > 0:
                assert lengths.size() == indices.size()
                lengths -= is_word_start[indices + 1].long()
                uncompleted = lengths >= 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                lengths = lengths[uncompleted]
                if self.replace_length != -1:
                    # delete token
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    source[indices] = self.mask_idx
                    source[indices[mask_random]] = torch.randint(
                        4, len(self.tgt_dict) - self.code_dict_size - self.num_bins, size=(mask_random.sum(),)
                    )
        else:
            # A bit faster when all lengths are 1
            while indices.size(0) > 0:
                uncompleted = is_word_start[indices + 1] == 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                if self.replace_length != -1:
                    # delete token
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    source[indices] = self.mask_idx
                    source[indices[mask_random]] = torch.randint(
                        4, len(self.tgt_dict) - self.code_dict_size - self.num_bins, size=(mask_random.sum(),)
                    )

                assert source_length - 1 not in indices

        source = source[to_keep]

        if num_inserts > 0:
            source = self.add_insertion_noise(source, num_inserts / source.size(0))

        return source

    def add_insertion_noise(self, tokens, p):
        if p == 0.0:
            return tokens

        num_tokens = len(tokens)
        n = int(math.ceil(num_tokens * p))

        noise_indices = torch.randperm(num_tokens + n - 2)[:n] + 1
        noise_mask = torch.zeros(size=(num_tokens + n,), dtype=torch.bool)
        noise_mask[noise_indices] = 1
        result = torch.LongTensor(n + len(tokens)).fill_(-1)

        num_random = int(math.ceil(n * self.random_ratio))
        result[noise_indices[num_random:]] = self.mask_idx
        result[noise_indices[:num_random]] = torch.randint(
            low=4, high=len(self.tgt_dict) - self.code_dict_size - self.num_bins, size=(num_random,)
        )

        result[~noise_mask] = tokens

        assert (result >= 0).all()
        return result

    def collater(self, samples, pad_to_length=None):
        """Merge samples of different tasks to form two mini-batches.
        Args:
            samples (List[Tuple]): samples to collate
        Returns:
            Tuple[dict]: two mini-batch containing the data of different tasks
        """

        samples_v1 = []

        for sample in samples:
            samples_v1 += sample
        res_v1 = self.collate(samples_v1, pad_idx=self.src_dict.pad(), eos_idx=self.eos)
        return res_v1
