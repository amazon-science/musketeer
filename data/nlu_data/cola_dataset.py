# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

import logging
import warnings
import torch
import numpy as np

from data import data_utils
from data.ofa_dataset import OFADataset

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


def collate(samples, pad_idx, eos_idx):
    if len(samples) == 0:
        return {}

    def merge(key):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx=eos_idx,
        )

    src_tokens = merge("source")
    src_lengths = torch.LongTensor([s["source"].ne(pad_idx).long().sum() for s in samples])

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
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "prev_output_tokens": prev_output_tokens
        },
        "ref_dict": ref_dict,
        "constraint_masks": constraint_masks,
        "target": target,
    }

    return batch


class COLADataset(OFADataset):
    def __init__(
        self,
        split,
        dataset,
        bpe,
        src_dict,
        tgt_dict=None,
        max_src_length=512,
        max_tgt_length=30,
        constraint_trie=None,
        prompt_type="none",
            description='base'
    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.constraint_trie = constraint_trie
        self.prompt_type = prompt_type
        self.description = description

    def __getitem__(self, index):
        sentence, label = self.dataset[index]
        if label == '0':
            label = 'no'
        elif label == '1':
            label = 'yes'
        else:
            raise NotImplementedError

        sentence = ' '.join(sentence.lower().strip().split()[:self.max_src_length])




        if self.description == 'base':
            src_item = self.encode_text(' is the text " {} " grammatically correct?'.format(sentence))
        elif self.description == 'onehot':
            src_item = self.encode_text('000000001 {}'.format(sentence))
        elif self.description == 'annotation_n':

            src_item = self.encode_text(
                'Dataset Description: CoLA (Corpus of Linguistic Acceptability) is a dataset for natural language understanding, specifically for evaluating the grammatical acceptability of a sentence. It contains over 10,000 sentences, each annotated with a binary label indicating whether the sentence is grammatically acceptable or not.'
                'The annotation process for CoLA involves two steps: (1) collecting a set of sentences and (2) annotating the sentences with grammatical acceptability labels.'
                'Collecting a set of sentences: The organizers of CoLA collected a set of sentences from a variety of sources, including books, websites, and other text corpora. They selected sentences that were likely to be grammatically challenging, such as sentences with complex syntactic structures or sentences that are grammatically ambiguous.'
                'Annotating the sentences with grammatical acceptability labels: Once the sentences have been collected, human annotators are asked to label each sentence as either "acceptable" or "unacceptable" based on its grammaticality.'
                'Input format: A Task Prompt ,  a question description text  and  a description image'
                'Output format: Yes or No'
                'Output description:  The input text is grammatically correct or not' \
                'Prompt: is the text " {} " grammatically correct?'.format(sentence))


        elif self.description == 'annotation_ohot_easy':

            src_item = self.encode_text(
                'Dataset Description: CoLA (Corpus of Linguistic Acceptability) is a dataset for natural language understanding, specifically for evaluating the grammatical acceptability of a sentence. It contains over 10,000 sentences, each annotated with a binary label indicating whether the sentence is grammatically acceptable or not.'
                'The annotation process for CoLA involves two steps: (1) collecting a set of sentences and (2) annotating the sentences with grammatical acceptability labels.'
                'Collecting a set of sentences: The organizers of CoLA collected a set of sentences from a variety of sources, including books, websites, and other text corpora. They selected sentences that were likely to be grammatically challenging, such as sentences with complex syntactic structures or sentences that are grammatically ambiguous.'
                'Annotating the sentences with grammatical acceptability labels: Once the sentences have been collected, human annotators are asked to label each sentence as either "acceptable" or "unacceptable" based on its grammaticality.'
                'Input format: 100 '
                'Output format: 10 '
                'Output description:  The input text is grammatically correct or not' \
                'Prompt: is the text " {} " grammatically correct?'.format(sentence))


        elif self.description == 'annotation_ohot_hard':

            src_item = self.encode_text(
                'Dataset Description: CoLA (Corpus of Linguistic Acceptability) is a dataset for natural language understanding, specifically for evaluating the grammatical acceptability of a sentence. It contains over 10,000 sentences, each annotated with a binary label indicating whether the sentence is grammatically acceptable or not.'
                'The annotation process for CoLA involves two steps: (1) collecting a set of sentences and (2) annotating the sentences with grammatical acceptability labels.'
                'Collecting a set of sentences: The organizers of CoLA collected a set of sentences from a variety of sources, including books, websites, and other text corpora. They selected sentences that were likely to be grammatically challenging, such as sentences with complex syntactic structures or sentences that are grammatically ambiguous.'
                'Annotating the sentences with grammatical acceptability labels: Once the sentences have been collected, human annotators are asked to label each sentence as either "acceptable" or "unacceptable" based on its grammaticality.'
                'Input format: 100 '
                'Output format: 00100 '
                'Output description:  The input text is grammatically correct or not' \
                'Prompt: is the text " {} " grammatically correct?'.format(sentence))



        tgt_item = self.encode_text(" {}".format(label))
        assert tgt_item.size(0) == 1
        ref_dict = {label: 1.0}

        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        if self.prompt_type == 'none':
            prev_output_item = self.bos_item
            target_item = tgt_item
        elif self.prompt_type == 'src':
            prev_output_item = src_item.clone()
            target_item = torch.cat([prev_output_item[1:], tgt_item])
        elif self.prompt_type == 'prev_output':
            prev_output_item = src_item[:-1].clone()
            target_item = torch.cat([prev_output_item[1:], tgt_item])
        else:
            raise NotImplementedError
        target_item[:-1] = self.tgt_dict.pad()

        example = {
            "source": src_item,
            "target": target_item,
            "prev_output_tokens": prev_output_item,
            "ref_dict": ref_dict,
        }
        if self.constraint_trie is not None:
            constraint_mask = torch.zeros((len(prev_output_item), len(self.tgt_dict))).bool()
            constraint_nodes = self.constraint_trie.get_next_layer(self.bos_item.tolist())
            constraint_mask[-1][constraint_nodes] = True
            example["constraint_mask"] = constraint_mask
        return example

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch containing the data of the task
        """
        return collate(samples, pad_idx=self.pad, eos_idx=self.eos)
