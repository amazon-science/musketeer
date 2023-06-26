# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.
# -----------------------------------------------------------------------------
# Modifications made by Zhaoyang Zhang on 2022.12.25:
#
# - Add support for dataloader with rich task descriptions.
# -----------------------------------------------------------------------------
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

    target_strs = np.array([s["target_str"] for s in samples])

    batch = {
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "prev_output_tokens": prev_output_tokens
        },
        "target": target,
        "target_strs": target_strs
    }

    return batch


class SummaryDataset(OFADataset):
    def __init__(
            self,
            split,
            dataset,
            bpe,
            src_dict,
            tgt_dict=None,
            code_dict_size=8192,
            num_bins=1000,
            max_src_length=512,
            max_tgt_length=128,
            noise_ratio=0.0,
            description='base'
    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.code_dict_size = code_dict_size
        self.num_bins = num_bins
        self.noise_ratio = noise_ratio
        self.description = description
        if type(bpe).__name__ == 'GPT2BPE':
            if self.description == 'base':
                self.prompt = ' what is the summary of article " {} "?'
            elif self.description == 'tep':
                self.prompt = 'Dataset description: Gigaword is a large-scale dataset for natural language processing tasks, such as language modeling and machine translation. It contains over 5 billion words of text, drawn from a variety of sources, including news articles, books, and websites.The annotation process for Gigaword involves collecting text from a variety of sources and ensuring that it is accurately' \
                              ' transcribed and formatted. The text is then divided into smaller units, such as sentences or paragraphs, and annotated with additional information, such as part-of-speech tags or named entity tags. ' \
                              'Input format: Text' \
                              'Output format: Text' \
                              'Output description: summary of input text' \
                              'prompt: what is the summary of article " {} "? '

            elif self.description == 'wiki-tep':
                self.prompt = 'Given a document, selecting a subset of the words or sentences which best represents a summary of the document.' \
                              'Dataset description: Gigaword is a large-scale dataset for natural language processing tasks, such as language modeling and machine translation. It contains over 5 billion words of text, drawn from a variety of sources, including news articles, books, and websites.The annotation process for Gigaword involves collecting text from a variety of sources and ensuring that it is accurately' \
                              ' transcribed and formatted. The text is then divided into smaller units, such as sentences or paragraphs, and annotated with additional information, such as part-of-speech tags or named entity tags. ' \
                              'Input format: Text' \
                              'Output format: Text' \
                              'Output description: summary of input text' \
                              'prompt: what is the summary of article " {} "? '

            elif self.description == 'annotation':
                self.prompt = \
                    'Dataset description: Gigaword is a large-scale dataset for natural language processing tasks, such as language modeling and machine translation. It contains over 5 billion words of text, drawn from a variety of sources, including news articles, books, and websites.The annotation process for Gigaword involves collecting text from a variety of sources and ensuring that it is accurately' \
                    ' transcribed and formatted. The text is then divided into smaller units, such as sentences or paragraphs, and annotated with additional information, such as part-of-speech tags or named entity tags. ' \
                    'Input format: Text' \
                    'Output format: Text' \
                    'Output description: summary of input text' \
                    'prompt: what is the summary of article " {} "? '


            elif self.description == 'wiki':
                self.prompt = \
                    'Given a document, selecting a subset of the words or sentences which best represents a summary of the document.' \
                    'prompt: what is the summary of article " {} "? '


            elif description == 'onehot':
                self.prompt = '1000000 {}'



        elif type(bpe).__name__ == 'BertBPE':

            self.prompt = "{} 请用一个句子简单总结上文："

    def __getitem__(self, index):
        source, target = self.dataset[index]
        target_str = target.lower()

        source = self.pre_caption(source, max_words=self.max_src_length)
        target = self.pre_caption(target, max_words=self.max_tgt_length)
        source = source.replace('<unk>', 'unk')
        target = target.replace('<unk>', 'unk')

        src_item = self.encode_text(
            self.prompt.format(source),
            length=self.max_src_length
        )
        tgt_item = self.encode_text('{}'.format(target))
        noise_tgt_item = self.add_noise_to_tgt(tgt_item.clone(), self.noise_ratio)

        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, noise_tgt_item])

        example = {
            "source": src_item,
            "target": target_item,
            "prev_output_tokens": prev_output_item,
            "target_str": target_str
        }
        return example

    def add_noise_to_tgt(self, target, p):
        noise_indices = torch.FloatTensor(target.size(0)).uniform_() < p
        target[noise_indices] = torch.randint(
            4, len(self.src_dict) - self.code_dict_size - self.num_bins, size=(noise_indices.sum(),)
        )
        return target

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch containing the data of the task
        """
        return collate(samples, pad_idx=self.pad, eos_idx=self.eos)
