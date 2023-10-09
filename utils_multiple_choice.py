# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Multiple choice fine-tuning: utilities to work with multiple choice tasks of reading comprehension  """

from __future__ import absolute_import, division, print_function
from importlib.resources import path


import logging
import os
from pickletools import read_decimalnl_long
import sys
from io import open
import json
import csv
import glob
from regex import P
import tqdm
import jsonlines
from typing import List
from transformers import PreTrainedTokenizer
import random
import re
#from mctest import parse_mc
from matplotlib import pyplot as plt

from transformers.data.metrics import pearson_and_spearman

logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for multiple choice"""

    def __init__(self, example_id, question, contexts, endings, label=None, nsp_label=None, context_sents=None):
        """Constructs a InputExample.

        Args:
            example_id: Unique id for the example.
            contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
            question: string. The untokenized text of the second sequence (question).
            endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.example_id = example_id
        self.question = question
        self.contexts = contexts
        self.endings = endings
        self.label = label


class InputFeatures(object):
    def __init__(self,
                 example_id,  
                 choices_features, 
                 label, 
                 pq_end_pos=None 

    ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
            }
            for input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label
        self.pq_end_pos=pq_end_pos


class DataProcessor(object):
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class RaceProcessor(DataProcessor):
    """Processor for the RACE data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        high = os.path.join(data_dir, 'train/high')
        middle = os.path.join(data_dir, 'train/middle')
        return self._create_examples(high + middle, 'train')

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        high = os.path.join(data_dir, 'dev/high')
        middle = os.path.join(data_dir, 'dev/middle')
        return self._create_examples(high + middle, 'dev')

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        high = os.path.join(data_dir, 'test/high')
        middle = os.path.join(data_dir, 'test/middle')
        return self._create_examples(high + middle, 'test')

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
  
        for (k, data_raw) in enumerate(lines):  
            race_id = "%s-%s" % (set_type, data_raw["race_id"]) 
            article = data_raw["article"]
            for i in range(len(data_raw["answers"])):  
                truth = str(ord(data_raw['answers'][i]) - ord('A')) 
                question = data_raw['questions'][i]  
                options = data_raw['options'][i] 
                examples.append(
                    InputExample(
                        example_id=race_id,
                        question=question,
                        contexts=[article, article, article, article], 
                        endings=[options[0], options[1], options[2], options[3]],
                        label=truth))
                
        return examples


class DreamProcessor(DataProcessor):
    """Processor for the SWAG data set."""
    def __init__(self):
        self.data_pos={"train":0,"dev":1,"test":2}
        self.D = [[], [], []]

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(data_dir, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(data_dir, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(data_dir, "test")
    def get_labels(self):
        """See base class."""
        return ["0", "1", "2"]

    def _create_examples(self, data_dir: str, type: str):
        """Creates examples for the training and dev sets."""    

        if len(self.D[self.data_pos[type]])==0:
            random.seed(42)
            for sid in range(3):
                with open([data_dir + "/" + "train.json", data_dir + "/"  + "dev.json",
                           data_dir + "/" + "test.json"][sid], "r") as f:
                    data = json.load(f)
                    if sid == 0:
                        random.shuffle(data)
                    for i in range(len(data)):
                        for j in range(len(data[i][1])):
                            d = ['\n'.join(data[i][0]).lower(), data[i][1][j]["question"].lower()]
                            for k in range(len(data[i][1][j]["choice"])):
                                d += [data[i][1][j]["choice"][k].lower()]
                            d += [data[i][1][j]["answer"].lower()]
                            self.D[sid] += [d]
        data=self.D[self.data_pos[type]]
        examples = []
        for (i, d) in enumerate(data):
            for k in range(3):
                if data[i][2 + k] == data[i][5]:
                    answer = str(k)

            label = answer
            guid = "%s-%s-%s" % (type, i, k)

            text_a = data[i][0]

            text_c = data[i][1]
            examples.append(
                InputExample(example_id=guid,contexts=[text_a,text_a,text_a],question=text_c,endings=[data[i][2],data[i][3],data[i][4]],label=label))

        return examples

class MctestProcessor(DataProcessor):
    """Processor for the SWAG data set."""
    def __init__(self):
        self.data_pos={"train":0,"dev":1,"test":2}
        self.D = [[], [], []]

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self. _create_examples(data_dir, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(data_dir, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(data_dir, "test")
    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]
    
    def _read_csv(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines
    
    def _read_txt(self, input_dir):
        lines = []
        files = glob.glob(input_dir + "/*txt")
        for file in tqdm.tqdm(files, desc="read files"):# tqdm 进度条库，加载一个进度条信息
            with open(file, 'r', encoding='utf-8') as fin:
                data_raw = json.load(fin)
                data_raw["mc_id"] = file
                lines.append(data_raw)
        return lines
    
    def _create_examples(self, data_dir: str, type: str):
        """Creates examples for the training and dev sets."""

        pref="mc160."
        #pref="mc500."

        file_data_path = os.path.join(data_dir , type)
        filr_datas = self._read_txt(file_data_path)
        examples = []
        for (k, data_raw) in enumerate(filr_datas):  
            mc_id = "%s-%s" % (type, data_raw["mc_id"]) 
            article = data_raw["article"]
            for i in range(len(data_raw["answers"])):   
                truth = str(ord(data_raw['answers'][i]) - ord('A')) 
                question = data_raw['questions'][i]  
                options = data_raw['options'][i]  
                examples.append(
                    InputExample(
                        example_id=mc_id,
                        question=question,
                        contexts=[article, article, article, article], 
                        endings=[options[0], options[1], options[2], options[3]],
                        label=truth))
                
        return examples

def read_race(path):
    with open(path, 'r', encoding='utf_8') as f:
        data_all = json.load(f)
        article = []
        question = []
        st = []
        ct1 = []
        ct2 = []
        ct3 = []
        ct4 = []
        y = []
        q_id = []
        for instance in data_all:

            ct1.append(' '.join(instance['options'][0]))
            ct2.append(' '.join(instance['options'][1]))
            ct3.append(' '.join(instance['options'][2]))
            ct4.append(' '.join(instance['options'][3]))
            question.append(' '.join(instance['question']))
            # q_id.append(instance['q_id'])
            q_id.append(0)
            art = instance['article']
            l = []
            for i in art: l += i
            article.append(' '.join(l))
            # article.append(' '.join(instance['article']))
            y.append(instance['ground_truth'])
        return article, question, ct1, ct2, ct3, ct4, y, q_id

def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
    pad_token_segment_id=0,
    pad_on_left=False,
    pad_token=0,
    mask_padding_with_zero=True,
    truncation_strategy='longest_first'
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(tqdm.tqdm(examples, desc="convert examples to features")):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_features = []
        # record end positions of two parts which need interaction such as Passage and Question, for later separating them
        pq_end_pos=[]
        for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)): 
            text_a=context
            text_b = example.question + " " + ending
            special_tok_len=3 

            sep_tok_len=1 
            t_q_len=len(tokenizer.tokenize(example.question))
            t_o_len=len(tokenizer.tokenize(text_b))-t_q_len
            context_max_len=max_length-special_tok_len-t_q_len-t_o_len
            t_c_len=len(tokenizer.tokenize(context))
  
            if t_c_len>context_max_len:
                t_c_len=context_max_len
  

            assert(t_q_len+t_o_len+t_c_len<=max_length)

            inputs = tokenizer.encode_plus(
                text_a,
                text_b,
                add_special_tokens=True,
                max_length=max_length,
                truncation_strategy=truncation_strategy
            )  
            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

            assert(len(input_ids[t_c_len+t_q_len+t_o_len:])==special_tok_len)

            t_pq_end_pos=[1 + t_c_len - 1, 1 + t_c_len + sep_tok_len + t_q_len + t_o_len - 1]

            pq_end_pos.append(t_pq_end_pos)
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            pad_token=tokenizer.pad_token_id

            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length) 
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_length
            assert len(attention_mask) == max_length
            assert len(token_type_ids) == max_length
            choices_features.append((input_ids, attention_mask, token_type_ids))
            
        label = label_map[example.label]
    
        features.append(
            InputFeatures(
                example_id=example.example_id,
                choices_features=choices_features,
                label=label,
                pq_end_pos=pq_end_pos
            )
        )

    return features


processors = {
    "race": RaceProcessor,
    "dream": DreamProcessor,
    "mctest": MctestProcessor
}


MULTIPLE_CHOICE_TASKS_NUM_LABELS = {
    "race", 4,
    "dream", 3,
    "mctest", 4
}
