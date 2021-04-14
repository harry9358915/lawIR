from typing import List, Tuple, Dict, Any

import json
import os
import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer

class longformerMaxpDataset(Dataset):
    def __init__(
        self,
        dataset: str,
        tokenizer: AutoTokenizer,
        mode: str,
        query_max_len: int = 32,
        doc_max_len: int = 256,
        max_input: int = 1280000,
        task: str = 'ranking'
    ) -> None:
        self._dataset = dataset
        self._tokenizer = tokenizer
        self._mode = mode
        self._query_max_len = query_max_len
        self._doc_max_len = doc_max_len
        self._seq_max_len = query_max_len + doc_max_len + 3
        self._max_input = max_input
        self._task = task

        if self._seq_max_len > 4096:
            raise ValueError('query_max_len + doc_max_len + 3 > 4096.')
        if self._mode == 'train':
            dataset = dataset.split(",")
            traindataset,train_id = dataset[0],dataset[1]
            subdirs = os.listdir(traindataset)
            self._queries = {}
            self._docs = {}
            for task_id in subdirs:
                candidates_list = os.listdir(traindataset + "/" + task_id + '/candidates')
                with open(traindataset+ "/" + task_id+ "/base_case.txt", mode="r", encoding="utf-8") as file:
                    context=[]
                    for line in file.readlines():
                        context.append(line.rstrip())
                    context = ' '.join([i for i in context])
                    self._queries[task_id] = context
                for candidatesfile in candidates_list:
                    context=[]
                    with open(traindataset+ "/" + task_id+ '/candidates/' + candidatesfile, mode="r", encoding="utf-8") as file:
                        for line in file.readlines():
                            context.append(line.rstrip())
                        context = ' '.join([i for i in context])
                        self._docs[task_id+os.path.splitext(candidatesfile)[0]] = context
            with open(train_id,'r') as file:
                self._examples = []
                for i,line in enumerate(file):
                    if i>=self._max_input:
                        break
                    line = line.strip().split("\t")
                    line[1] = "{:0>3d}".format(int(line[1]))
                    line[1] = line[0]+line[1]
                    if self._task == 'ranking':
                        self._examples.append({'query_id': line[0], 'doc_pos_id': line[1], 'doc_neg_id': line[2]})
                    elif self._task == 'classification':
                        self._examples.append({'query_id': line[0], 'doc_id': line[1], 'label': int(line[2])})
                    else:
                        raise ValueError('Task must be `ranking` or `classification`.')
        elif self._mode == 'dev':
            dataset = dataset.split(",")
            devdataset,dev_id = dataset[0],dataset[1]
            subdirs = os.listdir(devdataset)
            self._queries = {}
            self._docs = {}
            for task_id in subdirs:
                candidates_list = os.listdir(devdataset+ "/" + task_id + '/candidates')
                with open(devdataset+ "/" + task_id+ "/base_case.txt", mode="r", encoding="utf-8") as file:
                    context=[]
                    for line in file.readlines():
                        context.append(line.rstrip())
                    context = ' '.join([i for i in context])
                    self._queries[task_id] = context
                for candidatesfile in candidates_list:
                    context=[]
                    with open(devdataset+ "/" + task_id+ '/candidates/' + candidatesfile, mode="r", encoding="utf-8") as file:
                        for line in file.readlines():
                            context.append(line.rstrip())
                        context = ' '.join([i for i in context])
                        self._docs[task_id+os.path.splitext(candidatesfile)[0]] = context
            with open(dev_id,'r') as file:
                self._examples = []
                for i,line in enumerate(file):
                    if i>=self._max_input:
                        break
                    line = line.strip().split("\t")
                    line[1] = "{:0>3d}".format(int(line[1]))
                    line[1] = line[0]+line[1]
                    self._examples.append({'label': line[4], 'query_id': line[0], 'doc_id': line[1], 'retrieval_score': float(line[3])})            
        elif self._mode == 'test':
            dataset = dataset.split(",")
            testdataset,test_id = dataset[0],dataset[1]
            subdirs = os.listdir(testdataset)
            self._queries = {}
            self._docs = {}
            for task_id in subdirs:
                candidates_list = os.listdir(testdataset+ "/"+task_id + '/candidates')
                with open(testdataset+"/"+ task_id+ "/base_case.txt", mode="r", encoding="utf-8") as file:
                    context=[]
                    for line in file.readlines():
                        context.append(line.rstrip())
                    context = ' '.join([i for i in context])
                    self._queries[task_id] = context
                for candidatesfile in candidates_list:
                    with open(testdataset+"/"+ task_id+ '/candidates/' + candidatesfile, mode="r", encoding="utf-8") as file:
                        context=[]
                        for line in file.readlines():
                            context.append(line.rstrip())
                        context = ' '.join([i for i in context])
                        self._docs[task_id+os.path.splitext(candidatesfile)[0]] = context
            with open(test_id,'r') as file:
                self._examples = []
                for i,line in enumerate(file):
                    if i>=self._max_input:
                        break
                    line = line.strip().split("\t")
                    line[1] = "{:0>3d}".format(int(line[1]))
                    line[1] = line[0]+line[1]
                    
                    if (i%100)>=25:
                        continue
                    
                    self._examples.append({'query_id': line[0], 'doc_id': line[1], 'retrieval_score': float(line[2])})
        self._count = len(self._examples)

    def pack_bert_features(self, query_tokens: List[str], doc_tokens: List[str]):
        input_tokens = [self._tokenizer.cls_token] + query_tokens + [self._tokenizer.sep_token] + doc_tokens + [self._tokenizer.sep_token]
        input_ids = self._tokenizer.convert_tokens_to_ids(input_tokens)
        segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(doc_tokens) + 1)
        input_mask = [1] * len(input_tokens)

        #longformer attention_mask
        global_attention_mask = [0] * len(input_tokens)
        global_attention_mask[0] = 1

        padding_len = self._seq_max_len - len(input_ids)
        input_ids = input_ids + [self._tokenizer.pad_token_id] * padding_len
        input_mask = input_mask + [0] * padding_len
        segment_ids = segment_ids + [0] * padding_len
        global_attention_mask = global_attention_mask + [0] * padding_len
        
        try:
            assert len(input_ids) == self._seq_max_len
            assert len(input_mask) == self._seq_max_len
            assert len(segment_ids) == self._seq_max_len
            assert len(global_attention_mask) == self._seq_max_len
        except:
            print(len(input_ids), padding_len, self._seq_max_len, len(input_mask), len(segment_ids), len(doc_tokens))
            exit()

        return input_ids, input_mask, segment_ids, global_attention_mask
        
    def collate(self, batch: Dict[str, Any]):
        if self._mode == 'train':
            if self._task == 'ranking':
                input_ids_pos = torch.tensor([item['input_ids_pos'] for item in batch])
                segment_ids_pos = torch.tensor([item['segment_ids_pos'] for item in batch])
                input_mask_pos = torch.tensor([item['input_mask_pos'] for item in batch])
                input_ids_neg = torch.tensor([item['input_ids_neg'] for item in batch])
                segment_ids_neg = torch.tensor([item['segment_ids_neg'] for item in batch])
                input_mask_neg = torch.tensor([item['input_mask_neg'] for item in batch])
                return {'input_ids_pos': input_ids_pos, 'segment_ids_pos': segment_ids_pos, 'input_mask_pos': input_mask_pos,
                        'input_ids_neg': input_ids_neg, 'segment_ids_neg': segment_ids_neg, 'input_mask_neg': input_mask_neg}
            elif self._task == 'classification':
                input_ids = torch.tensor([item['input_ids'] for item in batch])
                segment_ids = torch.tensor([item['segment_ids'] for item in batch])
                input_mask = torch.tensor([item['input_mask'] for item in batch])
                global_attention_mask = torch.tensor([item['global_attention_mask'] for item in batch])
                label = torch.tensor([item['label'] for item in batch])
                return {'input_ids': input_ids, 'segment_ids': segment_ids, 'input_mask': input_mask,'global_attention_mask': global_attention_mask, 'label': label}
            else:
                raise ValueError('Task must be `ranking` or `classification`.')  
        elif self._mode == 'dev':
            query_id = [item['query_id'] for item in batch]
            doc_id = [item['doc_id'] for item in batch]
            label = [item['label'] for item in batch]
            retrieval_score = torch.tensor([item['retrieval_score'] for item in batch])
            input_ids = torch.tensor([item['input_ids'] for item in batch])
            segment_ids = torch.tensor([item['segment_ids'] for item in batch])
            global_attention_mask = torch.tensor([item['global_attention_mask'] for item in batch])
            input_mask = torch.tensor([item['input_mask'] for item in batch])
            return {'query_id': query_id, 'doc_id': doc_id, 'label': label, 'retrieval_score': retrieval_score,
                    'input_ids': input_ids, 'segment_ids': segment_ids, 'input_mask': input_mask,
                    'global_attention_mask': global_attention_mask}
        elif self._mode == 'test':
            query_id = [item['query_id'] for item in batch]
            doc_id = [item['doc_id'] for item in batch]
            retrieval_score = torch.tensor([item['retrieval_score'] for item in batch])
            input_ids = torch.tensor([item['input_ids'] for item in batch])
            segment_ids = torch.tensor([item['segment_ids'] for item in batch])
            global_attention_mask = torch.tensor([item['global_attention_mask'] for item in batch])
            input_mask = torch.tensor([item['input_mask'] for item in batch])
            return {'query_id': query_id, 'doc_id': doc_id, 'retrieval_score': retrieval_score,
                    'input_ids': input_ids, 'segment_ids': segment_ids, 'input_mask': input_mask,
                    'global_attention_mask': global_attention_mask}
        else:
            raise ValueError('Mode must be `train`, `dev` or `test`.')               
    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self._examples[index]
        index=str(index)
        example['query'] = self._queries[example['query_id']]
        if self._mode == 'train' and self._task == 'ranking':
            example['doc_pos'] = self._docs[example['doc_pos_id']]
            example['doc_neg'] = self._docs[example['doc_neg_id']]
        else:
            example['doc'] = self._docs[example['doc_id']]
        if self._mode == 'train':
            if self._task == 'ranking':
                query_tokens = self._tokenizer.tokenize(example['query'])[:self._query_max_len]
                doc_tokens_pos = self._tokenizer.tokenize(example['doc_pos'])#[:self._seq_max_len-len(query_tokens)-3]
                doc_tokens_neg = self._tokenizer.tokenize(example['doc_neg'])#[:self._seq_max_len-len(query_tokens)-3]

                pas_max_len = self._seq_max_len-len(query_tokens)-3
                input_ids_poss, input_mask_poss, segment_ids_poss = [], [], []
                input_ids_negs, input_mask_negs, segment_ids_negs = [], [], []
                for i in range(3):
                    input_ids_pos, input_mask_pos, segment_ids_pos = self.pack_bert_features(query_tokens, doc_tokens_pos[i*pas_max_len:(i+1)*pas_max_len])
                    input_ids_neg, input_mask_neg, segment_ids_neg = self.pack_bert_features(query_tokens, doc_tokens_neg[i*pas_max_len:(i+1)*pas_max_len])
                    
                    input_ids_poss += input_ids_pos
                    input_mask_poss += input_mask_pos
                    segment_ids_poss += segment_ids_pos
                    input_ids_negs += input_ids_neg
                    input_mask_negs += input_mask_neg
                    segment_ids_negs += segment_ids_neg
                return {'input_ids_pos': input_ids_poss, 'segment_ids_pos': segment_ids_poss, 'input_mask_pos': input_mask_poss,
                        'input_ids_neg': input_ids_negs, 'segment_ids_neg': segment_ids_negs, 'input_mask_neg': input_mask_negs}
            elif self._task == 'classification':
                query_tokens = self._tokenizer.tokenize(example['query'])[:self._query_max_len]
                doc_tokens = self._tokenizer.tokenize(example['doc'])#[:self._seq_max_len-len(query_tokens)-3]
                
                pas_max_len = self._seq_max_len-len(query_tokens)-3
                input_idss, input_masks, segment_idss, global_attention_masks = [], [], [], []
                for i in range(3):
                    input_ids, input_mask, segment_ids, global_attention_mask = self.pack_bert_features(query_tokens, doc_tokens[i*pas_max_len:(i+1)*pas_max_len])

                    input_idss += input_ids
                    input_masks += input_mask
                    segment_idss += segment_ids
                    global_attention_masks +=global_attention_mask
                return {'input_ids': input_idss, 'segment_ids': segment_idss, 'input_mask': input_masks, 'global_attention_mask': global_attention_masks, 'label': example['label']}
            else:
                raise ValueError('Task must be `ranking` or `classification`.')
        elif self._mode == 'dev':
            query_tokens = self._tokenizer.tokenize(example['query'])[:self._query_max_len]
            doc_tokens = self._tokenizer.tokenize(example['doc'])#[:self._seq_max_len-len(query_tokens)-3]

            pas_max_len = self._seq_max_len-len(query_tokens)-3
            input_idss, input_masks, segment_idss,global_attention_masks = [], [], [],[]
            for i in range(3):
                input_ids, input_mask, segment_ids, global_attention_mask = self.pack_bert_features(query_tokens, doc_tokens[i*pas_max_len:(i+1)*pas_max_len])

                input_idss += input_ids
                input_masks += input_mask
                segment_idss += segment_ids
                global_attention_masks += global_attention_mask
            return {'query_id': example['query_id'], 'doc_id': example['doc_id'], 'label': example['label'], 'retrieval_score': example['retrieval_score'],
                    'input_ids': input_idss, 'input_mask': input_masks, 'segment_ids': segment_idss,'global_attention_mask': global_attention_masks}
        elif self._mode == 'test':
            query_tokens = self._tokenizer.tokenize(example['query'])[:self._query_max_len]
            doc_tokens = self._tokenizer.tokenize(example['doc'])#[:self._seq_max_len-len(query_tokens)-3]

            pas_max_len = self._seq_max_len-len(query_tokens)-3
            input_idss, input_masks, segment_idss, global_attention_masks = [], [], [], []
            for i in range(3):
                input_ids, input_mask, segment_ids, global_attention_mask = self.pack_bert_features(query_tokens, doc_tokens[i*pas_max_len:(i+1)*pas_max_len])

                input_idss += input_ids
                input_masks += input_mask
                segment_idss += segment_ids
                global_attention_masks += global_attention_mask
            return {'query_id': example['query_id'], 'doc_id': example['doc_id'], 'retrieval_score': example['retrieval_score'],
                    'input_ids': input_idss, 'input_mask': input_masks, 'segment_ids': segment_idss, 'global_attention_mask': global_attention_masks}
        else:
            raise ValueError('Mode must be `train`, `dev` or `test`.')

    def __len__(self) -> int:
        return self._count