from transformers import AutoTokenizer, AutoModel
import torch
import os
import json
import numpy as np
import pandas as pd
import re
from glob import glob
from nltk import word_tokenize
import string
from vncorenlp import VnCoreNLP

rdrsegmenter = VnCoreNLP(address="http://172.26.33.174", port=2311)
print('Vncorenlp loaded')

def segment_text(sen):
    ##Segment
    seg = rdrsegmenter.tokenize(sen)
    if len(seg) == 0:
        sen = ' '
    else: 
        seg = [item for sublist in seg for item in sublist]
        sen = ' '.join(seg)
    return sen

def preprocess(x, max_length=-1, remove_puncts=False):
    x = nltk_tokenize(x)
    x = x.replace("\n", " ")
    if remove_puncts:
        x = "".join([i for i in x if i not in string.punctuation])
    if max_length > 0:
        x = " ".join(x.split()[:max_length])
    return x

def nltk_tokenize(x):
    return " ".join(word_tokenize(strip_context(x))).strip()

def strip_context(text): 
    text = text.replace('\n', ' ') 
    text = re.sub(r'\s+', ' ', text) 
    text = text.strip() 
    return text

def batched(input_id_chunks, mask_chunk, n):
    """Batch data into tuples of length n. The last batch may be shorter."""
    # batched('ABCDEFG', 3) --> ABC DEF G
    input_id_chunks = list(input_id_chunks)
    mask_chunk = list(mask_chunk)
    
    if n < 1:
        raise ValueError('n must be at least one')
    
    for i in range(len(input_id_chunks)):
        input_id_chunks[i] = torch.cat([torch.Tensor([0]), input_id_chunks[i], torch.Tensor([2])])
        mask_chunk[i] = torch.cat([torch.Tensor([1]), mask_chunk[i], torch.Tensor([1])])
        pad_len = n - input_id_chunks[i].shape[0]
        if pad_len > 0:
            input_id_chunks[i] = torch.cat([
                input_id_chunks[i], torch.Tensor([1] * pad_len)
            ])
            mask_chunk[i] = torch.cat([
                mask_chunk[i], torch.Tensor([0] * pad_len)
            ])
    input_ids = torch.stack(input_id_chunks)
    attention_mask = torch.stack(mask_chunk)
    input_dict = {
        'input_ids': input_ids.long(),
        'attention_mask': attention_mask.int()
    }
    return input_dict
    
def chunked_tokens(text, tokenizer, chunk_length):
    tokens = tokenizer(text, add_special_tokens=False, return_tensors='pt')
    
    input_id_chunks = tokens['input_ids'][0].split(chunk_length - 2)
    mask_chunk = tokens['attention_mask'][0].split(chunk_length - 2)
    input_dict = batched(input_id_chunks, mask_chunk, chunk_length)
    return input_dict