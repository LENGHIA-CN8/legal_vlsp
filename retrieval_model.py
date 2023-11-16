from transformers import AutoTokenizer, AutoModel
import torch
import os
import json
import numpy as np
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
from text_utils import chunked_tokens

class Retrieval_model:
    def __init__(self):
        super(Retrieval_model, self).__init__()
        auth_token = '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'
        self.tokenizer = AutoTokenizer.from_pretrained('bkai-foundation-models/vietnamese-bi-encoder', use_auth_token=auth_token)
        self.model = AutoModel.from_pretrained('bkai-foundation-models/vietnamese-bi-encoder', use_auth_token=auth_token)
        
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embedding(self, sen, model, max_length, device='cpu'):  
        encoded_input = chunked_tokens(sen, self.tokenizer, max_length)

        model.to(device)
        for k, v in encoded_input.items():
            encoded_input[k] = v.to(device)
        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)
        if model_output[0].shape[0] > 1:
            print(sen)
        # Perform pooling. In this case, mean pooling.
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings

    def search_related(self, df, query, n=3, pprint=True):
        df = df.copy()
        product_embedding = self.get_embedding(
            query, self.model, 256, 'cuda:1'
        )
        df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, product_embedding.to('cpu')))

        results = (
            df.sort_values("similarity", ascending=False)
            .head(n)
        )
        if pprint:
            print(results.head())
        return results