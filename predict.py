from text_utils import preprocess
from retrieval_model import Retrieval_model
from text_utils import segment_text
import pandas as pd
from bm25_model import BM25Gensim
import json
import ast
import numpy as np

encoder_model = Retrieval_model()
bm25_model_stage1 = BM25Gensim("./bm25/bm25_stage1/")
legal_cleaned_path =  "./processed_data/legal_df_embedding_numpy.csv"
legal_df = pd.read_csv(legal_cleaned_path)
legal_df['embedding'] = legal_df['embedding'].apply(ast.literal_eval)
legal_df['embedding'] = legal_df['embedding'].apply(lambda x: np.array([x]))

def convert_json_to_df(json_data):
    new_list = []
    for data in json_data:
        new_dict = {}
        for k in data.keys():
            if k == 'legal_passages':
                if len(data['legal_passages']) > 1:
                    print(data['legal_passages'])
                new_dict['law_id'] = data['legal_passages'][0]['law_id']
                new_dict['article_id'] = data['legal_passages'][0]['article_id']
            else:
                new_dict[k] = data[k]
        new_list.append(new_dict)
    new_df = pd.DataFrame(new_list)
    return new_df

def agrr_score(row):
    if np.isnan(row['similarity']):
        row['total_score'] = row['bm25_score'] + 0.001
        row['similarity'] = 0.001
        return row
    if np.isnan(row['bm25_score']):
        row['similarity'] = row['similarity'][0][0]
        row['total_score'] = row['similarity'] + 0.001
        row['bm25_score'] = 0.001
        return row
    row['similarity'] = row['similarity'][0][0]
    row['total_score'] = row['bm25_score'] + row['similarity']
    return row

def retrieval_stage(statement, legal_df, bm25_model_stage1, encoder_model):
    query = preprocess(statement).lower()
    top_n, bm25_scores = bm25_model_stage1.get_topk_stage1(query, 10)
    res = legal_df.loc[top_n]
    res['bm25_score'] = bm25_scores
    query = segment_text(statement)
    res2 = encoder_model.search_related(legal_df.copy(), query, 10, False)
    merged_df = pd.merge(res, res2, on=['law_id', 'article_id', 'segment','text', 'bm25_text'], suffixes=('_df1', '_df2'), how='outer')
    merged_df = merged_df.apply(agrr_score, axis=1)
    results = (
            merged_df.sort_values("total_score", ascending=False).head(5)
    )
    
    return results

if __name__ == "__main__":
    with open('./VLSP2023-LTER-Data/test.json') as f:
        test_data = json.load(f)
    test_df = convert_json_to_df(test_data)
    idx = 0
    statement = test_df.iloc[idx]['statement']
    print(statement)
    print(test_df.iloc[idx])
    rank_df = retrieval_stage(statement, legal_df, bm25_model_stage1, encoder_model)
    print(rank_df)