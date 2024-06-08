

from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction import _stop_words
import numpy as np
import pickle
import torch
import string
# from llmjudge import judge

MAX_SEQ_LENGTH = 256
NUM_RESULTS = 5
TOP_K = 32

def read_data(filepath):
    result = []
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
        result = [entry for entry in data]

    return result


def tokenizer(text):
    tokenized_doc = []
    for token in text.lower().split():
        token = token.strip(string.punctuation)

        if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
            tokenized_doc.append(token)

    return tokenized_doc



def semantic_search(queries, questions, model="msmarco-distilbert-base-v4"):
    sem_list = []

    biencoder = SentenceTransformer(model)
    #biencoder.max_seq_length = MAX_SEQ_LENGTH
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    corpus_embeddings = biencoder.encode(questions, convert_to_tensor=True)

    for index, query in queries:
        question_embedding = biencoder.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=TOP_K)
        hits = hits[0]

        cross_inp = [[query, questions[hit['corpus_id']]] for hit in hits]
        cross_scores = cross_encoder.predict(cross_inp)

        for idx in range(len(cross_scores)):
            hits[idx]['cross-score'] = cross_scores[idx]

        hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
        sem_list.append((index, [questions[hit['corpus_id']] for hit in hits[:NUM_RESULTS]]))

    return sem_list

def semantic_search_No_Cross_Encoder(queries, questions, model="msmarco-distilbert-base-v4"):
    sem_list = []

    biencoder = SentenceTransformer(model)
    biencoder.max_seq_length = MAX_SEQ_LENGTH

    corpus_embeddings = biencoder.encode(questions, convert_to_tensor=True)

    for index, query in queries:
        question_embedding = biencoder.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=TOP_K)
        hits = hits[0]
        sem_list.append((index, [questions[hit['corpus_id']] for hit in hits[:NUM_RESULTS]]))
    return sem_list

def read_llm_data(pickle_name):
    #takes in [(synthetic query, doc, ans), ...] returns [(snthetic query, doc), ...]
    with open(pickle_name, 'rb') as file:
        data = pickle.load(file)

    return [(data[i][0], data[i][1]) for i in range(len(data))]

def main():

    train_raw = read_llm_data("llm_query_train_sols.pkl")
    test_raw = read_llm_data("llm_query_test_sols.pkl")
    valid_raw = read_llm_data("llm_query_valid_sols.pkl")

    queries = [test_raw[i][0] for i in range(len(test_raw))]
    questions = [entry[1] for entry in train_raw]
    new_query_index = [(i, test_raw[i][0]) for i in range(len(test_raw))]

    print(questions[:3])
    print(new_query_index[:3])


    # #EXPERIMENT 1: Finetuned WITH CROSS ENCODER
    # sem_list = semantic_search(new_query_index, questions)
    #
    # result = [(queries[sem_list[i][0]], sem_list[i][1]) for i in range(len(sem_list))] #range(len(sem_list))
    # with open('finetune_marco_ce_large_synthetic5.pkl', 'wb') as file:
    #     pickle.dump(result, file)
    #
    # print(result[:3])

    # # #EXPERIMENT 2: Finetuned NO CROSS ENCODER
    # sem_list = semantic_search(new_query_index, questions, model='fine_tuned_marco_LARGEmodel2')
    #
    # result = [(queries[sem_list[i][0]], sem_list[i][1]) for i in range(len(sem_list))] #range(len(sem_list))
    # with open('realfinetune_marco_ce_large_synthetic5.pkl', 'wb') as file:
    #     pickle.dump(result, file)
    #
    # print(result[:3])



    # #EXPERIMENT 3: NO FINETUNE NO CROSS ENCODER
    sem_list = semantic_search_No_Cross_Encoder(new_query_index, questions)

    result = [(queries[sem_list[i][0]], sem_list[i][1]) for i in range(len(sem_list))] #range(len(sem_list))
    with open('pretrain_marco_NOce_large_synthetic5.pkl', 'wb') as file:
        pickle.dump(result, file)
    print(result[:3])


    # #EXPERIMENT 4: Finetuned NO CROSS ENCODER
    sem_list = semantic_search_No_Cross_Encoder(new_query_index, questions, model='fine_tuned_marco_LARGEmodel2')

    result = [(queries[sem_list[i][0]], sem_list[i][1]) for i in range(len(sem_list))] #range(len(sem_list))
    with open('finetune_marco_NOce_large_synthetic5.pkl', 'wb') as file:
        pickle.dump(result, file)

    print(result[:3])





if __name__ == '__main__':
    main()

