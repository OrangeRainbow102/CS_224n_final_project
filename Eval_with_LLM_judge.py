

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
NUM_RESULTS = 10
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
    biencoder.max_seq_length = MAX_SEQ_LENGTH
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


def main():
    fine_tuned_model = SentenceTransformer('fine_tuned_small_synthetic')

    with open('test_data.pkl', 'rb') as file:
        test_data = pickle.load(file)

    if torch.cuda.is_available():
        print("Warning: No GPU found")

    queries_filepath = "test_queries_100.pkl"
    questions_filepath = "train_data.pkl"

    questions = read_data(questions_filepath)

    queries = read_data(queries_filepath)

    print(queries[:3])

    new_query_index = [(query[0], query[2]) for query in queries]

    sem_list = semantic_search(new_query_index, questions)
    result = [(queries[sem_list[i][0]][2], sem_list[i][1]) for i in range(2)] #range(len(sem_list))

    print(result[:5])

    with open('pretrain_baseline_ce.pkl', 'wb') as file:
        pickle.dump(result, file)


    # print(result)
    #
    # percision = judge(result)
    # print(percision)





if __name__ == '__main__':
    main()

