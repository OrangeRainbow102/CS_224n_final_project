"""
Author: Mack Smith

This script contains the retriever and re-ranker implementations

For retrieving, we use an efficient Bi-Encoder to find similar questions from our 
collection of practice problems

For reranking, we re-rank the hits from the Bi-Encoder using a Cross-Encoder
"""

from sentence_transformers import SentenceTransformer, CrossEncoder, util
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction import _stop_words
import numpy as np
import pickle
import torch
import string

MAX_SEQ_LENGTH = 256
NUM_RESULTS = 10
TOP_K = 32

def read_questions(questions_filepath):
    questions = []
    with open(questions_filepath, 'rb') as file:
        data = pickle.load(file)
        questions = [question for question in data]

    return questions


def read_queries(queries_filepath):
    queries = []
    with open(queries_filepath, 'rb') as file:
        data = pickle.load(file)
        queries = [(query[1], query[2]) for query in data]

    return queries


def tokenizer(text):
    tokenized_doc = []
    for token in text.lower().split():
        token = token.strip(string.punctuation)

        if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
            tokenized_doc.append(token)

    return tokenized_doc


def lexical_search(queries, questions):
    lex_list = []

    tokenized_corpus = []
    for question in questions:
        tokenized_corpus.append(tokenizer(question))

    bm25 = BM25Okapi(tokenized_corpus)

    for _, new_query in queries:
        bm25_scores = bm25.get_scores(tokenizer(new_query))
        top_n = np.argpartition(bm25_scores, -NUM_RESULTS)[-NUM_RESULTS:]
        bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
        bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)

        lex_list.append((new_query, [questions[hit['corpus_id']] for hit in bm25_hits[NUM_RESULTS:]]))

    return lex_list


def semantic_search(queries, questions):
    sem_list = []

    biencoder = SentenceTransformer("msmarco-distilbert-base-v4")
    biencoder.max_seq_length = MAX_SEQ_LENGTH
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    corpus_embeddings = biencoder.encode(questions, convert_to_tensor=True)

    for _, new_query in queries:
        question_embedding = biencoder.encode(new_query, convert_to_tensor=True)
        hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=TOP_K)
        hits = hits[0]

        cross_inp = [[new_query, questions[hit['corpus_id']]] for hit in hits]
        cross_scores = cross_encoder.predict(cross_inp)

        for idx in range(len(cross_scores)):
            hits[idx]['cross-score'] = cross_scores[idx]

        sem_list.append((new_query, [questions[hit['corpus_id']] for hit in hits[:NUM_RESULTS]]))
    
    return sem_list

def main():
    if torch.cuda.is_available():
        print("Warning: No GPU found")
    
    queries_filepath = "test_queries_100.pkl"
    questions_filepath = "train_data.pkl"

    questions = read_questions(questions_filepath)
    queries = read_queries(queries_filepath)

    lex_list = lexical_search(queries, questions)
    sem_list = semantic_search(queries, questions)

    with open('lexical_search_results.pkl', 'wb') as file:
        pickle.dump(lex_list, file)

    with open('semantic_search_results.pkl', 'wb') as file:
        pickle.dump(sem_list, file)


if __name__ == '__main__':
    main()