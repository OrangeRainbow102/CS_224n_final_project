

from SciqDataReader import Sciq_reader
import numpy as np
from Sentence_Transformer import Sentence_Transformer
from util import *
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from nltk.translate.bleu_score import sentence_bleu


def main():

    train_pickle = "llm_query_train_sols.pkl"
    test_pickle = "llm_query_test_sols.pkl"
    num_results = 5
    # train_filename = "SciQ_dataset/train.json"
    # test_filename = "SciQ_dataset/test.json"
    # train_subset_size = 1083
    # test_subset_size = 97
    # random_seed = 42
    # num_results = 15
    #
    # train_reader = Sciq_reader(train_filename, use_random_seed=True, random_seed=random_seed)
    # train_reader.read_data()
    # train_reader.read_data_to_list()
    # train_data = train_reader.problem_list[:train_subset_size]

    # test_reader = Sciq_reader(test_filename, use_random_seed=True, random_seed=random_seed)
    # test_reader.read_data()
    # test_reader.read_data_to_list()
    # test_data = test_reader.problem_list[:test_subset_size]

    with open(train_pickle, 'rb') as file:
        train_raw = pickle.load(file)

    with open(test_pickle, 'rb') as file:
        test_raw = pickle.load(file)


    train_data = [train_raw[i][1] for i in range(len(train_raw))]
    test_data = [test_raw[i][0] for i in range(len(test_raw))]

    train_subset_size = len(train_data)
    test_subset_size = len(test_data)
    print("Length of train set : ", len(train_data))
    print("Ex of train is : ", train_data[:3])
    print("Length of test set : ", len(test_data))
    print("Ex of test is : ", test_data[:3])

    #This line is fast but no need to run over and over -- functions from util.py
    generate_Sentence_embeddings_from_list(train_data, "SBERT_subset_train_emb.npy")
    generate_Sentence_embeddings_from_list(test_data, "SBERT_subset_test_emb.npy")

    # #The only line you have to change to switch from BERT to Sentence is this file that's being loaded
    train_embeddings = np.load("SBERT_subset_train_emb.npy") # 410 x 384
    test_embeddings = np.load("SBERT_subset_test_emb.npy") # 105 x 384

    print("train info : ")
    print(type(train_embeddings))
    print(train_embeddings.shape)

    print("test info : ")
    print(type(test_embeddings))
    print(test_embeddings.shape)

    scores_matrix = np.zeros((test_subset_size, train_subset_size))
    for i in range(test_subset_size):
        if i % 5 == 0:
            print("We are calculating for query number ", i, "in test dataset of length ", test_subset_size)
        for j in range(train_subset_size):
            if j % 1000 == 0:
                print("We are calculating for query number ", i, "iteration ", j ,"in test dataset of length ", test_subset_size)

            score = cosine_similarity(test_embeddings[i].reshape(1, -1), train_embeddings[j].reshape(1, -1))
            scores_matrix[i][j] = score[0,0]

    print(scores_matrix.shape)

    matches_indicies = get_top_n_indicies(scores_matrix, num_results)

    print("Shape of matrch indicies : ", matches_indicies.shape)

    results = construct_matches(train_data, test_data, matches_indicies)
    #
    # print(type(results))
    # print(len(results))
    # print(results[:5])

    with open('old_baseline_large_synthetic5.pkl', 'wb') as file:
        # Serialize and save the list
        pickle.dump(results, file)

    #print("BLEU Baseline Score is : ", calc_BLEU_score(results))
    print(results[:3])
    print(len(results))
    print(len(results[0][1]))

def calc_BLEU_score(results):
    import sacrebleu
    from evaluate import load
    sacrebleu = load("sacrebleu")
    # Results is a list of tuples where each tuple is of the form (query, [result1, result2, ...])
    BLEU_total = 0
    num_results_per_test = len(results[0][1])
    num_tests = len(results)

    for i in range(len(results)):
        query = results[i][0]
        #print(query)
        matches = results[i][1]
        #print(matches)
        for j in range(len(matches)):
            predictions = matches[j]
            #print(predictions)
            res = sacrebleu.compute(predictions=[predictions], references=[[query]])
            BLEU_total += res["score"]
    return BLEU_total / (num_tests * num_results_per_test)





if __name__ == '__main__':
    main()


