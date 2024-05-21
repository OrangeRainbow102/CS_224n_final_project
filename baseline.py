

from SciqDataReader import Sciq_reader
from BERT_encoder import BERT_encoder
from sklearn.cluster import KMeans
import numpy as np
from itertools import permutations
from Sentence_Transformer import Sentence_Transformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from util import *



def main():

    train_filename = "SciQ_dataset/train.json"
    test_filename = "SciQ_dataset/test.json"
    train_subset_size = 1000
    test_subset_size = 100
    random_seed = 42
    num_results = 10

    train_reader = Sciq_reader(train_filename, use_random_seed=True, random_seed=random_seed)
    train_reader.read_data()
    train_reader.read_data_to_list()
    train, test, train_labels, test_labels = data_reader.split_train_test(train_percent=train_split, verbose=False)

    #example_query = test[5]
    example_query = "A car travels at a constant speed of 20 meters per second. It accelerates uniformly at a rate of 2 meters per second squared for 10 seconds and then continues at this new constant speed. What is the total distance traveled by the car during the first 20 seconds of its motion?"
    print("Example query is : ", example_query)

    #This line is fast but no need to run over and over -- functions from util.py
    generate_Sentence_embeddings_from_list(train, "generator_train_emb.npy")
    generate_Sentence_embeddings_from_list([example_query], "generator_test_emb.npy")

    # #The only line you have to change to switch from BERT to Sentence is this file that's being loaded
    train_embeddings = np.load("generator_train_emb.npy") # 410 x 384
    test_embeddings = np.load("generator_test_emb.npy") # 105 x 384

    #This line takes a while, do ones and remember params
    # branch, thresh = compute_BIRCH_params(train_embeddings,k,train_labels)
    # print("Best branching parameter is : ", branch, "Best threshold parameter is : ", thresh)
    #

    #train_clust_labels = run_BIRCH(train_embeddings,k,branch,thresh, use_random_seed=True, random_seed=random_seed)

    train_clust_labels = run_Hierarchical(train_embeddings,k, use_random_seed=True, random_seed=random_seed)

    #accuracy, corrected_train_clust_labels = calculate_accuracy(train_clust_labels,train_labels)
    #print("Accuracy of Initial clustering step is : " , accuracy)

    corrected_train_clust_labels = train_clust_labels
    predicted_test_labels = implement_basic_nn(train_embeddings, corrected_train_clust_labels, test_embeddings)
    #nn_accuracy = calc_nn_acc(predicted_test_labels, test_labels)
    #print("Overall neural network accuracy is : ", nn_accuracy)


    result_embeds = get_embeddings(train_embeddings, corrected_train_clust_labels, predicted_test_labels[0])
    print("Number of initial Results is : ")
    print(len(result_embeds))


    search_results = get_results(train, result_embeds, test_embeddings[0], num_results)

    print("Search Results are : ")
    print(search_results)


if __name__ == '__main__':
    main()