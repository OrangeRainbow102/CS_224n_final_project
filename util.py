

from SciqDataReader import Sciq_reader
import numpy as np
from itertools import permutations
from Sentence_Transformer import Sentence_Transformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.neural_network import MLPClassifier
from BERT_encoder import BERT_encoder
from sklearn.metrics.pairwise import cosine_similarity


def generate_BERT_embeddings_from_list(problem_list, save_name):
    #Uses the class JEEBench_reader defined in JEEBench_reader.py to read in data
    #Uses the class BERT_encoder defined in BERT_encoder.py to generate the embeddings
    #saves the embeddings to a numpy matrix of dimensions n x embedding length -- for JEEBench data is 515 x 768

    encoder = BERT_encoder()
    problem_embeddings = encoder.generate_embeddings(problem_list)
    np.save(save_name, problem_embeddings)


def generate_Sentence_embeddings_from_data(filename, save_name):
    #Uses the class JEEBench_reader defined in JEEBench_reader.py to read in data
    #Uses the class BERT_encoder defined in BERT_encoder.py to generate the embeddings
    #saves the embeddings to a numpy matrix of dimensions n x embedding length -- for JEEBench data is 515 x 768

    data_reader = Sciq_reader(filename)
    data_reader.read_data()
    data_reader.read_data_to_list()

    encoder = Sentence_Transformer()
    problem_embeddings = encoder.generate_embeddings(data_reader.problem_list)
    np.save(save_name, problem_embeddings)

def generate_Sentence_embeddings_from_list(problem_list, save_name):
    encoder = Sentence_Transformer()
    problem_embeddings = encoder.generate_embeddings(problem_list)
    np.save(save_name, problem_embeddings)


def calculate_accuracy(predicted_labels, actual_labels):
    #We don't know the mapping of labels outputted by k means and the actual labels given in the dataset
    #This function checks all the possible mappings and outputs the accuracy for the optimal mapping

    accuracies = []
    corrected_labels = []
    map = [0,1,2]
    all_mappings = list(permutations(map))
    for i in range(len(all_mappings)):
        accuracy = calc_accuracy_for_mapping(predicted_labels, actual_labels, all_mappings[i])[0]
        corrected_label = calc_accuracy_for_mapping(predicted_labels, actual_labels, all_mappings[i])[1]
        accuracies.append(accuracy)
        corrected_labels.append(corrected_label)

    max_index = find_max_index(accuracies)
    max_mapping = corrected_labels[max_index]

    return max(accuracies), max_mapping

def find_max_index(list):
    max_accuracy = 0
    max_index = 0
    for i in range(len(list)):
        if (list[i] > max_accuracy):
            max_accuracy = list[i]
            max_index = i
    return max_index


def calc_accuracy_for_mapping(predicted_labels, actual_labels, mapping):
    #Mapping is a tuple . For example, (1,0,2) means a zero in predicted labels maps to 1 in actual labels,
    #1 in predicted labels maps to 0 in actual labels, etc...
    n = predicted_labels.shape[0]
    corrected_predicted_labels = []
    for i in range(n):
        if predicted_labels[i] == 0:
            corrected_predicted_labels.append(mapping[0])
        elif predicted_labels[i] == 1:
            corrected_predicted_labels.append(mapping[1])
        else:
            corrected_predicted_labels.append(mapping[2])

    num_correct = 0

    for i in range(n):
        if corrected_predicted_labels[i] == actual_labels[i]:
            num_correct += 1
    return (num_correct / n, corrected_predicted_labels)

def run_Hierarchical(embeddings, k, use_random_seed=False, random_seed=42):

    if use_random_seed:
        np.random.seed(random_seed)
    # Create KMeans instance
    hierarchical_cluster = AgglomerativeClustering(n_clusters=k, linkage='ward')

    # Fit the data to the model
    predicted_labels = hierarchical_cluster.fit_predict(embeddings)

    return predicted_labels

def run_BIRCH(embeddings, k, branch, thresh, use_random_seed=True, random_seed=42):
    if (use_random_seed):
        np.random.seed(random_seed)
    # Creating the BIRCH clustering model
    model = Birch(branching_factor=branch, n_clusters=k, threshold=thresh)

    # Fit the data (Training)
    model.fit(embeddings)

    # Predict the same data
    pred = model.predict(embeddings
                         )
    return pred


def compute_BIRCH_params(embeddings, k, actual_labels):
    #Search for best BIRCH parameters:
    best_accuracy = 0
    for i in range(10):
        for j in range(10):
            branch = i*10 + 2
            thresh = 0.1 + 0.1*(j)
            predicted_labels = run_BIRCH(embeddings, k, branch, thresh)

            accuracy = calculate_accuracy(predicted_labels, actual_labels)[0]
            #print("accuracy for BIRCH is : ", accuracy)
            print("for i : ", 2, "for j : ", j, "the parameters are branching factor: ", branch, "threshhold : ", thresh, "Accuracy: ", accuracy)
            if accuracy > best_accuracy:
                best_params = (branch, thresh, accuracy)
                best_accuracy = accuracy
    best_branch = best_params[0]
    best_thresh = best_params[1]
    return best_branch, best_thresh

def implement_basic_nn(train, train_labels, test):
    NN = MLPClassifier()
    NN.fit(train, train_labels)
    predicted_labels = NN.predict(test)
    return predicted_labels


def calc_nn_acc(pred_labels, true_labels):
    n_correct = 0
    for i in range(len(true_labels)):
        if pred_labels[i] == true_labels[i]:
            n_correct += 1
    return n_correct / len(true_labels)



def get_embeddings(train_embeddings, corrected_train_clust_labels, index):
    results = np.zeros(train_embeddings.shape[1])
    for i in range(len(corrected_train_clust_labels)):
        if corrected_train_clust_labels[i] == index:
            results = np.vstack((results, train_embeddings[i]))
    return results[1:]

# def get_embeddings(train_embeddings, corrected_train_clust_labels, index):
#     results = np.zeros(train_embeddings.shape[1])
#     for i in range(len(corrected_train_clust_labels)):
#         if corrected_train_clust_labels[i] == index:
#             results = np.vstack((results, train_embeddings[i]))
#     return results[1:]

def get_embeddings(train_embeddings, corrected_train_clust_labels, index):
    results = []
    for i in range(len(corrected_train_clust_labels)):
        if corrected_train_clust_labels[i] == index:
            results.append((train_embeddings[i], i))
    return results

def get_results(train, embedding_matches, test_embedding, num_results):
    prelim_results = []
    results = []

    for i in range(len(embedding_matches)):
        score = calc_cosine_similarity(test_embedding, embedding_matches[i][0])
        prelim_results.append((score, embedding_matches[i][1]))


    prelim_sorted = sorted(prelim_results, key = lambda x : x[0])

    top_results = prelim_sorted[-num_results : ]

    for i in range(len(top_results)):
        idx = top_results[i][1]
        results.append(train[idx])
    return results

def calc_cosine_similarity(embedding1, embedding2):
    return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))


def get_top_n_indicies(np_array, n):
    #n is the top number of matches
    #Assumes that the i, jth entry is the cosine similarity between the ith test vector and the jth training vector
    best_matches = np.argsort(np_array, axis=1)
    return best_matches[:, -n:]

def construct_matches(train_data, test_data, top_indicies):
    num_test = top_indicies.shape[0]
    num_matches = top_indicies.shape[1]
    results = []
    for i in range(num_test):
        matches = []
        for j in range(num_matches):
            match = top_indicies[i][j]
            matches.append(train_data[match])
        results.append((test_data[i], matches))
    return results