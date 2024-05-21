import json
import numpy as np
import random
import pickle

#This file defines a class that is meant to read in the JEEBench data into more easily handled formats like lists
def main():
    #Example usage of class
    filename = "SciQ_dataset/train.json"
    data_reader = Sciq_reader(filename)
    data_reader.read_data()
    data_reader.read_data_to_list()

    print(len(data_reader.problem_list))


    data = data_reader.problem_list
    #Print a few Examples
    print(data[:5])

    #Print Examples with Solutions
    data_sols = data_reader.prob_list_w_answers
    print(data_sols[:5])

    total_length = 0
    for i in range(len(data)):
        total_length += len(data[i])
    print("Average length of query is : ", total_length / len(data))

    with open('train_data.pkl', 'wb') as file:
        # Serialize and save the list
        pickle.dump(data, file)

    with open('train_data.pkl', 'rb') as file:
        # Load and deserialize the list
        train_data_test = pickle.load(file)

    print(train_data_test[:5])

class Sciq_reader:
    def __init__(self, filename, use_random_seed=False, random_seed = 42):
        self.filename = filename
        self.problem_list = []   #list where each element is a string with problem text
        self.prob_list_w_answers = [] #list where each element is a tuple :  (problem text, subject label)
        self.use_random_seed = use_random_seed
        self.random_seed = random_seed

    def read_data(self):
        with open(self.filename, 'r') as json_file:
            self.data = json.load(json_file)

    def read_data_to_list(self):
        for point in self.data:
            self.problem_list.append(point["question"])
            self.prob_list_w_answers.append((point["question"], point["correct_answer"]))

    def get_rand_subset(self, num_examples):
        shuffled_list = self.problem_list
        random.shuffle(shuffled_list)
        return shuffled_list[:num_examples]


if __name__ == '__main__':
    main()