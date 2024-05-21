import json
import numpy as np
import random
import pickle

#This file defines a class that is meant to read in the JEEBench data into more easily handled formats like lists
def main():
    #Example usage of class
    train_file = "SciQ_dataset/train.json"
    test_file = "SciQ_dataset/test.json"
    valid_file = "SciQ_dataset/valid.json"

    train_reader = Sciq_reader(train_file)
    train_reader.read_data()
    train_reader.read_data_to_list()
    train_data = train_reader.problem_list
    train_sols = train_reader.prob_list_w_answers

    test_reader = Sciq_reader(test_file)
    test_reader.read_data()
    test_reader.read_data_to_list()
    test_data = test_reader.problem_list
    test_sols = test_reader.prob_list_w_answers

    valid_reader = Sciq_reader(valid_file)
    valid_reader.read_data()
    valid_reader.read_data_to_list()
    valid_data = valid_reader.problem_list
    valid_sols = valid_reader.prob_list_w_answers

    # print("Length of Training Dataset is : ", len(train_reader.problem_list))
    #
    # train_data = train_reader.problem_list
    #Print a few Examples
    print("First few Training Examples are : ", train_data[:5])
    print("First few Testing Examples are : ", test_data[:5])
    print("First few Validation Examples are : ", valid_data[:5])

    print("First few Training Examples Sols are : ", train_sols[:5])
    print("First few Testing Examples Sols are : ", test_sols[:5])
    print("First few Validation Sols are : ", valid_sols[:5])
    #
    # #Print Examples with Solutions
    # train_sols = train_reader.prob_list_w_answers
    # print(train_sols[:5])

    with open('train_data.pkl', 'wb') as file:
        # Serialize and save the list
        pickle.dump(train_data, file)

    with open('test_data.pkl', 'wb') as file:
        # Serialize and save the list
        pickle.dump(test_data, file)

    with open('valid_data.pkl', 'wb') as file:
        # Serialize and save the list
        pickle.dump(valid_data, file)

    with open('train_sols.pkl', 'wb') as file:
        # Serialize and save the list
        pickle.dump(train_sols, file)

    with open('test_sols.pkl', 'wb') as file:
        # Serialize and save the list
        pickle.dump(test_sols, file)

    with open('valid_sols.pkl', 'wb') as file:
        # Serialize and save the list
        pickle.dump(valid_sols, file)


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
        random.seed(self.random_seed)
        shuffled_list = self.problem_list
        random.shuffle(shuffled_list)
        return shuffled_list[:num_examples]


if __name__ == '__main__':
    main()