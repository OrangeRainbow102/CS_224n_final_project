import pickle


def main():
    # with open("train_queries_100.pkl", 'rb') as file:
    #     train = pickle.load(file)
    #
    # with open("test_queries_100.pkl", 'rb') as file:
    #     test = pickle.load(file)
    #
    # with open("valid_queries_100.pkl", 'rb') as file:
    #     valid = pickle.load(file)
    #
    #
    # print("num train is : ", len(train))
    # print("num test is : ", len(test))
    # print("num valid is : ", len(valid))
    #
    # with open("llm_query_train_sols.pkl", 'rb') as file:
    #     train = pickle.load(file)
    #
    # with open("llm_query_test_sols.pkl", 'rb') as file:
    #     test = pickle.load(file)
    #
    # with open("llm_query_valid_sols.pkl", 'rb') as file:
    #     valid = pickle.load(file)
    #
    #
    # print(train[:3])
    # train_new = read_llm_data("llm_query_train_sols.pkl")
    # print(train_new[:3])
    #
    # print("test length is : ", len(test))

    with open("finetune_marco_ce_large_synthetic5.pkl", 'rb') as file:
        test = pickle.load(file)

    print(len(test))
    print(len(test[0][1]))
    print(test[:3])



def read_llm_data(pickle_name):
    #takes in [(synthetic query, doc, ans), ...] returns [(snthetic query, doc), ...]
    with open(pickle_name, 'rb') as file:
        data = pickle.load(file)

    return [(data[i][0], data[i][1]) for i in range(len(data))]




if __name__ == '__main__':
    main()