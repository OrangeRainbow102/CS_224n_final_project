import pickle



def main():
    with open('finetune_NOce_small_synthetic5.pkl', 'rb') as file:
        results1 = pickle.load(file)
    print(len(results1))

    with open('finetune_ce_small_synthetic5.pkl', 'rb') as file:
        results2 = pickle.load(file)
    print(len(results2))

    with open('pretrained_NOce_small_synthetic5.pkl', 'rb') as file:
        results3 = pickle.load(file)
    print(len(results3))

    with open('pretrain_ce_small_synthetic5.pkl', 'rb') as file:
        results4 = pickle.load(file)
    print(len(results4))

    if(results1 == results2):
        print("they are the same")

    if (results3 == results4):
        print("these are also the same")

    with open('baseline_results_1.pkl', 'rb') as file:
        bas_results = pickle.load(file)

    print(bas_results[:3])


    with open('pretrain_all-MiniLM-L6-v2_small_synthetic5.pkl', 'rb') as file:
        mini_results = pickle.load(file)

    with open('old_baseline_small_synthetic5.pkl', 'rb') as file:
        baseline_res = pickle.load(file)

    assert(mini_results != baseline_res)

    print(len(mini_results))
    print(len(baseline_res))



if __name__ == '__main__':
    main()

