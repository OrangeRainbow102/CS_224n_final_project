

import pickle
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

def main():
    with open('old_baseline_small_synthetic5.pkl', 'rb') as file:
        old_baseline_res = pickle.load(file)

    print("Old Baseline SacreBLEU Score is : ", calc_BLEU_score(old_baseline_res))

    with open('pretrain_ce_small_synthetic.pkl', 'rb') as file:
        msmarco_res = pickle.load(file)

    print("MSmarco SacreBLEU Score is : ", calc_BLEU_score(msmarco_res))

    with open('finetune_ce_small_synthetic5.pkl', 'rb') as file:
        mscarco_finetune_res = pickle.load(file)

    print("MSmarco finetune SacreBLEU Score is : ", calc_BLEU_score(mscarco_finetune_res))



    with open('pretrain_all-MiniLM-L6-v2_small_synthetic5.pkl', 'rb') as file:
        mini_res = pickle.load(file)

    print("mini SacreBLEU Score is : ", calc_BLEU_score(mini_res))

    with open('pretrain_allenai-specter_small_synthetic5.pkl', 'rb') as file:
        allen_res = pickle.load(file)

    print("allen SacreBLEU Score is : ", calc_BLEU_score(allen_res))







if __name__ == '__main__':
    main()