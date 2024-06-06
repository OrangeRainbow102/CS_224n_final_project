

import pickle
import torch
from sentence_transformers import InputExample, datasets
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import wandb
from sentence_transformers import SentenceTransformer, losses



import pickle
import torch
from sentence_transformers import InputExample, datasets
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import wandb
from sentence_transformers import SentenceTransformer, losses
def load_synthetic_small_data(orig_data, rewritten_data):
    with open(rewritten_data, 'rb')as file:
        rewrittens = pickle.load(file)

    with open(orig_data, 'rb') as file:
        orig = pickle.load(file)


    rewrittens.sort(key = lambda x: x[0])

    data = [] #Formatted (Synthetic Query, Document)
    for i in range(len(rewrittens)):
        data.append((rewrittens[i][2], orig[rewrittens[i][0]]))
    return data

def get_eval_dictionaries(eval_data):
    #data is of type [(query1, doc1), (query2, doc2), ...]
    #returns a dictionary mapping query keys to queries
    queries = {}
    documents = {}
    for i in range(len(eval_data)):
        queries[str(i)] = eval_data[i][0]
        documents[str(i)] = eval_data[i][1]

    relevent_docs = {}
    for i in range(len(eval_data)):
        relevent_docs[str(i)] = set(str(i))

    return queries, documents, relevent_docs

def load_binary_dataset_retriever(train_data, test_data, val_data, batch_size=8):

    train_examples = []
    # Creating the Input Examples with format (anchor, positive, negative)
    for entry in train_data:
        train_examples.append(InputExample(texts=[entry[0], entry[1]]))

    test_examples = []
    for entry in test_data:
        test_examples.append(InputExample(texts=[entry[0], entry[1]]))

    val_examples = []
    for entry in val_data:
        val_examples.append(InputExample(texts=[entry[0], entry[1]]))

    train_dataloader = datasets.NoDuplicatesDataLoader(train_examples, batch_size=batch_size)

    return train_dataloader, test_examples, val_examples


def callback_model(score, epoch, steps):
    wandb.log({"train/epoch": epoch,
               "train/steps": steps,
               "val/score": score})
def main():

    val_data = load_synthetic_small_data("valid_data.pkl", "valid_queries_100.pkl")

    results = []
    for i in range(40):
        filename = 'training_small_synthetic_checkpoints/checkpoint-' + str((i + 1) * 10)
        model = SentenceTransformer(filename)

        val_queries, val_corpus, val_relevent_docs = get_eval_dictionaries(val_data)
        val_evaluator = InformationRetrievalEvaluator(queries=val_queries, corpus=val_corpus, relevant_docs=val_relevent_docs, map_at_k=[10])
        result = model.evaluate(val_evaluator)

        results.append(result)

    with open('training_data_small_synthetic.pkl', 'wb') as file:
        pickle.dump(results, file)






if __name__ == '__main__':
    main()