
import pickle
import torch
from sentence_transformers import InputExample, datasets
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import wandb
from sentence_transformers import SentenceTransformer, losses
import matplotlib.pyplot as plt


def read_data(data, metric):
    met_data = []

    for i in range(len(data)):
        met_data.append(data[i][metric])

        #Examples:
        # cos_acc5.append(data[i]['cosine_accuracy@5'])
        # cos_map10.append(data[i]['cosine_map@10'])
        # cos_pre5.append(data[i]['cosine_precision@5'])
        # cos_pre10.append(data[i]['cosine_precision@10'])
    return met_data
def graph_data(y_data, title, save_path, y_label='Cosine Percision', label='Cosine Percision @5'):
    training_steps = [(i + 1)*10 for i in range(len(y_data))]

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the data
    ax.plot(training_steps, y_data, label=label, color='b', linewidth=2, linestyle='-')

    # Customize the plot
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Training Steps', fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Customize the ticks
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=8)

    # Show the plot
    plt.savefig(save_path)




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

def collect_data(checkpoint_foder, num_checkpoints):
    val_data = load_synthetic_small_data("valid_data.pkl", "valid_queries_100.pkl")

    results = []
    for i in range(num_checkpoints):
        filename = checkpoint_foder + '/checkpoint-' + str((i + 1) * 10)
        model = SentenceTransformer(filename)

        val_queries, val_corpus, val_relevent_docs = get_eval_dictionaries(val_data)
        val_evaluator = InformationRetrievalEvaluator(queries=val_queries, corpus=val_corpus,
                                                      relevant_docs=val_relevent_docs, map_at_k=[10])
        result = model.evaluate(val_evaluator)
        results.append(result)
    return results


def main():
    # mini_raw_data = collect_data('all-MiniLM-L6-v2_checkpoints', num_checkpoints=40)
    # mini_data = read_data(mini_raw_data, 'cosine_precision@5')
    # graph_data(mini_data, title="all-MiniLM-L6-v2 Small Synthetic Training Performance",
    #            save_path="mini_cospre5.png")

    allen_raw_data = collect_data('allenai-specter_checkpoints', num_checkpoints=40)
    allen_data = read_data(allen_raw_data, 'cosine_precision@5')
    graph_data(allen_data, title="AllenAI-SPECTER Small Synthetic Training Performance",
               save_path="allen_cospre5.png")






    # with open('training_data_small_synthetic.pkl', 'rb') as file:
    #     data = pickle.load(file)
    #
    # cos_acc1 = []
    # cos_acc5 = []
    # cos_map10 = []
    # cos_pre5 = []
    # cos_pre10 = []
    #
    # for i in range(len(data)):
    #     cos_acc1.append(data[i]['cosine_accuracy@1'])
    #     cos_acc5.append(data[i]['cosine_accuracy@5'])
    #     cos_map10.append(data[i]['cosine_map@10'])
    #     cos_pre5.append(data[i]['cosine_precision@5'])
    #     cos_pre10.append(data[i]['cosine_precision@10'])
    #
    # training_steps = [(i + 1)*10 for i in range(len(cos_acc1))]
    #
    # print(data[0])
    #
    # # Create the plot
    # fig, ax = plt.subplots(figsize=(10, 6))
    #
    # # Plot the data
    # ax.plot(training_steps, cos_pre5, label='Cosine Precision k = 5', color='b', linewidth=2, linestyle='-')
    #
    # # Customize the plot
    # ax.set_title('MS Marco Small Synthetic Dataset Training Performance', fontsize=16, fontweight='bold')
    # ax.set_xlabel('Training Steps', fontsize=14)
    # ax.set_ylabel('Cosine Precision', fontsize=14)
    # ax.legend(loc='upper right', fontsize=12)
    # ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    #
    # # Customize the ticks
    # ax.tick_params(axis='both', which='major', labelsize=12)
    # ax.tick_params(axis='both', which='minor', labelsize=8)
    #
    # # Show the plot
    # plt.savefig('cospre5_synthetic_small.png')










if __name__ == '__main__':
    main()