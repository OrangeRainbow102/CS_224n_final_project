
import pickle
import torch
from sentence_transformers import InputExample, datasets
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import wandb
from sentence_transformers import SentenceTransformer, losses
import matplotlib.pyplot as plt


def main():
    with open('training_data_small_synthetic.pkl', 'rb') as file:
        data = pickle.load(file)

    cos_acc1 = []
    cos_acc5 = []
    cos_map10 = []
    cos_pre5 = []
    cos_pre10 = []

    for i in range(len(data)):
        cos_acc1.append(data[i]['cosine_accuracy@1'])
        cos_acc5.append(data[i]['cosine_accuracy@5'])
        cos_map10.append(data[i]['cosine_map@10'])
        cos_pre5.append(data[i]['cosine_precision@5'])
        cos_pre10.append(data[i]['cosine_precision@10'])

    training_steps = [(i + 1)*10 for i in range(len(cos_acc1))]

    print(data[0])

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the data
    ax.plot(training_steps, cos_pre5, label='Cosine Precision k = 10', color='b', linewidth=2, linestyle='-')

    # Customize the plot
    ax.set_title('Training Performance on Evaluation Set', fontsize=16, fontweight='bold')
    ax.set_xlabel('Training Steps', fontsize=14)
    ax.set_ylabel('Cosine Precision', fontsize=14)
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Customize the ticks
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=8)

    # Show the plot
    plt.savefig('cospre10_synthetic_small.png')










if __name__ == '__main__':
    main()