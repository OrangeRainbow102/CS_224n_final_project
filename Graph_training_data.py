
import pickle
import torch
from sentence_transformers import InputExample, datasets
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import wandb
from sentence_transformers import SentenceTransformer, losses



def main():
    with open('training_data_small_synthetic.pkl', 'rb') as file:
        data = pickle.load(file)

    cos_acc10 = []
    cos_map10 = []

    print(data[:2])









if __name__ == '__main__':
    main()