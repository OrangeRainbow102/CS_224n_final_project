
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

    #See this : https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss

    model_name = "allenai-specter"
    checkpoint_path = "allenai-specter_checkpoints"
    model_save_name = "fine_tuned_allenai-specter_small_synthetic"

    print("MODEL IS : ", model_name)

    train_data = load_synthetic_small_data("train_data.pkl", "train_queries_100.pkl")
    test_data = load_synthetic_small_data("test_data.pkl", "test_queries_100.pkl")
    val_data = load_synthetic_small_data("valid_data.pkl", "valid_queries_100.pkl")


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dloader, test_examples, val_examples = load_binary_dataset_retriever(train_data, test_data, val_data)

    print(len(train_dloader))
    val_queries, val_corpus, val_relevent_docs = get_eval_dictionaries(val_data)
    val_evaluator = InformationRetrievalEvaluator(queries=val_queries, corpus=val_corpus, relevant_docs=val_relevent_docs, map_at_k=[10])
    #val_evaluator.primary_metric = 'cosine_accuracy@10'


    test_queries, test_corpus, test_relevent_docs = get_eval_dictionaries(test_data)
    test_evaluator = InformationRetrievalEvaluator(queries=test_queries, corpus=test_corpus, relevant_docs=test_relevent_docs, map_at_k=[10])
    #test_evaluator.primary_metric = 'cosine_accuracy@10'



    model = SentenceTransformer(model_name).to(device)
    loss = losses.MultipleNegativesRankingLoss(model)


    num_epochs = 3
    warmup_steps = int(len(train_dloader) * num_epochs * 0.1)


    result_pre_fine_tuning = model.evaluate(test_evaluator)
    print("Pre-Fine Tuning Evaluation on Test Set: ", result_pre_fine_tuning)

    valresult_pre_fine_tunign = model.evaluate(val_evaluator)
    print("Pre-Fine Tuning Eval on Eval Set : ", valresult_pre_fine_tunign)
    print("Primary metric is : ", val_evaluator.primary_metric)

    #model.save('pre-trained_small_synthetic')

    model.fit(train_objectives=[(train_dloader, loss)],
              epochs=num_epochs,
              warmup_steps=warmup_steps,
              show_progress_bar=True,
              evaluator=val_evaluator,
              evaluation_steps=10,
              callback=callback_model,
              checkpoint_save_steps=10,
              checkpoint_path=checkpoint_path)


    result_post_fine_tuning = model.evaluate(test_evaluator)
    print("Post-Fine Tuning Evaluation on Test Set : ", result_post_fine_tuning)
    #
    #
    # #REMEMBER TO UNCOMMENT
    # model.save(model_save_name)


if __name__ == '__main__':
    main()


