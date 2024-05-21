

# importing libraries
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

#Citation: #Code mostly taken from https://www.geeksforgeeks.org/how-to-generate-word-embedding-using-bert/
def main():

    # Example usage
    text_list = [
        "This is the first sentence.",
        "This is the second sentence.",
        "And this is the third sentence."
    ]

    text_list_2 = ["GeeksforGeeks is a computer science portal",
                    "GeeksforGeeks is a technology website"]

    encoder = BERT_encoder()
    problem_embeddings = encoder.generate_embeddings(text_list)

    # print(type(problem_embeddings))
    # print(problem_embeddings.shape)
    print(problem_embeddings)

    cosine_similarity = encoder.calc_cosine_similarity(text_list[0], text_list[1])
    print("Cosine similarity Score for first example is : ", cosine_similarity)

    cosine_similarity = encoder.calc_cosine_similarity(text_list_2[0], text_list_2[1])
    print("Cosine similarity Score for second example is : ", cosine_similarity)


    # ex[0] is physics, ex[1] is physics, ex[2] is chemistry, ex[3] is math
    example_questions = [
        "The diameter of a cylinder is measured using a vernier callipers with no zero error. It is found that the zero of the vernier scale lies between $5.10 cm$ and $5.15 cm$ of the main scale. The vernier scale has 50 division equivalent to $2.45 cm$. The $24^{\\text {th }}$ division of the vernier scale exactly coincides with one of the main scale divisions. The diameter of the cylinder is:\n\n(A) $5.112 cm$\n(B) $5.124 cm$\n(C) $5.136 cm$\n(D) $5.148 cm$",
        "A horizontal stretched string, fixed at two ends, is vibrating in its fifth harmonic according to the equation, $y(x$, $\\mathrm{t})=(0.01 \\mathrm{~m}) \\sin \\left[\\left(62.8 \\mathrm{~m}^{-1}\\right) \\mathrm{x}\\right] \\cos \\left[\\left(628 \\mathrm{~s}^{-1}\\right) \\mathrm{t}\\right]$. Assuming $\\pi=3.14$, the correct statement(s) is (are):\n\n(A) The number of nodes is 5.\n(B) The length of the string is $0.25 \\mathrm{~m}$.\n(C) The maximum displacement of the midpoint of the string its equilibrium position is $0.01 \\mathrm{~m}$.\n(D) The fundamental frequency is $100 \\mathrm{~Hz}$.",
        "Concentrated nitric acid, upon long standing, turns yellow-brown due to the formation of\n\n(A) NO\n(B) $\\mathrm{NO}_{2}$\n(C) $\\mathrm{N}_{2} \\mathrm{O}$\n(D) $\\mathrm{N}_{2} \\mathrm{O}_{4}$",
        "Perpendiculars are drawn from points on the line $\\frac{x+2}{2}=\\frac{y+1}{-1}=\\frac{z}{3}$ to the plane $x+y+z=3$. The feet of perpendiculars lie on the line\n\n(A) $\\frac{x}{5}=\\frac{y-1}{8}=\\frac{z-2}{-13}$\n(B) $\\frac{x}{2}=\\frac{y-1}{3}=\\frac{z-2}{-5}$\n(C) $\\frac{x}{4}=\\frac{y-1}{3}=\\frac{z-2}{-7}$\n(D) $\\frac{x}{2}=\\frac{y-1}{-7}=\\frac{z-2}{5}$"
    ]
    # Example sentence for similarity comparison physics -physics #Similarity Score : 0.862
    cosine_similarity = encoder.calc_cosine_similarity(example_questions[0], example_questions[1])
    print("Cosine similarity Score for phys-phys is : ", cosine_similarity)

    #Physics Chemistry : 0.746
    cosine_similarity = encoder.calc_cosine_similarity(example_questions[0], example_questions[2])
    print("Cosine similarity Score for phys-chem is : ", cosine_similarity)

    #Physics Math : 0.758
    cosine_similarity = encoder.calc_cosine_similarity(example_questions[0], example_questions[3])
    print("Cosine similarity Score for phys-math is : ", cosine_similarity)

    # Chem Math : 0.876
    cosine_similarity = encoder.calc_cosine_similarity(example_questions[2], example_questions[3])
    print("Cosine similarity Score for chem-math is : ", cosine_similarity)


class BERT_encoder:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def generate_embeddings(self, problem_strings):
        #input parameter problem_strings is a list where list[i] = "problem text here"
        #if len(problem_strings) = n, output is a numpy array of shape n x embedding_length

        problem_embeddings = []

        for problem in problem_strings:
            # Tokenize and encode text
            encoding = self.tokenizer.encode_plus(
                problem,
                padding=True,
                truncation=True,
                return_tensors='pt',
                add_special_tokens=True
            )

            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']

            # Calc embeddings
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                prob_embedding = outputs.last_hidden_state.mean(dim=1)  # Average pooling along the sequence length dimension
                problem_embeddings.append(prob_embedding)

        problem_embeddings = torch.cat(problem_embeddings, dim=0)
        return problem_embeddings.numpy()

    def calc_cosine_similarity(self, string_1, string_2):
        #Function for sanity check + debugging : takes in two strings outputs cosine similarity of their embeddings
        list = [string_1, string_2]
        embeddings = self.generate_embeddings(list)
        print("embeddings " , embeddings.shape)
        similarity_score = cosine_similarity(embeddings[0].reshape(1,-1), embeddings[1].reshape(1,-1))
        return similarity_score

if __name__ == '__main__':
    main()
