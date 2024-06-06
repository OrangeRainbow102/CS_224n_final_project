from together import Together
import sys
import numpy as np
import time

names = {"chase": "query_generation/chase_together_key.txt"}
default_name = "chase"

prompt_prefix = "Respond only yes or no: is the question \""
prompt_middle = "\" relevant for this query \""
prompt_end = "\"?"

with open(names[default_name], 'r+') as f:
    key = f.readlines()[0]
    key = key.replace("\n", "")
    client = Together(api_key=key)


def calc_percision(bool_array):
    return np.sum(bool_array) / len(bool_array)


def judge(query_top_k, user="chase"):
    results = []
    if user != default_name:
        with open(names[user], 'r+') as f:
            key = f.readlines()[0]
            key = key.replace("\n", "")
            client = Together(api_key=key)
    for query, top_k in query_top_k:
        for question in top_k:
            time.sleep(.8)
            msgs = [prompt_prefix + question + prompt_middle + query + prompt_end]
            response = client.chat.completions.create(
                model="mistralai/Mistral-7B-Instruct-v0.3",
                messages=msgs)
            answer = response.choices[0].message.content.lower()
            if "yes" in answer:
                results.append(1)
            elif "no" in answer:
                results.append(0)
            else:
                print("Error adding 0")
                results.append(0)
    percision = calc_percision(results)
    return percision


def main():
    pass


if __name__ == "__main__":
    main()