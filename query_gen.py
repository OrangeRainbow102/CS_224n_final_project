from together import Together
from multiprocessing import Process
import sys
import numpy as np
import time
import os
import pickle

topic_prefix = "As briefly as possible state the topic of the following question and answer pairing(add no filler text); "
rephrase_prompt = "Briefly Rephrase the following question one time(add no filler text): "
names = {"chase" : "query_generation/chase_together_key.txt", "ron": "query_generation/ron_together_key.txt"}
file_name = "valid_sols.pkl"
def rephrase(rephrase_data, pid):
    #with open(file, "rb") as f:
    #    rephrase_data = pickle.load(f)
    with open(names["chase"], 'r+') as f:
        key = f.readlines()[0]
        key = key.replace("\n", "")
        client_chase = Together(api_key=key)
    with open(names["ron"], 'r+') as f:
        key = f.readlines()[0]
        key = key.replace("\n", "")
        client_ron = Together(api_key=key)
    results = []
    if os.path.isfile("llm_query_" + str(pid) +"_" + file_name):
        with open("llm_query_" + str(pid) +"_" + file_name, "rb") as f:
            results = pickle.load(f)
    while len(results) < len(rephrase_data):
        question, ans = rephrase_data[len(results)]
        time.sleep(.1)
        if True:
            client = client_chase
        else:
            client = client_ron
        prompt = topic_prefix + " Question: " + question + " Answer: "
        msgs = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
    		model="meta-llama/Llama-3-8b-chat-hf",
    		messages=msgs)    
        topic = response.choices[0].message.content

        time.sleep(.1)

        msgs = [{"role": "user", "content": rephrase_prompt + "Can you find more questions about " + topic + "?" }]
        response = client.chat.completions.create(
    		model="meta-llama/Llama-3-8b-chat-hf",
    		messages=msgs)   
        query = response.choices[0].message.content
        results.append((query, question, ans))
        #print(results[-1])
        if len(results) % 25 == 0:
            print("PID: " + str(pid),len(results))
            with open("llm_query_" + str(pid) +"_" + file_name, 'wb') as f:
                pickle.dump(results, f)
    with open("llm_query_" + str(pid) +"_" + file_name, 'wb') as f:
        pickle.dump(results, f)


def main():
    with open(file_name, "rb") as f:
        rephrase_data = pickle.load(f)
    num_workers = 5
    data_slices = []
    for i in range(num_workers):
        data_slices.append([])
    for i in range(len(rephrase_data)):
        index = i % num_workers
        data_slices[index].append(rephrase_data[i])
    pids = []
    for pid in range(num_workers):
        p = Process(target=rephrase, args=(data_slices[pid], pid,))
        p.start()
        pids.append(p)
    for pid in pids:
        pid.join()

    result_slices = []
    for pid in range(num_workers):
        with open("llm_query_" + str(pid) +"_" + file_name, "rb") as f:
            slice = pickle.load(f)
            result_slices.append(slice)
    final = []
    while len(final) < len(rephrase_data):
        for index in range(num_workers):
            if len(result_slices[index]) == 0:
                continue
            final.append(result_slices[index][0])
            result_slices[index] = result_slices[index][1:]
    with open("llm_query_"+ file_name, 'wb') as f:
        pickle.dump(final, f)


    """
    with open(names["chase"], 'r+') as f:
        key = f.readlines()[0]
        key = key.replace("\n", "")
        client_chase = Together(api_key=key)
    msgs = [{"role": "user", "content": "Rephrase the following question one time: What are ribosomes known for?"}]
    start = time.time()
    response = client_chase.chat.completions.create(
    		model="mistralai/Mistral-7B-Instruct-v0.3",
    		messages=msgs)
    end = time.time()
    print(end - start, response.choices[0].message.content)
    """

if __name__ == "__main__":
    main()