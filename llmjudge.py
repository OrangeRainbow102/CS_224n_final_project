from together import Together
import sys
import numpy as np
import time
import pickle


names = {"chase" : "query_generation/chase_together_key.txt", "ron": "query_generation/ron_together_key.txt"}
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
	count = 0
	with open(names[default_name], 'r+') as f:
		key = f.readlines()[0]
		key = key.replace("\n", "")
		client_chase = Together(api_key=key)
	results = []
	with open(names["ron"], 'r+') as f:
		key = f.readlines()[0]
		key = key.replace("\n", "")
		client_ron = Together(api_key=key)
	for query, top_k in query_top_k:
		if count % 5 == 0: print(count)
		count += 1
		index = 0
		for question in top_k:
			time.sleep(.04)
			if True:#index % 2 == 0:
				client = client_chase
			else:
				client = client_ron
			index += 1
			prompt = prompt_prefix + question + prompt_middle + query + prompt_end
			msgs = [{"role": "user", "content": prompt}]
			response = client.chat.completions.create(
    		model="mistralai/Mistral-7B-Instruct-v0.3",
    		messages=msgs)
			answer = response.choices[0].message.content.lower()
			#print(answer)
			if "yes" in answer:
				results.append(1)
				print("1:", question, "||" , query)
			elif "no" in answer:
				results.append(0)
				print("0:", question, "||" , query)
			else:
				print("Error adding 0")
				results.append(0)		
	percision = calc_percision(results)
	return percision
		
def main():
    #files = ["pretrain_baseline_ce.pkl", "pretrain_ce_small_synthetic5.pkl", "finetune_ce_small_synthetic5.pkl", "old_baseline_small_synthetic5.pkl"]
	#files = ["pretrain_all-MiniLM-L6-v2_small_synthetic5.pkl","finetune_all-MiniLM-L6-v2_small_synthetic5.pkl","pretrain_allenai-specter_small_synthetic5.pkl", "finetune_allenai-specter_small_synthetic5.pkl"]
	#files = ["pretrain_msmarco_CE_small_synthetic.pkl", "fine_tuned_msmarco_CE_small_synthetic.pkl"]
	#files = ["pretrain_multi-qa-mpnet-base-cos-v1_small_synthetic.pkl", "pretrain_multi-qa-MiniLM-L6-cos-v1_small_synthetic.pkl", "pretrain_all-msmarco-distilbert-base-v4_no_max_small_synthetic.pkl", "pretrain_all-MiniLM-L6-v2_no_max_small_synthetic.pkl"]
	files = ["pretrain_all-msmarco-distilbert-base-v4_no_max_small_synthetic.pkl"]
	for file in files:
		print("Start " + file)
		with open(file, 'rb') as f:
			test_data = pickle.load(f)
		precise = judge(test_data)
		with open("results.txt", "r+") as f:
			lines = f.readlines()
		lines.append(file + ": " + str(precise) + "\n")
		with open("results.txt", "w") as f:
			f.writelines(lines)
			f.truncate()	
		print(precise)
		

if __name__ == "__main__":
	main()