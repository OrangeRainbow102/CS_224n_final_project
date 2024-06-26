from together import Together
import sys
import numpy as np
import time
import pickle
from multiprocessing import Process


names = {"chase" : "query_generation/chase_together_key.txt", "ron": "query_generation/ron_together_key.txt"}
default_name = "chase"
	
prompt_prefix = "Respond only yes or no: is the question \"" 
prompt_middle = "\" relevant for this query \"" 
prompt_end = "\"?"

with open(names[default_name], 'r+') as f:
    key = f.readlines()[0]
    key = key.replace("\n", "")
    client = Together(api_key=key)
def calc_percision(num_workers):
	total = 0
	hits = 0
	for pid in range(num_workers):
		with open("intermidiate_state_" + str(pid)+ "_llm_judge.pkl", 'rb') as f:
			binary_list = pickle.load(f)
		total += len(binary_list)
		hits += np.sum(binary_list)
	return hits/ total
	


def judge(query_top_k, pid, user="chase"):
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
		if count % 5 == 0: print("PID:", pid, count)
		count += 1
		index = 0
		for question in top_k:
			start = time.time()
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
				#print("1:", question, "||" , query)
			elif "no" in answer:
				results.append(0)
				#print("0:", question, "||" , query)
			else:
				print("Error adding 0")
				results.append(0)
			end = time.time()
			time.sleep(max(.55 - (end - start), 0))
	with open("intermidiate_state_" + str(pid)+ "_llm_judge.pkl", 'wb') as f:
		pickle.dump(results, f)		
	#percision = calc_percision(results)

	return None
		
def main():
    #files = ["pretrain_baseline_ce.pkl", "pretrain_ce_small_synthetic5.pkl", "finetune_ce_small_synthetic5.pkl", "old_baseline_small_synthetic5.pkl"]
	#files = ["pretrain_all-MiniLM-L6-v2_small_synthetic5.pkl","finetune_all-MiniLM-L6-v2_small_synthetic5.pkl","pretrain_allenai-specter_small_synthetic5.pkl", "finetune_allenai-specter_small_synthetic5.pkl"]
	#files = ["pretrain_msmarco_CE_small_synthetic.pkl", "fine_tuned_msmarco_CE_small_synthetic.pkl"]
	#files = ["pretrain_multi-qa-mpnet-base-cos-v1_small_synthetic.pkl", "pretrain_multi-qa-MiniLM-L6-cos-v1_small_synthetic.pkl", "pretrain_all-msmarco-distilbert-base-v4_no_max_small_synthetic.pkl", "pretrain_all-MiniLM-L6-v2_no_max_small_synthetic.pkl"]
	files = ["finetune_marco_ce_large_synthetic5.pkl"]
	num_workers = 10

	
	for file in files:
		print("Start " + file)
		with open(file, 'rb') as f:
			test_data = pickle.load(f)
		data_slices = []
		for i in range(num_workers):
			data_slices.append([])
		for i in range(len(test_data)):
			index = i % num_workers
			data_slices[index].append(test_data[i])
		pids = []
		for pid in range(num_workers):
			p = Process(target=judge, args=(data_slices[pid], pid, ))
			p.start()
			pids.append(p)
		for pid in pids:
			pid.join()

		precise = calc_percision(num_workers)
		with open("results.txt", "r+") as f:
			lines = f.readlines()
		lines.append(file + ": " + str(precise) + "\n")
		with open("results.txt", "w") as f:
			f.writelines(lines)
			f.truncate()	
		print(precise)

		

if __name__ == "__main__":
	main()