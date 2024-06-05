import os
from together import Together
import sys
import time

rephrase_prefix = "As briefly as possible state the topic of the following question: "
small_question_data = [
			"What part of the cell is responsible for ATP creation",
			"What is the apporximate speed of light",
			"Which gas is the most abundant in our atmosphere",
			"Which male organs are responsible for sexual reproduction"
			]

if len(sys.argv) != 2:
	print("Error please enter the name of whose together key to use")
	sys.exit()
names = {"chase" : "chase_together_key.txt"}
if sys.argv[1] not in names.keys():
	print("Error: pleasae enter a valid name")
def main():
	with open(names[sys.argv[1]], 'r+') as f:
		key = f.readlines()[0]
	key = key.replace("\n", "")
	client = Together(api_key=key)
	start = time.time()
	msgs = []
	for question in small_question_data:
		msgs.append({"role": "user", "content": rephrase_prefix + question})
	response = client.chat.completions.create(
    		model="mistralai/Mistral-7B-Instruct-v0.3",
    		messages=msgs)
	end = time.time()
	print(msgs)
	for i in range(len(response.choices)):
		print(response.choices[0].message.content)
	print("Elapsed time:", end - start)
	

if __name__ == "__main__":
	main()
