import os
from together import Together
import sys
import time

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
	response = client.chat.completions.create(
    		model="meta-llama/Llama-3-8b-chat-hf",
    		messages=[{"role": "user", "content": "What are some fun things to do in New York"}])
	end = time.time()
	print(response.choices[0].message.content)
	print("Elapsed time:", end - start)
	

if __name__ == "__main__":
	main()
