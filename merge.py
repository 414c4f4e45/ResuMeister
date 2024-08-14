import json
import os
import glob

folder = "jsonl_files/"
folder_path = os.path.abspath(folder)

def combine_jsonl_files(input_dir, output_file):
	with open(output_file, 'w') as outfile:
		for filename in glob.glob(f"{input_dir}/*.jsonl"):
			with open(filename, 'r') as infile:
				for line in infile:
					data = json.loads(line)
					#data['question'] = data['question'].replace('\"',"'")
					#data['question'] = data['question'].replace('\u2019',"'")
					#data['answer'] = data['answer'].replace('\u2014',",")
					#data['answer'] = data['answer'].replace('\"',",")
					data['context'] = filename.split("/")[-1][:-6]
					outfile.write(json.dumps(data) + '\n')

output_file = "combined_data.jsonl"
combine_jsonl_files(folder_path, output_file)

