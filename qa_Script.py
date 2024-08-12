import os
import json

def txt_to_jsonl(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    qna_pairs = []
    question = None

    for line in lines:
        line = line.strip()
        if line.startswith('Q:'):
            if question is not None:
                # Add the previous Q&A pair to the list
                qna_pairs.append({"question": question, "answer": answer})
            # Start a new Q&A pair
            question = line[2:].strip()
        elif line.startswith('A:'):
            answer = line[2:].strip()
    
    # Add the last Q&A pair to the list
    if question is not None and answer is not None:
        qna_pairs.append({"question": question, "answer": answer})

    with open(output_file, 'w') as jsonl_file:
        for qna in qna_pairs:
            jsonl_file.write(json.dumps(qna) + '\n')

def convert_all_txt_to_jsonl(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, filename.replace('.txt', '.jsonl'))
            txt_to_jsonl(input_file, output_file)
            print(f"Converted {input_file} to {output_file}")

# Example usage:
input_folder = 'QA'          # Replace with the path to your input folder
output_folder = 'jsonl_files' # Replace with the path to your output folder
convert_all_txt_to_jsonl(input_folder, output_folder)

