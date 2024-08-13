import json
import os

def merge_files(f1_folder, f2_folder, output_folder):
    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # List all files in the F1 folder
    for f1_file in os.listdir(f1_folder):
        if f1_file.endswith('.jsonl'):
            base_name = f1_file.rsplit('.', 1)[0]
            f2_file = f"{base_name}_context.jsonl"
            f1_path = os.path.join(f1_folder, f1_file)
            f2_path = os.path.join(f2_folder, f2_file)
            output_path = os.path.join(output_folder, f"{base_name}_merged.jsonl")
            
            try:
                # Read lines from F1 and F2 files
                with open(f1_path, 'r') as file1, open(f2_path, 'r') as file2:
                    lines_f1 = file1.readlines()
                    lines_f2 = file2.readlines()

                    # Ensure both files have the same number of lines
                    if len(lines_f1) != len(lines_f2):
                        print(f"Warning: Files {f1_file} and {f2_file} have different number of lines.")
                        continue

                    # Merge lines and write to output file
                    with open(output_path, 'w') as outfile:
                        for line_f1, line_f2 in zip(lines_f1, lines_f2):
                            data_f1 = json.loads(line_f1)
                            data_f2 = json.loads(line_f2)
                            
                            # Combine the JSON objects
                            combined_data = {
                                "input_text": data_f2.get("input_text"),
                                "job_role": data_f2.get("job_role"),
                                "question": data_f1.get("question"),
                                "answer": data_f1.get("answer")
                            }

                            # Write the combined JSON object to the output file
                            outfile.write(json.dumps(combined_data) + '\n')
            
            except Exception as e:
                print(f"Error processing files {f1_file} and {f2_file}: {e}")
                continue
            
            print(f"Merged file created: {output_path}")

# Define your folders here
F1_folder = 'jsonl_files'
F2_folder = 'new_dataset'
output_folder = 'Merged'

merge_files(F1_folder, F2_folder, output_folder)

