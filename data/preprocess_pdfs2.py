
# I Combined all .txt files in data/processed_data into a single corpus.txt file.

import os

processed_folder_path = 'data/processed_data' 
output_filepath = 'data/corpus.txt'

all_text_content = []

print("Reading files and collecting content...")
for filename in os.listdir(processed_folder_path):
    if filename.endswith('.txt'):
        filepath = os.path.join(processed_folder_path, filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            all_text_content.append(f.read())

print(f"Combining content from {len(all_text_content)} files...")
with open(output_filepath, 'w', encoding='utf-8') as f:
    f.write("\n\n".join(all_text_content))

print(f"Successfully created the corpus at: {output_filepath}")
