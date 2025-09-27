
# I Extracted text from PDFs and save as .txt files into data/processed_data. 

import os
import PyPDF2

pdf_folder_path = 'data/raw_data'
text_folder_path = 'data/processed_data'


for filename in os.listdir(pdf_folder_path):
    if filename.endswith('.pdf'):
        pdf_path = os.path.join(pdf_folder_path, filename)
        
        print(f"Processing: {filename}...")

        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            full_text = ""
            
            for page in pdf_reader.pages:
                full_text += page.extract_text() + "\n"

        text_filename = filename.replace('.pdf', '.txt')
        text_path = os.path.join(text_folder_path, text_filename)
        
        with open(text_path, 'w', encoding='utf-8') as text_file:
            text_file.write(full_text)

print("\nProcessing complete!")