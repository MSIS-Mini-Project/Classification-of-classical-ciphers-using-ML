import os
import re


def clean_text(text):

    cleaned_text = re.sub(r'[^A-Za-z]+', '', text)

    cleaned_text = cleaned_text.replace(" ", "")

    return cleaned_text
def clean_files_in_directory(directory):

    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)

        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                content = file.read()

            cleaned_content = clean_text(content)

            with open(file_path, 'w') as file:
                file.write(cleaned_content)

            print(f"Cleaned file: {file_name}")

directory = "Plaintext\PT_vigenere"
clean_files_in_directory(directory)
print("All files in the directory have been cleaned.")
