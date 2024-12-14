import os

def convert_uppercase_to_lowercase(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    content_lowercase = content.lower()

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content_lowercase)

directory_path = "Plaintext/PT_columnar"

for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)

    if os.path.isfile(file_path) and filename.endswith('.txt'):
        convert_uppercase_to_lowercase(file_path)
        print(f"Processed file: {filename}")