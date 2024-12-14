import os
import pandas as pd
from collections import Counter
from math import log2


def vigenere_encrypt(plaintext, key):
    key = key.upper()
    encrypted_text = []
    key_index = 0

    for char in plaintext:
        if char.isalpha():
            shift = ord(key[key_index % len(key)]) - ord('A')
            if char.isupper():
                encrypted_char = chr(((ord(char) - ord('A') + shift) % 26) + ord('A'))
            else:
                encrypted_char = chr(((ord(char) - ord('a') + shift) % 26) + ord('a'))
            key_index += 1
        else:
            encrypted_char = char
        encrypted_text.append(encrypted_char)

    return ''.join(encrypted_text)

def calculate_entropy(text):
    frequency = Counter(text)
    text_length = len(text)
    entropy = -sum((count / text_length) * log2(count / text_length) for count in frequency.values())
    return entropy


def calculate_ioc(text):
    frequency = Counter(text)
    n = len(text)
    ioc = sum(f * (f - 1) for f in frequency.values()) / (n * (n - 1)) if n > 1 else 0
    return ioc


def append_to_excel(data, excel_file):
    if not os.path.exists(excel_file):
        # If the Excel file doesn't exist, create a new one
        new_df = pd.DataFrame(data, columns=["Plaintext", "Ciphertext", "Encryption Type", "Length", "Entropy", "IOC"])
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            new_df.to_excel(writer, sheet_name="Sheet1", index=False)
        print(f"Excel file created and data written to {excel_file}")
    else:

        existing_df = pd.read_excel(excel_file, sheet_name="Sheet1")
        new_df = pd.DataFrame(data, columns=["Plaintext", "Ciphertext", "Encryption Type", "Length", "Entropy", "IOC"])


        final_df = pd.concat([existing_df, new_df], ignore_index=True)


        with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            final_df.to_excel(writer, sheet_name="Sheet1", index=False)
        print(f"Data appended to existing Excel file: {excel_file}")


def encrypt_files(input_dir, output_dir, key, excel_file):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_to_append = []

    for filename in os.listdir(input_dir):
        input_file_path = os.path.join(input_dir, filename)

        if os.path.isfile(input_file_path):
            with open(input_file_path, 'r', encoding='utf-8') as file:
                plaintext = file.read().strip()

            encrypted_text = vigenere_encrypt(plaintext, key)
            length = len(encrypted_text)
            entropy = calculate_entropy(encrypted_text)
            ioc = calculate_ioc(encrypted_text)
            encryption_type = "Vigenère Cipher"


            data_to_append.append([plaintext, encrypted_text, encryption_type, length, entropy, ioc])


            output_file_path = os.path.join(output_dir, f"encrypted_{filename}")
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                output_file.write(encrypted_text)

            print(f"Encrypted {filename} and saved to {output_file_path}")


    append_to_excel(data_to_append, excel_file)


if __name__ == "__main__":
    input_directory = 'Plaintext/PT_vigenere'
    output_directory = 'dump'
    encryption_key = 'YOURKEY'
    excel_file = 'DATASET/DATASET_CIPHER.xlsx'

    encrypt_files(input_directory, output_directory, encryption_key, excel_file)
