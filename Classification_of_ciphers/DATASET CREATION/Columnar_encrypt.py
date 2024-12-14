import os
import math
import itertools
import pandas as pd
from collections import Counter
from math import log2


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


def generate_all_keys(length=5):
    digits = [str(i) for i in range(1, length + 1)]
    permutations = itertools.permutations(digits)
    keys = [''.join(p) for p in permutations]
    return keys


def columnar_transposition_encrypt(plaintext, key):
    plaintext = plaintext.replace(" ", "").lower()
    num_columns = len(key)
    num_rows = math.ceil(len(plaintext) / num_columns)

    grid = [['' for _ in range(num_columns)] for _ in range(num_rows)]

    for i, char in enumerate(plaintext):
        row = i // num_columns
        col = i % num_columns
        grid[row][col] = char

    key_order = sorted(range(len(key)), key=lambda x: key[x])

    ciphertext = []
    for col_index in key_order:
        column_chars = [grid[row][col_index] for row in range(num_rows)]
        ciphertext.append(''.join(column_chars))

    return ''.join(ciphertext)


def append_to_excel(data, excel_file):
    if not os.path.exists(excel_file):

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

def encrypt_files(input_dir, output_dir, excel_file):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_to_append = []

    for filename in os.listdir(input_dir):
        input_file_path = os.path.join(input_dir, filename)

        if os.path.isfile(input_file_path):
            with open(input_file_path, 'r', encoding='utf-8') as file:
                plaintext = file.read().strip()


            all_keys = generate_all_keys()

            for key in all_keys:
                encrypted_text = columnar_transposition_encrypt(plaintext, key)
                length = len(encrypted_text)
                entropy = calculate_entropy(encrypted_text)
                ioc = calculate_ioc(encrypted_text)
                encryption_type = "Columnar Transposition"


                data_to_append.append([plaintext, encrypted_text, encryption_type, length, entropy, ioc])


                output_file_path = os.path.join(output_dir, f"encrypted_{filename}_{key}")
                with open(output_file_path, 'w', encoding='utf-8') as output_file:
                    output_file.write(encrypted_text)

                print(f"Encrypted {filename} and saved to {output_file_path}")


    append_to_excel(data_to_append, excel_file)


if __name__ == "__main__":
    input_directory = 'Plaintext/PT_columnar'
    output_directory = 'dump'
    excel_file = 'DATASET/DATASET_CIPHER.xlsx'

    encrypt_files(input_directory, output_directory, excel_file)
