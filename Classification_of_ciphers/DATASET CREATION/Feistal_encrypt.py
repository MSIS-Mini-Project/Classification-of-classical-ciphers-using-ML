import os
import pandas as pd
from collections import Counter
from math import log2


def feistel_round(left, right, subkey):
    return right, (left + f_function(right, subkey)) % 26

def f_function(value, subkey):
    return (value + subkey) % 26

def feistel_encrypt(plaintext, keys, rounds=16):
    left = plaintext[0]
    right = plaintext[1]

    for i in range(rounds):
        subkey = keys[i % len(keys)]
        left, right = feistel_round(left, right, subkey)

    return right, left

def encrypt_file(input_file_path, keys):

    with open(input_file_path, 'r', encoding='utf-8') as file:
        plaintext = [ord(c) - ord('a') for c in file.read().lower() if c.isalpha()]

    if len(plaintext) % 2 != 0:
        plaintext.append(0)

    left = plaintext[:len(plaintext) // 2]
    right = plaintext[len(plaintext) // 2:]

    ciphertext = []
    for l, r in zip(left, right):
        encrypted_pair = feistel_encrypt((l, r), keys)
        ciphertext.extend(encrypted_pair)

    encrypted_text = ''.join(chr(c + ord('a')) for c in ciphertext)
    return plaintext, encrypted_text


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

#
def encrypt_files(input_dir, output_dir, keys):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_to_append = []
    excel_file = "DATASET/DATASET_CIPHER.xlsx"

    for filename in os.listdir(input_dir):
        input_file_path = os.path.join(input_dir, filename)

        if os.path.isfile(input_file_path):
            plaintext, encrypted_content = encrypt_file(input_file_path, keys)

            length = len(encrypted_content)
            entropy = calculate_entropy(encrypted_content)
            ioc = calculate_ioc(encrypted_content)
            encryption_type = "Feistel Cipher"


            data_to_append.append([''.join(chr(c + ord('a')) for c in plaintext), encrypted_content, encryption_type, length, entropy, ioc])

            output_file_path = os.path.join(output_dir, filename)
            with open(output_file_path, 'w', encoding='utf-8') as encrypted_file:
                encrypted_file.write(encrypted_content)

            print(f"Encrypted {filename} and saved to {output_file_path}")


    append_to_excel(data_to_append, excel_file)

if __name__ == "__main__":
    input_directory = 'Plaintext/PT_feistal'
    output_directory = '/dump'
    keys = [10, 20, 30, 40]

    encrypt_files(input_directory, output_directory, keys)
