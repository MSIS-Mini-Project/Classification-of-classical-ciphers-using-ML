import os
import random
import string
import pandas as pd
from collections import Counter
from math import log2


def generate_random_key(length):
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(length))


def vernam_encrypt(plain_text, key):
    plain_text = plain_text.lower()
    key = key.lower()

    cipher_text = []

    for pt_char, key_char in zip(plain_text, key):
        if pt_char in string.ascii_lowercase:
            pt_value = ord(pt_char) - ord('a')
            key_value = ord(key_char) - ord('a')
            cipher_value = (pt_value + key_value) % 26
            cipher_char = chr(cipher_value + ord('a'))
            cipher_text.append(cipher_char)
        else:
            cipher_text.append(pt_char)

    return ''.join(cipher_text)


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

def encrypt_files_in_directory(input_dir, output_dir, excel_file):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_to_append = []

    for filename in os.listdir(input_dir):
        input_file_path = os.path.join(input_dir, filename)

        if os.path.isfile(input_file_path) and filename.endswith('.txt'):
            with open(input_file_path, 'r') as file:
                plaintext = file.read().strip()

            key = generate_random_key(len(plaintext))
            ciphertext = vernam_encrypt(plaintext, key)
            length = len(ciphertext)
            entropy = calculate_entropy(ciphertext)
            ioc = calculate_ioc(ciphertext)
            encryption_type = "Vernam Cipher"


            data_to_append.append([plaintext, ciphertext, encryption_type, length, entropy, ioc])


            output_file_path = os.path.join(output_dir, f"encrypted_{filename}")
            with open(output_file_path, 'w') as output_file:
                output_file.write(ciphertext)

            print(f"Encrypted file saved as: {output_file_path}")


    append_to_excel(data_to_append, excel_file)


if __name__ == "__main__":
    input_directory = "/Users/sid/Downloads/CPP/CpP/bakra/validation/val_vernam"
    output_directory = "/Users/sid/Downloads/CPP/CpP/dump"
    excel_file = "/Users/sid/Downloads/CPP/CpP/bakra/validation/val.xlsx"

    encrypt_files_in_directory(input_directory, output_directory, excel_file)
