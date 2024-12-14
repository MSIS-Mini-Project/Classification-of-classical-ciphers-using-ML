import pandas as pd
from collections import Counter
import math
import re


df = pd.read_excel("DATASET/DATASET_CIPHER.xlsx", sheet_name="Sheet1", engine='openpyxl')


def calculate_bigram_frequencies(text):
    bigrams = [text[i:i+2] for i in range(len(text) - 1)]
    return Counter(bigrams)

def calculate_trigram_frequencies(text):
    trigrams = [text[i:i+3] for i in range(len(text) - 2)]
    return Counter(trigrams)

def calculate_ioc(text):
    n = len(text)
    if n <= 1:
        return 0
    freq = Counter(text)
    return sum(f * (f - 1) for f in freq.values()) / (n * (n - 1))

def calculate_entropy(text):
    n = len(text)
    if n == 0:
        return 0
    freq = Counter(text)
    return -sum((count / n) * math.log(count / n, 2) for count in freq.values())

def calculate_unigram_variance(text):
    freq = Counter(text)
    n = len(text)
    if n == 0:
        return 0
    probabilities = [count / n for count in freq.values()]
    mean_prob = sum(probabilities) / len(probabilities)
    return sum((p - mean_prob) ** 2 for p in probabilities) / len(probabilities)

def find_longest_unique_substring(text):
    seen = {}
    start = max_length = 0
    for i, char in enumerate(text):
        if char in seen and seen[char] >= start:
            start = seen[char] + 1
        seen[char] = i
        max_length = max(max_length, i - start + 1)
    return max_length

def calculate_repeated_ngram_count(text, n):
    ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
    freq = Counter(ngrams)
    return sum(1 for count in freq.values() if count > 1)

def extract_features(row):
    ciphertext = row['Ciphertext']
    cleaned_text = re.sub(r'[^a-zA-Z]', '', ciphertext).lower()


    features = {'Length': len(cleaned_text)}


    letter_counts = Counter(cleaned_text)
    features.update({f'Freq_{chr(i)}': letter_counts.get(chr(i), 0) for i in range(ord('a'), ord('z') + 1)})


    bigram_features = ['TH', 'HE', 'IN', 'ER', 'AN']
    trigram_features = ['THE', 'AND', 'ING', 'HER', 'HAT']
    bigram_counts = calculate_bigram_frequencies(cleaned_text)
    trigram_counts = calculate_trigram_frequencies(cleaned_text)

    for bigram in bigram_features:
        features[f'bigram_{bigram}'] = bigram_counts.get(bigram.lower(), 0)

    for trigram in trigram_features:
        features[f'trigram_{trigram}'] = trigram_counts.get(trigram.lower(), 0)


    features['bigram_ratio_TH_HE'] = (bigram_counts.get('th', 0) + 1) / (bigram_counts.get('he', 0) + 1)
    features['bigram_ratio_ER_AN'] = (bigram_counts.get('er', 0) + 1) / (bigram_counts.get('an', 0) + 1)

    features['average_bigram_freq'] = sum(bigram_counts.values()) / len(bigram_counts) if bigram_counts else 0
    features['average_trigram_freq'] = sum(trigram_counts.values()) / len(trigram_counts) if trigram_counts else 0


    features['unique_bigrams'] = len(bigram_counts)
    features['unique_trigrams'] = len(trigram_counts)





    features['unigram_variance'] = calculate_unigram_variance(cleaned_text)


    features['longest_unique_substring'] = find_longest_unique_substring(cleaned_text)


    features['repeated_4gram_count'] = calculate_repeated_ngram_count(cleaned_text, 4)

    return features


feature_data = df.apply(extract_features, axis=1, result_type='expand')


df = pd.concat([df, feature_data], axis=1)


df.to_excel("DATASET/DATASET_CIPHER.xlsx", index=False)


