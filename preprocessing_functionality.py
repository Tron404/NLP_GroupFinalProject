import os
import tokenize
import io
import re
import gensim.downloader as api
import pandas as pd
import numpy as np
from numpy import asarray
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec
from keras.utils import pad_sequences
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
from nltk import FreqDist

data_file = "ProblemSolutionPythonV3.csv" # obtained from Kaggle
preprocessed_file = "preprocessed_text.csv"
preprocessed_folder = "./processed"
pretrained_w2v = "glove-wiki-gigaword-50"
pretrained_w2v_file = str(pretrained_w2v) + "-word2vec.txt" 
w2v_filename =  'embedding_w2v_matrix.npy'

def load_data(data_file):
    """Loads data from a csv file"""
    data = pd.read_csv(data_file)
    data.set_index('Unnamed: 0', inplace=True, drop=True)
    data = data.dropna()  
    data = data.reset_index(drop=True)  

    # Making sure all is good here
    print(f"Data downloaded. Shape: {data.shape}")
    # Look for the head
    print(f"Look at the head:\n {data.head()}")
        
    return data

def preprocess_problem_data(text):
    """Preprocesses problem text"""
    REGEX = re.compile(r'[^\w\s]')
    LEMMATIZER = WordNetLemmatizer()
    STOP_WORDS = set(stopwords.words('english'))

    text = text.lower()
    text = REGEX.sub('', text)
    words = text.split()
    words = [LEMMATIZER.lemmatize(word) for word in words if word.isalpha() and word not in STOP_WORDS]
    
    return words

def preprocess_code_data(code):
    """Preprocesses code text"""
    INCLUDE_TOKENS = set((tokenize.NAME, tokenize.OP, tokenize.NUMBER, tokenize.STRING))

    tokens = tokenize.generate_tokens(io.StringIO(code).readline)
    tokenized_code = []
    try: 
        for token in tokens:
            if token.type in INCLUDE_TOKENS:
                tokenized_code.append(token.string)
    except (tokenize.TokenError, IndentationError):
        return None
    return tokenized_code

def save_preprocessed_data(data, preprocessed_file, preprocessed_folder):
    """Saves preprocessed data to a csv file"""
    if not os.path.exists(preprocessed_folder):
        os.makedirs(preprocessed_folder)
    path = os.path.join(preprocessed_folder, preprocessed_file)
    data.to_csv(path, sep=',', index_label='Id')

def process_data(data_file, preprocessed_file, preprocessed_folder):
    """Loads and preprocesses data"""
    data = load_data(data_file)
    data = apply_preprocessing(data)
    save_preprocessed_data(data, preprocessed_file, preprocessed_folder)
    return data

def split_data(data):
    """Splits data into training and test sets"""
    x_train, x_test, y_train, y_test = train_test_split(data['Problem'], data['Python Code'], test_size=0.2, random_state=42)
    
    return x_train, x_test, y_train, y_test

def build_vocabulary(x_train):
    """Builds vocabulary from training data"""
    x_train_text = [' '.join(tokens) for tokens in x_train]
    vocab_freq = Counter()
    for text in x_train_text:
        words = text.split()  # Split the text into individual words
        vocab_freq.update(words)

    return vocab_freq

def filter_vocab(vocab_freq, min_rank_cutoff=15, max_rank_cutoff=1000):
    """Filters vocabulary based on frequency cutoffs"""
    word_freq = np.array(list(vocab_freq.values()))
    word_freq_sorted = np.sort(word_freq)[::-1]  # sort in descending order

    min_freq = word_freq_sorted[max_rank_cutoff-1] 
    max_freq = word_freq_sorted[min_rank_cutoff-1]  
    
    vocab = {token: freq for token, freq in vocab_freq.items() if min_freq <= freq <= max_freq}
    
    return vocab

def filter_data_by_freq(data, vocab):
    """Filters data based on vocabulary"""
    filtered_data = []
    for line in data:
        filtered = ' '.join([token for token in line if token in vocab])
        filtered_data.append(filtered)
    return pd.Series(filtered_data, index=data.index)

def apply_preprocessing(data):
    data['Problem'] = data['Problem'].apply(preprocess_problem_data)

    x_train, x_test, y_train, y_test = split_data(data)
    vocab_freq = build_vocabulary(x_train)
    vocab = filter_vocab(vocab_freq)
    x_train, x_test = filter_data_by_freq(x_train, vocab), filter_data_by_freq(x_test, vocab)

    data['Python Code'] = data['Python Code'].apply(preprocess_code_data)

    data.dropna(subset=['Problem', 'Python Code'], inplace=True) 
    data.reset_index(drop=True, inplace=True)
    print(f"Data pre-processed. Shape: {data.shape}")
    print(f"Look how clean it looks now!: \n {data.head()}")

    return x_train, x_test, y_train, y_test, len(vocab)

def get_tokenizer_and_maxlen(x_train):
    """Tokenizes the data and finds the max sequence length"""
    tokenizer = Tokenizer()
    tokenizer.filters = ""  # Remove default filters
    tokenizer.lower = False  
    tokenizer.fit_on_texts(x_train) 

    max_seq_len = max([len(line.split()) for line in x_train])
    return tokenizer, max_seq_len

def encode_and_pad_data(text_line, max_len, tokenizer):
    """Encodes and pads the data"""
    encoded = tokenizer.texts_to_sequences(text_line)
    padded = pad_sequences(encoded, maxlen=max_len, padding='post')
    return padded

DIM = 50

import os
import numpy as np
from gensim.models import KeyedVectors
from numpy import asarray

def load_embedding(preprocessed_folder, pretrained_w2v_file):
    """Loads word embeddings from a pretrained Word2Vec model file or downloads it if it doesn't exist"""
    file_path = os.path.join(preprocessed_folder, pretrained_w2v_file)
    if not os.path.exists(file_path):
        print("Downloading model...")
        model = api.load("glove-wiki-gigaword-50")
        model.save_word2vec_format(file_path, binary=False)
    else:
        print("Loading Word2Vec model...")
    file = open(file_path, 'r', encoding='utf8')
    lines = file.readlines()[1:]
    file.close()

    # Map words to vectors
    embedding = dict()
    for line in lines:
        parts = line.split()
        # key is string word, value is numpy array for vector
        embedding[parts[0]] = asarray(parts[1:], dtype='float32')

    return embedding


def build_matrix(embedding, tokenizer, vocab_length):
    """Builds an embedding matrix for a given tokenizer and vocabulary length using the provided word embeddings"""
    DIM = len(next(iter(embedding.values())))  # Assuming all vectors have the same dimensionality
    total_count = 0
    na_count = 0

    matrix = np.zeros((vocab_length + 1, DIM))  # +1 for unknown words
    for token, i in tokenizer.word_index.items():
        if token in embedding.keys():
            matrix[i] = embedding.get(token)
        else:
            na_count += 1
        total_count += 1
    print(f'NA/All words: {str(na_count)}/{total_count}')
    print(f"Matrix shape: {matrix.shape}")

    # Save the matrix
    # with open(os.path.join(preprocessed_folder, pretrained_w2v_file), 'w') as file:
    #     file.write('\n'.join(' '.join(str(x) for x in line) for line in matrix))

    return matrix


def main_preprocess(data_file, preprocessed_file, preprocessed_folder):
    """Main function to load and preprocess data"""
    data = load_data(data_file)
    x_train, x_test, y_train, y_test, vocab_len = apply_preprocessing(data)
    # save_preprocessed_data(data, preprocessed_file, preprocessed_folder)

    tokenizer, max_seq_len = get_tokenizer_and_maxlen(x_train)

    x_train_encoded = encode_and_pad_data(x_train, max_seq_len, tokenizer)
    x_test_encoded = encode_and_pad_data(x_test, max_seq_len, tokenizer)
    y_train_encoded = encode_and_pad_data(y_train, max_seq_len, tokenizer)
    y_test_encoded = encode_and_pad_data(y_test, max_seq_len, tokenizer)

    embedding = load_embedding(preprocessed_folder, pretrained_w2v_file)
    embedding_matrix = build_matrix(embedding, tokenizer, vocab_len)

    return x_train_encoded, x_test_encoded, y_train_encoded, y_test_encoded, embedding_matrix
