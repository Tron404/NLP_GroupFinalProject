import pandas as pd
import numpy as np
import tensorflow as tf
import unicodedata
import io
import re
import os
from numpy import asarray
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import tokenize
import gensim.downloader as api

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from preprocessing_functionality import *

class NMTDataset:
    def __init__(self, problem_type):
        self.problem_type = problem_type
        self.inp_lang_tokenizer = None
        self.targ_lang_tokenizer = None
        self.file_path = "./processed/"
        self.DIM = 50
        self.model = f"glove-wiki-gigaword-{self.DIM}-word2vec"
        self.embedding_model = self.load_embedding_model()
        self.test_size = 0.2

    # turn problem descriptions into clean tokens
    def preprocess_problem_data(self, text):
        # Convert to lowercase
        text = text.lower()
        # remove punctuation from text
        punc_clean = re.sub(r'[^\w\s]', '', text)
        # split tokens
        words = punc_clean.split()
        # remove remaining tokens that are not alphabetic
        words = [word for word in words if word.isalpha()]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        # NOTE: we do not filter out short tokens since there are special chars like x and y
        # lemmatize the tokens
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]

        return words
    
    def preprocess_code_data(self, code):
        # Convert code to format accepted by Python's own tokenizer that can read Python syntax
        tokens = tokenize.generate_tokens(io.StringIO(code).readline)

        tokenized_code = []
        try: 
            for token in tokens:
                # Include only the code tokens and exclude space, commets etc.
                if token.type in (tokenize.NAME, tokenize.OP, tokenize.NUMBER, tokenize.STRING):
                    tokenized_code.append(token.string)
        
        # Skip code that ends with an error, e.g. missing some closing brackets.
        except tokenize.TokenError:
            return None
        except IndentationError:
            return None
        
        return tokenized_code
    
    def add_sos_eos(self, tokens):
        tokens = [' <sos>'] + tokens + ['<eos> ']
        return tokens
    
    def add_pad(self, tokens, max_length):
        while len(tokens) < max_length:
            tokens.append("<pad>")
        
        return tokens
    
    def create_dataset(self, path, num_examples):
        df = pd.read_csv(path, sep="ă")

        df = df.iloc[:num_examples]
        
        # Remove NA
        df.dropna(subset=['Problem', 'Python Code'], inplace=True)  # remove any rows with missing values
        df = df[df['Python Code'].str.len() > 1]
        df.reset_index(drop=True, inplace=True)

        # only if data is not processed yet
        ##################################
        # df["Problem"] = df["Problem"].apply(self.preprocess_problem_data)
        # df["Python Code"] = df["Python Code"].apply(self.preprocess_code_data)


        # # add <sos> and <eos> tags
        # df["Problem"] = df["Problem"].apply(self.add_sos_eos)
        # df["Python Code"] = df["Python Code"].apply(self.add_sos_eos)

        # add padding
        # max_length_input = np.max([len(text) for text in df["Problem"]])
        # max_length_target = np.max([len(text) for text in df["Python Code"]])

        # df["Problem"] = df["Problem"].apply(self.add_pad, args=(max_length_input,))
        # df["Python Code"] = df["Python Code"].apply(self.add_pad, args=(max_length_target,))

        # df["Problem"] = [" ".join(text) for text in df["Problem"]]
        # df["Python Code"] = [" ".join(text) for text in df["Python Code"]]
        ##################################

        df.dropna(subset=['Problem', 'Python Code'], inplace=True)  # remove any rows with missing values

        return df["Problem"].to_list(), df["Python Code"].to_list()
    
    def build_vocabulary(self, data_lang):
        tokenizer = tf.keras.preprocessing.text.Tokenizer(filters="", oov_token="<oov>")
        tokenizer.fit_on_texts(data_lang)

        return tokenizer
    
    def preprocess_one_sentence(self, text):
        text = self.preprocess_problem_data(text)
        text = self.add_sos_eos(text)

        text = " ".join(text)

        return self.generate_embeddings([text])
    
    def generate_embeddings(self, data):
        embedding_data = []
        for data_point in data:
            vect_emb = []
            aux = data_point.split(" ")

            for word in aux:
                if word in self.embedding_model.keys():
                    vect_emb.append(np.asarray(self.embedding_model.get(word)).astype("float32"))
                else:
                    vect_emb.append(np.random.rand(self.DIM).astype("float32"))

            embedding_data.append(np.mean(vect_emb, axis=0))

        return embedding_data

    def load_embedding_model(self):

        if not os.path.exists(self.file_path):
            print("Downloading model...")
            model = api.load(self.model)
            model.save_word2vec_format(self.file_path, binary=False)
        # else
        print("Loading w2v model...")
        file = open(self.file_path + self.model + ".txt", 'r', encoding='utf8')
        lines = file.readlines()[1:]
        file.close()
        
        # Map words to vectors
        embedding = dict()
        for line in lines:
            parts = line.split()
            # key is string word, value is numpy array for vector
            embedding[parts[0]] = asarray(parts[1:], dtype='float32')
        
        return embedding

    def load_dataset_embeddings(self, path, num_examples=None):
        inp_lang, targ_lang = self.create_dataset(path, num_examples)

        self.inp_lang_tokenizer, self.targ_lang_tokenizer = self.build_vocabulary(inp_lang), self.build_vocabulary(targ_lang)

        input_tensor = self.generate_embeddings(inp_lang)
        target_tensor = self.generate_embeddings(targ_lang)

        return input_tensor, target_tensor

    def call(self, num_examples, file_path, BUFFER_SIZE, BATCH_SIZE):
        input_tensor, target_tensor = self.load_dataset_embeddings(file_path, num_examples)

        input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=self.test_size)

        train_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train))
        train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

        val_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_val, target_tensor_val))
        val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)

        return train_dataset, val_dataset,self.inp_lang_tokenizer, self.targ_lang_tokenizer 
