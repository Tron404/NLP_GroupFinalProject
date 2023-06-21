import pandas as pd
import numpy as np
import tensorflow as tf
import io
import re
import os
from numpy import asarray
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import tokenize
import gensim.downloader as api

from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self, processed=True):
        self.inp_lang_tokenizer = None
        self.targ_lang_tokenizer = None
        self.file_path = "./processed/"
        self.DIM = 300
        self.model = f"glove-wiki-gigaword-{self.DIM}"
        self.embedding_model = self.load_embedding_model()

        self.max_length_input = -1
        self.max_length_output = -1
        self.processed = processed

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
        tokens = ['<sos>'] + tokens + ['<eos>']
        return tokens
    
    def process_dataset(self, df):
        df["Problem"] = df["Problem"].apply(self.preprocess_problem_data)
        df["Python Code"] = df["Python Code"].apply(self.preprocess_code_data)


        # add <sos> and <eos> tags
        df["Problem"] = df["Problem"].apply(self.add_sos_eos)
        df["Python Code"] = df["Python Code"].apply(self.add_sos_eos)

        df["Problem"] = [" ".join(text) for text in df["Problem"]]
        df["Python Code"] = [" ".join(text) for text in df["Python Code"]]

        return df
    
    def create_dataset(self, path, num_examples):
        df = pd.read_csv(path, sep="ă")

        df = df.iloc[:num_examples]
        
        # Remove NA
        df.dropna(subset=['Problem', 'Python Code'], inplace=True)  # remove any rows with missing values
        df = df[df['Python Code'].str.len() > 1]
        df.reset_index(drop=True, inplace=True)

        # only if data is not processed yet
        if not self.processed:
            df = self.process_dataset(df)

        df.dropna(subset=['Problem', 'Python Code'], inplace=True)  # remove any rows with missing values

        return df["Problem"].to_list(), df["Python Code"].to_list()
    
    def build_vocabulary(self, data_lang):
        tokenizer = tf.keras.preprocessing.text.Tokenizer(filters="", oov_token="<oov>")
        tokenizer.fit_on_texts(data_lang)

        tensor = tokenizer.texts_to_sequences(data_lang)
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding="post")

        return tensor, tokenizer

    def load_embedding_model(self):
        if not os.path.exists(self.file_path + self.model):
            print("Downloading model...")
            model = api.load(self.model)
            model.save_word2vec_format(self.file_path + self.model + ".txt", binary=False)
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
    
    def process_one_sentence(self, text):
        text = self.preprocess_problem_data(text)
        text = self.add_sos_eos(text)

        print(f"Processed sentence: {text}")

        inputs = [self.inp_lang_tokenizer.word_index[i] if i in self.inp_lang_tokenizer.word_index else self.inp_lang_tokenizer.word_index["<oov>"] for i in text]
        print(inputs)
        print(self.inp_lang_tokenizer.word_index["<sos>"], self.inp_lang_tokenizer.word_index["<eos>"], self.inp_lang_tokenizer.word_index["<oov>"])
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=self.max_length_input, padding='post')

        return inputs
    
    def build_embedding_matrix(self, tokenizer):
        matrix = np.random.rand(len(tokenizer.get_config()["word_counts"]) + 1, self.DIM) # +1 for <oov>
        
        for token, i in tokenizer.word_index.items():
            if token in self.embedding_model.keys():
                matrix[i] = self.embedding_model.get(token)
            # else the embedding will be random

        return matrix

    def load_dataset_embeddings(self, path, num_examples=None):
        inp_lang, targ_lang = self.create_dataset(path, num_examples)

        input_tensor, self.inp_lang_tokenizer = self.build_vocabulary(inp_lang)
        target_tensor, self.targ_lang_tokenizer = self.build_vocabulary(targ_lang)

        return input_tensor, target_tensor

    def call(self, num_examples, file_path, BUFFER_SIZE, BATCH_SIZE):
        input_tensor, target_tensor = self.load_dataset_embeddings(file_path, num_examples)

        input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, train_size=0.6)
        input_tensor_val, input_tensor_test, target_tensor_val, target_tensor_test = train_test_split(input_tensor_val, target_tensor_val, test_size=0.5)

        train_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train))
        train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

        val_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_val, target_tensor_val))
        val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)

        test_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_test, target_tensor_test))
        test_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)

        example_input_batch, example_target_batch = next(iter(train_dataset))
        self.max_length_input = example_input_batch.shape[1]
        self.max_length_output = example_target_batch.shape[1]

        self.vocab_input_size = len(self.inp_lang_tokenizer.get_config()["word_counts"]) + 1 # for <oov>
        self.vocab_target_size = len(self.targ_lang_tokenizer.get_config()["word_counts"]) + 1

        return train_dataset, val_dataset, test_dataset, self.inp_lang_tokenizer, self.targ_lang_tokenizer 
