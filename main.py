from dataloader import *
from architecture import *
from training import *
from evaluation_metrics import *

import matplotlib.pyplot as plt
import dill
import sys
import re
import pickle

import nltk

BUFFER_SIZE = 4000
BATCH_SIZE = 30
num_examples = 3000

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

file = "processed/preprocessed_text.csv"

tf.keras.backend.clear_session()

dataset_creator = NMTDataset('NL-PL')
train_dataset, val_dataset, inp_lang_tokenizer, target_lang_tokenizer = dataset_creator.call(num_examples, file, BUFFER_SIZE, BATCH_SIZE)

example_input_batch, example_target_batch = next(iter(train_dataset))

vocab_inp_size = len(inp_lang_tokenizer.get_config()) + 1
vocab_tar_size = len(target_lang_tokenizer.get_config()) + 1
max_length_input = example_input_batch.shape[1]
max_length_output = example_target_batch.shape[1]

embedding_dim = 50
units = 512
steps_per_epoch = num_examples//BATCH_SIZE

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE, num_layers=7)
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE, max_length_input, max_length_output)

lstm_model = LSTM_custom(encoder, decoder, units, max_length_input, dataset_creator, BATCH_SIZE)

lstm_model.train(train_dataset, val_dataset, 5, steps_per_epoch, patience=5)
hist = lstm_model.get_training_history()

y_loss = hist[:,0]
y_val_loss = hist[:,1]

plt.plot(y_loss, label="Training loss")
plt.plot(y_val_loss, label="Validation loss")
plt.legend()
plt.show()

problem_condtions = [    # Put problem descriptions here (later) # preprocess input more to be seq2seq
    "Calculate the sum of two numbers"
]

problem_solutions = [    # Put their python code solutions here (later)
    "def add_numbers(a, b):\n    return a + b"
]

# problem_condtions = [    # Put problem descriptions here (later) # preprocess input more to be seq2seq
#     "numpy"
# ]

# problem_solutions = [    # Put their python code solutions here (later)
#     "np"
# ]


evaluator = Evaluator(problem_condtions, problem_solutions, lstm_model, inp_lang_tokenizer, target_lang_tokenizer)
print(evaluator.bleu_scores())
print(evaluator.meteor_scores())
print(evaluator.code_bert_scores())