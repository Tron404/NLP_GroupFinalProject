from dataloader import *
from architecture import *
from training import *
from evaluation_metrics import *

import matplotlib.pyplot as plt

import nltk

BUFFER_SIZE = 600
BATCH_SIZE = 2
num_examples = 50
TRAIN = True
file = "processed/preprocessed_text.csv"

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

tf.keras.backend.clear_session()

dataset_creator = NMTDataset('NL-PL')
train_dataset, val_dataset, inp_lang, targ_lang = dataset_creator.call(num_examples, file, BUFFER_SIZE, BATCH_SIZE)

example_input_batch, example_target_batch = next(iter(train_dataset))

vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1
max_length_input = example_input_batch.shape[1]
max_length_output = example_target_batch.shape[1]

embedding_dim = 50
units = 50
steps_per_epoch = num_examples//BATCH_SIZE

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE, max_length_input, max_length_output, 'luong')

lstm_model = LSTM_custom(encoder, decoder, units, max_length_input, dataset_creator, BATCH_SIZE)

if TRAIN:
    lstm_model.train(train_dataset, val_dataset, 2, steps_per_epoch, patience=5)
    lstm_model.load_model("training_checkpoints/")
else:
    lstm_model.load_model("training_checkpoints/")

hist = lstm_model.get_training_history()
y_loss = hist[:,0]
y_val_loss = hist[:,1]

# plt.plot(y_loss, label="Training loss")
# plt.plot(y_val_loss, label="Validation loss")
# plt.legend()
# plt.show()

problem_condtions = [    # Put problem descriptions here (later) # preprocess input more to be seq2seq
    "write python program find maximum value"
]

problem_solutions = [    # Put their python code solutions here (later)
    "def maximum(a,b): return a if a > b else b"
]

evaluator = Evaluator(problem_solutions, problem_condtions, lstm_model, inp_lang, targ_lang)
print(evaluator.bleu_scores())
print(evaluator.meteor_scores())
print(evaluator.code_bert_scores())