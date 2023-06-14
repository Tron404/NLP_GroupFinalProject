from dataloader import *
from architecture import *
from training import *
from evaluation_metrics import *

import nltk

BUFFER_SIZE = 62000
BATCH_SIZE = 100
num_examples = 60000
TRAIN = False
nltk.download('punkt')
nltk.download('wordnet')

tf.keras.backend.clear_session()

dataset_creator = NMTDataset('en-ron')
train_dataset, val_dataset, inp_lang, targ_lang = dataset_creator.call(num_examples, "ron.txt", BUFFER_SIZE, BATCH_SIZE)

example_input_batch, example_target_batch = next(iter(train_dataset))

vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1
max_length_input = example_input_batch.shape[1]
max_length_output = example_target_batch.shape[1]

embedding_dim = 768
units = 512
steps_per_epoch = num_examples//BATCH_SIZE

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE, max_length_input, max_length_output, 'luong')

lstm_model = LSTM_custom(encoder, decoder, units, max_length_input, dataset_creator, BATCH_SIZE)

lstm_model.train(train_dataset, val_dataset, 2, steps_per_epoch, patience=5)

problem_condtions = [    # Put problem descriptions here (later)
    "Buna!",
    "Ce mai faci?",
    "A nins.",
    "Vreau unul!"
]

problem_solutions = [    # Put their python code solutions here (later)
    "Hello!",
    "How are you?",
    "It had snowed.",
    "I want one!"
]

# evaluator = Evaluator(problem_solutions, problem_condtions, lstm_model, inp_lang, targ_lang)
# print(evaluator.bleu_scores())
# print(evaluator.meteor_scores())
# print(evaluator.code_bert_scores())