from dataloader import *
from architecture import *
from training import *
from evaluation_metrics import *

import matplotlib.pyplot as plt
import pickle

import nltk

BUFFER_SIZE = 4000
BATCH_SIZE = 5
num_examples = 2000

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

file = "processed/preprocessed_text.csv"

tf.keras.backend.clear_session()

dataset_creator = Dataset(processed=True)
train_dataset, val_dataset, test_dataset, inp_lang_tokenizer, target_lang_tokenizer = dataset_creator.call(num_examples, file, BUFFER_SIZE, BATCH_SIZE)
embedding_matrix_input = dataset_creator.build_embedding_matrix(inp_lang_tokenizer)
embedding_matrix_target = dataset_creator.build_embedding_matrix(target_lang_tokenizer)

example_input_batch, example_target_batch = next(iter(train_dataset))

print(embedding_matrix_input.shape, embedding_matrix_target)

vocab_inp_size = len(inp_lang_tokenizer.get_config()["word_counts"]) + 3 # for <eos>, <sos>, <oov>
vocab_tar_size = len(target_lang_tokenizer.get_config()["word_counts"]) + 3 # for <eos>, <sos>, <oov>
max_length_input = example_input_batch.shape[1]
max_length_output = example_target_batch.shape[1]

print(example_input_batch.shape, example_target_batch.shape)

embedding_dim = 50
units = 20
steps_per_epoch = num_examples//BATCH_SIZE

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE, embedding_matrix_input, num_layers=1)
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE, max_length_input, max_length_output, embedding_matrix_target)

lstm_model = LSTM_custom(encoder, decoder, units, max_length_input, dataset_creator, BATCH_SIZE)

TRAIN = True
if TRAIN == True:
    lstm_model.train(train_dataset, val_dataset, 15, steps_per_epoch, patience=5)
    hist = lstm_model.get_training_history()
else:
    lstm_model.load_model("training_checkpoints")
    hist = pickle.load(open("training_history", "rb"))

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


evaluator = Evaluator(problem_condtions, problem_solutions, lstm_model, target_lang_tokenizer)
print(evaluator.bleu_scores())
print(evaluator.meteor_scores())
print(evaluator.code_bert_scores())