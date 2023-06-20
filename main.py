from dataloader import *
from architecture import *
from training import *
from evaluation_metrics import *

import matplotlib.pyplot as plt
import dill
import sys

import nltk

def read_embeddings(file):
    glove_embeddings = {}
    with open(file, "r") as f:
        no_lines = 1
        line = f.readline() # read header line
        line = re.sub(r"\n", "", line)
        total_lines, embedding_dim = line.split(" ")
        total_lines, embedding_dim = int(total_lines), int(embedding_dim)
        line = f.readline() # actual first line
        while line != "":
            sys.stdout.write("\r" + f"Read line {no_lines}/{total_lines}") 
            line = re.sub(r"\n", "", line)
            line = line.split(" ")
            glove_embeddings[line[0]] = list(map(float, line[1:])) # one word per line
            line = f.readline()
            no_lines += 1

    return glove_embeddings

BUFFER_SIZE = 4000
BATCH_SIZE = 24
num_examples = 3000


glove_embeddings = read_embeddings("processed/glove-wiki-gigaword-50-word2vec.txt")
exit()

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

file = "processed/preprocessed_text.csv"



tf.keras.backend.clear_session()

dataset_creator = NMTDataset('NL-PL')
train_dataset, val_dataset, inp_lang, targ_lang = dataset_creator.call(num_examples, file, BUFFER_SIZE, BATCH_SIZE)

example_input_batch, example_target_batch = next(iter(train_dataset))

vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1
max_length_input = example_input_batch.shape[1]
max_length_output = example_target_batch.shape[1]

embedding_dim = 700
units = 512
steps_per_epoch = num_examples//BATCH_SIZE

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE, num_layers=20)
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE, max_length_input, max_length_output)

lstm_model = LSTM_custom(encoder, decoder, units, max_length_input, dataset_creator, BATCH_SIZE)

TRAIN =  not True
if TRAIN:
    lstm_model.train(train_dataset, val_dataset, 5, steps_per_epoch, patience=5)
    hist = lstm_model.get_training_history()
    decoder.save_weights("decoder.h5")
else:
    # lstm_model.load_model("training_checkpoints/")
    lstm_model.decoder.build((max_length_input, embedding_dim))
    lstm_model.decoder.load_weights("decoder.h5")
    # lstm_model.load_weights("test.h5")
    hist = dill.load(open("training_history", "rb"))

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


evaluator = Evaluator(problem_condtions, problem_solutions, lstm_model, inp_lang, targ_lang)
print(evaluator.bleu_scores())
print(evaluator.meteor_scores())
print(evaluator.code_bert_scores())