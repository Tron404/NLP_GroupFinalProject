from dataloader import *
from architecture import *
from training import *
from preprocessing_functionality import *

BUFFER_SIZE = 62000
BATCH_SIZE = 100
num_examples = 60000

tf.keras.backend.clear_session()
#### getting our actual data set
x_train_encoded, x_test_encoded, y_train_encoded, y_test_encoded, embedding_matrix = main_preprocess(data_file, preprocessed_file, preprocessed_folder)
####

dataset_creator = NMTDataset('en-ron')
train_dataset, val_dataset, inp_lang, targ_lang = dataset_creator.call(num_examples, "ron.txt", BUFFER_SIZE, BATCH_SIZE)

example_input_batch, example_target_batch = next(iter(train_dataset))
print(example_input_batch.shape, example_target_batch.shape)

vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1
max_length_input = example_input_batch.shape[1]
max_length_output = example_target_batch.shape[1]

embedding_dim = 768
units = 512
steps_per_epoch = num_examples//BATCH_SIZE

print(max_length_input)

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE, max_length_input, max_length_output, 'luong')

lstm_model = LSTM_custom(encoder, decoder, units, max_length_input, dataset_creator, BATCH_SIZE)

lstm_model.train(train_dataset, val_dataset, 150, steps_per_epoch, patience=5)

lstm_model.translate(inp_lang, targ_lang, u"Buna!")
