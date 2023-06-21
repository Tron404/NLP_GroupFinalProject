import tensorflow as tf
import tensorflow_addons as tfa

# import keras_nlp

class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, embeddings, num_layers = 1):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embeddings])

    ##-------- LSTM layer in Encoder ------- ##
    self.lstm_layers = []
    for idx in range(num_layers):
      self.lstm_layers.append(tf.keras.layers.LSTM(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform',
                                   name=f"LSTM{idx}",
                                   dropout=0.1
                                  #  recurrent_regularizer="l1"
                                   )
                                )
      
    self.batch_norm = tf.keras.layers.BatchNormalization()
      
  def call(self, x, hidden):
    x = self.embedding(x)
    for lstm_layer in self.lstm_layers:
      output, h, c = lstm_layer(x, initial_state = hidden)
      hidden = [h, c]
      x = output
    
    norm_output = self.batch_norm(output)

    return norm_output, h, c

  def initialize_hidden_state(self):
    return [tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units))]


class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, max_length_input, max_length_output, embeddings):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units

    self.max_length_input = max_length_input
    self.max_length_output = max_length_output

    self.embeddings = embeddings

    # Embedding Layer
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights = [self.embeddings])

    # Define the fundamental cell for decoder recurrent structure
    self.decoder_rnn_cell = tf.keras.layers.LSTMCell(self.dec_units)

    #Final Dense layer on which softmax will be applied
    self.fc = tf.keras.layers.Dense(vocab_size)

    # Sampler
    self.sampler = tfa.seq2seq.sampler.TrainingSampler()

    # Create attention mechanism with memory = None
    self.attention_mechanism = self.build_attention_mechanism(self.dec_units, 
                                                              None, self.batch_sz*[self.max_length_input])

    # Wrap attention mechanism with the fundamental rnn cell of decoder
    self.rnn_cell = self.build_rnn_cell()
    # self.rnn_cell = tf.keras.layers.RNN(self.rnn_cell, return_state=True)

    # Define the decoder with respect to fundamental rnn cell
    self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, sampler=self.sampler, output_layer=self.fc)
  
  def get_weights(self):
    return self.embedding.weights

  def build_rnn_cell(self):
    return tfa.seq2seq.AttentionWrapper(self.decoder_rnn_cell, self.attention_mechanism, attention_layer_size=self.dec_units)

  def build_attention_mechanism(self, dec_units, memory, memory_sequence_length):
    # ------------- #
    # dec_units: final dimension of attention outputs 
    # memory: encoder hidden states of shape (batch_size, max_length_input, enc_units)
    # memory_sequence_length: 1d array of shape (batch_size) with every element set to max_length_input (for masking purpose)

    return tfa.seq2seq.LuongAttention(units=dec_units, memory=memory, memory_sequence_length=memory_sequence_length, scale=True) # scale = True --> global attention

  def build_initial_state(self, batch_sz, encoder_state, Dtype):
    decoder_initial_state = self.rnn_cell.get_initial_state(batch_size=batch_sz, dtype=Dtype)
    decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
    return decoder_initial_state

  def call(self, inputs, initial_state):
    x = self.embedding(inputs)
    outputs, _, _ = self.decoder(x, initial_state=initial_state, sequence_length=self.batch_sz*[self.max_length_output-1])
    return outputs
  