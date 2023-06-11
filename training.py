import tensorflow as tf
import tensorflow_addons as tfa
import os
import time
import numpy as np

from tqdm import tqdm

class LSTM_custom:
    def __init__(self, encoder, decoder, units, max_length_input, dataset_creator, batch_size):
        self.encoder = encoder
        self.decoder = decoder
        self.BATCH_SIZE = batch_size
        self.units = units
        self.max_length_input = max_length_input
        self.dataset_creator = dataset_creator
        self.optimizer = tf.keras.optimizers.Adam()

        self.checkpoint = None

    def loss_function(self, real, pred):
        # real shape = (BATCH_SIZE, max_length_output)
        # pred shape = (BATCH_SIZE, max_length_output, tar_vocab_size )
        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        loss = cross_entropy(y_true=real, y_pred=pred)
        mask = tf.logical_not(tf.math.equal(real,0))   #output 0 for y=0 else output 1
        mask = tf.cast(mask, dtype=loss.dtype)  
        loss = mask * loss
        loss = tf.reduce_mean(loss)
        return loss
    
    @tf.function
    def train_step(self, inp, targ, enc_hidden):
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_h, enc_c = self.encoder(inp, enc_hidden)

            dec_input = targ[ : , :-1 ] # Ignore <end> token
            real = targ[ : , 1: ]         # ignore <start> token

            # Set the AttentionMechanism object with encoder_outputs
            self.decoder.attention_mechanism.setup_memory(enc_output)

            # Create AttentionWrapperState as initial_state for decoder
            decoder_initial_state = self.decoder.build_initial_state(self.BATCH_SIZE, [enc_h, enc_c], tf.float32)
            pred = self.decoder(dec_input, decoder_initial_state)
            logits = pred.rnn_output
            loss = self.loss_function(real, logits)

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss
    
    @tf.function
    def validation_step(self, inp, targ, enc_hidden):
        enc_output, enc_h, enc_c = self.encoder(inp, enc_hidden)
        dec_input = targ[ : , :-1 ] # Ignore <end> token
        real = targ[ : , 1: ]         # ignore <start> token

        # Set the AttentionMechanism object with encoder_outputs
        self.decoder.attention_mechanism.setup_memory(enc_output)

        # Create AttentionWrapperState as initial_state for decoder
        decoder_initial_state = self.decoder.build_initial_state(self.BATCH_SIZE, [enc_h, enc_c], tf.float32)
        pred = self.decoder(dec_input, decoder_initial_state)
        logits = pred.rnn_output
        loss = self.loss_function(real, logits)

        return loss

    def train(self, train_dataset, val_dataset, EPOCHS, steps_per_epoch, patience = None):
        checkpoint_prefix = os.path.join('./training_checkpoints', "ckpt")
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                        encoder=self.encoder,
                                        decoder=self.decoder)
        
        minimum_epoch_loss = np.inf
        patient_round = 0
        for epoch in range(EPOCHS):
            start = time.time()

            enc_hidden = self.encoder.initialize_hidden_state()
            # print(enc_hidden[0].shape, enc_hidden[1].shape)

            """@TODO: maybe shuffle data before each batch computation """
            input_progressBar = tqdm(enumerate(train_dataset.take(steps_per_epoch)))

            total_loss = 0
            for batch, (inp, targ) in input_progressBar:
                batch_loss = self.train_step(inp, targ, enc_hidden)

                total_loss += batch_loss

                input_progressBar.set_description(f"Epoch: {epoch+1} === Loss: {batch_loss.numpy()} === Batch: {batch+1}/{steps_per_epoch}")

            # saving (checkpoint) the model every 2 epochs
            if (epoch + 1) % 2 == 0:
                self.checkpoint.save(file_prefix = checkpoint_prefix)

            val_size = int(steps_per_epoch * 0.25)
            val_dataset = val_dataset.shuffle(val_size, reshuffle_each_iteration=True)
            val_progressBar = tqdm(enumerate(val_dataset.take(val_size)))

            total_val_loss = 0
            for batch, (val_inp, val_targ) in val_progressBar:
                val_batch_loss = self.validation_step(val_inp, val_targ, enc_hidden)

                total_val_loss += val_batch_loss

                val_progressBar.set_description(f"Epoch: {epoch+1} === Val loss: {val_batch_loss.numpy()} === Batch: {batch+1}/{val_size}")

            loss = total_loss / steps_per_epoch
            loss_val = total_val_loss / val_size

            if patience is not None:
                if loss_val < minimum_epoch_loss:
                    minimum_epoch_loss = loss_val
                    patient_round = 0
                else: 
                    patient_round += 1

                if patient_round >= patience:
                    print(f"Stopping training early; loss has not improved in {patient_round} epochs")
                    break

            print('Epoch {} Loss {:.4f} Val loss {:.4f}'.format(epoch + 1, loss, loss_val))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    def __evaluate_sentence(self, inp_lang, targ_lang, sentence):
        sentence = self.dataset_creator.preprocess_sentence(sentence)

        inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                                maxlen=self.max_length_input,
                                                                padding='post')
        inputs = tf.convert_to_tensor(inputs)
        inference_batch_size = inputs.shape[0]
        result = ''

        enc_start_state = [tf.zeros((inference_batch_size, self.units)), tf.zeros((inference_batch_size,self.units))]
        enc_out, enc_h, enc_c = self.encoder(inputs, enc_start_state)

        dec_h = enc_h
        dec_c = enc_c

        start_tokens = tf.fill([inference_batch_size], targ_lang.word_index['<start>'])
        end_token = targ_lang.word_index['<end>']

        greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler()

        # Instantiate BasicDecoder object
        decoder_instance = tfa.seq2seq.BasicDecoder(cell=self.decoder.rnn_cell, sampler=greedy_sampler, output_layer=self.decoder.fc)
        # Setup Memory in decoder stack
        self.decoder.attention_mechanism.setup_memory(enc_out)

        # set decoder_initial_state
        decoder_initial_state = self.decoder.build_initial_state(inference_batch_size, [enc_h, enc_c], tf.float32)


        ### Since the BasicDecoder wraps around Decoder's rnn cell only, you have to ensure that the inputs to BasicDecoder 
        ### decoding step is output of embedding layer. tfa.seq2seq.GreedyEmbeddingSampler() takes care of this. 
        ### You only need to get the weights of embedding layer, which can be done by decoder.embedding.variables[0] and pass this callabble to BasicDecoder's call() function

        decoder_embedding_matrix = self.decoder.embedding.variables[0]

        outputs, _, _ = decoder_instance(decoder_embedding_matrix, start_tokens = start_tokens, end_token= end_token, initial_state=decoder_initial_state)
        return outputs.sample_id.numpy()
    
    def translate(self, inp_lang, targ_lang, sentence):
        result = self.__evaluate_sentence(inp_lang, targ_lang, sentence)
        print(result)
        result = targ_lang.sequences_to_texts(result)
        print('Input: %s' % (sentence))
        print('Predicted translation: {}'.format(result))

    def load_model(self, path):
        # restoring the latest checkpoint in checkpoint_dir
        self.checkpoint.restore(tf.train.latest_checkpoint(path))