import pickle
import tensorflow as tf
import tensorflow_addons as tfa
import time
import numpy as np

from tqdm import tqdm

class LSTM_custom(tf.keras.Model):
    def __init__(self, encoder, decoder, units, max_length_input, dataset_creator, batch_size):
        super(LSTM_custom, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.BATCH_SIZE = batch_size
        self.units = units
        self.max_length_input = max_length_input
        self.dataset_creator = dataset_creator

        initial_learning_rate = 1e-3
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=10000,
            decay_rate=0.96,
            staircase=True)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


        self.history = []
        self.checkpoint = self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                        encoder=self.encoder,
                                        decoder=self.decoder
                                        )
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, directory="./training_checkpoints", max_to_keep=3)
        

    @tf.function
    def loss_function(self, real, pred):
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

            dec_input = targ[ : , :-1 ] # Ignore <eos> token
            real = targ[ : , 1: ]         # ignore <sos> token

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
        dec_input = targ[ : , :-1 ] # Ignore <eos> token
        real = targ[ : , 1: ]         # ignore <sos> token

        # Set the AttentionMechanism object with encoder_outputs
        self.decoder.attention_mechanism.setup_memory(enc_output)

        # Create AttentionWrapperState as initial_state for decoder
        decoder_initial_state = self.decoder.build_initial_state(self.BATCH_SIZE, [enc_h, enc_c], tf.float32)
        pred = self.decoder(dec_input, decoder_initial_state)
        logits = pred.rnn_output
        loss = self.loss_function(real, logits)

        return loss

    def train(self, train_dataset, val_dataset, EPOCHS, steps_per_epoch, patience = None):
        minimum_epoch_loss = np.inf
        patient_round = 0

        for epoch in range(EPOCHS):
            start = time.time()

            enc_hidden = self.encoder.initialize_hidden_state()

            train_dataset = train_dataset.shuffle(steps_per_epoch)
            train_batch_data = train_dataset.take(steps_per_epoch)
            input_progressBar = tqdm(enumerate(train_batch_data))

            total_loss = 0
            for batch, (inp, targ) in input_progressBar: # mini-batch approach
                batch_loss = self.train_step(inp, targ, enc_hidden)

                total_loss += batch_loss

                input_progressBar.set_description(f"Epoch: {epoch+1} === Loss: {total_loss/(batch+1):.3f} === Batch: {batch+1}/{len(train_batch_data)} === Patience: {patient_round}")

            val_dataset = val_dataset.shuffle(steps_per_epoch, reshuffle_each_iteration=True)
            val_batch_data = val_dataset.take(steps_per_epoch)
            val_progressBar = tqdm(enumerate(val_batch_data))

            total_val_loss = 0
            for batch, (val_inp, val_targ) in val_progressBar:
                val_batch_loss = self.validation_step(val_inp, val_targ, enc_hidden)

                total_val_loss += val_batch_loss

                val_progressBar.set_description(f"Epoch: {epoch+1} === Val loss: {total_val_loss/(batch+1):.3f} === Batch: {batch+1}/{len(val_batch_data)}")

            loss = total_loss / steps_per_epoch
            loss_val = total_val_loss / steps_per_epoch

            self.history.append([loss, loss_val])
            self.checkpoint_manager.save() # save the last 3 checkpoints

            if patience is not None:
                if loss_val < minimum_epoch_loss:
                    minimum_epoch_loss = loss_val
                    patient_round = 0
                else: 
                    patient_round += 1

                if patient_round >= patience:
                    print(f"Stopping training early; loss has not improved in {patient_round} epochs")
                    break

            print('Epoch {} Loss {:.3f} Val loss {:.3f}'.format(epoch + 1, loss, loss_val))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
        
        pickle.dump(np.asarray(self.history), open("training_history", "wb"))
        self.save_weights("weights.h5")

    def get_training_history(self):
        return np.asarray(self.history)

    def evaluate_sentence(self, targ_lang, sentence):
        inputs = self.dataset_creator.process_one_sentence(sentence)

        inputs = tf.convert_to_tensor(inputs)
        inference_batch_size = inputs.shape[0]

        enc_start_state = [tf.zeros((inference_batch_size, self.units)), tf.zeros((inference_batch_size,self.units))]
        enc_out, enc_h, enc_c = self.encoder(inputs, enc_start_state)

        start_tokens = tf.fill([inference_batch_size], targ_lang.word_index['<sos>'])
        end_token = targ_lang.word_index['<eos>']

        # greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler()
        greedy_sampler = tfa.seq2seq.SampleEmbeddingSampler(self.decoder.embedding)

        # Instantiate BasicDecoder object
        # DO NOT REMOVE MAXIMUM ITERATIONS!!! - the decoder will get stuck in an infinite loop otherwise
        decoder_instance = tfa.seq2seq.BasicDecoder(cell=self.decoder.rnn_cell, sampler=greedy_sampler, output_layer=self.decoder.fc) # change sampler from training to greedy to extract result from embedding
        # # Setup Memory in decoder stack
        self.decoder.attention_mechanism.setup_memory(enc_out)

        # # set decoder_initial_state
        decoder_initial_state = self.decoder.build_initial_state(inference_batch_size, [enc_h, enc_c], tf.float32)

        decoder_embedding_matrix = self.decoder.embedding.variables[0]

        outputs, _, _ = decoder_instance(decoder_embedding_matrix, start_tokens = start_tokens, end_token = end_token, initial_state=decoder_initial_state)
        
        return outputs.sample_id.numpy()
    
    def translate(self, inp_lang, targ_lang, sentence):
        result = self.evaluate_sentence(inp_lang, targ_lang, sentence)
        print(result)
        result = targ_lang.sequences_to_texts(result)
        print('Input: %s' % (sentence))
        print('Predicted translation: {}'.format(result))

    def load_model(self, path):
        # restoring the latest checkpoint in checkpoint_dir
        self.checkpoint.restore(tf.train.latest_checkpoint(path))