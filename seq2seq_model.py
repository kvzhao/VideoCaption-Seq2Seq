import tensorflow as tf

class Seq2Seq(object):
    def __init__(self):
        pass

    def model_fn(self, mode, features, labels, params):
        self.mode = mode
        self.params = params

        self._init_placeholder(features, labels)
        self.build_graph()

        if self.mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                        mode=mode,
                        predictions={"prediction": self.prediction})
        else:
            return tf.estimator.EstimatorSpec(
                        mode=mode,
                        predictions=self.decoder_train_pred,
                        loss=self.loss,
                        train_op=self.train_op)

    def build_graph(self):
        print ("Start building the Seq2Seq model:")
        self._build_embed()
        if self.params.bidir:
            self._build_bidirectional_encoder()
        else:
            self._build_encoder()
        self._build_decoder()

        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self._build_loss()
            self._build_optimizer()

    def _init_placeholder(self, features, captions):
        print ("build placehodlers...")

        self.encoder_video_inputs= features

        # Used for prediction
        if type(features) == dict:
            self.encoder_video_inputs = features["input_data"]

        self.encoder_video_lengths = [self.params.max_video_length] * self.params.batch_size

        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self.decoder_sequence_inputs = captions

            self.decoder_input_lengths = tf.reduce_sum(
                            tf.to_int32(tf.not_equal(
                                    self.decoder_sequence_inputs, 
                                    self.params.PAD_ID)), axis=1,
                                    name="decoder_input_lengths")
            decoder_shift_by_one = tf.slice(self.decoder_sequence_inputs, [0, 1],
                                            [self.params.batch_size, self.params.max_seq_length-1])
            pad_tokens = tf.zeros(shape=[self.params.batch_size, 1], dtype=tf.int32)

            self.decoder_sequence_targets = tf.concat([decoder_shift_by_one, pad_tokens], axis=1)


    def _build_embed(self):
        print ("build embedding...")
        with tf.device("/cpu:0"):
            with tf.variable_scope("embeddings"):
                embed_dim = self.params.num_units
                # visual embedding
                self.video_embedding_weight = tf.get_variable(
                        name="video_embedding_weight",
                        shape=[self.params.frame_dim, embed_dim],
                        dtype=tf.float32)
                self.video_embedding_bias= tf.get_variable(
                        name="video_embedding_bias",
                        shape=[embed_dim],
                        dtype=tf.float32)

                flatten_video = tf.reshape(self.encoder_video_inputs,
                                    shape=[-1, self.params.frame_dim])

                print ("Shape of the flatten vidoe: {}".format(flatten_video.get_shape()))

                self.encoder_emb_inp = tf.nn.xw_plus_b(
                        flatten_video,
                        self.video_embedding_weight,
                        self.video_embedding_bias)

                print ("Shape of embedding weight: {}".format(self.video_embedding_weight.get_shape()))
                print ("Shape of the encoder : {}".format(self.encoder_emb_inp.get_shape()))

                # reshape back to time series
                self.encoder_emb_inp = tf.reshape(
                        self.encoder_emb_inp,
                        [self.params.batch_size, self.params.max_video_length, embed_dim])

                print ("Shape of reshaped encoder : {}".format(self.encoder_emb_inp.get_shape()))

                self.embedding_decoder = tf.get_variable(
                        name="embedding_decoder",
                        shape=[self.params.vocab_size, embed_dim],
                        dtype=tf.float32)

            # seqeuence embedding
                if self.mode != tf.estimator.ModeKeys.PREDICT:

                    self.decoder_emb_inp = tf.nn.embedding_lookup(
                        self.embedding_decoder, self.decoder_sequence_inputs)

    def _build_encoder(self):
        """Build Encoder
        """
        print ("building encoder...")
        with tf.variable_scope("encoder"):
            cells = self._build_rnn_cells()
            self.encoder_outputs, self.encoder_final_states = tf.nn.dynamic_rnn(
                cells,
                self.encoder_emb_inp,
                sequence_length=self.encoder_video_lengths,
                dtype=tf.float32,
                time_major=False)

    def _build_bidirectional_encoder(self):
        print ("building bidirectional encoder...")
        with tf.variable_scope("bidirectional_encoder"):
            forward_cells = self._build_rnn_cells()
            backward_cells = self._build_rnn_cells()
            bi_outputs, self.encoder_final_states = tf.nn.bidirectional_dynamic_rnn(
                forward_cells,
                backward_cells,
                self.encoder_emb_inp,
                sequence_length=self.encoder_video_lengths,
                dtype=tf.float32,
                time_major=False
            )
            self.encoder_outputs = tf.concat(bi_outputs, -1)

    def _build_decoder(self):
        """Build Decoder
            Attention Mechanisms
        """
        def decode(helper=None, scope="decode"):
            """Decode with attention mechanism.
            """
            with tf.variable_scope(scope):
                if self.params.atten == "bahdanau":
                    print ("Bahdanau Attention Mechanism")
                    attention_machanism = tf.contrib.seq2seq.BahdanauAttention(
                        num_units=self.params.num_units, memory=self.encoder_outputs,
                        memory_sequence_length=self.encoder_video_lengths)
                elif self.params.atten == "luong":
                    print ("Luong Attention Mechanism")
                    attention_machanism = tf.contrib.seq2seq.LuongAttention(
                        num_units=self.params.num_units, memory=self.encoder_outputs,
                        memory_sequence_length=self.encoder_video_lengths)
                else:
                    pass
                
                cells = self._build_rnn_cells()

                alignment_history = (self.mode == tf.estimator.ModeKeys.PREDICT and self.params.beam_width==0)

                attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                    cells,
                    attention_machanism,
                    attention_layer_size=self.params.num_units,
                    alignment_history=alignment_history,
                    name="attention"
                )

                out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                    attn_cell, self.params.vocab_size)
                
                decoder_initial_state = out_cell.zero_state(self.params.batch_size, dtype=tf.float32)
                decoder_initial_state.clone(cell_state=self.encoder_final_states)

                if self.mode == tf.estimator.ModeKeys.PREDICT:

                    maximum_iterations = tf.round(tf.reduce_max(self.encoder_video_lengths) * 2)

                    if helper is None:
                        decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                            cell=out_cell,
                            embedding=self.embedding_decoder,
                            start_tokens=tf.fill([self.params.batch_size], self.params.BOS_ID),
                            end_token=self.params.EOS_ID,
                            initial_state=(decoder_initial_state),
                            beam_width=self.params.beam_width,
                            length_penalty_weight=self.params.length_penalty_weight)
                        outputs = tf.seq2seq.dynamic_decode(
                            decoder=decoder,
                            output_time_major=False,
                            impute_finished=False,
                            maximum_iterations=maximum_iterations)
                    else:
                        decoder = tf.contrib.seq2seq.BasicDecoder(
                            cell=out_cell,
                            helper=helper,
                            initial_state=(decoder_initial_state))
                        outputs = tf.contrib.seq2seq.dynamic_decode(
                            decoder=decoder,
                            output_time_major=False,
                            impute_finished=True,
                            maximum_iterations=maximum_iterations)

                else:
                    decoder = tf.contrib.seq2seq.BasicDecoder(
                        cell=out_cell,
                        helper=helper,
                        initial_state=(decoder_initial_state))
                    
                    outputs = tf.contrib.seq2seq.dynamic_decode(
                        decoder=decoder,
                        output_time_major=False,
                        swap_memory=True)

                return outputs[0]

        print ("building decoder...")
        with tf.variable_scope("decoder"):
            if self.mode == tf.estimator.ModeKeys.PREDICT:
                """ PREDICTION """
                self.params.beam_width = 0  # can not understand now
                if self.params.beam_width > 1:
                    self.decoder_pred_outputs = decode()
                    self.prediction = self.decoder_pred_outputs.predicted_ids
                else:
                    self.pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                        embedding=self.embedding_decoder,
                        start_tokens=tf.fill([self.params.batch_size], self.params.BOS_ID),
                        end_token=self.params.EOS_ID)
                    self.decoder_pred_outputs = decode(helper=self.pred_helper)
                    self.prediction = self.decoder_pred_outputs.sample_id
            else:
                """ TRAINING & EVALUATION """
                if self.params.sched_sample:
                    print ("with scheduled sampling")
                    #TODO: sampling prob decay

                    sample_prob = tf.train.exponential_decay(1.0, tf.train.get_global_step(), 600, 0.9)

                    self.train_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
                                            inputs=self.decoder_emb_inp,
                                            sequence_length=self.decoder_input_lengths,
                                            embedding=self.embedding_decoder,
                                            sampling_probability=sample_prob,
                                            time_major=False)

                else:
                    self.train_helper = tf.contrib.seq2seq.TrainingHelper(
                                            inputs=self.decoder_emb_inp,
                                            sequence_length=self.decoder_input_lengths)

                self.decoder_train_outputs = decode(self.train_helper, "decode")
                self.decoder_train_logits = self.decoder_train_outputs.rnn_output

        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self.decoder_train_pred = tf.argmax(self.decoder_train_logits[0], axis=1, name="train/pred_0")

    def _build_rnn_cells(self):
        stacked_cells = []

        for _ in range(self.params.num_layers):
            cell = self._single_cell(self.params.cell_type, self.params.dropout)
            stacked_cells.append(cell)
        
        return tf.nn.rnn_cell.MultiRNNCell(cells=stacked_cells, state_is_tuple=True)


    def _single_cell(self, cell_type, dropout):
        """
            * TODO: ortho init
        """
        if cell_type == "GRU":
            single_cell = tf.contrib.rnn.GRUCell(self.params.num_units)
        elif cell_type == "LSTM":
            if self.params.ortho_init:
                single_cell = tf.contrib.rnn.LSTMCell(self.params.num_units,
                    initializer=tf.orthogonal_initializer(),
                    forget_bias=1.0)
            else:
                single_cell = tf.contrib.rnn.LSTMCell(self.params.num_units, forget_bias=1.0)
        elif cell_type == "LAYER_NORM_LSTM":
            single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.params.num_units,
                forget_bias=1.0,
                layer_norm=True)
        elif cell_type == "NAS":
            single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.params.num_units)
        else:
            raise ValueError("Unkown cell type. {cell_type}")
        
        if dropout > 0.0:
            single_cell = tf.contrib.rnn.DropoutWrapper(
                cell=single_cell, input_keep_prob=(1.0-dropout))
        
        return single_cell

    def _build_loss(self):
        print ("building sequence loss...")
        pad_num = self.params.max_seq_length - tf.shape(self.decoder_train_logits)[1]
        zero_padding = tf.zeros(
            [self.params.batch_size, pad_num, self.params.vocab_size])
        zero_padded_logits = tf.concat([self.decoder_train_logits, zero_padding], axis=1)

        weight_masks = tf.sequence_mask(
                lengths=self.decoder_input_lengths,
                maxlen=self.params.max_seq_length,
                dtype=tf.float32, name="masks")
        
        self.loss = tf.contrib.seq2seq.sequence_loss(
                logits=zero_padded_logits,
                targets=self.decoder_sequence_targets,
                weights=weight_masks,
                name="loss")

    def _build_optimizer(self):
        print ("building {} optimizer...".format(self.params.optimizer_type))
        self.train_op = tf.contrib.layers.optimize_loss(
            self.loss, tf.train.get_global_step(),
            optimizer=self.params.optimizer_type,
            learning_rate=self.params.learning_rate,
            learning_rate_decay_fn=lambda lr, gstep: tf.train.exponential_decay(
            learning_rate=lr,
            global_step=gstep,
            decay_steps=self.params.decay_steps,
            decay_rate=0.95,
            staircase=self.params.stair),
            summaries=["loss", "learning_rate"],
            name="train_op")