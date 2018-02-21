"""Configurations
"""

class Configurations:
    def __init__(self):

        # data
        self.PAD_ID = 0
        self.UNK_ID = 1
        self.START_ID = 2
        self.EOS_ID = 3

        # todo
        self.vocab_size = 47691

        self.max_seq_length = 200

        self.frame_dim = 4096
        self.frame_steps = 80

        self.processed_path = "data/processed"

        # seq2seq model
        #self.embed_share = True
        self.embed_dim = 256
        self.num_layers = 2
        self.num_units = 128
        self.cell_type = "LSTM"
        self.dropout = 0.25
        self.beam_width = 5
        self.length_penalty_weight = 1.0

        # training
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.train_steps = 100000
        self.min_eval_frequency = 1000
        self.save_every = 1000
        self.loss_hook_n_iter = 1000
        self.check_hook_n_iter = 1000

        # utils
        self.model_dir = "logs/seq2seq"