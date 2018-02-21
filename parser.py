import tensorflow as tf

print ("Version of tensorflow is {}".format(tf.__version__))

""" GENERAL SETTINGS """
tf.app.flags.DEFINE_string("mode", "train_and_evaluate", "Mode (train/test/train_and_evaluate)")

tf.app.flags.DEFINE_string("model", "seq2seq", "Choose the type of model. (seq2seq/s2vt)")
tf.app.flags.DEFINE_string("logdir", "logs/s2s_baseline", "Name of output folder")
tf.app.flags.DEFINE_string("task_name", "seq2seq", "Name of this training task")
tf.app.flags.DEFINE_bool("reset", False, "Training start from stratch")
tf.app.flags.DEFINE_bool("comp", False, "Whether make comparison table")

# PATHS
tf.app.flags.DEFINE_string("vocab_path", "processed/vocab_wc3", "Preprocessed vocab")

tf.app.flags.DEFINE_string("train_feature_path", "MLDS_hw2_data/training_data/feat", "Folder containing features")
tf.app.flags.DEFINE_string("train_vid_path", "processed/train_wc3sent5_vid.enc", "File of video indices")
tf.app.flags.DEFINE_string("train_capid_path", "processed/train_wc3sent5_capid.dec", "File of captions")

tf.app.flags.DEFINE_string("test_feature_path", "MLDS_hw2_data/testing_data/feat", "Folder containing features")
tf.app.flags.DEFINE_string("test_vid_path", "processed/test_wc3sent5_vid.enc", "File of video indices")
tf.app.flags.DEFINE_string("test_capid_path", "processed/test_wc3sent5_capid.dec", "File of captions")

tf.app.flags.DEFINE_string("out_name", "sample_output_testset.txt", "Output path")

""" DATASET SPECS """
tf.app.flags.DEFINE_integer("frame_dim", 4096, "Dimension of the video features")
tf.app.flags.DEFINE_integer("frame_steps", 80, "Time steps of the input video")
tf.app.flags.DEFINE_integer("down_sample", 1, "Interval of the down sampling")
tf.app.flags.DEFINE_integer("max_seq_length", 95, "Maximum length of the captions")
tf.app.flags.DEFINE_integer("max_video_length", 80, "Maximum length of input video features")
tf.app.flags.DEFINE_integer("vocab_size", 2996, "Size of the vocab (6158/2996)")

tf.app.flags.DEFINE_integer("PAD_ID", 0, "PAD_ID")
tf.app.flags.DEFINE_integer("UNK_ID", 1, "UNK_ID")
tf.app.flags.DEFINE_integer("BOS_ID", 2, "BOS_ID")
tf.app.flags.DEFINE_integer("EOS_ID", 3, "EOS_ID")

""" ARCHITECTURE """
tf.app.flags.DEFINE_integer("num_units", 128, "Hidden unit size of RNN")
tf.app.flags.DEFINE_integer("num_layers", 1, "Depth of the RNN")
tf.app.flags.DEFINE_string("cell_type", "LSTM", "Type of the RNN Cell")
tf.app.flags.DEFINE_float("dropout", 0.5, "Dropout probability")

tf.app.flags.DEFINE_bool("ortho_init", True, "Orthogonal initialization on LSTM Cell")
tf.app.flags.DEFINE_bool("bidir", True, "Set True to build bi-directional encoder")
tf.app.flags.DEFINE_string("atten", "luong", "Types of attention mechanism (bahdanau/luong)")

""" TRAINING """
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size")
tf.app.flags.DEFINE_float("length_penalty_weight", 1.0, "Length penalty used in decoder ")
tf.app.flags.DEFINE_integer("beam_width", 10, "Beam search width")
tf.app.flags.DEFINE_bool("sched_sample", False, "Enable Scheduled Sampling in training stage")
#tf.app.flags.DEFINE_string("sched_decay", "linear", "Strategy of scheduled samping probability decay")

tf.app.flags.DEFINE_string("optimizer_type", "RMSProp", "Optimizer can be choosen as SGD/ADAM/RMSProp")
tf.app.flags.DEFINE_float("learning_rate", 1e-3, "Initial learning rate")
tf.app.flags.DEFINE_float("decay_rate", 0.25, "Decay rate")
tf.app.flags.DEFINE_integer("train_steps", 6000, "Number of training steps(200 epochs, roughly)")
tf.app.flags.DEFINE_integer("decay_steps", 1000, "Decay after # of steps")
tf.app.flags.DEFINE_bool("stair", True, "")

tf.app.flags.DEFINE_integer("save_checkpoints_steps", 200, "# of times after checkpoints are ready")
tf.app.flags.DEFINE_integer("min_eval_frequency", 1, "# of times after checkpoints are ready")
tf.app.flags.DEFINE_integer("check_hook_n_iter", 100, "")

FLAGS = tf.app.flags.FLAGS