import tensorflow as tf
import os, sys
import json
import shutil

import seq2seq_model

def load_model(sess, model_type, saved_path, mode):
    class Config:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    
    # Load configuration from trained model
    with open("/".join([saved_path, "config.json"]), "r") as file:
        saved = json.load(file)
    config = Config(**saved)

    if model_type == "seq2seq":
        model = seq2seq_model(config, mode)
        model.build()
    elif model_type == "s2vt":
        pass



    print ("Restore from previous results...")
    ckpt = tf.train.get_checkpoint_state(saved_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print ("saved path: {}".format(saved_path))
        model.restore(sess, saved_path)
        print ("Model reloaded from {}".format(ckpt.model_checkpoint_path))
    else:
        raise ValueError("FAIL TO LOAD CHECKPOINTS!")
    return model


def create_model(sess, model_type, FLAGS, mode):
    """Create model only used for train mode.
    """

    if model_type == "seq2seq":
        model = seq2seq_model.Seq2Seq(FLAGS, mode)
        model.build()
    elif model_type == "s2vt":
        pass

    # create task file
    model_path = os.path.join(FLAGS.logdir, FLAGS.task_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        os.makedirs(model_path+"/eval")
        print ("Save model to {}".format(model_path))
        # Build new model from scratch using FLAGS configurations and Save configurations
    elif (FLAGS.reset):
        shutil.rmtree(model_path)
        os.makedirs(model_path)
        print ("Remove existing model at {} and restart.".format(model_path))
    else:
        #ERROR
        raise ValueError("Fail to create the new model.")

    # Save the current configurations
    config = dict(FLAGS.__flags.items())
    with open("/".join([model_path, "config.json"]), "w") as file:
        json.dump(config, file)

    # initialize variables
    sess.run(tf.global_variables_initializer())

    return model