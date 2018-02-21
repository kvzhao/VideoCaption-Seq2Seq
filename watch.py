"""
    Give the video list and feature path

    Assign by this two flags:
        --test_feature_path
        --test_vid_path
"""

import json
import os, sys
import numpy as np
import tensorflow as tf

import dataset

from pprint import pprint

import seq2seq_model
import s2vt_model
import os
import json
import preprocess_data
from parser import FLAGS
from hook import get_rev_vocab
from bleu_eval import BLEU

# for evaluating BLEU score
TESTING_LABEL="MLDS_hw2_data/testing_label.json"

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def _make_estimator(params):

    print ("Params are recovered: {}".format(params))

    run_config = tf.contrib.learn.RunConfig(
            model_dir=FLAGS.logdir,
            session_config=tf.ConfigProto(
                device_count={"GPU": 0}
            )
    )

    if params.model == "seq2seq":
        model = seq2seq_model.Seq2Seq()
    elif params.model == "s2vt":
        model = s2vt_model.S2VT()

    return tf.estimator.Estimator(
            model_fn=model.model_fn,
            model_dir=FLAGS.logdir,
            params=params,
            config=run_config)

def make_submit(predictions, vid_list):
    with open(FLAGS.out_name, "w") as f:
        # Note: Orders are not checked
        for vid, pred in zip(vid_list, predictions):
            pred.rstrip(".")
            f.write("{},{}\n".format(vid, pred))

def make_compare_table(predictions, vid_list, ground_path):
    # reading the ground truth
    with open(ground_path, 'rb') as f:
        grounds = [x.decode("utf-8").strip() for x in f.readlines()]

    # TODO: Put outcomes in dict
    result={}
    FLAGS.logdir = FLAGS.logdir.rstrip("/")
    with open(FLAGS.logdir + "_report.log", "w") as f:
        # Note: Orders are not checked
        for vid, pred, tgt in zip(vid_list, predictions, grounds):
            print ("Video: {}".format(vid))
            print ("\t Target: {}".format(tgt))
            print ("\t Predict: {}".format(pred))
            f.write("Video: {}\n".format(vid))
            f.write("\t Target: {}\n".format(tgt))
            f.write("\t Predict: {}\n".format(pred))
            pred.rstrip(".")
            result[vid] = pred

    test = json.load(open('MLDS_hw2_data/testing_label.json','r'))

    """ Calculate BLEU Score: from demo code"""
    bleu=[]
    for item in test:
        score_per_video = []
        for caption in item['caption']:
            caption = caption.rstrip('.')
            score_per_video.append(BLEU(result[item['id']],caption))
        bleu.append(sum(score_per_video)/len(score_per_video))
    average = sum(bleu) / len(bleu)
    print("Originally, average bleu score is " + str(average))
    bleu=[]
    for item in test:
        score_per_video = []
        captions = [x.rstrip('.') for x in item['caption']]
        score_per_video.append(BLEU(result[item['id']],captions,True))
        bleu.append(score_per_video[0])
    average = sum(bleu) / len(bleu)
    print("By another method, average bleu score is " + str(average))

def watch_videos(features, vocab, params):
    """
        args:
            params (Hparams)
    """

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"input_data": features}, # input feature is a dict
        batch_size=params.batch_size,
        num_epochs=1,
        shuffle=False
    )

    estimator = _make_estimator(params)
    results = estimator.predict(input_fn=predict_input_fn)

    predictions = np.array([res["prediction"] for res in results])

    rev_vocab = get_rev_vocab(vocab)

    def to_str(sequence):
        tokens = [
            rev_vocab.get(x, "") for x in sequence if (x != FLAGS.PAD_ID and
                                                        x != FLAGS.EOS_ID and 
                                                        x != 5)
        ]
        return " ".join(tokens)

    return [to_str(pred) for pred in predictions]


def main():

    """ Load testing dataset """
    vocab = preprocess_data.load_vocab(FLAGS.vocab_path)

    vid_path = FLAGS.test_vid_path
    feat_path = FLAGS.test_feature_path

    features, video_ids = dataset.get_video_features(
        feat_path, vid_path
    )

    num_videos = len(features)
    print ("{} videos ({})".format(num_videos, features.shape))

    """ Load parameters """

    # Load configuration from trained model
    with open("/".join([FLAGS.logdir, "hparams.json"]), "r") as file:
        saved = json.load(file)
    params = Config(**saved).__dict__
    pprint (params)

    params = tf.contrib.training.HParams(**params)
    params.batch_size = num_videos

    captions = watch_videos(features, vocab, params)
    make_submit(captions, video_ids)

    if FLAGS.comp:
        make_compare_table(captions, video_ids, "processed/test_normal_cap")

    print ("Done.")

if __name__ == "__main__":
    main()
