""" Seq2Seq Video Caption
"""
import os
import json
import tensorflow as tf
import experiment
from parser import FLAGS

from pprint import pprint

def main(mode):
    print ("mode: {}".format(mode))


    params = dict(FLAGS.__flags.items())

    # Save params to logdir
    if not os.path.exists(FLAGS.logdir):
        os.makedirs(FLAGS.logdir)
    with open("/".join([FLAGS.logdir, "hparams.json"]), "w") as file:
        json.dump(params, file)

    params = tf.contrib.training.HParams(**params)

    pprint (params)

    """RunConfig:
        https://www.tensorflow.org/api_docs/python/tf/contrib/learn/RunConfig
    """

    run_config = tf.contrib.learn.RunConfig(
            model_dir=params.logdir,
            save_checkpoints_steps=params.save_checkpoints_steps,
            gpu_memory_fraction=0.5,
            save_summary_steps=100
    )

    """Runner 
        https://www.tensorflow.org/api_docs/python/tf/contrib/learn/learn_runner/run
    """

    tf.contrib.learn.learn_runner.run(
        experiment_fn=experiment.experiment_fn,
        run_config=run_config,
        schedule=mode,
        hparams=params
    )

if __name__ == "__main__":
    main(FLAGS.mode)