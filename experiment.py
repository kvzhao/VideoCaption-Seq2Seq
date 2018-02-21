import tensorflow as tf

import seq2seq_model
import s2vt_model
import dataset
import hook 

from preprocess_data import load_vocab

def experiment_fn(run_config, params):
    """ Experiemnt API """

    if params.model == "seq2seq":
        model = seq2seq_model.Seq2Seq()
    elif params.model == "s2vt":
        model = s2vt_model.S2VT()

    estimator = tf.estimator.Estimator(
            model_fn=model.model_fn,
            model_dir=params.logdir,
            params=params,
            config=run_config
    )

    vocab = load_vocab(params.vocab_path)

    train_videos, train_captions = dataset.data_reader(
            params.train_feature_path,
            params.train_vid_path,
            params.train_capid_path)

    test_videos, test_captions = dataset.data_reader(
            params.test_feature_path,
            params.test_vid_path,
            params.test_capid_path)
    
    train_input_fn, train_input_hook = dataset.get_train_inputs(
        train_videos, train_captions)
    
    test_input_fn, test_input_hook = dataset.get_test_inputs(
        test_videos, test_captions)

    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=train_input_fn,
        eval_input_fn=test_input_fn,
        train_steps=params.train_steps,
        min_eval_frequency=params.min_eval_frequency,
        train_monitors=[
            train_input_hook,
            hook.print_variables(variables=["Train_Data/caption_0", "train/pred_0"],
                        vocab=vocab,
                        every_n_iter=params.check_hook_n_iter)
        ],
        eval_hooks=[
            test_input_hook,
            hook.print_variables(variables=["Test_Data/caption_0", "train/pred_0"],
                        vocab=vocab,
                        every_n_iter=params.check_hook_n_iter)
        ]
    )

    return experiment