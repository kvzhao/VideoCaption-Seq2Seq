import tensorflow as tf                                        
from parser import FLAGS

def print_variables(variables, vocab=None, every_n_iter=100):
    """Print given tensor n local steps or n sec"""
    return tf.train.LoggingTensorHook(variables, 
                                    every_n_iter=every_n_iter,
                                    formatter=format_variable(variables, vocab=vocab))


def format_variable(keys, vocab=None):
    """format the input variables"""
    rev_vocab = get_rev_vocab(vocab)

    def to_str(seqeunce):
        tokens = [rev_vocab.get(x, '') for x in seqeunce if x != FLAGS.PAD_ID]
        return " ".join(tokens)

    def format(values):
        result=[]
        for key in keys:
            if vocab is None:
                result.append("{} = {}".format(key, values[key]))
            else:
                result.append("{} = {}".format(key, to_str(values[key])))
        try:
            print("\n - ".join(result))
        except:
            pass

    return format


def get_rev_vocab(vocab):
    if vocab is None:
        return None
    return {idx: key for key, idx in vocab.items()}