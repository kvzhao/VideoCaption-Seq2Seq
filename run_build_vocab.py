"""Build Vocab
"""

import os
import argparse
from preprocess_data import build_vocab, token2id

parser = argparse.ArgumentParser()

parser.add_argument("--out_path", dest="out_path", default="vocab", type=str, 
                    help="postfix")

parser.add_argument("--data_dir", dest="data_dir", default="MLDS_hw2_data", type=str, 
                    help="postfix")

parser.add_argument("--wcount", dest="wcount", default=3, type=int, 
                    help="Word counts threshold")

FLAGS = parser.parse_args()


def main():
    print ("Building vocab")
    build_vocab(data_dir=FLAGS.data_dir, out_path=FLAGS.out_path, threshold=FLAGS.wcount)


if __name__=="__main__":
    main()