""" The dataset preprocessing 
	Use training_id.txt and testing_id.txt as root (list of video filenames)
"""
import os
import json
import re
import random

def basic_tokenizer(line, normalize_digits=True):
    """ A basic tokenizer to tokenize text into tokens.
    Feel free to change this to suit your need. """
    line = re.sub('<u>', '', line)
    line = re.sub('</u>', '', line)
    line = re.sub('\[', '', line)
    line = re.sub('\]', '', line)
    words = []
    _WORD_SPLIT = re.compile("([.,!?\"'-<>:;)(])")
    #_WORD_SPLIT = re.compile(b"([.,!?\"'-<>:;)(])")
    _DIGIT_RE = re.compile(r"\d")
    for fragment in line.strip().lower().split():
        for token in re.split(_WORD_SPLIT, fragment):
            if not token:
                continue
            if normalize_digits:
                token = re.sub(_DIGIT_RE, '#', token)
                #token = re.sub(_DIGIT_RE, b'#', token)
            words.append(token)
    return words

def build_vocab(data_dir, out_path, threshold=0):
    """Build the vocab from dataset (which hierarchy we already know)
        Args:
            data_dir: path to dataset
            out_path: output path of vocab
    """
    out_folder = "processed"
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)
    
    out_path = out_folder + "/" + out_path

    path_training_label = os.path.join(data_dir, "training_label.json")
    path_test_label = os.path.join(data_dir, "testing_label.json")

    training_labels = json.load(open(path_training_label, "rb"))
    testing_labels = json.load(open(path_test_label, "rb"))

    labels = training_labels + testing_labels

    vocab = {}

    for video_cpations in labels:
        for captions in video_cpations["caption"]:
            for token in basic_tokenizer(captions):
                if not token in vocab:
                    vocab[token] = 0
                vocab[token] += 1

    sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)
    with open(out_path, 'w') as f:
        f.write('<pad>' + '\n')
        f.write('<unk>' + '\n')
        f.write('<bos>' + '\n')
        f.write('<eos>' + '\n') 
        index = 4
        for word in sorted_vocab:
            f.write(word + '\n')
            if vocab[word] < threshold:
                break
            index += 1
    print ("total {} words in {}".format(index, out_path))

def load_vocab(vocab_path):
    with open(vocab_path, 'rb') as f:
        #words = f.read().splitlines()
        words = [x.decode("utf-8").strip() for x in f.readlines()]
    return {words[i]: i for i in range(len(words))}

def sentence2id(vocab, line):
    return [vocab.get(token, vocab['<unk>']) for token in basic_tokenizer(line)]

def token2id(label_path, vocab_path, num_caps, outname):
    """ Convert all the tokens in the data into their corresponding
    index in the vocabulary. 
        Args:
            label_path: Path to the label in dataset
            num_caps: maximum number of repeating captions
            vocab_path: vocab
            outname: prefix of output file name

        strategies of choosing captions:
            * min length caption
            * max length caption
            * repeating captions
            * middle lenght caption
    """
    out_folder = "processed"
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)

    out_capid_path = out_folder + "/" + outname + '_capid.dec'
    out_cap_path = out_folder + "/" + outname + '_cap'
    out_vid_path = out_folder + "/" + outname + '_vid.enc'

    out_vid_list = []
    out_cap_list = []
    out_capid_list = []

    vocab = load_vocab(os.path.join(vocab_path))
    labels = json.load(open(label_path, "rb"))

    for video_captions in labels:
        video_id = video_captions["id"]
        captions = video_captions["caption"]
        max_num_caps = len(captions)
        sample_caps = min([max_num_caps, num_caps])

        ## Strategies are imposed here
        for _ in range(sample_caps):
            cap = random.choice(captions)
            capid = sentence2id(vocab, cap)

            out_vid_list.append(video_id)
            out_cap_list.append(cap)
            # Add BOS and EOS in front and end of the capation id
            out_capid_list.append([vocab["<bos>"]] + capid + [vocab["<eos>"]])

    with open(out_cap_path, "w") as out_file:
        for cap in out_cap_list:
            out_file.write(cap + "\n")
        print ("save to {}".format(out_cap_path))

    with open(out_capid_path, "w") as out_file:
        for capid in out_capid_list:
            out_file.write(' '.join(str(id_) for id_ in capid) + "\n")
        print ("save to {}".format(out_capid_path))

    with open(out_vid_path, "w") as out_file:
        for vid in out_vid_list:
            out_file.write(vid + "\n")
        print ("save to {}".format(out_vid_path))

def extract_training_ids(file_path, save_path):
	labels = json.load(open(file_path, "rb"))
	ids=[lab["id"] for lab in labels]
	with open(save_path, "wb") as file:
		for idx in ids:
			file.write(bytes("%s\n" % idx, "utf-8"))
