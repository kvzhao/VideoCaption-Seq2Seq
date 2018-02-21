#!/bin/bash
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1hCy_Y3xOx7xbP5dOI8hrumhgr0GUYh3E" -O mymodel/model.ckpt-2401.data-00000-of-00002
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1SSJwHdAckE0SPzra9GIFdazJC7XGDglB" -O mymodel/model.ckpt-2401.data-00001-of-00002
python watch.py --logdir=mymodel --test_vid_path=$1/testing_id.txt --test_feature_path=$1/testing_data/feat --out_name=$2
python watch.py --logdir=mymodel --test_vid_path=$1/peer_review_id.txt --test_feature_path=$1/peer_review/feat --out_name=$3

#$1: the data directory, 
#$2: test data output filename 
#$3: peer review output filename 
#Ex: ./hw2_seq2seq.sh myData/  sample_output_testset.txt sample_output_peer_review.txt