import torch 
import torch.nn as nn
from models import *
from features import laserembs, encode_labels
import pickle
import numpy as np
import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--data_file', type=str, help='data file')
parser.add_argument('--out_file', type=str, help='data file')
args = parser.parse_args()

datafile = args.data_file
outfile = args.out_file

data = pd.read_csv(datafile, sep="\t" , names=['set', 'label', 'text'])
texts = data['text'].tolist()
string_labels = data["label"].tolist()

print("encoding labels...")
encoded_labels = encode_labels(string_labels)
np.save(outfile+"labels.npy", encoded_labels)


print("making laser embeddings...")
vectors = laserembs(texts)
np.save(outfile+"features.npy", vectors)







