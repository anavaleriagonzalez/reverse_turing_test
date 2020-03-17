import torch 
import torch.nn as nn
from models import *
from features import laserembs, encode_labels, Bert
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

if "fine" in datafile:
    with open(datafile, "r") as f:
        lines =  f.readlines()
        labels = [line.strip().split()[0] for line in lines]
        text = [" ".join(line.strip().split()[1::]) for line in lines]

    data = pd.DataFrame()
    data['text']= text
    data['label'] = labels

   
else:
    data = pd.read_csv(datafile, sep="\t" , names=['text', 'label'])

print("len of data ", len(data))
texts = data['text'].tolist()
string_labels = data["label"].tolist()


print("making embeddings...")
vectors = Bert(texts)
np.save(outfile+"features_bert.npy", vectors)







