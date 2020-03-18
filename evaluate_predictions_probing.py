import torch 
import torch.nn as nn
from models import *
from lime import lime_text
from lime.lime_text import LimeTextExplainer

from lime.lime_text import LimeTextExplainer
from tqdm import tqdm
from features import laserembs, encode_labels
import pickle
import numpy as np
import argparse
import os
import pandas as pd
import utils

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description='Process some integers.')

#network params
parser.add_argument('--dataset', type=str, help='data set')
parser.add_argument('--input_size', type=int, help='input size for aux task', default=1024)
parser.add_argument('--mlp2', type= str, default="True")

parser.add_argument('--hidden_size', type=int, help='hidden size', default=100)



args = parser.parse_args()

mlp2 = args.mlp2

dataset= args.dataset
# Hyper parameters
hidden_size = args.hidden_size

#input sizes are the same
input_size = args.input_size
output_path = "dummy"
datafile = "data/probing/"+dataset+".txt"
encoded_labels_file = "data/probing_features/"+dataset+"_labels.npy"
embeddings_file = "data/probing_features/"+dataset+"_features.npy"

data = pd.read_csv(datafile, sep="\t" , names=['set', 'label', 'text'])

encoded_labels = np.load(encoded_labels_file)
vectors = np.load(embeddings_file)

num_classes = len(set(encoded_labels))
data['encoded_labels'] = list(encoded_labels)
data['features'] = list(vectors)

test = data[data["set"]=="te"]
dev = data[data["set"]=="va"]


if "obj" in dataset:
    sents = test['text'].tolist()[1::]
    test_eval = test['features'].tolist()[1::]
else:
    sents = test['text'].tolist()
    test_eval = test['features'].tolist()



class_names = list(set(test['label'].tolist()))
print(class_names)

test = list(zip(torch.tensor(test['features'].tolist()), torch.tensor(test['encoded_labels'].tolist())))
dev = list(zip(torch.tensor(dev['features'].tolist()), torch.tensor(dev['encoded_labels'].tolist())))

if mlp2 == "True":
    model = MLP2(input_size,hidden_size,num_classes)
else:
    print("loading single layered MLP")
    model = MLP(input_size,hidden_size,num_classes)

model.load_state_dict(torch.load("model_outputs/"+dataset+'/model.ckpt'))

dev_loader = torch.utils.data.DataLoader(dataset=dev,
                                          batch_size=len(dev), 
                                          shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test,
                                          batch_size=len(test), 
                                          shuffle=False)



best_score = 0


best_score, model, preds = utils.evaluate(output_path,  device,model, test_loader,best_score, dev=False)

text2features = dict(zip(sents,test_eval))


def predict_prob(item):
   
    x = []
    for i, t in tqdm(enumerate(item)):
        x.append(laserembs(t))


    x = torch.tensor(x)
    x= x.to(device).float()

    outputs = model(x)
    sm = torch.nn.Softmax()
    prob = sm(outputs).squeeze(1)
    

    return prob.detach().numpy()



explainer = LimeTextExplainer()


exp = explainer.explain_instance(sents[0], predict_prob, num_features=10, num_samples=500)

print(exp.as_list())