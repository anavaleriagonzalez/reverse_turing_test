import torch 
import torch.nn as nn
from models import *
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
parser.add_argument('--num_epochs', type=int, help='number of epochs to train for', default=250)
parser.add_argument('--dataset', type=str, help='data set')
parser.add_argument('--input_size', type=int, help='input size for aux task', default=1024)

parser.add_argument('--hidden_size', type=int, help='hidden size', default=100)
parser.add_argument('--batch_size', type=int, help='batch size', default=100)
parser.add_argument('--lr', type=float, help='learning rate', default=0.001)


parser.add_argument('--model_output', type=str, help='path to model checkpoints')

args = parser.parse_args()

dataset= args.dataset
# Hyper parameters
num_epochs = args.num_epochs
batch_size = args.batch_size
learning_rate = args.lr
hidden_size = args.hidden_size


#input sizes are the same
input_size = args.input_size




output_path = args.model_output

print("model_output: ", output_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)

trainfile = "data/downstream/SST/"+dataset+"/sentiment-train"
testfile = "data/downstream/SST/"+dataset+"/sentiment-test"
devfile = "data/downstream/SST/"+dataset+"/sentiment-dev"

train_embeddings = "data/downstream_features/SST_"+dataset+"_trainfeatures.npy"
test_embeddings = "data/downstream_features/SST_"+dataset+"_testfeatures.npy"
dev_embeddings = "data/downstream_features/SST_"+dataset+"_devfeatures.npy"


if "fine" in dataset:
    with open(trainfile, "r") as f:
        lines =  f.readlines()
        labels = [int(line.strip().split()[0]) for line in lines]
        text = [" ".join(line.strip().split()[1::]) for line in lines]

    train = pd.DataFrame()
    train['text']= text
    train['label'] = labels

    with open(testfile, "r") as f:
        lines =  f.readlines()
        labels = [int(line.strip().split()[0]) for line in lines]
        text = [" ".join(line.strip().split()[1::]) for line in lines]

    test = pd.DataFrame()
    test['text']= text
    test['label'] = labels

    with open(devfile, "r") as f:
        lines =  f.readlines()
        labels = [int(line.strip().split()[0]) for line in lines]
        text = [" ".join(line.strip().split()[1::]) for line in lines]

    dev = pd.DataFrame()
    dev['text']= text
    dev['label'] = labels
   

else:
    train = pd.read_csv(trainfile, sep="\t" , names=['text', 'label'])
    test = pd.read_csv(testfile, sep="\t" , names=['text', 'label'])
    dev = pd.read_csv(devfile, sep="\t" , names=['text', 'label'])


train_labels = torch.tensor(train['label'].tolist())
test_labels = torch.tensor(test['label'].tolist())
dev_labels = torch.tensor(dev['label'].tolist())

trainvectors = torch.tensor(np.load(train_embeddings))
testvectors = torch.tensor(np.load(test_embeddings))
devvectors = torch.tensor(np.load(dev_embeddings))


num_classes = len(set(train['label'].tolist()))




train = list(zip(trainvectors, train_labels))
test = list(zip(testvectors, test_labels))
dev = list(zip(devvectors, dev_labels))


model = MLP2(input_size,hidden_size,num_classes)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


train_loader = torch.utils.data.DataLoader(dataset=train,
                                           batch_size=batch_size, 
                                           shuffle=False)


dev_loader = torch.utils.data.DataLoader(dataset=dev,
                                          batch_size=batch_size, 
                                          shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test,
                                          batch_size=batch_size, 
                                          shuffle=False)


total_step = len(train_loader)

best_score = 0
print("starting training....")
for epoch in range(num_epochs):
    
    for i, (x, y) in enumerate(train_loader):
        x= x.to(device).float()

        y = y.to(device)
        outputs = model(x)
        loss = criterion(outputs, y)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 200 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))          

    best_score, model, preds = utils.evaluate(output_path,  device,model, dev_loader,best_score, dev=True)

print("Best dev score: ", best_score)

best_score, model, preds = utils.evaluate(output_path,  device,model, test_loader,best_score, dev=False)




