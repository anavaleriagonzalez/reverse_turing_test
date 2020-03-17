from sklearn.model_selection import cross_val_score
import torch 

import numpy as np
import glob
import json
import re, ast
import pandas as pd 
import argparse


def evaluate(output_path,device,model, data_loader,best_score,  dev=True):
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
       
        preds = []

        for i ,  (x, y)  in enumerate(data_loader):
            x= x.to(device).float()
            label = y.to(device)

            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            preds.append(predicted)
            total += label.size(0)

            correct += (predicted == y).sum().item()
 
        if dev == False:
            print('Test Accuracy of the model : {} %'.format(100 * correct / total))
            return best_score, model, preds



    
        if (correct / total) > best_score:

            best_score = (correct / total) 

            print('Dev Accuracy of the model : {} %'.format(100 * correct / total))
            torch.save(model.state_dict(), output_path+'model.ckpt')
            print("Score improved, model saved...")
        else:
            print('Dev Accuracy of the model : {} %'.format(100 * correct / total))
            print("Score not improved, model not saved, loading previous")
            
            model.load_state_dict(torch.load(output_path+'model.ckpt'))



        
        return best_score, model, preds
