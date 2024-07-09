import pandas as pd
from nltk_utils import tokenize,stem,bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

df = pd.read_json("intents.json")
print(df)
all_words = []
tags = []
xy = []
for intent in df["intents"]:
    tag = intent["tag"]
    tags.append(tag)
    for pattren in intent["pattrens"]:
        w = tokenize(pattren)
        all_words.extend(w)
        xy.append((w,tag))
ignore_words = [',','.','?','!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))
x_train = []
y_train = []
for (pattren_sentense,tag) in xy:
    bag = bag_of_words(pattren_sentense,all_words)
    x_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)
x_train = np.array(x_train)
y_train = np.array(y_train)
  
class ChatDataset (Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = x_train
        self.y_data = y_train
#dataset [idx]
def getitem_(self, index):
    return self.x_data[idx], self.y_data[idx]
def _len_(self):
    return self.n_samples
dataset = ChatDataset()

# Hyperparameters
batch_size = 8
dataset = ChatDataset()
train_loader = DataLoader(dataset-dataset, batch_size=batch_size, shuffle=True, num_workers=2)

        
        
        