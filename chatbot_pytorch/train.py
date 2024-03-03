import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk_utils import tokenize, stem, bag_of_words
from model import NeuralNet

# Read json file
with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

# Populate the all_words, tags, and xy lists
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        tokens = tokenize(pattern)
        all_words.extend(tokens)
        xy.append((tokens, tag))

# Clean and sort all_words and tags
ignore_punctuations = ['?', '!', '.']
all_words = [stem(w) for w in all_words if w not in ignore_punctuations]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = [] # (len(xy), len(all_words))
y_train = [] # len(tags)

for tokens, tag in xy:
    bag = bag_of_words(tokens, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self, input_data, target_data):
        self.n_samples = len(input_data)
        self.x_data = input_data
        self.y_data = target_data

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

# Hyperparameters
batch_size = 8
input_size = len(X_train[0])
hidden_size = 12
output_size = len(tags)
epochs = 1000
learning_rate = 1e-3

dataset = ChatDataset(X_train, y_train)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = NeuralNet(input_size, hidden_size, output_size).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    for X, y in train_loader:
        X = X.to(device)
        y = y.type(torch.LongTensor).to(device)

        # forward pass
        pred = model(X)
        loss = loss_fn(pred, y)

        # backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'epoch: {epoch + 1:>4d} / {epochs}, loss = {loss.item():.4f}')

print(f'final loss = {loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags,
}

FILE = "data.pth"

torch.save(data, FILE)

print(f"Training complete. Model saved to {FILE}")

