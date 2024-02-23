import torch
import random
import json
import numpy as np
from model import NeuralNet
from nltk_utils import tokenize, stem, bag_of_words

# Load data file
data = torch.load('data.pth')

# Unpack data file
input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['all_words']
tags = data['tags']

# Configure and load the model state
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(data['model_state'])

model.eval()

bot_name = "🐱Cy"
threshold = 0.75
# Read json file
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Predict and classify to which tag the sentence belongs to
def classify(sentence):
    with torch.no_grad():
        token = tokenize(sentence)
        X = bag_of_words(token, all_words)
        X = X.reshape((1, X.shape[0]))
        X = torch.from_numpy(X).to(device)
        pred = model(X)

        probs = torch.softmax(pred, dim=1)
        prob_idx = probs.argmax(1).item()
        prob = probs[0][prob_idx]
        tag = tags[prob_idx]
        return (tag, prob)

# Generate a response given a tag
def response(sentence, uuid="123"):
    tag, prob = classify(sentence)

    if prob > threshold:
        for intent in intents['intents']:
            if tag == intent['tag']:
                result = random.choice(intent['responses'])
                return tag, result
    return None

# Command-line chatbot
print("Let's chat! Type 'quit' to exit, or type a goodbye message.")
while True:
    sentence = input("You: ")
    if sentence == "quit":
        break
    else:
        tag, result = response(sentence)

        if result is None:
            print(f"{bot_name}: I don't understand...")
        elif tag == 'goodbye':
            print(f"{bot_name}: {result}")
            break
        else:
            print(f"{bot_name}: {result}")
    
