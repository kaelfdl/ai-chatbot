import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import random
import numpy as np
import torch
import torch.nn as nn
from chatbot_pytorch.model import NeuralNet
from chatbot_pytorch.nltk_utils import bag_of_words, tokenize

# Locate data file and json file
parent_dir = os.path.dirname(os.path.pardir)
data_path = os.path.join(parent_dir, 'chatbot_pytorch/data.pth')
json_path = os.path.join(parent_dir, 'chatbot_pytorch/intents.json')

# Device
device = 'cuda' if torch.cuda.is_available else 'cpu'

# Read intents file

class Assistant():
    def __init__(self, threshold):
        self._load_json()
        self.threshold = threshold
        self.data = torch.load(data_path)

        self.model_state = self.data['model_state']
        self.input_size = self.data['input_size']
        self.hidden_size = self.data['hidden_size']
        self.output_size = self.data['output_size']
        self.all_words = self.data['all_words']
        self.tags = self.data['tags']

        self.model = NeuralNet(self.input_size, self.hidden_size, self.output_size).to(device)
        self.model.load_state_dict(self.model_state)

        self.model.eval()


    def __str__(self): 
        return self.model

    def _load_json(self):
        with open(json_path, 'r') as f:
            self.intents = json.load(f)

    def predict(self, x):
        with torch.no_grad():
            x = np.reshape(x, (1, x.shape[0]))
            x = torch.from_numpy(x).to(device)
            pred = self.model(x)
            
            probs = torch.softmax(pred, dim=1)
            prob_idx = probs.argmax(1)
            prob = probs[0][prob_idx]
            tag = self.tags[prob_idx]
            
            return tag, prob

    def respond(self, text):
        x = self._encode_text(text)
        tag, prob = self.predict(x)
        
        if prob > self.threshold:
            for intent in self.intents['intents']:
                if tag == intent['tag']:
                    response = random.choice(intent['responses'])
                    return response
        else:
            return "I don't understand"
        
    def _encode_text(self, text):
        tokens = tokenize(text)
        bag = bag_of_words(tokens, self.all_words)
        return bag


