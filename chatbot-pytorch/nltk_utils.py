# NLP techniques to encode input data
import nltk
import numpy as np
from nltk.stem import PorterStemmer

# nltk.download('punkt')

stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    words = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for i, word in enumerate(all_words):
        if word in words:
            bag[i] = 1.0
    return bag