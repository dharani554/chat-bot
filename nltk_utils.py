import nltk
import numpy as np
nltk.download("punkt")
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
def tokenize(sentense):
    return nltk.word_tokenize(sentense)
def stem(word):
    return stemmer.stem(word.lower())
def bag_of_words(tokenize_sentense,all_words):
    tokenize_sentense = [stem(w) for w in tokenize_sentense]
    bag = np.zeros(len(all_words),dtype = np.float32)
    #print("the bag of words:",bag)
    for idx,w in enumerate(all_words):
        if w in tokenize_sentense:
            bag[idx] = 2.0
        print(bag[idx])
    return bag
    


