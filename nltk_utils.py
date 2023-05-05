# Natural language processing is a subfield of computer science and artificial intelligence 
# concerned with the interactions between computers and human language.

import numpy as np
# NumPy is a Python library used for working with arrays.
# It has collection of mathematical functions to operate on these arrays.

import nltk
# nltk (Natural Language Toolkit) is Python Library to work with human language data.
# It is added for using tokenization and stemming

#nltk.download('punkt')
# punkt is a package with a pretrained tokenizer

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    """
    split sentence into array of words/tokens
    
    "courses and programs do you have?" 
    ["courses","and","programs","do","you","offer","?"]
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    Find the root form of the word
    
    ["likes", "liked", "likely"]
    ["like", "like", "like"]
    """
    return stemmer.stem(word.lower())



def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["courses", "programs", "you", "have"]
    words = ["hi", "courses", "I", "program", "you", "thank", "have"]
    bag   = [  0 ,    1     ,  0 ,     1    ,   1  ,    0   ,    1  ]
    """
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]

    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)

    for idx, w in enumerate(words):   # enumerate in used here to keep count of iterations.
        if w in sentence_words: 
            bag[idx] = 1

    return bag
