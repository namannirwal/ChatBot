import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from speech import voicetotext

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Check whether Gpu in available

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

# Extracting the data which was saved in data.pth
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# passing values to model
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "BOT"

# Voice Based
# print("Hi! (type 'quit' to exit) and 'voice' for voice input")
# while True:
    
#     # sentence = voicetotext()  
    
#     sentence = input('You: ')
#     if sentence == "voice":
#         sentence = voicetotext()
#     if sentence == "quit":
#         break

#     sentence = tokenize(sentence)
#     X = bag_of_words(sentence, all_words)
#     X = X.reshape(1, X.shape[0])
#     X = torch.from_numpy(X).to(device)

#     output = model(X)
#     _, predicted = torch.max(output, dim=1)

#     tag = tags[predicted.item()]

#     probs = torch.softmax(output, dim=1)
#     prob = probs[0][predicted.item()]
#     if prob.item() > 0.75:
#         for intent in intents['intents']:
#             if tag == intent["tag"]:
#                 print(f"{bot_name}: {random.choice(intent['responses'])}")
#     else:
#         print(f"{bot_name}: For Further details, Visit AICTE Official Website")


# ------------------This Function is made for GUI----------
def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1) # returns max value stored in output

    tag = tags[predicted.item()] # predicted tag

    probs = torch.softmax(output, dim=1) 
    # The input values can be positive, negative, zero but the softmax transforms them between 0 and 1
    # so that they can be interpreted as probabilities.
    
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return "For Further details, Visit AICTE Official Website"
