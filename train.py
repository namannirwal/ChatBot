import numpy as np
import random
import json

import torch
# PyTorch is an open source machine learning framework 
# used for applications such as natural language processing.

import torch.nn as nn
# PyTorch provides the torch.nn module to help us in creating and training of the neural network.

from torch.utils.data import Dataset, DataLoader
# Dataset is argument of DataLoader constructor which indicates a dataset object to load from.
# Dataloader allows us to iterate through the dataset in batches.

from nltk_utils import bag_of_words, tokenize, stem
# importing functions from nltk_utils.

from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)
# json file is opened in read mode and loaded into intents.

all_words = []  # to store patterns
tags = []       # to store tags
xy = []         # pattern and its corresponding tag

# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)

        # add to our words list
        all_words.extend(w)

        # add to xy pair
        xy.append((w, tag))

# these characters will be ignored and will not be added in words
ignore_words = ['?', '.', '!']

# stem and lower each word
all_words = [stem(w) for w in all_words if w not in ignore_words]

# sort and remove dublicates using set
all_words = sorted(set(all_words))
tags = sorted(set(tags))


# create training data
X_train = []  # Contains bag of words for each pattern_sentence
y_train = []  # Contains index of tags
for (pattern_sentence, tag) in xy:
    
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)


X_train = np.array(X_train)   # Converted to Numpy array because they are fast.
y_train = np.array(y_train)

# Hyper-parameters 
num_epochs = 1000             # number of pases
batch_size = 8                # number of training examples utilized in one iteration.
learning_rate = 0.001         # learning rate controls how quickly the model is adapted to the problem
input_size = len(X_train[0])   # length of bag of words
hidden_size = 8
output_size = len(tags)        # number of tags


class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # if gpu is available 

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()  # A perfect model has a cross-entropy loss of 0.

# Adam is used as an optimization technique for gradient descent.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 


# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
       
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
