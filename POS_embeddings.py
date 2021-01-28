# This script can be used to perform POS-tagging with word embeddings 
# The more elaborate documentation from NN_numbers is not repeated here 

# path to the word embeddings model
path = '/Users/alessandrapolimeno/Documents/VU/models/sonar-160.tar'

import pandas as pd
import numpy as np 

from sklearn.metrics import confusion_matrix

from class_NN import neuralNetwork
from utils import extract_info, get_embeddings


### LOAD DATA 
# load training data 
path = "data/pos_data_train.csv"
with open(path, "r") as infile: 
    train_data = infile.readlines()

# load test data
path = "data/pos_data_test.csv"
with open(path, "r") as infile: 
    test_data = infile.readlines()
    
train_data = train_data[:1000]
test_data = test_data[:100]
    
# for obtaining the language model
data_all = test_data + train_data 

### EXTRACT INFORMATION 
# get relevant information from datasets 
data_all, tokens, pos_tags, targets = extract_info(data_all)
data_tr, tokens_tr, pos_tags_tr, targets_tr = extract_info(train_data)
data_te, tokens_te, pos_tags_te, targets_te = extract_info(test_data)

### INITIALIZE NN 
# number of nodes 
input_nodes = 160
# is the number of the vocabulary (dimensions)
hidden_nodes = 150
output_nodes = 12 # equals number of possible labels 

# learning rate 
learning_rate = 0.3

# create instance of neural network 
nn = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)


### TRAINING 
# get word embeddings for tokens 
embeddings_model = get_embeddings(path, tokens)

epochs = 5
for e in range(epochs):
    for inp, targ in zip(tokens_tr , targets_tr):
        inp = inp.lower()
        if inp in embeddings_model:
            embedding = embeddings_model[inp]
            nn.train(embedding, targ) 
        else:
            pass

### TESTING 
scorecard = []
predicted_labels = []
selected_targets = []
for inp, targ in zip(tokens_te, targets_te):
    inp = inp.lower()
    if inp in embeddings_model[0]:
        embedding = embeddings_model[inp]
        print(embedding)

        # query network 
        outputs = nn.query(embedding)

        # highest number == label 
        label = np.argmax(outputs)
        predicted_labels.append(label)
        selected_targets.append(targ.index(max(targ)))

        # append correct / incorrect to scorecard 
        #print(targ.index(max(targ)))
        if label == targ.index(max(targ)):
            scorecard.append(1)
        else:
            scorecard.append(0)
            
    else:
        print('nee')

        
### EVALUATION 
print('scorecard', scorecard)
scorecard_array = np.asarray(scorecard)
print(f"accuracy = ", scorecard_array.sum() / scorecard_array.size)
print()

# confusion matrix 
cm = confusion_matrix(selected_targets, predicted_labels)
df_confusion = pd.DataFrame.from_records(cm)
print('confusion matrix:')
print(df_confusion)
#print(df_confusion.to_latex())