# This script applies the POS-tagging experiment 
# I did not repeat the documentation for the code 
# that I already included in the NN_numbers.py file

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, f1_score

from class_NN import neuralNetwork
from utils import extract_info

# load training data 
path = "data/pos_data_train.csv"
with open(path, "r") as infile: 
    train_data = infile.readlines()

# load test data
path = "data/pos_data_test.csv"
with open(path, "r") as infile: 
    test_data = infile.readlines()

# For the training data experiment,
# take a subset of the training data here:
#train_data = train_data[:1000]

    
### Training data preparation 
# extract all relevant information from the dataset 
# specify the 
data_train, tokens_tr, pos_tags_tr, targets_tr = extract_info(train_data)

# vectorize all tokens 
vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w\w+\b|!|.|:|\?|\"|\'")
inputs_tr = vectorizer.fit_transform(tokens_tr)
inputs_tr = inputs_tr.todense() + 0.01
inputs_tr[inputs_tr > 1 ] = 0.99

### Initiate NN 
# get the number of dimensions in a variable 
full_shape = inputs_tr.shape
dim = full_shape[1]

# number of nodes 
input_nodes = dim # is the number of the vocabulary (dimensions)
hidden_nodes = 150
output_nodes = 12 # equals number of possible labels 

# learning rate 
learning_rate = 0.3

# create instance of neural network 
nn = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)


### Train NN 
epochs = 3
for e in range(epochs):
    for inp, targ in zip(inputs_tr, targets_tr):
        nn.train(inp, targ)
        
### Test data preparation
# extract data from file 
data_test, tokens_te, pos_tags_te, targets_te = extract_info(test_data)

# vectorize all tokens 
inputs_te = vectorizer.transform(tokens_te)
inputs_te = inputs_te.todense() + 0.01
inputs_te[inputs_te > 1 ] = 0.99

### Evaluation
scorecard = []
predicted_labels = []
for inp, targ in zip(inputs_te, targets_te):
    # query network 
    outputs = nn.query(inp)

    # highest number == label 
    label = np.argmax(outputs)
    predicted_labels.append(label)

    # append correct / incorrect to scorecard 
    #print(targ.index(max(targ)))
    if label == targ.index(max(targ)):
        scorecard.append(1)
    else:
        scorecard.append(0)
        
# get accuracy
scorecard_array = np.asarray(scorecard)
print(f"accuracy = ", scorecard_array.sum() / scorecard_array.size)
print()

### Confusion matrix 
# extract target labels
targets_cm = []
for targ in targets_te:
    targets_cm.append(targ.index(max(targ)))

# print confustion matrix 
cm = confusion_matrix(targets_cm, predicted_labels)
df_confusion = pd.DataFrame.from_records(cm)
#print(df_confusion.to_latex())
print('Confusion matrix:')
print(df_confusion)
print()
print('done')