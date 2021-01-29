import pandas as pd
import numpy as np 

from sklearn.metrics import confusion_matrix

from class_NN import neuralNetwork
from utils import extract_info, get_embeddings

# insert path to embeddings model here
embeddings_path = "/Users/alessandrapolimeno/Documents/VU/models/sonar-160.tar"

### LOAD DATA 
# training data 
train_path = "data/pos_data_train.csv"
with open(train_path, "r") as infile: 
    train_data = infile.readlines()

# test data
test_path = "data/pos_data_test.csv"
with open(test_path, "r") as infile: 
    test_data = infile.readlines()
    
# take subsets of data for testing (optional)
#train_data = train_data[:1000]
#test_data = test_data[:100]

data_all = test_data + train_data # for training the embeddings model on 

### EXTRACT INFORMATION 
# get relevant information from datasets 
data_all, tokens, pos_tags, targets = extract_info(data_all)
data_tr, tokens_tr, pos_tags_tr, targets_tr = extract_info(train_data)
data_te, tokens_te, pos_tags_te, targets_te = extract_info(test_data)

### INITIATE NN 
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
embeddings_model = get_embeddings(embeddings_path, tokens)

epochs = 5 # can be experimented with 
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
    try: 
        embedding = embeddings_model[inp]
        
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
            
    except: 
        pass

    
scorecard_array = np.asarray(scorecard)
print(f"accuracy = ", scorecard_array.sum() / scorecard_array.size)
print()

### EVALUATION

# Find out the labels that are present (optional)
# target_set = set()
# for num in selected_targets:
#    target_set.add(num)
# print(target_set)

# print confustion matrix 
cm = confusion_matrix(selected_targets, predicted_labels)
df_confusion = pd.DataFrame.from_records(cm)
print('confusion matrix:')
print(df_confusion)
#print(df_confusion.to_latex())

print('done!')