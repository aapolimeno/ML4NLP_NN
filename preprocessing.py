from datasets import load_dataset
from utils import preprocess
import csv

dataset = load_dataset("conll2002", "nl")

# see structure of dataset
print("structure of the data set:")
print(dataset)

##### DEV SET 
# extract tokens + pos tags
pos_tags_dev = dataset['validation']['pos_tags']
tokens_dev = dataset['validation']['tokens']
# get separate lists for pos_tags and tokens
pos_list_dev, tok_list_dev = preprocess(pos_tags_dev, tokens_dev)

    
##### TRAIN SET 
pos_tags_train = dataset['train']['pos_tags']
tokens_train = dataset['train']['tokens']
# get separate lists for pos_tags and tokens
pos_list_train, tok_list_train = preprocess(pos_tags_train, tokens_train)


#### TEST SET 
pos_tags_test = dataset['test']['pos_tags']
tokens_test = dataset['test']['tokens']
# get separate lists for pos_tags and tokens
pos_list_test, tok_list_test = preprocess(pos_tags_test, tokens_test)

# match each token with the corresponding POS and add them to a list
data_dev = [ [tok_list_dev[i], pos_list_dev[i]] for i in range(len(tok_list_dev)) ]
data_train = [ [tok_list_train[i], pos_list_train[i]] for i in range(len(tok_list_train)) ]
data_test = [ [tok_list_test[i], pos_list_test[i]] for i in range(len(tok_list_test)) ]

# write the data to a csv file 
filepath = "data/pos_data_train.csv"
with open(filepath, 'w', newline = "") as outfile:
    writer = csv.writer(outfile, delimiter=';')
    writer.writerows(data_train)

filepath = "data/pos_data_test.csv"
with open(filepath, 'w', newline = "") as outfile:
    writer = csv.writer(outfile, delimiter=';')
    writer.writerows(data_test)

filepath = "data/pos_data_dev.csv"
with open(filepath, 'w', newline = "") as outfile:
    writer = csv.writer(outfile, delimiter=';')
    writer.writerows(data_dev)

print('done, you can find your .csv files in the data folder')