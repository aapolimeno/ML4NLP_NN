#### This script applies the neural network to the task of recognizing handwritten numbers

import matplotlib.pyplot
#%matplotlib inline
import numpy
from class_NN import neuralNetwork

# number of nodes 
input_nodes = 784 # should be equal to the number of dimensions of the input vector
hidden_nodes = 100 # is not fixed, could be experimented with
output_nodes = 10 # should be equal to the number of possible output labels 

# learning rate 
learning_rate = 0.2 # is not fixed, could be experimented with

# create instance of neural network 
nn = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

#### load training data 
training_path = "mnist/mnist_train_100.csv"
with open(training_path, "r") as infile: 
    training_data = infile.readlines()

#### load test data
test_path = "mnist/mnist_test_10.csv"
with open(test_path, "r") as infile: 
    test_data = infile.readlines()
    
#### start training 
# in this loop, each datapoint in the data set is split to obtain
# the input (which is transformed to fall within the 0.1 - 0.99 range)
# and the target, which takes the form of an array with 10 positions, 
# with only the gold label filled with a high number (0.99), and the rest
# filled with a low number (0.01). 

epochs = 5

for e in range(epochs):
    for record in training_data: 
        # split on commas
        all_values = record.split(',')
        
        # scale inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.1
        #print(len(inputs))
        # create target outputs 
        targets = numpy.zeros(output_nodes) + 0.01
        #print(len(targets))
        # select relevant target label for the current record
        targets[int(all_values[0])] = 0.99
       # nn.train(inputs, targets)
    
#### test nn with a scorecard 
# the following code loops through all data points in the test data,
# and splits them in order to extract the gold label and the input vector.
# the input vector is converted to an array, and is transformed 
# to match the activation function (range 0.1 - 0.99)
# then, the network is queried, resulting in the output of the network. 
# the output consists of an array of probabilities of each label.
# the highest of which corresponds to the label that is predicted by the nn.
# if the predicted label is equal to the gold label, a 1 is added to the scorecard.
# otherwise, a 0 is added. the accuracy is calculated by taking the sum of the numbers 
# in the scorecard, divided by the total number of scorecard entries. This results 
# in the proportion of correct answers, in other words: the system's accuracy. 

scorecard = []
for record in test_data: 
    # split on commas
    all_values = record.split(',')
    # gold label 
    correct_label = int(all_values[0])
    
    # scale inputs
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.1
    # query the network 
    outputs = nn.query(inputs)
    
    # the highest number == label
    label = numpy.argmax(outputs)
    #print(label, "network's answer")
    
    # append correct / incorrect to list 
    if label == correct_label: 
        scorecard.append(1)
    else:
        scorecard.append(0)
    # pass 
    

# Calculate performance (% correct answers)
scorecard_array = numpy.asarray(scorecard)
print("accuracy = ", scorecard_array.sum() / scorecard_array.size)
