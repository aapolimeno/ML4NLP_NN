from numpy import array
from numpy import asarray
from numpy import zeros

def preprocess(pos_tags, tokens):
    """
    Get the tokens and POS-tags from a dataset and return them as lists
    
    """
    pos_list = []
    tok_list = []

    for seq in pos_tags: 
        for entry in seq:
            pos_list.append(entry)

    for sentence in tokens: 
        for word in sentence: 
            tok_list.append(word)
            
    return pos_list, tok_list 


def extract_info(data, output_nodes = 12):
    """
    This function extracts all information that is required for training and testing the neural network.
    
    :param data: 
    :param output_nodes: number of output nodes of the NN (12 in the current case )
    
    :return: data_out, tokens, pos_tags, targets --> all in separate lists. 
    
    """
    data_out = []
    tokens = []
    pos_tags = []
    targets = []

    for pair in data:
        all_values = pair.strip('\n')
        all_values = all_values.split(';')
        data_out.append(all_values)

        # extract tokens for vectorization
        token = all_values[0]
        tokens.append(token)

        # extract pos-tags 
        pos_tag = all_values[1]
        pos_tags.append(pos_tag)

        # extract targets 
        target = zeros(output_nodes) + 0.01
        target[int(pos_tag)] = 0.99
        targets.append(list(target))
        
    return data_out, tokens, pos_tags, targets 


def get_embeddings(embeddings_path, tokens, dimension=160):
    """
    This code was taken and adapted from the following tutorial: 
    https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
    (accessed 24 Jan 2021)
    
    :param path: path to downloaded embeddings 
    :param tokens: list containing all tokens that occur in the data
    :param dimension: default = 160 
    
    :return embeddings_dictionary: dict containing embeddings for each token (key = token, values = embeddings)
    
    """
    
    embeddings_dictionary = dict()
    
    file = open(embeddings_path)

    for line in file:
        records = line.split()
        if records:
            word = records[0]
            if word in tokens:
                try:
                    vector_dimensions = asarray(records[1:], dtype='float32')
                except:
                    pass
            else: 
                vector_dimensions = [0]*dimension
            embeddings_dictionary[word] = vector_dimensions
    file.close()
    
    return embeddings_dictionary

