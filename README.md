# Neural Network for POS-tagging

## Files: 
- utils.py contains functions that were used 
- class_NN.py contains the code of the neural network 
- NN_numbers.py contains the number recognition task, where the most elaborate documentation is added. Most of the code is reused for the POS-tagging task, but the extensive explanations are not repeated there for the sake of readability. 
- see the rest of the files below.  

## step 1: preproccesing.py
This file loads in all data sets, splits them, and writes them out. A path is hardcoded at the end of the code to the 'data' file included in this repository, where the data is saved. 

## step 2: POS_experiment.py 
If the data is saved in the default place, there is no need to change any paths here. 

## step 3: POS_embeddings.ipynb
I could not manage to convert this notebook to a script, because it gave me a lot of weird errors. You can just run them via Jupyter Notebook. You should specify the path to your embeddings model at the top of the notebook, right under the import statements. Again, the paths to the data should be fine. 
