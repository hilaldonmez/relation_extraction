# Relation Extraction from Biomedical Documents
Binary relation extraction for chemical and protein interactions from biomedical documents. We aim to identify whether a sentence states there is a relation between biochemicals or not.

# Model
Several Transformers-based models with different optimizers, learning rate, and weight decay were finetuned on ChemProt data set to find the relations between chemical and protein entities in biomedical literature. 
* BioBERT
* SciBERT

We achieved the best F1-score from BioBERT-based model with a binary classification layer with softmax activation and share our best model via Google Drive.

In order to run scripts from scratch, please follow these:
* Install [Genia Sentence Splitter](http://www.nactem.ac.uk/y-matsu/geniass/)
* Download our pretrained binary relation extraction model from [Google Drive](https://drive.google.com/file/d/19_eKPAAwug49JNlNneJoNVhl5VDMP1VH/view?usp=sharing) and run the relevant script to find related protein - compound pairs in ChemPROT. Put the model into a new folder named models
* Install [ChemProt](https://biocreative.bioinformatics.udel.edu/news/corpora/chemprot-corpus-biocreative-vi/) and put the dataset into a new folder named data 

# How to get relations from the trained model
* You can directly use splitted sentences from GENIA sentence splitter in pickles folder or you can split abstracts into sentences with geniass_sentence_splitter function in sentence_splitter.py
* Follow the steps in Relation Extraction Train & Predictions in notebooks folder to train a model and to get the predictions from the trained model that was shared in Google Drive. 
