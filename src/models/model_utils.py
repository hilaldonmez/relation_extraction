import pandas as pd
import numpy as np
import json
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score, roc_auc_score
import keras
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup, AdamW, AutoModelWithLMHead, AutoTokenizer, AutoModelForSequenceClassification, AutoModel, BertForSequenceClassification
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
from tqdm import tqdm, trange
import src.data.config as config


if torch.cuda.is_available():
    print('cuda')

# specify GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(config.Pickle_Path.train_tagged_sentence_path.value, 'rb') as f:
    df_base_train = pickle.load(f)

with open(config.Pickle_Path.dev_tagged_sentence_path.value, 'rb') as f:
    df_base_dev = pickle.load(f)

with open(config.Pickle_Path.test_tagged_sentence_path.value, 'rb') as f:
    df_base_test = pickle.load(f)

    
def change_label(df_base):
    """ Change a label from 11 classes to binary classes

    Args:
        df_base: a dataframe containing entities and preprocessed sentences
       
    Returns:
        a dataframe with changed label for binary classification
        
    """
 
    df_base['label_'] = 1
    df_base.loc[df_base['relation_group'] == 'CPR:10' , 'label_'] = 0
    df_base.loc[df_base['relation_group'] == 'NOT' , 'label_'] = 0

    return df_base


def to_ids(sentences, tokenizer):
    """ Change id of tokens in sentences considering Transformer-based tokenizer

    Args:
        sentences: a dataframe containing preprocessed sentences
        tokenizer: Transformer-based tokenizer
       
    Returns:
        a list of ids that are changed considering usable ids in tokenizer 
        
    """

    fc_id = 101
    e1_id = 1
    end_e1_id = 2
    e2_id = 3
    end_e2_id = 4
    fc_end = 102
    input_ids = []

    for example_sent in sentences:

        input_id = []  
        if example_sent.find('<e1>') < example_sent.find('<e2>'):

            first = example_sent[:example_sent.find('<e1>')]
            entity_1 =  example_sent[(example_sent.find('<e1>') + 4):example_sent.find('</e1>')]
            inner = example_sent[(example_sent.find('</e1>') + 5):example_sent.find('<e2>')]
            entity_2 = example_sent[(example_sent.find('<e2>') + 4):example_sent.find('</e2>')]
            last = example_sent[(example_sent.find('</e2>') + 5):]

            input_id = ([fc_id] + tokenizer.encode(first, add_special_tokens = False) + [e1_id] 
                      + tokenizer.encode(entity_1, add_special_tokens = False) + [end_e1_id] 
                      + tokenizer.encode(inner, add_special_tokens = False) + [e2_id] 
                      + tokenizer.encode(entity_2, add_special_tokens = False) + [end_e2_id]   
                      + tokenizer.encode(last, add_special_tokens = False) + [fc_end] )

        else:
      
            first = example_sent[:example_sent.find('<e2>')]
            entity_2 =  example_sent[(example_sent.find('<e2>') + 4):example_sent.find('</e2>')]
            inner = example_sent[(example_sent.find('</e2>') + 5):example_sent.find('<e1>')]
            entity_1 = example_sent[(example_sent.find('<e1>') + 4):example_sent.find('</e1>')]
            last = example_sent[(example_sent.find('</e1>') + 5):]

            input_id = ([fc_id] + tokenizer.encode(first, add_special_tokens = False) + [e2_id] 
                      + tokenizer.encode(entity_2, add_special_tokens = False) + [end_e2_id] 
                      + tokenizer.encode(inner, add_special_tokens = False) + [e1_id] 
                      + tokenizer.encode(entity_1, add_special_tokens = False) + [end_e1_id]   
                      + tokenizer.encode(last, add_special_tokens = False) + [fc_end] )
        
        input_ids.append(input_id)
    
    input_ids = pad_sequences(input_ids, maxlen = 200, dtype="long", truncating="post", padding="post")
    return input_ids


def create_attention(input_ids):
    """ Create an attention matrix to encode which tokens are used or not used

    Args:
        input_ids: a list of ids that are changed considering usable ids in tokenizer 
        
    Returns:
        a list of ids that are considered as attention matrix
        
    """
    # Create attention masks
    attention_masks = []
    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)

    return attention_masks



def prepare_inputs(df_base_train, df_base_dev, df_base_test, batch_size=16):
    """ Create an attention matrix to encode which tokens are used or not used

    Args:
        df_base_train: a dataframe containing preprocessed sentences and entities for training set
        df_base_dev: a dataframe containing preprocessed sentences and entities for development set
        df_base_test: a dataframe containing preprocessed sentences and entities for test set
        batch_size: batch size
        
    Returns:
        train_dataloader, val_dataloader, test_dataloader that are Torch Data Loader to train model with train, dev, and test set accordingly.
        
    """

    tokenizer = AutoTokenizer.from_pretrained("monologg/biobert_v1.1_pubmed")

    df_base_train = change_label(df_base_train)
    df_base_dev = change_label(df_base_dev)
    df_base_test = change_label(df_base_test)

    df_base_train = df_base_train.drop_duplicates(subset = ['article_id', 'sent_id', 'ARG1_num', 'ARG2_num', 'label_'] , keep='first')
    df_base_dev = df_base_dev.drop_duplicates(subset = ['article_id', 'sent_id', 'ARG1_num', 'ARG2_num', 'label_'] , keep='first')
    df_base_test = df_base_test.drop_duplicates(subset = ['article_id', 'sent_id', 'ARG1_num', 'ARG2_num', 'label_'] , keep='first')

    train_article_id = list(df_base_train['article_id'])
    train_arg1 = list(df_base_train['ARG1_num'])
    train_arg2 = list(df_base_train['ARG2_num'])
    train_sentence = list(df_base_train['changed_sent'])
    train_label = list(df_base_train['label_'])


    val_article_id = list(df_base_dev['article_id'])
    val_arg1 = list(df_base_dev['ARG1_num'])
    val_arg2 = list(df_base_dev['ARG2_num'])
    val_sentence = list(df_base_dev['changed_sent'])
    val_label = list(df_base_dev['label_'])

    test_article_id = list(df_base_test['article_id'])
    test_arg1 = list(df_base_test['ARG1_num'])
    test_arg2 = list(df_base_test['ARG2_num'])
    test_sentence = list(df_base_test['changed_sent'])
    test_label = list(df_base_test['label_'])

    input_ids = to_ids(train_sentence, tokenizer)
    val_input_ids = to_ids(val_sentence, tokenizer)
    test_input_ids = to_ids(test_sentence, tokenizer)

    attention_masks = create_attention(input_ids)
    val_attention_masks = create_attention(val_input_ids)
    test_attention_masks = create_attention(test_input_ids)


    # Train Data
    train_article_ids = torch.tensor(train_article_id).to(device)
    train_arg1s = torch.tensor(train_arg1).to(device)
    train_arg2s = torch.tensor(train_arg2).to(device)

    train_inputs = torch.tensor(input_ids).to(device)
    train_labels = torch.tensor(train_label).to(device)
    train_masks = torch.tensor(attention_masks).to(device)

    # Create an iterator of our data with torch DataLoader 
    train_data = TensorDataset(train_inputs, train_masks, train_labels, train_article_ids, train_arg1s, train_arg2s)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Validation Data
    val_article_ids = torch.tensor(val_article_id).to(device)
    val_arg1s = torch.tensor(val_arg1).to(device)
    val_arg2s = torch.tensor(val_arg2).to(device)

    val_inputs = torch.tensor(val_input_ids).to(device)
    val_labels = torch.tensor(val_label).to(device)
    val_masks = torch.tensor(val_attention_masks).to(device)

    # Create an iterator of our data with torch DataLoader 
    val_data = TensorDataset(val_inputs, val_masks, val_labels, val_article_ids, val_arg1s, val_arg2s)
    val_sampler = RandomSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    # Test Data
    test_article_ids = torch.tensor(test_article_id).to(device)
    test_arg1s = torch.tensor(test_arg1).to(device)
    test_arg2s = torch.tensor(test_arg2).to(device)
    test_labels = torch.tensor(test_label).to(device)
    test_inputs = torch.tensor(test_input_ids).to(device)
    test_masks = torch.tensor(test_attention_masks).to(device)

    # Create an iterator of our data with torch DataLoader 
    test_data = TensorDataset(test_inputs, test_masks, test_labels, test_article_ids, test_arg1s, test_arg2s)
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader


