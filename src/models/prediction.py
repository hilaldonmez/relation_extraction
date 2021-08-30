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
import src.models.model_utils as utility
from src.models.train_binary_classification import Bert_Chemprot

if torch.cuda.is_available():
    print('cuda')

# specify GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(config.Pickle_Path.train_tagged_sentence_path.value, 'rb') as f:
    df_base_only_train = pickle.load(f)

with open(config.Pickle_Path.dev_tagged_sentence_path.value, 'rb') as f:
    df_base_only_dev = pickle.load(f)

with open(config.Pickle_Path.test_tagged_sentence_path.value, 'rb') as f:
    df_base_only_test = pickle.load(f)


def prediction(write_file, save_dir="model_12092020_2_classes_bert.pt", batch_size=16):
    """ Predict whether a sentence has a relation or not by a trained model on ChemProt training set

    Args:
        write_file: a path and a file name to write predictions
        save_dir: a path for the saved model
        batch_size: batch size
       
    Returns:
        None, write sentence into pickle file considering data type

    """
    input_path=config.Pickle_Path.test_tagged_sentence_path.value
    with open(input_path, 'rb') as f:
        df_base_only = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained("monologg/biobert_v1.1_pubmed")
    test_article_id = list(df_base_only['article_id'])
    test_arg1 = list(df_base_only['ARG1_num'])
    test_arg2 = list(df_base_only['ARG2_num'])
    test_sentence = list(df_base_only['changed_sent'])
    test_sent_id = list(df_base_only['sent_id'])
    test_input_ids = utility.to_ids(test_sentence, tokenizer)
    test_attention_masks = utility.create_attention(test_input_ids)

    
    # Test Data
    test_article_ids = torch.tensor(test_article_id).to(device)
    test_arg1s = torch.tensor(test_arg1).to(device)
    test_arg2s = torch.tensor(test_arg2).to(device)
    test_inputs = torch.tensor(test_input_ids).to(device)
    test_masks = torch.tensor(test_attention_masks).to(device)
    test_sent_ids = torch.tensor(test_sent_id).to(device)

    # Create an iterator of our data with torch DataLoader 
    test_data = TensorDataset(test_inputs, test_masks, test_article_ids, test_sent_ids, test_arg1s, test_arg2s)
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    num_training_steps = 3000
    num_warmup_steps = 400
    max_grad_norm = 1.0
    warmup_proportion = float(num_warmup_steps) / float(num_training_steps)  # 0.1

    bert_chemprot = Bert_Chemprot().to(device)
    optimizer = AdamW(bert_chemprot.parameters(), lr=0.00003,  correct_bias=False, weight_decay=0.1)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)  # PyTorch scheduler
    bert_chemprot = Bert_Chemprot().to(device)

    checkpoint = torch.load(save_dir)
    bert_chemprot.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss = checkpoint['loss']

    ## Prediction on test set
    # Put model in evaluation mode
    bert_chemprot.eval()
    test_preds = []
    test_arg1 = []
    test_arg2 = []
    test_article_id_ = []
    test_sent_ids = []

    # Predict 
    for step, batch in enumerate(test_dataloader): 
        b_input_ids, b_input_mask, b_article_ids, b_sent_ids, b_arg1, b_arg2 = batch
        with torch.no_grad(): 
            # Forward pass
            pred_labels = bert_chemprot.forward(b_input_ids, b_input_mask)
            _, predicted = torch.max(pred_labels.data, 1)   

            test_preds = test_preds + predicted.tolist()
            test_arg1 = test_arg1 + b_arg1.tolist()
            test_arg2 = test_arg2 + b_arg2.tolist()
            test_article_id_ = test_article_id_ + b_article_ids.tolist()
            test_sent_ids = test_sent_ids + b_sent_ids.tolist()
        
    pred_test = pd.DataFrame({'Pred': np.array(test_preds).flatten(), 'article_id': np.array(test_article_id_).flatten(), 'ARG1_num': np.array(test_arg1).flatten(), 'ARG2_num': np.array(test_arg2).flatten(), 'sent_id': np.array(test_sent_ids).flatten()})
        
    df_final = df_base_only.merge(pred_test, on = ['article_id', 'ARG1_num', 'ARG2_num', 'sent_id'], how = 'left')
  
    # write output file
    df_final.to_csv(write_file + '.csv')
    with open(write_file + '.json', 'w') as handle:
        df_final.to_json(handle)
    

    return df_final