#!pip3 install transformers==3.5.1

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

if torch.cuda.is_available():
    print('cuda')

# specify GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(config.Pickle_Path.train_tagged_sentence_path.value, 'rb') as f:
    df_base_only = pickle.load(f)

with open(config.Pickle_Path.dev_tagged_sentence_path.value, 'rb') as f:
    df_base_only_dev = pickle.load(f)

with open(config.Pickle_Path.test_tagged_sentence_path.value, 'rb') as f:
    df_base_only_test = pickle.load(f)

    
    
class Bert_Chemprot(nn.Module):
    def __init__(self):
        super(Bert_Chemprot, self).__init__()
        num_labels = 2
        self.net_bert = AutoModel.from_pretrained("monologg/biobert_v1.1_pubmed")
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, x, attention):
        x, _ = self.net_bert(x, attention)
        x = x[:, 0, :]
        x = self.classifier(x)

        return x


def train_model(folder_name, file_date, save_dir, learning_rate=0.00002, wd=0.3):
    """ Train binary classification model based on Transformers with ChemProt relations

    Args:
        folder_name: folder path to save predictions from a trained model
        file_date: date to save predictions
        save_dir: file name to save a trained model
        learning_rate: learning rate for a model
        wd: weight decay for a model

    Returns:
        None, save a trained model into a pickle file

    """
    
    train_dataloader, val_dataloader, test_dataloader = utility.prepare_inputs(df_base_only, df_base_only_dev, df_base_only_test)
    weights = [1]
    weights.append(df_base_only['label_'].value_counts()[1]/(df_base_only['label_'].value_counts()[0]))
    weights = torch.Tensor(weights).to(device)

    for i in range(10):
        bert_chemprot = Bert_Chemprot().to(device)
        criterion = nn.CrossEntropyLoss(weight = weights)
        #criterion = nn.CrossEntropyLoss()
        num_training_steps = 3000
        num_warmup_steps = 400
        max_grad_norm = 1.0
        warmup_proportion = float(num_warmup_steps) / float(num_training_steps)  # 0.1


        optimizer = AdamW(bert_chemprot.parameters(), lr=learning_rate,  correct_bias=False, weight_decay = wd)  # To reproduce BertAdam specific behavior set correct_bias=False
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)  # PyTorch scheduler

        train_loss_plot = []
        train_acc_plot = []

        # Number of training epochs 
        epochs = 4
        epoch = 0
        max_grad_norm = 1.0
        best_val_loss = 100000
        # BERT training loop
        for _ in trange(epochs, desc="Epoch"):  

            ## TRAINING
            # Set our model to training mode
            bert_chemprot.train() 

            train_loss_set = []
            train_acc_set = []
            train_preds = []
            train_label = []

            # Tracking variables
            tr_loss, tr_accuracy = 0, 0
            nb_tr_examples, nb_tr_steps = 0, 0


            # Train the data for one epoch
            for step, batch in enumerate(train_dataloader):

                b_input_ids, b_input_mask, b_labels, b_article_ids, b_arg1, b_arg2 = batch

                optimizer.zero_grad()

                # Forward pass
                pred_labels = bert_chemprot.forward(b_input_ids, b_input_mask)

                loss = criterion(pred_labels ,b_labels)

                _, predicted = torch.max(pred_labels.data, 1)   
                acc = (predicted == b_labels).sum().item() / len(predicted)
                train_preds.append(predicted.tolist())
                train_label.append(b_labels.tolist())
        

                if step % 50 == 49:
                    train_loss_plot.append(np.mean(train_loss_set))
                    train_acc_plot.append(np.mean(train_acc_set))
                    print('Loss: ', np.mean(train_loss_set))
                    print('Acc: ', np.mean(train_acc_set))

                    f1_micro = f1_score(np.array(train_label).flatten(), np.array(train_preds).flatten(), average='micro')
                    print('Micro F1: ', f1_micro)

                train_loss_set.append(loss.item())  
                train_acc_set.append(acc)  
        
                # Backward pass
                loss.backward()
                # Update parameters and take a step using the computed gradient
                optimizer.step()
                # Update tracking variables
                scheduler.step()

                tr_loss += loss.item()
                tr_accuracy += acc
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1


            bert_chemprot.eval()
            val_preds = []
            val_label = []
            val_arg1 = []
            val_arg2 = []
            val_article_id = []

            # Predict 
            for step, batch in enumerate(val_dataloader): 

                b_input_ids, b_input_mask, b_labels, b_article_ids, b_arg1, b_arg2 = batch
        
                val_loss_set = []
                val_acc_set = []
                # Tracking variables
                val_loss, val_accuracy = 0, 0
                nb_val_examples, nb_val_steps = 0, 0

                with torch.no_grad():
                    # Forward pass
                    pred_labels = bert_chemprot.forward(b_input_ids, b_input_mask)

                    loss = criterion(pred_labels ,b_labels)

                    _, predicted = torch.max(pred_labels.data, 1)   
                    acc = (predicted == b_labels).sum().item() / len(predicted)
                    val_preds = val_preds + predicted.tolist()
                    val_label = val_label + b_labels.tolist()
                    val_arg1 = val_arg1 + b_arg1.tolist()
                    val_arg2 = val_arg2 + b_arg2.tolist()
                    val_article_id = val_article_id + b_article_ids.tolist()

                    if step % 50 == 49:
                        print('Loss: ', np.mean(val_loss_set))
                        print('Acc: ', np.mean(val_acc_set))
                        f1_micro = f1_score(np.array(val_label).flatten(), np.array(val_preds).flatten(), average='micro')
                        print('Micro F1: ', f1_micro)

                    val_loss_set.append(loss.item())  
                    val_acc_set.append(acc)  

                    val_loss += loss.item()
                    val_accuracy += acc
                    nb_val_examples += b_input_ids.size(0)
                    nb_val_steps += 1

            print("Validation Accuracy: {}".format(val_accuracy/nb_val_steps))
            print("Validation Loss: {}".format(val_loss/nb_val_steps))

            if val_loss < best_val_loss:
                pred = pd.DataFrame({'Label': np.array(val_label).flatten(), 'Pred': np.array(val_preds).flatten(), 'article_id': np.array(val_article_id).flatten(), 'arg1': np.array(val_arg1).flatten(), 'arg2': np.array(val_arg2).flatten() })
                pred.to_csv(folder_name + 'y_dev_pred_'+file_date+'_bert_2_classes_'+ str(i) +'_bert.csv')

                pd.DataFrame(np.array(val_preds).flatten()).to_csv(folder_name + 'y_dev_pred_'+file_date+'_bert_2_classes'+ str(i) +'_bert.csv')
                pd.DataFrame(np.array(val_label).flatten()).to_csv(folder_name +'y_dev_'+file_date+'_bert_2_classes'+ str(i) +'_bert.csv')


                torch.save({        
                      #'epoch': epoch+1,
                      'model_state_dict': bert_chemprot.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'loss': loss,
                      }, save_dir)

                best_val_loss = val_loss
                # pred test
                bert_chemprot.eval()
                test_preds = []
                test_label = []
                test_arg1 = []
                test_arg2 = []
                test_article_id = []

            # Predict 
            for step, batch in enumerate(test_dataloader): 
            
                b_input_ids, b_input_mask, b_labels, b_article_ids, b_arg1, b_arg2 = batch
            
                with torch.no_grad():
                  # Forward pass
                    pred_labels = bert_chemprot.forward(b_input_ids, b_input_mask)
                    _, predicted = torch.max(pred_labels.data, 1)   

                    test_preds = test_preds + predicted.tolist()
                    test_arg1 = test_arg1 + b_arg1.tolist()
                    test_arg2 = test_arg2 + b_arg2.tolist()
                    test_article_id = test_article_id + b_article_ids.tolist()
                    test_label = test_label + b_labels.tolist()

                pred = pd.DataFrame({'Pred': np.array(test_preds).flatten(), 'article_id': np.array(test_article_id).flatten(), 'arg1': np.array(test_arg1).flatten(), 'arg2': np.array(test_arg2).flatten() })
                pred.to_csv(folder_name+'y_test_pred_'+file_date+'_bert_2_classes_'+ str(i) +'_bert_.csv')

                pd.DataFrame(np.array(test_preds).flatten()).to_csv(folder_name+'y_test_pred_'+file_date+'_bert_2_classes'+ str(i) +'_bert.csv')
                pd.DataFrame(np.array(test_label).flatten()).to_csv(folder_name+'y_test_'+file_date+'_bert_2_classes'+ str(i) +'_bert.csv')


        print('Epoch: ', epoch )
        epoch = epoch + 1

