import pickle
import pandas as pd
import numpy as np
import os
import subprocess
import src.data.config as config


def geniass_sentence_splitter(df_abst, input_folder_path="input_dev/", output_folder_path="output_dev/", data_set_type='dev'):
    """ Split abstracts into sentences by GENIA Sentence Splitter

    Args:
        df_abst: Abstracts in ChemPROT
        input_folder_path: input file path for each abstract
        output_folder_path: output file path for each abstract
        data_set_type: Type of data set (train, dev, test)

    Returns:
        None, write sentence into pickle file considering data type

    """
    
    colnames=['article_id', 'title', 'abstract']
    
    for i in df_abst.index:
        file = open( input_folder_path + str(df_abst.loc[i,'article_id']) + ".txt", "w")
        article_id = df_abst.loc[i,'article_id']
        inp = list(df_abst[df_abst['article_id'] == article_id]['title'] + ' ' + df_abst[df_abst['article_id'] == article_id]['abstract'])[0]
        file.write(inp)
        
    for i in df_abst.index:
        cmd = "./geniass " + input_folder_path + str(df_abst.loc[i,'article_id']) + ".txt " + output_folder_path + str(df_abst.loc[i,'article_id']) + "_o.txt"
        output = subprocess.check_output(cmd, shell=True)
        lines = output.decode().split("\n")
        output = [i.split('\t')[0] for i in lines if i.split('\t')[0]]
        file.close()
        
    df_sentence = df_abst[['article_id']].copy()
    df_sentence['sent_id'] = 0
    df_sentence['sentence'] = ''

    for i in df_abst.index:
        file = open(output_folder_path + str(df_abst.loc[i,'article_id']) + "_o.txt", "r")
        sent_list = file.readlines()
        df_temp = pd.DataFrame({'article_id': df_abst.loc[i,'article_id'], 'sent_id': np.arange(len(sent_list)) })
        
        for j in range(len(sent_list)):
            df_temp.loc[df_temp['sent_id'] == j, 'sentence'] = sent_list[j].replace('\n', '')
            
        df_sentence = df_sentence.append([df_temp])

    df_sentence.drop_duplicates(subset = ['article_id', 'sent_id'], keep='last', inplace=True)
    
    if data_set_type == 'train':
         pickle_path = config.Pickle_Path.train_sentence_split_path.value
    elif data_set_type == 'dev':
        pickle_path = config.Pickle_Path.dev_sentence_split_path.value
    elif data_set_type == 'test':
        pickle_path = config.Pickle_Path.test_sentence_split_path.value
   
    with open(pickle_path, 'wb' ) as f:
        pickle.dump(df_sentence, f, protocol=pickle.HIGHEST_PROTOCOL)
   