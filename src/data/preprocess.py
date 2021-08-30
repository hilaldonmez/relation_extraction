import pandas as pd
import numpy as np
import pickle
import src.data.config as config
import src.data.read_data as rd


def change_arg_column(instance):
    """ Change ARG1 and ARG2 columns

    Args:
        instance: A row of pandas dataframe
        

    Returns:
        Return a dataframe by changing argument columns

    """
    if int(instance['arg1'][1:]) < int(instance['arg2'][1:]):
        instance['ARG1'] = 'Arg1:' + instance['arg1']
        instance['ARG2'] = 'Arg2:' + instance['arg2']
        
    else:
        instance['ARG1'] = 'Arg1:' + instance['arg2']
        instance['ARG2'] = 'Arg2:' + instance['arg1']
    return instance


def extract_entity_num(instance):
    """ Extract entity number within ARG1 and ARG2

    Args:
        instance: A row of pandas dataframe
        

    Returns:
        Return a dataframe by adding two number columns

    """
    if int(instance['arg1'][1:]) < int(instance['arg2'][1:]):
        instance['ARG1_num'] = int(instance['arg1'][1:])
        instance['ARG2_num'] = int(instance['arg2'][1:])
        
    else:
        instance['ARG1_num'] = int(instance['arg2'][1:])
        instance['ARG2_num'] = int(instance['arg1'][1:])
    return instance


def duplicate_process(df_relation):
    """ Merge duplicated relations for the same entities in the same sentence

    Args:
        df_relation: raw relation data frame 
        

    Returns:
        Return a relation dataframe munipulating relations

    """
    
    not_duplicated = df_relation.drop_duplicates(subset = ['article_id', 'arg1', 'arg2'] , keep=False) 
    duplicated = df_relation[df_relation.duplicated(['article_id', 'arg1', 'arg2'], keep=False)]
    duplicated_arguments = duplicated.groupby(['article_id', 'arg1', 'arg2']).count().reset_index()
    duplicated_arguments = duplicated_arguments[duplicated_arguments['relation_group'] > 1][['article_id', 'arg1', 'arg2']]

    five_class = ['CPR:3', 'CPR:4', 'CPR:5', 'CPR:6', 'CPR:9']
    df_duplicated_process = duplicated_arguments[['article_id', 'arg1', 'arg2']]
    col_names =  ['article_id', 'arg1', 'arg2', 'relation_group']
    final_duplicated = pd.DataFrame(columns = col_names)

    for j in range(df_duplicated_process.shape[0]):
        article_id = df_duplicated_process.iloc[j]['article_id'] 
        arg1 = df_duplicated_process.iloc[j]['arg1'] 
        arg2 = df_duplicated_process.iloc[j]['arg2'] 

        df_inner = duplicated.loc[(duplicated['article_id'] == article_id)&(duplicated['arg1'] == arg1)&(duplicated['arg2'] == arg2)]
        count = df_inner.count()['article_id']

        rel_1 = df_inner.iloc[0]['relation_group']
        rel_2 = df_inner.iloc[1]['relation_group']  

        if (rel_1 in five_class) and (rel_2 not in five_class) :
            final_duplicated = final_duplicated.append(df_inner.iloc[0], ignore_index=True)
        elif (rel_1 not in five_class) and (rel_2 in five_class) :
            final_duplicated = final_duplicated.append(df_inner.iloc[1], ignore_index=True)
        elif (rel_1 in five_class) and (rel_2  in five_class) :
            final_duplicated = final_duplicated.append(df_inner.iloc[0], ignore_index=True)
            final_duplicated = final_duplicated.append(df_inner.iloc[1], ignore_index=True)
        else:
            df_inner.iloc[1, df_inner.columns.get_loc('relation_group')] = 'NOT'
            final_duplicated = final_duplicated.append(df_inner.iloc[1], ignore_index=True)


    final_duplicated = final_duplicated.drop_duplicates(subset = ['article_id', 'arg1', 'arg2', 'relation_group'], keep = 'first')
    df_relation_ = not_duplicated.append(final_duplicated, ignore_index=True)
    df_relation_ = df_relation_.astype({"article_id": np.int64})
    
    return df_relation_


def sentence_entity_matching(df_ent, df_sentences, df_relation):
    """ Match the entities in sentences with their relations and get only sentence having chemical and protein together

    Args:
        df_ent: raw entity data frame 
        df_relation: raw relation data frame 
        df_sentences: raw data frame including sentences of abstracts 
        
    Returns:
        Return sentence dataframe having chemical and protein entities, a dataframe which will be labeled with integer encoding

    """

    df_ent_ = df_ent[['article_id','number', 'entity_mention', 'start_offset', 'end_offset', 'type']]
    df_ent_ = df_ent_.rename(columns={"number": "arg1", 'entity_mention': 'entity_mention_1', 'start_offset': 'start_offset_1' , 'end_offset': 'end_offset_1', 'type': 'type_1'})
    df_merge = df_relation.merge(df_ent_, on = ['article_id','arg1'], how = 'left')

    df_ent_ = df_ent[['article_id','number', 'entity_mention', 'start_offset', 'end_offset', 'type']]
    df_ent_ = df_ent_.rename(columns={"number": "arg2", 'entity_mention': 'entity_mention_2', 'start_offset': 'start_offset_2' , 'end_offset': 'end_offset_2', 'type': 'type_2'})
    df_merge = df_merge.merge(df_ent_, on = ['article_id','arg2'], how = 'left')

    df_sentences['start_index'] = 0
    df_sentences['end_index'] = 0

    df_each_doc = df_sentences.groupby('article_id')['sent_id'].agg(['count']).reset_index()

    for j in range(df_each_doc.shape[0]):
        article_id = df_each_doc.iloc[j]['article_id'] 
        sent_count = df_each_doc.iloc[j]['count'] 
        start_index = 0

        sentence = df_sentences.loc[(df_sentences['article_id'] == article_id)&(df_sentences['sent_id'] == 0),'sentence'].iloc[0]
        df_sentences.loc[(df_sentences['article_id'] == article_id)&(df_sentences['sent_id'] == 0),'start_index'] = start_index
        df_sentences.loc[(df_sentences['article_id'] == article_id)&(df_sentences['sent_id'] == 0),'end_index'] =  start_index + len(sentence.lstrip()) - 1
        start_index = start_index + len(sentence.lstrip()) 

        for i in range(1,sent_count):
            sentence = df_sentences.loc[(df_sentences['article_id'] == article_id)&(df_sentences['sent_id'] == i),'sentence'].iloc[0]
            df_sentences.loc[(df_sentences['article_id'] == article_id)&(df_sentences['sent_id'] == i),'start_index'] = start_index
            df_sentences.loc[(df_sentences['article_id'] == article_id)&(df_sentences['sent_id'] == i),'end_index'] =  start_index + len(sentence.lstrip()) 
            start_index = start_index + len(sentence.lstrip()) + 1

    df_sentences.drop_duplicates(subset = ['article_id', 'sent_id'], keep = 'last', inplace = True) 

    df_ent_chem = df_ent[df_ent['type'] == 'CHEMICAL'][['article_id','number', 'entity_mention', 'start_offset', 'end_offset', 'type']]
    df_ent_chem = df_ent_chem.rename(columns={"number": "arg1", 'entity_mention': 'entity_mention_1', 'start_offset': 'start_offset_1' , 'end_offset': 'end_offset_1', 'type': 'type_1'})

    df_ent_gen = df_ent[df_ent['type'] != 'CHEMICAL'][['article_id','number', 'entity_mention', 'start_offset', 'end_offset', 'type']]
    df_ent_gen = df_ent_gen.rename(columns={"number": "arg2", 'entity_mention': 'entity_mention_2', 'start_offset': 'start_offset_2' , 'end_offset': 'end_offset_2', 'type': 'type_2'})

    df_sentences.reset_index(drop=True, inplace=True)
    df_check = df_sentences.merge(df_ent_chem, on=['article_id'], how='left')
    df_check['is_contain_chem'] = 0
    df_check.loc[((df_check['start_offset_1'] >= df_check['start_index']) & (df_check['end_offset_1'] <= df_check['end_index'])) ,'is_contain_chem' ] = 1

    df_check = df_check.merge(df_ent_gen, on=['article_id'], how='left')
    df_check['is_contain_gen'] = 0
    df_check.loc[((df_check['start_offset_2'] >= df_check['start_index']) & (df_check['end_offset_2'] <= df_check['end_index'])) ,'is_contain_gen' ] = 1
    df_check = df_check[(df_check['is_contain_chem'] == 1)&(df_check['is_contain_gen'] == 1) ]

    df_sent_including_entities = df_check.copy()
    df_check.reset_index(drop=True, inplace=True)
    df_merge.reset_index(drop=True, inplace=True)
    df_check = df_check[['article_id', 'sent_id', 'start_index', 'end_index']]
    df_label = df_check.merge(df_merge, on=['article_id'], how='left')
    
    return df_sent_including_entities, df_label
    
    
def give_label_for_sentence(df_sent_including_entities, df_label):
    """ Give labels in sentences with their relations 

    Args:
        df_sent_including_entities: sentence dataframe having chemical and protein entities 
        df_label: a dataframe which will be labeled integer encoding
        
    Returns:
        Return a final dataframe with entity numbers
        
    """

    df_labeled_relation_sentence = df_label[((df_label['start_offset_1'] >= df_label['start_index']) & (df_label['end_offset_2'] <= df_label['end_index']))]
    df_labeled_relation_sentence['is_relation'] = 1

    df_sent_including_entities.drop(['is_contain_chem', 'is_contain_gen' ], axis=1, inplace=True)

    merge_columns = list(df_sent_including_entities.columns)
    merge_columns.remove('sentence')

    df_final = df_sent_including_entities.merge(df_labeled_relation_sentence, on=merge_columns, how='left')
    df_final.fillna({'is_relation': 0, 'relation_group': 'NOT', 'eval_type': 'N', 'relation': 'NOT'}, inplace=True)
    df_final.drop_duplicates(keep = 'first', inplace = True)
    df_final = df_final.apply(lambda x: change_arg_column(x), axis=1)
    df_final = df_final.apply(lambda x: extract_entity_num(x), axis=1)
    
    return df_final


def assing_label(df_final):   
    """ Give labels in sentences with integer encoding

    Args:
        df_label: a dataframe which will be labeled integer encoding
        
    Returns:
        Return a final dataframe with labels
        
    """
    df_final['label'] = 11
    df_final.loc[df_final['relation_group'] == 'CPR:0', 'label'] = 0
    df_final.loc[df_final['relation_group'] == 'CPR:1', 'label'] = 1
    df_final.loc[df_final['relation_group'] == 'CPR:2', 'label'] = 2
    df_final.loc[df_final['relation_group'] == 'CPR:3', 'label'] = 3
    df_final.loc[df_final['relation_group'] == 'CPR:4', 'label'] = 4
    df_final.loc[df_final['relation_group'] == 'CPR:5', 'label'] = 5
    df_final.loc[df_final['relation_group'] == 'CPR:6', 'label'] = 6
    df_final.loc[df_final['relation_group'] == 'CPR:7', 'label'] = 7
    df_final.loc[df_final['relation_group'] == 'CPR:8', 'label'] = 8
    df_final.loc[df_final['relation_group'] == 'CPR:9', 'label'] = 9
    df_final.loc[df_final['relation_group'] == 'CPR:10', 'label'] = 10
    
    return df_final

def change_entities_chem(sentence, chem_1, chem_2, pro_1, pro_2):
    """ Write entity tag in sentences having chemical first

    Args:
        sentence: a sentence in an abstract
        chem_1: start offset of a chemical in a sentence
        chem_2: end offset of a chemical in a sentence
        pro_1: start offset of a chemical in a sentence
        pro_2 end offset of a chemical in a sentence
        
    Returns:
        Return a sentence with <e1> and <e2> tags
        
    """
    
    first_part = sentence[:chem_1] + ' <e1> ' + sentence[chem_1:chem_2] + ' </e1> ' + sentence[chem_2:pro_1]
    last_part = ' <e2> ' + sentence[pro_1 :pro_2] + ' </e2> ' + sentence[pro_2:]
    
    return first_part + last_part


def change_entities_pro(sentence, chem_1, chem_2, pro_1, pro_2):
    """ Write entity tag in sentences having protein first

    Args:
        sentence: a sentence in an abstract
        chem_1: start offset of a chemical in a sentence
        chem_2: end offset of a chemical in a sentence
        pro_1: start offset of a chemical in a sentence
        pro_2 end offset of a chemical in a sentence
        
    Returns:
        Return a sentence with <e1> and <e2> tags
        
    """
    
    first_part = sentence[:chem_1] + ' <e2> ' + sentence[chem_1:chem_2] + ' </e2> ' + sentence[chem_2:pro_1]
    last_part = ' <e1> ' + sentence[pro_1:pro_2] + ' </e1> ' + sentence[pro_2:]
    
    return first_part + last_part


def change_entities_nested(sentence, chem_1, chem_2, pro_1, pro_2):
    """ Write entity tag in sentences having nested chemical and protein entities

    Args:
        sentence: a sentence in an abstract
        chem_1: start offset of a chemical in a sentence
        chem_2: end offset of a chemical in a sentence
        pro_1: start offset of a chemical in a sentence
        pro_2 end offset of a chemical in a sentence
        
    Returns:
        Return a sentence with <e1> and <e2> tags
        
    """
    
    if chem_2 < pro_2:
        first_part = sentence[:chem_1] + ' <e1> <e2> ' + sentence[chem_1:chem_2] + ' </e1> ' + sentence[chem_2:pro_2 ] + ' </e2> ' + sentence[pro_2:] 
        
    else:    
        first_part = sentence[:pro_1] + ' <e2> <e1> ' + sentence[pro_1:pro_2] + ' </e2> ' + sentence[pro_2:chem_2] + ' </e1> ' + sentence[chem_2:] 
        
    return first_part


def add_tags_for_entities(df_sent):
    """ Add entity tags in a sentence dataframe

    Args:
        df_sent: a sentence dataframe with label and entities
        
    Returns:
        Return a sentence dataframe with <e1> and <e2> tags
        
    """
    df_sent['chem_1'] = (df_sent['start_offset_1'].astype(int) - df_sent['start_index']).astype(int) - 1
    df_sent['chem_2'] = (df_sent['end_offset_1'] - df_sent['start_index']).astype(int) - 1

    df_sent['pro_1'] = (df_sent['start_offset_2'] - df_sent['start_index']).astype(int) - 1
    df_sent['pro_2'] = (df_sent['end_offset_2'] - df_sent['start_index']).astype(int) - 1

    df_sent.loc[df_sent['start_index'] == 0, 'chem_1'] = (df_sent['start_offset_1'].astype(int) - df_sent['start_index']).astype(int) 
    df_sent.loc[df_sent['start_index'] == 0, 'chem_2'] = (df_sent['end_offset_1'] - df_sent['start_index']).astype(int) 

    df_sent.loc[df_sent['start_index'] == 0, 'pro_1'] = (df_sent['start_offset_2'] - df_sent['start_index']).astype(int) 
    df_sent.loc[df_sent['start_index'] == 0, 'pro_2'] = (df_sent['end_offset_2'] - df_sent['start_index']).astype(int)

    df_sent.loc[df_sent['start_offset_1'] <  df_sent['start_offset_2'], 'changed_sent'] = df_sent[['chem_1', 'chem_2', 'pro_1', 'pro_2', 'sentence']].apply(lambda x: change_entities_chem(x.sentence, x.chem_1, x.chem_2, x.pro_1, x.pro_2), axis=1)
    df_sent.loc[df_sent['start_offset_1'] >  df_sent['start_offset_2'], 'changed_sent'] = df_sent[['chem_1', 'chem_2', 'pro_1', 'pro_2', 'sentence']].apply(lambda x: change_entities_pro(x.sentence, x.pro_1, x.pro_2, x.chem_1, x.chem_2), axis=1)
    df_sent.loc[df_sent['start_offset_1'] ==  df_sent['start_offset_2'], 'changed_sent'] = df_sent[['chem_1', 'chem_2', 'pro_1', 'pro_2', 'sentence']].apply(lambda x: change_entities_nested(x.sentence, x.chem_1, x.chem_2, x.pro_1, x.pro_2), axis=1)
    
    return df_sent


def preprocess(data_set_type, sentence_split=False):
    """ Apply all preprocessing steps

    Args:
        data_set_type: data set type (train, dev, test)
        
    Returns:
        Return a dataframe with preprocessed sentence
        
    """
    
    df_ent, df_abst, df_relation, df_gold = rd.read_all_data(data_set_type)
    
    if not sentence_split:
        if data_set_type == 'train':
            with open(config.Pickle_Path.train_sentence_split_path.value, 'rb') as handle:
                  df_sentences = pickle.load(handle)

        elif data_set_type == 'dev':
            with open(config.Pickle_Path.dev_sentence_split_path.value, 'rb') as handle:
                  df_sentences = pickle.load(handle)

        elif data_set_type == 'test':
            with open(config.Pickle_Path.test_sentence_split_path.value, 'rb') as handle:
                  df_sentences = pickle.load(handle)
     
    else:
        return ('Run sentence splitter')

    df_relation_ = duplicate_process(df_relation)
    df_sent_including_entities, df_label = sentence_entity_matching(df_ent, df_sentences, df_relation_)
    df_final = give_label_for_sentence(df_sent_including_entities, df_label)
    df_final = assing_label(df_final)
    df_final = add_tags_for_entities(df_final)
    
    return df_final

