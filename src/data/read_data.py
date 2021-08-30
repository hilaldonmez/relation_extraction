import pandas as pd
import src.data.config as config

def read_data(data_title, data_set_type):  
    """ Read data of selected data type (train, dev, test) and data title

    Args:
        data_title: Data type in ChemPROT (entity, abstract, relations, gold-standard)
        data_set_type: Type of data set (train, dev, test)

    Returns:
        Return selected data in pandas dataframe

    """
    
    
    if data_set_type == 'train':
        data = config.Data.train.value
        
    elif data_set_type == 'dev':
        data = config.Data.dev.value
        
    elif data_set_type == 'test':
        data = config.Data.test.value
        
    entity_path = data['entity'] 
    abstract_path = data['abstract'] 
    relations_path = data['relations'] 
    gold_standard_path = data['gold_standard'] 
        
    
    if data_title == 'entity':
        colnames=['article_id', 'number', 'type', 'start_offset', 'end_offset', 'entity_mention'] 
        return  pd.read_csv(entity_path, sep="\t", names=colnames, header=None)

    elif data_title == 'abstract':
        colnames=['article_id', 'title', 'abstract']
        return pd.read_csv(abstract_path, sep="\t", names=colnames, header=None)

    elif data_title == 'relation':
        colnames=['article_id', 'relation_group', 'eval_type', 'relation', 'arg1', 'arg2']
        df_relation = pd.read_csv(relations_path, sep="\t", names=colnames, header=None)
        df_relation['arg1'] = df_relation['arg1'].map(lambda x: str(x)[5:])
        df_relation['arg2'] = df_relation['arg2'].map(lambda x: str(x)[5:])
        return df_relation
    
    elif data_title == 'gold-standard':
        colnames=['article_id', 'relation_group', 'arg1', 'arg2']
        df_gold = pd.read_csv(gold_standard_path, sep="\t", names=colnames, header=None)
        df_gold['arg1'] = df_gold['arg1'].map(lambda x: str(x)[5:])
        df_gold['arg2'] = df_gold['arg2'].map(lambda x: str(x)[5:])
        return df_gold
    
def read_all_data(data_set_type):   
    """ Read all data of selected data type (train, dev, test) 

    Args:
        data_set_type: Type of data set (train, dev, test)

    Returns:
        Return all data in pandas dataframe

    """
    
    df_ent = read_data('entity', data_set_type)
    df_abst = read_data('abstract', data_set_type)
    df_relation = read_data('relation', data_set_type)
    df_gold = read_data('gold-standard', data_set_type)
    
    return df_ent, df_abst, df_relation, df_gold
    
