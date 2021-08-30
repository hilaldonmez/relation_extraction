import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def performance_measure(y_actual, y_pred, class_no):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)): 
        if y_actual[i] == y_pred[i] == class_no:
            TP += 1
        if y_pred[i] == class_no and y_actual[i] != y_pred[i]:
            FP += 1
        if (y_actual[i] != class_no) & (y_pred[i] != class_no):
            TN += 1
        if y_pred[i] != class_no and y_actual[i] == class_no:
            FN += 1

    return(class_no, TP, FP, TN, FN)

def write_evaluation(filename, y_test, y_pred):
    y_test = y_test.drop('Unnamed: 0', axis = 1)
    y_test.rename(columns = {'0': 'label'}, inplace = True)

    y_pred = y_pred.drop('Unnamed: 0', axis = 1)
    y_pred.rename(columns = {'0': 'label'}, inplace = True)

    y_test_final = list(y_test['label'])
    y_pred_final = list(y_pred['label'])

    full_labels = pd.DataFrame(y_test_final, columns=['test'])
    full_labels['pred'] = y_pred_final

    # TP, FP, TN, FN
    final_result = []
    for i in range(2):
        final_result.append(performance_measure(y_test_final, y_pred_final, i))

    df_final_result = pd.DataFrame(final_result, columns =['Group', 'TP', 'FP', 'TN', 'FN'])
    df_final_result['precision'] = df_final_result['TP'] / (df_final_result['TP'] + df_final_result['FP'])
    df_final_result['recall'] = df_final_result['TP'] / (df_final_result['TP'] + df_final_result['FN'])
    df_final_result['F_score'] = (2 * df_final_result['precision'] * df_final_result['recall'])  / (df_final_result['precision'] + df_final_result['recall'])
    df_final_result

    test_count = pd.DataFrame(y_test['label'].value_counts())
    test_count.reset_index(level=0, inplace=True)
    test_count.rename(columns = {'index': 'Group', 'label': 'test_count'}, inplace = True)

    pred_count = pd.DataFrame(y_pred['label'].value_counts())
    pred_count.reset_index(level=0, inplace=True)
    pred_count.rename(columns = {'index': 'Group', 'label': 'pred_count'}, inplace = True)

    df_final_result = df_final_result.merge(test_count, on = 'Group', how = 'left')
    df_final_result = df_final_result.merge(pred_count, on = 'Group', how = 'left')

    df_final_result['sum_TP'] = sum(df_final_result['TP'])
    df_final_result['sum_FP'] = sum(df_final_result['FP'])
    df_final_result['sum_FN'] = sum(df_final_result['FN'])

    df_final_result['micro_precision'] = df_final_result['sum_TP'] / (df_final_result['sum_TP'] + df_final_result['sum_FP'])
    df_final_result['micro_recall'] = df_final_result['sum_TP'] / (df_final_result['sum_TP'] + df_final_result['sum_FN'])

    test_sum = sum(df_final_result['test_count'])
    df_final_result['macro_precision'] = df_final_result['precision'].mean()
    df_final_result['macro_recall'] = df_final_result['recall'].mean()
    df_final_result['macro_F_score'] = df_final_result['F_score'].mean()

    df_final_result.to_csv(filename)


def calculate_multiple_evaluation(bert_type , loss_func, set_name = 'dev'):
    df_final = pd.DataFrame() 
    for t in range(1,4):
        for z in range(1,4):
            precision, recall, f_score = [], [], []
            for i in range(10):
                filename = './/experiment//multiple_run//2_class_'+bert_type+'_tum_sentence//'+bert_type+'_weighted_'+loss_func+'_wd_0'+str(t)+'_lr_0'+ str(z)+'//'+set_name+'//evaluation//evaluation_13102020_'+bert_type+'_2_classes_'+str(i)+'.csv'
                df = pd.read_csv(filename)
                precision.append(df.loc[1]['precision'])
                recall.append(df.loc[1]['recall'])
                f_score.append(df.loc[1]['F_score'])

            bert = pd.DataFrame(f_score, columns =['f_score'])
            bert['precision'] = precision
            bert['recall'] = recall

            f = [str(round(bert.describe()['f_score']['mean'],3)) + '+' + str(round(bert.describe()['f_score']['std'], 3))]
            p = [str(round(bert.describe()['precision']['mean'],3)) + '+' + str(round(bert.describe()['precision']['std'], 3))]
            r = [str(round(bert.describe()['recall']['mean'],3)) + '+' + str(round(bert.describe()['recall']['std'], 3))]

            model_loss = [loss_func]
            model_stat = pd.DataFrame(model_loss, columns =['model_loss_func'])
            model_stat['bert_type'] = bert_type
            model_stat['wd'] = t
            model_stat['lr'] = z
            model_stat['set'] = set_name
            model_stat['f1'] = f
            model_stat['precision'] = p
            model_stat['recall'] = r
            
            df_final = df_final.append(model_stat, ignore_index = True)
    return df_final


def total_write_files(bert_type, loss_func, folder_path = './/experiment//multiple_run//2_class_scibert_tum_sentence'):
    for t in range(1,4):
        for z in range(1,4):
            for i in range(10):
                y_test = pd.read_csv(open(folder_path + '//'+ bert_type +'_weighted_'+loss_func+'_wd_0'+str(t)+'_lr_0'+ str(z)+'//dev//y_dev_13102020_bert_2_classes'+str(i)+'_bert.csv','rb'))
                y_pred = pd.read_csv(folder_path + '//'+ bert_type +'_weighted_'+loss_func+'_wd_0'+str(t)+'_lr_0'+ str(z)+'//dev//y_dev_pred_13102020_bert_2_classes'+str(i)+'_bert.csv')
                filename = folder_path + '//'+ bert_type +'_weighted_'+loss_func+'_wd_0'+str(t)+'_lr_0'+ str(z)+'//dev//evaluation//evaluation_13102020_'+bert_type+'_2_classes_'+str(i)+'.csv'
                write_evaluation(filename, y_test, y_pred)

                y_test = pd.read_csv(open(folder_path + '//'+ bert_type +'_weighted_'+loss_func+'_wd_0'+str(t)+'_lr_0'+ str(z)+'//test//y_test_13102020_bert_2_classes'+str(i)+'_bert.csv','rb'))
                y_pred = pd.read_csv(folder_path + '//'+ bert_type +'_weighted_'+loss_func+'_wd_0'+str(t)+'_lr_0'+ str(z)+'//test//y_test_pred_13102020_bert_2_classes'+str(i)+'_bert.csv')
                filename = folder_path + '//'+ bert_type +'_weighted_'+loss_func+'_wd_0'+str(t)+'_lr_0'+ str(z)+'//test//evaluation//evaluation_13102020_'+bert_type+'_2_classes_'+str(i)+'.csv'
                write_evaluation(filename, y_test, y_pred)

def get_final_results(bert_type):
    total_write_files(bert_type, 'adamw', folder_path = './/experiment//multiple_run//2_class_'+ bert_type +'_tum_sentence')
    total_write_files(bert_type, 'adam', folder_path = './/experiment//multiple_run//2_class_'+ bert_type +'_tum_sentence')
    total_write_files(bert_type, 'sgd', folder_path = './/experiment//multiple_run//2_class_'+ bert_type +'_tum_sentence')
    
    df_final = pd.DataFrame()
    for z in ['adam', 'adamw', 'sgd']:
        for i in ['dev', 'test']:
            df = calculate_multiple_evaluation(bert_type, z, i)
            df_final = df_final.append(df, ignore_index = True)
    return df_final


def get_total_results(bert_type = 'scibert', filename = './/experiment//multiple_run//2_class_scibert_tum_sentence//evaluation.csv'):
	df_result = get_final_results(bert_type)
	df_result.to_csv(filename)
