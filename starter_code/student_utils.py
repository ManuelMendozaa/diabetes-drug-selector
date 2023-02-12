import pandas as pd
import numpy as np
import os
import tensorflow as tf
import functools

####### STUDENTS FILL THIS OUT ######
#Question 3
def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''
    target_columns = ['NDC_Code', 'Non-proprietary Name']
    
    aux_df = pd.merge(df, ndc_df[target_columns], left_on='ndc_code', right_on='NDC_Code')
    aux_df['generic_drug_name'] = aux_df['Non-proprietary Name']
    
    aux_df.drop(columns=target_columns, inplace=True)

    return aux_df

#Question 4
def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''
    aux_df = df.sort_values('encounter_id').groupby('patient_nbr')
    first_encounter_indexes = aux_df.head(1).index
    first_encounter_df = df[df.index.isin(first_encounter_indexes)]
    return first_encounter_df

#Question 6
def patient_dataset_splitter(df, patient_key='patient_nbr', train_split=0.6, val_split=0.2):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id
    train_split: percentage of dataframe intended for training
    val_split: percentage of dataframe intended for validation

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    assert(train_split + val_split < 1)
    
    # Extract patients df
    patients_df = df[patient_key]
    
    # Compute basic values and shuffle
    patients = patients_df.unique()
    n_patients = len(patients)
    np.random.shuffle(patients)
    
    # Compute set thresholds
    train_threshold = int(n_patients * train_split)
    val_threshold = int(n_patients * (train_split + val_split))
    
    # Split patient's identifiers for each set
    train_patients = patients[:train_threshold]
    val_patients = patients[train_threshold : val_threshold]
    test_patients = patients[val_threshold:]
    
    # Split dataframe
    train = df[patients_df.isin(train_patients)].reset_index(drop=True)
    val = df[patients_df.isin(val_patients)].reset_index(drop=True)
    test = df[patients_df.isin(test_patients)].reset_index(drop=True)
    
    return train, val, test

#Question 7

def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        '''
        Which TF function allows you to read from a text file and create a categorical feature
        You can use a pattern like this below...
        tf_categorical_feature_column = tf.feature_column.......

        '''
        tf_categorical_feature_column = tf.feature_column.categorical_column_with_vocabulary_file(c, vocab_file_path)
        tf_categorical_feature_column = tf.feature_column.indicator_column(tf_categorical_feature_column)
        output_tf_list.append(tf_categorical_feature_column)
    return output_tf_list

#Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/std

def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    normalizer_fn = functools.partial(normalize_numeric_with_zscore, mean=MEAN, std=STD)
    tf_numeric_feature = tf.feature_column.numeric_column(
        key=col, 
        default_value=default_value, 
        normalizer_fn=normalizer_fn, 
        dtype=tf.float64
    )
    return tf_numeric_feature

#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = diabetes_yhat.mean()
    s = diabetes_yhat.stddev()
    return m, s

# Question 10
def get_student_binary_prediction(df, col, threshold=5):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    threshold: binary threshold to separate classes
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    student_binary_prediction = df[col].apply(lambda x : 1 if x >= threshold else 0) 
    return student_binary_prediction
