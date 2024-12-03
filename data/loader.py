
import os
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

data_dir = Path(os.path.dirname(os.path.abspath(__file__)))

diabetic_data_raw = pd.read_csv(data_dir / 'diabetic_data.csv')
IDS_mapping = pd.read_csv(data_dir / 'IDS_mapping.csv')

# row id numbers, not necessary
diabetic_data_raw.drop(columns='encounter_id', inplace=True)

# not tracking individual patients
patient_nbr = diabetic_data_raw['patient_nbr']
diabetic_data_raw.drop(columns='patient_nbr', inplace=True)

# no cases where examide or citoglipton was even prescribed. provides 0 signal
diabetic_data_raw.drop(columns='examide', inplace=True)
diabetic_data_raw.drop(columns='citoglipton', inplace=True)

# pull out Y
readmitted = diabetic_data_raw[['readmitted',]]
diabetic_data_raw.drop(columns='readmitted', inplace=True)

def encode_and_partition(diabetic_data_raw):

    # fit on all data so all all possible states of a patient can be encoded
    one_hot_encoder_X = OneHotEncoder(handle_unknown='ignore')
    one_hot_encoder_X.fit(diabetic_data_raw)
    one_hot_encoder_y = OneHotEncoder()
    one_hot_encoder_y.fit(readmitted)
    
    partition_train = pd.read_csv( data_dir / 'partitions' / 'train.txt', names=['patient_nbr'])
    partition_test =  pd.read_csv( data_dir / 'partitions' / 'test.txt' , names=['patient_nbr'])
    partition_val =   pd.read_csv( data_dir / 'partitions' / 'val.txt'  , names=['patient_nbr'])
    
    train = one_hot_encoder_X.transform(
        diabetic_data_raw[patient_nbr.isin(partition_train.patient_nbr)]
    )
    train_y = one_hot_encoder_y.transform(
        readmitted[patient_nbr.isin(partition_train.patient_nbr)]
    ).toarray()
    
    test = one_hot_encoder_X.transform(
        diabetic_data_raw[patient_nbr.isin(partition_test.patient_nbr)]
    )
    test_y = one_hot_encoder_y.transform(
        readmitted[patient_nbr.isin(partition_test.patient_nbr)]
    ).toarray()
    
    val = one_hot_encoder_X.transform(
        diabetic_data_raw[patient_nbr.isin(partition_val.patient_nbr)]
    )
    val_y = one_hot_encoder_y.transform(
        readmitted[patient_nbr.isin(partition_val.patient_nbr)]
    ).toarray()

    return (train, train_y), (test, test_y), (val, val_y), one_hot_encoder_X, one_hot_encoder_y



