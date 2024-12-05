
import os
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset, DataLoader

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

def encode_and_partition(diabetic_data_raw, batch_size, labelled=False):

    # fit on all data so all all possible states of a patient can be encoded
    one_hot_encoder_X = OneHotEncoder(handle_unknown='ignore')
    one_hot_encoder_X.fit(diabetic_data_raw)
    one_hot_encoder_y = OneHotEncoder()
    one_hot_encoder_y.fit(readmitted)
    
    partition_train = pd.read_csv( data_dir / 'partitions' / 'train.txt', names=['patient_nbr'])
    partition_test =  pd.read_csv( data_dir / 'partitions' / 'test.txt' , names=['patient_nbr'])
    partition_val =   pd.read_csv( data_dir / 'partitions' / 'val.txt'  , names=['patient_nbr'])

    train_df = diabetic_data_raw[patient_nbr.isin(partition_train.patient_nbr)]
    train = one_hot_encoder_X.transform(train_df)
    train_y = one_hot_encoder_y.transform(
        readmitted[patient_nbr.isin(partition_train.patient_nbr)]
    ).toarray()

    test_df = diabetic_data_raw[patient_nbr.isin(partition_test.patient_nbr)]
    test = one_hot_encoder_X.transform(test_df)
    test_y = one_hot_encoder_y.transform(
        readmitted[patient_nbr.isin(partition_test.patient_nbr)]
    ).toarray()
    
    val = one_hot_encoder_X.transform(
        diabetic_data_raw[patient_nbr.isin(partition_val.patient_nbr)]
    )
    val_y = one_hot_encoder_y.transform(
        readmitted[patient_nbr.isin(partition_val.patient_nbr)]
    ).toarray()

    
    train_data = torch.from_numpy(train.toarray().astype(np.float32))
    if labelled:
        train_y_data = torch.from_numpy(train_y.astype(np.float32))
        train_dataset = TensorDataset(train_data, train_y_data)
    else:
        train_dataset = TensorDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    test_data = torch.from_numpy(test.toarray().astype(np.float32))
    if labelled:
        test_y_data = torch.from_numpy(test_y.astype(np.float32))
        test_dataset = TensorDataset(test_data, test_y_data)
    else:
        test_dataset = TensorDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    return (train, train_y), (test, test_y), (val, val_y), (one_hot_encoder_X, one_hot_encoder_y), (train_df, test_df, None), (train_dataset, test_dataset, None), (train_loader, test_loader, None)



