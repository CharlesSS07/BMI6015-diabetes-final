
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
diabetic_data_raw.drop(columns='encounter_id')

# not tracking individual patients
diabetic_data_raw.drop(columns='patient_nbr')

# no cases where examide or citoglipton was even perscribed. provides 0 signal
diabetic_data_raw.drop(columns='examide')
diabetic_data_raw.drop(columns='citoglipton')

# fit on all data so all possible states of a patient can be encoded
one_hot_diabetic_data = OneHotEncoder().fit_transform(diabetic_data_raw)

partition_train = pd.read_csv(data_dir / 'partitions' / 'train.txt', names=['patient_nbr'])
partition_test =  pd.read_csv(data_dir / 'partitions' / 'test.txt' , names=['patient_nbr'])
partition_val =   pd.read_csv(data_dir / 'partitions' / 'val.txt'  , names=['patient_nbr'])

train = diabetic_data[diabetic_data.patient_nbr.isin(trian_txt.patient_nbr)]