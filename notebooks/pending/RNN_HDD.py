
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

chunksize = 10 ** 6
dtype_dict = {'date': 'str', 'serial_number': 'str', 'model': 'str', 'capacity_bytes': 'int32', 'failure': 'bool', 'datacenter': 'str', 'cluster_id': 'int8', 'vault_id': 'int16', 'pod_id': 'int16', 'pod_slot_num': 'float32', 'is_legacy_format': 'bool', 'smart_1_normalized': 'float64', 'smart_1_raw': 'float64', 'smart_2_normalized': 'float64', 'smart_2_raw': 'float64', 'smart_3_normalized': 'float64', 'smart_3_raw': 'float64', 'smart_4_normalized': 'float64', 'smart_4_raw': 'float64', 'smart_5_normalized': 'float64', 'smart_5_raw': 'float64', 'smart_7_normalized': 'float64', 'smart_7_raw': 'float64', 'smart_8_normalized': 'float64', 'smart_8_raw': 'float64', 'smart_9_normalized': 'float64', 'smart_9_raw': 'float64', 'smart_10_normalized': 'float64', 'smart_10_raw': 'float64', 'smart_11_normalized': 'float64', 'smart_11_raw': 'float64', 'smart_12_normalized': 'float64', 'smart_12_raw': 'float64', 'smart_13_normalized': 'float64', 'smart_13_raw': 'float64', 'smart_15_normalized': 'float64', 'smart_15_raw': 'float64', 'smart_16_normalized': 'float64', 'smart_16_raw': 'float64', 'smart_17_normalized': 'float64', 'smart_17_raw': 'float64', 'smart_18_normalized': 'float64', 'smart_18_raw': 'float64', 'smart_22_normalized': 'float64', 'smart_22_raw': 'float64', 'smart_23_normalized': 'float64', 'smart_23_raw': 'float64', 'smart_24_normalized': 'float64', 'smart_24_raw': 'float64', 'smart_27_normalized': 'float64', 'smart_27_raw': 'float64', 'smart_71_normalized': 'float64', 'smart_71_raw': 'float64', 'smart_82_normalized': 'float64', 'smart_82_raw': 'float64', 'smart_90_normalized': 'float64', 'smart_90_raw': 'float64', 'smart_160_normalized': 'float64', 'smart_160_raw': 'float64', 'smart_161_normalized': 'float64', 'smart_161_raw': 'float64', 'smart_163_normalized': 'float64', 'smart_163_raw': 'float64', 'smart_164_normalized': 'float64', 'smart_164_raw': 'float64', 'smart_165_normalized': 'float64', 'smart_165_raw': 'float64', 'smart_166_normalized': 'float64', 'smart_166_raw': 'float64', 'smart_167_normalized': 'float64', 'smart_167_raw': 'float64', 'smart_168_normalized': 'float64', 'smart_168_raw': 'float64', 'smart_169_normalized': 'float64', 'smart_169_raw': 'float64', 'smart_170_normalized': 'float64', 'smart_170_raw': 'float64', 'smart_171_normalized': 'float64', 'smart_171_raw': 'float64', 'smart_172_normalized': 'float64', 'smart_172_raw': 'float64', 'smart_173_normalized': 'float64', 'smart_173_raw': 'float64', 'smart_174_normalized': 'float64', 'smart_174_raw': 'float64', 'smart_175_normalized': 'float64', 'smart_175_raw': 'float64', 'smart_176_normalized': 'float64', 'smart_176_raw': 'float64', 'smart_177_normalized': 'float64', 'smart_177_raw': 'float64', 'smart_178_normalized': 'float64', 'smart_178_raw': 'float64', 'smart_179_normalized': 'float64', 'smart_179_raw': 'float64', 'smart_180_normalized': 'float64', 'smart_180_raw': 'float64', 'smart_181_normalized': 'float64', 'smart_181_raw': 'float64', 'smart_182_normalized': 'float64', 'smart_182_raw': 'float64', 'smart_183_normalized': 'float64', 'smart_183_raw': 'float64', 'smart_184_normalized': 'float64', 'smart_184_raw': 'float64', 'smart_187_normalized': 'float64', 'smart_187_raw': 'float64', 'smart_188_normalized': 'float64', 'smart_188_raw': 'float64', 'smart_189_normalized': 'float64', 'smart_189_raw': 'float64', 'smart_190_normalized': 'float64', 'smart_190_raw': 'float64', 'smart_191_normalized': 'float64', 'smart_191_raw': 'float64', 'smart_192_normalized': 'float64', 'smart_192_raw': 'float64', 'smart_193_normalized': 'float64', 'smart_193_raw': 'float64', 'smart_194_normalized': 'float64', 'smart_194_raw': 'float64', 'smart_195_normalized': 'float64', 'smart_195_raw': 'float64', 'smart_196_normalized': 'float64', 'smart_196_raw': 'float64', 'smart_197_normalized': 'float64', 'smart_197_raw': 'float64', 'smart_198_normalized': 'float64', 'smart_198_raw': 'float64', 'smart_199_normalized': 'float64', 'smart_199_raw': 'float64', 'smart_200_normalized': 'float64', 'smart_200_raw': 'float64', 'smart_201_normalized': 'float64', 'smart_201_raw': 'float64', 'smart_202_normalized': 'float64', 'smart_202_raw': 'float64', 'smart_206_normalized': 'float64', 'smart_206_raw': 'float64', 'smart_210_normalized': 'float64', 'smart_210_raw': 'float64', 'smart_218_normalized': 'float64', 'smart_218_raw': 'float64', 'smart_220_normalized': 'float64', 'smart_220_raw': 'float64', 'smart_222_normalized': 'float64', 'smart_222_raw': 'float64', 'smart_223_normalized': 'float64', 'smart_223_raw': 'float64', 'smart_224_normalized': 'float64', 'smart_224_raw': 'float64', 'smart_225_normalized': 'float64', 'smart_225_raw': 'float64', 'smart_226_normalized': 'float64', 'smart_226_raw': 'float64', 'smart_230_normalized': 'float64', 'smart_230_raw': 'float64', 'smart_231_normalized': 'float64', 'smart_231_raw': 'float64', 'smart_232_normalized': 'float64', 'smart_232_raw': 'float64', 'smart_233_normalized': 'float64', 'smart_233_raw': 'float64', 'smart_234_normalized': 'float64', 'smart_234_raw': 'float64', 'smart_235_normalized': 'float64', 'smart_235_raw': 'float64', 'smart_240_normalized': 'float64', 'smart_240_raw': 'float64', 'smart_241_normalized': 'float64', 'smart_241_raw': 'float64', 'smart_242_normalized': 'float64', 'smart_242_raw': 'float64', 'smart_244_normalized': 'float64', 'smart_244_raw': 'float64', 'smart_245_normalized': 'float64', 'smart_245_raw': 'float64', 'smart_246_normalized': 'float64', 'smart_246_raw': 'float64', 'smart_247_normalized': 'float64', 'smart_247_raw': 'float64', 'smart_248_normalized': 'float64', 'smart_248_raw': 'float64', 'smart_250_normalized': 'float64', 'smart_250_raw': 'float64', 'smart_251_normalized': 'float64', 'smart_251_raw': 'float64', 'smart_252_normalized': 'float64', 'smart_252_raw': 'float64', 'smart_254_normalized': 'float64', 'smart_254_raw': 'float64', 'smart_255_normalized': 'float64', 'smart_255_raw': 'float64'}

def clean_data_drives(df):
    df.head()
    columns_to_delete = ['model','capacity_bytes','datacenter','cluster_id','vault_id','pod_id','pod_slot_num','is_legacy_format']
    smart_allowed = ['failure', 'serial_number']
    for column in df.columns:
        if column not in smart_allowed:
            columns_to_delete.append(column)
    df = df.drop(columns=columns_to_delete)
    df = df.fillna(0)

    return df

def get_failed_drives(file_path):
    aggregated_result = pd.DataFrame() 

    for chunk in pd.read_csv(file_path, chunksize=chunksize, dtype=dtype_dict):
        chunk = clean_data_drives(chunk)
        aggregated_result = pd.concat([aggregated_result, chunk[chunk['failure'] == 1]])
    return aggregated_result

def get_success_drives(file_path):
    aggregated_result = pd.DataFrame() 

    for chunk in pd.read_csv(file_path, chunksize=chunksize, dtype=dtype_dict):
        chunk = clean_data_drives(chunk)
        aggregated_result = pd.concat([aggregated_result, chunk[chunk['failure'] == 0]])
    return aggregated_result

def get_data_drives(folder_path, failed):
    df = pd.DataFrame()
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            if failed:
                df = pd.concat([df, get_failed_drives(file_path)])
            else:
                df = pd.concat([df, get_success_drives(file_path)])
            #print(file_path, ' done')
    return df

def cleandata_smart(df, failure_array, success_array):
    
    df.head()
    columns_to_delete = ['model','capacity_bytes','datacenter','cluster_id','vault_id','pod_id','pod_slot_num','is_legacy_format']
    smart_allowed = ['date', 'serial_number']
    rows_allowed = [1, 3, 5, 7, 9, 187, 189, 190, 195, 197] # https://www.cropel.com/library/smart-attribute-list.aspx (Interestingly the chosen attributes from RNN paper and CT paper are the same)
    for i in rows_allowed: smart_allowed.append(f'smart_{i}_normalized')
    for column in df.columns:
        if column != 'failure' and column not in smart_allowed and column != "smart_5_raw" and column != "smart_197_raw":
            columns_to_delete.append(column)


    df = df[df['serial_number'].isin(failure_array) | df['serial_number'].isin(success_array)]

    df = df.drop(columns=columns_to_delete)
    df = df.fillna(0)
    float_columns = df.select_dtypes(include=['float64']).columns
    df[float_columns] = df[float_columns].astype(int)

    return df

def get_failed(file_path, failure_array, success_array):
    aggregated_result = pd.DataFrame() 

    for chunk in pd.read_csv(file_path, chunksize=chunksize, dtype=dtype_dict):
        chunk = cleandata_smart(chunk, failure_array, success_array)
        aggregated_result = pd.concat([aggregated_result, chunk])
        
    return aggregated_result

def get_data(folder_path, verbose):
    failure_array = get_data_drives(folder_path, True)['serial_number'].values
    success_size = len(failure_array)
    success_array = get_data_drives(folder_path, False).head(success_size)['serial_number'].values

    df = pd.DataFrame()
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            df = pd.concat([df, get_failed(file_path, failure_array, success_array)])
            #df = pd.concat([df, process_chunks(file_path)])
            if verbose:
                print(file_path, ' done')
    return df

def process_data(df):
  df.sort_values(by=['serial_number', 'date'], ascending=[True, True], inplace=True)
  return df

class CustomDrives(Dataset):
    def __init__(self, day_num = None, folder_path = None, verbose = False):
        self.day_num = day_num
        self.folder_path = folder_path
        self.verbose = verbose

        original_df = get_data(folder_path, verbose)
        df = process_data(original_df)
        df = df.groupby('serial_number').filter(lambda x: len(x) > day_num)
        self.df = df

    def __len__(self):
        # column numbers (days) (each day is an array of the smart_data) * number of drives (serial_numbers)
        return self.day_num * len(self.df['serial_number'].unique())

    def __getitem__(self, serial_number, idx):
        # Return the date of today and the day to be predicted
        # TODO improve the method
        return self.df[self.df['serial_number'] == serial_number].iloc[idx], self.df[self.df['serial_number'] == serial_number].iloc[idx + 1]

    def gettestandtraindata(self):
        # Return the test and train data``
        train_ratio = 0.8

        train_df = self.df.groupby('serial_number').apply(lambda x: x.head(int(len(x) * train_ratio))).reset_index(drop=True)
        test_df = self.df.groupby('serial_number').apply(lambda x: x.tail(int(len(x) * (1 - train_ratio)))).reset_index(drop=True)
        return train_df, test_df

    def get_df(self):
        return self.df

path = 'data_Q4_2023_test'
test_path = 'data_2023_2'

custom_drives = CustomDrives(day_num = 10, folder_path=path)

training_data, test_data = custom_drives.gettestandtraindata()

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

grouped_df = custom_drives.get_df.groupby('serial_number')
for name, group in grouped_df:
  plt.plot(group['date'], group['smart_1_normalized'])

plt.xticks(rotation=45)

plt.ylabel('smart 1')

plt.title('Data by Drive')
plt.legend()
plt.show()

train_ratio = 0.8

train_df = df.groupby('serial_number').apply(lambda x: x.head(int(len(x) * train_ratio))).reset_index(drop=True)
test_df = df.groupby('serial_number').apply(lambda x: x.tail(int(len(x) * (1 - train_ratio)))).reset_index(drop=True)

dataset_train = train_df.iloc[:, 3:]
dataset_train = np.reshape(dataset_train, (-1,1))
dataset_train.shape

dataset_test = test_df.iloc[:, 3:]

dataset_test = np.reshape(dataset_test, (-1,1))
dataset_test.shape

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
# scaling dataset
scaled_train = scaler.fit_transform(dataset_train)

print(scaled_train[:5])
# Normalizing values between 0 and 1
scaled_test = scaler.fit_transform(dataset_test)
print(*scaled_test[:5]) #prints the first 5 rows of scaled_test


# ## References 

# Reference: https://medium.com/@VersuS_/coding-a-recurrent-neural-network-rnn-from-scratch-using-pytorch-a6c9fc8ed4a7
# 
# Reference: https://www.geeksforgeeks.org/implementing-recurrent-neural-networks-in-pytorch/
# 
# Reference: https://www.geeksforgeeks.org/time-series-forecasting-using-pytorch/
# 
# Reference: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
