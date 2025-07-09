# Step 1: Setup and Imports

import os
import sys
from typing import Optional
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import joblib
import json
from datetime import datetime
from sklearn.metrics import classification_report
import random
from joblib import load
import seaborn as sns
from feature_engine.imputation import RandomSampleImputer
import re
import shutil 

# Add project root and submodules to sys.path for imports
project_root = os.path.abspath(os.path.join(os.getcwd(), "../"))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "notebooks/LSTM"))
sys.path.append(os.path.join(project_root, "notebooks/CT"))

import notebooks.LSTM.smart as smart
import notebooks.CT.CT as CT
from notebooks.LSTM.smart import Net

post_proc_data_path = "../data/"
ct_data_path = "../data/a_test/processed_data"
lstm_data_path = "../data/a_test/processed_data"
days_to_train = 4
days_to_predict = 1
verbose = True

# Correlation matrix parameters
n_features = 10 # Number of features to select based on correlation

#LSTM parameters
num_features = 10 # For default SMART data, this should match the number of features in your dataset
n_neurons = 6
num_epochs = 4000
learning_rate = 0.01

lstm_param_grid = {
    'n_neurons': [4, 8, 16, 32],
    'learning_rate': [0.001, 0.003, 0.005, 0.01],
    'batch_size': [2, 4, 8, 16]
}

#CT parameters
ct_depth = 100
ct_leaf = 15

ct_param_grid = {
    'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200, None],
    'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50]
}


# Set device for torch: use GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

preprocess_path = '../data/validation_data/'

# Load data for correlation analysis
ct_raw_data = CT.importdata(preprocess_path, columns_to_delete=[])
preprocessing_df = ct_raw_data.copy()

columns_to_delete = [
    # Remove normalized and raw features in normalized_rows and raw_rows from columns_to_delete
    'datacenter', 'cluster_id', 'vault_id', 'pod_id', 'pod_slot_num', 'is_legacy_format',
    'smart_1_normalized','smart_1_raw','smart_2_normalized','smart_2_raw','smart_3_normalized','smart_3_raw',
    'smart_4_normalized','smart_4_raw','smart_5_normalized','smart_5_raw','smart_7_normalized','smart_7_raw',
    'smart_8_normalized','smart_8_raw','smart_9_normalized','smart_9_raw','smart_10_normalized','smart_10_raw',
    'smart_11_normalized','smart_11_raw','smart_12_normalized','smart_12_raw','smart_13_normalized','smart_13_raw',
    'smart_15_normalized','smart_15_raw','smart_16_normalized','smart_16_raw','smart_17_normalized','smart_17_raw',
    'smart_18_normalized','smart_18_raw','smart_22_normalized','smart_22_raw','smart_23_normalized','smart_23_raw',
    'smart_24_normalized','smart_24_raw','smart_160_normalized','smart_160_raw','smart_161_normalized','smart_161_raw',
    'smart_163_normalized','smart_163_raw','smart_164_normalized','smart_164_raw','smart_165_normalized','smart_165_raw',
    'smart_166_normalized','smart_166_raw','smart_167_normalized','smart_167_raw','smart_168_normalized','smart_168_raw',
    'smart_169_normalized','smart_169_raw','smart_170_normalized','smart_170_raw','smart_171_normalized','smart_171_raw',
    'smart_172_normalized','smart_172_raw','smart_173_normalized','smart_173_raw','smart_174_normalized','smart_174_raw',
    'smart_175_normalized','smart_175_raw','smart_176_normalized','smart_176_raw','smart_177_normalized','smart_177_raw',
    'smart_178_normalized','smart_178_raw','smart_179_normalized','smart_179_raw','smart_180_normalized','smart_180_raw',
    'smart_181_normalized','smart_181_raw','smart_182_normalized','smart_182_raw','smart_183_normalized','smart_183_raw',
    'smart_184_normalized','smart_184_raw','smart_187_normalized','smart_187_raw','smart_188_normalized','smart_188_raw',
    'smart_189_normalized','smart_189_raw','smart_190_normalized','smart_190_raw','smart_191_normalized','smart_191_raw',
    'smart_192_normalized','smart_192_raw','smart_193_normalized','smart_193_raw','smart_194_normalized','smart_194_raw',
    'smart_195_normalized','smart_195_raw','smart_196_normalized','smart_196_raw','smart_197_normalized','smart_197_raw',
    'smart_198_normalized','smart_198_raw','smart_199_normalized','smart_199_raw','smart_200_normalized','smart_200_raw',
    'smart_201_normalized','smart_201_raw','smart_202_normalized','smart_202_raw','smart_206_normalized','smart_206_raw',
    'smart_210_normalized','smart_210_raw','smart_218_normalized','smart_218_raw','smart_220_normalized','smart_220_raw',
    'smart_222_normalized','smart_222_raw','smart_223_normalized','smart_223_raw','smart_224_normalized','smart_224_raw',
    'smart_225_normalized','smart_225_raw','smart_226_normalized','smart_226_raw','smart_230_normalized','smart_230_raw',
    'smart_231_normalized','smart_231_raw','smart_232_normalized','smart_232_raw','smart_233_normalized','smart_233_raw',
    'smart_234_normalized','smart_234_raw','smart_235_normalized','smart_235_raw','smart_240_normalized','smart_240_raw',
    'smart_241_normalized','smart_241_raw','smart_242_normalized','smart_242_raw','smart_244_normalized','smart_244_raw',
    'smart_245_normalized','smart_245_raw','smart_246_normalized','smart_246_raw','smart_247_normalized','smart_247_raw',
    'smart_248_normalized','smart_248_raw','smart_250_normalized','smart_250_raw','smart_251_normalized','smart_251_raw',
    'smart_252_normalized','smart_252_raw','smart_254_normalized','smart_254_raw','smart_255_normalized','smart_255_raw'
]



try:
    for n in normalized_rows:
        columns_to_delete.remove(f"smart_{n}_normalized")
except Exception as e:
    normalized_rows = [187, 1, 3, 5, 195, 199, 194, 184, 189, 222]
    for n in normalized_rows:
        columns_to_delete.remove(f"smart_{n}_normalized")
raw_rows = []
df = preprocessing_df.drop(columns=columns_to_delete, errors='ignore')

nan_columns = df.columns[df.isna().all()].tolist()
print(f"Columns with only NaN values: {nan_columns}")

df = df.drop(columns=nan_columns)

nan_counts = df.iloc[:, :5].isna().sum()

output_dir = '../reports/figures/histogramas/'
os.makedirs(output_dir, exist_ok=True)

df_smart = df.iloc[:, 5:]
kurtosis_columns = [col for col in df_smart.columns if df_smart[col].kurtosis() < -1.0]
print(f"Columnas con kurtosis menor a -1.0: {kurtosis_columns}")

# Calcular el rango intercuartílico (IQR) para las columnas 6 en adelante
iqr_values = {}
outliers_dict = {}
no_kurtosis_df = df.drop(columns=kurtosis_columns)
for column in no_kurtosis_df.columns[5:]:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    iqr_values[column] = IQR
    outliers = df[(df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))]
    print(f"Outliers in column {column}: {outliers.shape[0]}")
    if not outliers.empty:
        outliers_dict[column] = outliers

        valores = {}
for column in df.columns[5:]:
    if column not in kurtosis_columns:
        if column in outliers:
            valores[column] = df[column].median()
        else:
            valores[column] = df[column].mean()

print(valores)

# Calcular la frecuencia de cada modelo
model_freq = df['model'].value_counts()

# Crear un diccionario para mapear cada modelo a su frecuencia
model_freq_dict = model_freq.to_dict()

# Crear una nueva columna categórica basada en la frecuencia de uso
df['model_freq'] = df['model'].map(model_freq_dict)

# Convertir la nueva columna a tipo categórico
df['model_freq'] = pd.cut(df['model_freq'], bins=[0, 10, 100, 1000, 10000, float('inf')],
                          labels=['Muy Bajo', 'Bajo', 'Medio', 'Alto', 'Muy Alto'])

print(df[['model', 'model_freq']].head())


def extract_brand(model):
    match = re.match(r'^[A-Za-z]+', model)
    return match.group(0) if match else None

df['brand'] = df['model'].apply(extract_brand)
print(df['brand'].drop_duplicates())

df = pd.get_dummies(df, columns=['brand', 'model_freq'])
print(df.head())

print(df['capacity_bytes'].unique())

# Replace -1 in 'capacity_bytes' with the correct value from the same serial_number

# First, find serial_numbers with more than one unique capacity (including -1)
mask_minus1 = df['capacity_bytes'] == '-1'
serials_with_minus1 = df.loc[mask_minus1, 'serial_number'].unique()

# For each affected serial_number, replace -1 with the other value
for serial in tqdm(serials_with_minus1, desc="Replacing -1 in capacity_bytes"):
    # Get all unique non -1 capacities for this serial_number
    capacities = df.loc[(df['serial_number'] == serial) & (df['capacity_bytes'] != '-1'), 'capacity_bytes'].unique()
    if len(capacities) == 1:
        correct_capacity = capacities[0]
        df.loc[(df['serial_number'] == serial) & (df['capacity_bytes'] == '-1'), 'capacity_bytes'] = correct_capacity

print("Replaced -1 values in 'capacity_bytes' where possible.")

print(df['capacity_bytes'].unique())

Var=df['capacity_bytes']                      # Variable categórica ordinal
n=Var.nunique()                     # Cardinalidad
lim_inf=(n-1)//2 if n%2!=0 else n-1 # Abs del límite inferior
step=1 if n%2!=0 else 2             # Intervalo
X=range(-lim_inf,lim_inf+1,step)
list(X)

beta0=0
beta1=1 # beta1>0  ->  codificación creciente. beta1<0  ->  codificación decreciente
y=[beta0+beta1*x for x in X]
y

# Asociamos estos valores a las categorías ordenadas de 'Var'
# Var.unique()
Var_ord = np.sort(Var.unique())
Var_ord

map_lin_CB={categoria:codificacion for categoria,codificacion in zip(Var_ord,y)}
map_lin_CB

df['Lin_capacity_bytes']=df['capacity_bytes'].map(map_lin_CB)

df.head()

# Store each day's data in a separate CSV file in ../data/processed_data
#output_dir = "../data/a_test/processed_data"
output_dir = "../data/validation_data/processed_data"

# Delete all contents of the output directory before storing new files
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

# Ensure 'date' column is in datetime format
df['date'] = pd.to_datetime(df['date'])
# Group by date and save each group to a separate CSV, overwriting if file exists
for day, group in df.groupby(df['date'].dt.date):
    filename = os.path.join(output_dir, f"data_{day}.csv")
    group.to_csv(filename, index=False, mode='w')  # mode='w' ensures overwrite
    print(f"Saved (overwritten) {filename}")
