
import os
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

dtype_dict = {'date': 'str', 'serial_number': 'str', 'model': 'str', 'capacity_bytes': 'int32', 'failure': 'bool', 'datacenter': 'str', 'cluster_id': 'int8', 'vault_id': 'int16', 'pod_id': 'int16', 'pod_slot_num': 'float32', 'is_legacy_format': 'bool', 'smart_1_normalized': 'float64', 'smart_1_raw': 'float64', 'smart_2_normalized': 'float64', 'smart_2_raw': 'float64', 'smart_3_normalized': 'float64', 'smart_3_raw': 'float64', 'smart_4_normalized': 'float64', 'smart_4_raw': 'float64', 'smart_5_normalized': 'float64', 'smart_5_raw': 'float64', 'smart_7_normalized': 'float64', 'smart_7_raw': 'float64', 'smart_8_normalized': 'float64', 'smart_8_raw': 'float64', 'smart_9_normalized': 'float64', 'smart_9_raw': 'float64', 'smart_10_normalized': 'float64', 'smart_10_raw': 'float64', 'smart_11_normalized': 'float64', 'smart_11_raw': 'float64', 'smart_12_normalized': 'float64', 'smart_12_raw': 'float64', 'smart_13_normalized': 'float64', 'smart_13_raw': 'float64', 'smart_15_normalized': 'float64', 'smart_15_raw': 'float64', 'smart_16_normalized': 'float64', 'smart_16_raw': 'float64', 'smart_17_normalized': 'float64', 'smart_17_raw': 'float64', 'smart_18_normalized': 'float64', 'smart_18_raw': 'float64', 'smart_22_normalized': 'float64', 'smart_22_raw': 'float64', 'smart_23_normalized': 'float64', 'smart_23_raw': 'float64', 'smart_24_normalized': 'float64', 'smart_24_raw': 'float64', 'smart_27_normalized': 'float64', 'smart_27_raw': 'float64', 'smart_71_normalized': 'float64', 'smart_71_raw': 'float64', 'smart_82_normalized': 'float64', 'smart_82_raw': 'float64', 'smart_90_normalized': 'float64', 'smart_90_raw': 'float64', 'smart_160_normalized': 'float64', 'smart_160_raw': 'float64', 'smart_161_normalized': 'float64', 'smart_161_raw': 'float64', 'smart_163_normalized': 'float64', 'smart_163_raw': 'float64', 'smart_164_normalized': 'float64', 'smart_164_raw': 'float64', 'smart_165_normalized': 'float64', 'smart_165_raw': 'float64', 'smart_166_normalized': 'float64', 'smart_166_raw': 'float64', 'smart_167_normalized': 'float64', 'smart_167_raw': 'float64', 'smart_168_normalized': 'float64', 'smart_168_raw': 'float64', 'smart_169_normalized': 'float64', 'smart_169_raw': 'float64', 'smart_170_normalized': 'float64', 'smart_170_raw': 'float64', 'smart_171_normalized': 'float64', 'smart_171_raw': 'float64', 'smart_172_normalized': 'float64', 'smart_172_raw': 'float64', 'smart_173_normalized': 'float64', 'smart_173_raw': 'float64', 'smart_174_normalized': 'float64', 'smart_174_raw': 'float64', 'smart_175_normalized': 'float64', 'smart_175_raw': 'float64', 'smart_176_normalized': 'float64', 'smart_176_raw': 'float64', 'smart_177_normalized': 'float64', 'smart_177_raw': 'float64', 'smart_178_normalized': 'float64', 'smart_178_raw': 'float64', 'smart_179_normalized': 'float64', 'smart_179_raw': 'float64', 'smart_180_normalized': 'float64', 'smart_180_raw': 'float64', 'smart_181_normalized': 'float64', 'smart_181_raw': 'float64', 'smart_182_normalized': 'float64', 'smart_182_raw': 'float64', 'smart_183_normalized': 'float64', 'smart_183_raw': 'float64', 'smart_184_normalized': 'float64', 'smart_184_raw': 'float64', 'smart_187_normalized': 'float64', 'smart_187_raw': 'float64', 'smart_188_normalized': 'float64', 'smart_188_raw': 'float64', 'smart_189_normalized': 'float64', 'smart_189_raw': 'float64', 'smart_190_normalized': 'float64', 'smart_190_raw': 'float64', 'smart_191_normalized': 'float64', 'smart_191_raw': 'float64', 'smart_192_normalized': 'float64', 'smart_192_raw': 'float64', 'smart_193_normalized': 'float64', 'smart_193_raw': 'float64', 'smart_194_normalized': 'float64', 'smart_194_raw': 'float64', 'smart_195_normalized': 'float64', 'smart_195_raw': 'float64', 'smart_196_normalized': 'float64', 'smart_196_raw': 'float64', 'smart_197_normalized': 'float64', 'smart_197_raw': 'float64', 'smart_198_normalized': 'float64', 'smart_198_raw': 'float64', 'smart_199_normalized': 'float64', 'smart_199_raw': 'float64', 'smart_200_normalized': 'float64', 'smart_200_raw': 'float64', 'smart_201_normalized': 'float64', 'smart_201_raw': 'float64', 'smart_202_normalized': 'float64', 'smart_202_raw': 'float64', 'smart_206_normalized': 'float64', 'smart_206_raw': 'float64', 'smart_210_normalized': 'float64', 'smart_210_raw': 'float64', 'smart_218_normalized': 'float64', 'smart_218_raw': 'float64', 'smart_220_normalized': 'float64', 'smart_220_raw': 'float64', 'smart_222_normalized': 'float64', 'smart_222_raw': 'float64', 'smart_223_normalized': 'float64', 'smart_223_raw': 'float64', 'smart_224_normalized': 'float64', 'smart_224_raw': 'float64', 'smart_225_normalized': 'float64', 'smart_225_raw': 'float64', 'smart_226_normalized': 'float64', 'smart_226_raw': 'float64', 'smart_230_normalized': 'float64', 'smart_230_raw': 'float64', 'smart_231_normalized': 'float64', 'smart_231_raw': 'float64', 'smart_232_normalized': 'float64', 'smart_232_raw': 'float64', 'smart_233_normalized': 'float64', 'smart_233_raw': 'float64', 'smart_234_normalized': 'float64', 'smart_234_raw': 'float64', 'smart_235_normalized': 'float64', 'smart_235_raw': 'float64', 'smart_240_normalized': 'float64', 'smart_240_raw': 'float64', 'smart_241_normalized': 'float64', 'smart_241_raw': 'float64', 'smart_242_normalized': 'float64', 'smart_242_raw': 'float64', 'smart_244_normalized': 'float64', 'smart_244_raw': 'float64', 'smart_245_normalized': 'float64', 'smart_245_raw': 'float64', 'smart_246_normalized': 'float64', 'smart_246_raw': 'float64', 'smart_247_normalized': 'float64', 'smart_247_raw': 'float64', 'smart_248_normalized': 'float64', 'smart_248_raw': 'float64', 'smart_250_normalized': 'float64', 'smart_250_raw': 'float64', 'smart_251_normalized': 'float64', 'smart_251_raw': 'float64', 'smart_252_normalized': 'float64', 'smart_252_raw': 'float64', 'smart_254_normalized': 'float64', 'smart_254_raw': 'float64', 'smart_255_normalized': 'float64', 'smart_255_raw': 'float64'}

def clean_data_drives(df):
    # Drop columns that are not relevant for the prediction
    columns_to_delete = ['model','capacity_bytes','datacenter','cluster_id','vault_id','pod_id','pod_slot_num','is_legacy_format']
    smart_allowed = ['failure', 'serial_number']
    for column in df.columns:
        if column not in smart_allowed:
            columns_to_delete.append(column)
    df = df.drop(columns=columns_to_delete)
    df = df.fillna(0)

    return df

def clean_data_smart(df):
    columns_to_delete = ['model','capacity_bytes','datacenter','cluster_id','vault_id','pod_id','pod_slot_num','is_legacy_format']
    smart_allowed = ['date', 'serial_number']
    rows_allowed = [1, 3, 5, 7, 9, 187, 189, 190, 195, 197] # https://www.cropel.com/library/smart-attribute-list.aspx (Interestingly the chosen attributes from RNN paper and CT paper are the same)
    for i in rows_allowed: smart_allowed.append(f'smart_{i}_normalized')
    for column in df.columns:
        if column != 'failure' and column not in smart_allowed and column != "smart_5_raw" and column != "smart_197_raw":
            columns_to_delete.append(column)

    df = df.drop(columns=columns_to_delete)
    df = df.fillna(0)
    float_columns   = df.select_dtypes(include=['float64']).columns
    df[float_columns] = df[float_columns].astype(int)

    return df

class _CustomDrives(Dataset):
    def __init__(self, 
                 root: str = '.',
                 train: bool = False,
                 input_len: int = 3,
                 label_len: int = 1,
                 verbose = False):
        super().__init__()
        
        #Input len is the past window of time and label len is the future window 
        self.input_len = input_len
        self.label_len = label_len
        
        self.dataset_path = root
        if not os.path.isdir(self.dataset_path):
            msg = f'Could not find the csv files in {self.dataset_path}. '
            raise FileNotFoundError(msg)

        data = []
        csv_files = [f for f in os.listdir(self.dataset_path) if f.endswith(".csv")]
        
        for file_name in tqdm(csv_files, desc="Loading CSV files"):
            df = clean_data_smart(pd.read_csv(os.path.join(self.dataset_path, file_name), dtype=dtype_dict))
            # this can be in threadpool
            data.append(df)
            if len(data) > 5: break # TODO:: remove this line for entire dataset loading
            if verbose: print(f'Loaded {file_name} with shape {df.shape}')
                 
        data = pd.concat(data)

        # sort to make consistent
        data.sort_values(by=['serial_number', 'date'], ascending=[True, True], inplace=True)

        if verbose: print(f'Loaded {len(data)} rows with shape {data.shape}')

        grouped = data.groupby('serial_number')

        if verbose: print(f'Loaded {len(grouped.groups)} serial numbers')

        # then we drop 20% of groups if traning otherwise we drop 80% for testing
        drop_ratio = int((len(grouped.groups)) * 0.8 if train else 0.2)
        if train:
            groups_to_keep = list(grouped.groups.keys())[drop_ratio:]
        else:
            groups_to_keep = list(grouped.groups.keys())[:-drop_ratio]
        if verbose: print(f'Loaded {len(groups_to_keep)} serial numbers after sampling')
        data =data[~data['serial_number'].isin(groups_to_keep)]
        grouped = data.groupby('serial_number')
        if verbose: print(f'Loaded {len(grouped.groups)} serial numbers after sampling')
        
        # Compute the size of each group and create a boolean mask for rows to keep
        group_sizes = grouped.size()
        groups_to_keep = group_sizes[group_sizes >= (input_len + label_len)].index
        self.data = data[data['serial_number'].isin(groups_to_keep)].groupby('serial_number')
        if verbose: print(f'Loaded {len(self.data.groups)} serial numbers after filtering')
        
        # need to to calculations to get the length of the dataset
        # we may start with a simple calculation and improve it later:
        # just the number of keys
        self.len = len(self.data.groups)
        self.list_of_keys = list(self.data.groups.keys())
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        values =  self.data.get_group(self.list_of_keys[idx])
        
        # We can return the drives that have n or more days
        # print(f'Number of rows: {len(values)}')
        train = values.iloc[:self.input_len]
        label = values.iloc[self.input_len:self.input_len + self.label_len]
        # print(f'train shape: {train.shape}, label shape: {label.shape}')
        return train, label
    
class CustomDrives(Dataset):
    def __init__(self, 
                 root: str = '.',
                 train: bool = False,
                 input_len: int = 3,
                 label_len: int = 1,
                 verbose = False):
        super().__init__()
        self.dataset = _CustomDrives(root=root, 
                                     train=train, 
                                     input_len=input_len,
                                     label_len=label_len,
                                     verbose=verbose)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        train, label = self.dataset[idx]
        train = train.drop(columns=['serial_number', 'date', 'failure']) # maybe failure?
        label = label.drop(columns=['serial_number', 'date', 'failure']) # maybe failure?
        # now convert to torch tensors
        # return shape [input_len, num_features], [label_len, num_features]
        return torch.from_numpy(train.values).double(), torch.from_numpy(label.values).double()

class Net(nn.Module):
    def __init__(self, n_neurons, features, days_to_predict=2):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size=features, hidden_size=n_neurons, batch_first=True)
        self.fc = nn.Linear(n_neurons, days_to_predict*features)
        self.days_to_predict = days_to_predict
        self.features = features

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the last output of the LSTM
        return out.view(-1, self.days_to_predict, self.features)
    
if __name__ == '__main__':

    days_to_train = 3
    days_to_predict = 2
    look_back = days_to_train + days_to_predict
    path = "../../data/data_Q4_2024"

    dataset_train = CustomDrives(root=path, 
                            train=True, 
                            input_len=days_to_train,
                            label_len=days_to_predict,
                            verbose=True)

    dataset_test = CustomDrives(root=path, 
                                train=False, 
                                input_len=days_to_train,
                                label_len=days_to_predict,
                                verbose=True)    

    train_loader = DataLoader(dataset_train, batch_size=3, shuffle=True)

    test_loader = DataLoader(dataset_test, batch_size=3, shuffle=True)

    num_features = 12
    n_neurons = 4

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    model = Net(n_neurons, num_features, days_to_predict).double()
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 1000

    train, labels = next(iter(train_loader))
    print(f'shape: {train.shape, labels.shape}')

    loss_curve = []
    minimum_loss = np.inf    # Load the model if it exists
    if os.path.exists('model.pth'):
        model.load_state_dict(torch.load('model.pth'))
        print("Model loaded successfully.")
    else:
        print("No saved model found. Training from scratch.")

    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        epoch_loss = 0
        num_batches = 0
        
        # Training loop with progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=False)
        for train_labels in train_pbar:
            model.zero_grad()
            train, labels = train_labels

            predictions = model(train)
            loss = loss_function(predictions, labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Update progress bar with current loss
            train_pbar.set_postfix({'Loss': f'{loss.item():.6f}'})

        # Validation loop with progress bar
        val_loss = 0
        val_batches = 0
        test_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", leave=False)
        
        with torch.no_grad():  # Don't compute gradients for validation
            for test_labels in test_pbar:
                test, labels = test_labels
                predictions = model(test)
                loss = loss_function(predictions, labels)
                
                val_loss += loss.item()
                val_batches += 1
                
                # Update progress bar with current validation loss
                test_pbar.set_postfix({'Val Loss': f'{loss.item():.6f}'})
        
        # Calculate average losses
        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        
        loss_curve.append(avg_val_loss)
        
        # Save best model
        if avg_val_loss < minimum_loss:
            minimum_loss = avg_val_loss
            torch.save(model.state_dict(), 'model.pth')
            
        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Save loss plot every 10 epochs
        if (epoch + 1) % 10 == 0:
            fig, ax = plt.subplots(1, 1, figsize=(15, 5))
            ax.plot(loss_curve, lw=2)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Validation Loss")
            ax.set_title("Training Progress")
            plt.savefig(f'loss_epoch_{epoch+1}.png')
            plt.close()  # Close the figure to free memory

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.plot(loss_curve, lw=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss (L1)")
    plt.savefig('final_loss.png')

    # Score the results with the testing set and plot 
    with torch.no_grad():
        test_predictions = model(dataset_train).squeeze()

    test_predictions = test_predictions.detach().numpy()

    x = np.arange(100 + look_back, 100 + look_back + len(test_predictions) * 0.5, 0.5)
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.plot(x, test_predictions, lw=2, label='Predictions')
    ax.set_xlabel("Time")
    ax.set_ylabel("Predicted Values")
    ax.legend()
    plt.savefig('test_predictions.png')
    plt.show()