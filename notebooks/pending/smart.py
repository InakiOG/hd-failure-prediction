from typing import Optional
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
import joblib
import json
from datetime import datetime

dtype_dict = {'date': 'str', 'serial_number': 'str', 'model': 'str', 'capacity_bytes': 'int32', 'failure': 'bool', 'datacenter': 'str', 'cluster_id': 'int8', 'vault_id': 'int16', 'pod_id': 'int16', 'pod_slot_num': 'float32', 'is_legacy_format': 'bool', 'smart_1_normalized': 'float64', 'smart_1_raw': 'float64', 'smart_2_normalized': 'float64', 'smart_2_raw': 'float64', 'smart_3_normalized': 'float64', 'smart_3_raw': 'float64', 'smart_4_normalized': 'float64', 'smart_4_raw': 'float64', 'smart_5_normalized': 'float64', 'smart_5_raw': 'float64', 'smart_7_normalized': 'float64', 'smart_7_raw': 'float64', 'smart_8_normalized': 'float64', 'smart_8_raw': 'float64', 'smart_9_normalized': 'float64', 'smart_9_raw': 'float64', 'smart_10_normalized': 'float64', 'smart_10_raw': 'float64', 'smart_11_normalized': 'float64', 'smart_11_raw': 'float64', 'smart_12_normalized': 'float64', 'smart_12_raw': 'float64', 'smart_13_normalized': 'float64', 'smart_13_raw': 'float64', 'smart_15_normalized': 'float64', 'smart_15_raw': 'float64', 'smart_16_normalized': 'float64', 'smart_16_raw': 'float64', 'smart_17_normalized': 'float64', 'smart_17_raw': 'float64', 'smart_18_normalized': 'float64', 'smart_18_raw': 'float64', 'smart_22_normalized': 'float64', 'smart_22_raw': 'float64', 'smart_23_normalized': 'float64', 'smart_23_raw': 'float64', 'smart_24_normalized': 'float64', 'smart_24_raw': 'float64', 'smart_27_normalized': 'float64', 'smart_27_raw': 'float64', 'smart_71_normalized': 'float64', 'smart_71_raw': 'float64', 'smart_82_normalized': 'float64', 'smart_82_raw': 'float64', 'smart_90_normalized': 'float64', 'smart_90_raw': 'float64', 'smart_160_normalized': 'float64', 'smart_160_raw': 'float64', 'smart_161_normalized': 'float64', 'smart_161_raw': 'float64', 'smart_163_normalized': 'float64', 'smart_163_raw': 'float64', 'smart_164_normalized': 'float64', 'smart_164_raw': 'float64', 'smart_165_normalized': 'float64', 'smart_165_raw': 'float64', 'smart_166_normalized': 'float64', 'smart_166_raw': 'float64', 'smart_167_normalized': 'float64', 'smart_167_raw': 'float64', 'smart_168_normalized': 'float64', 'smart_168_raw': 'float64', 'smart_169_normalized': 'float64', 'smart_169_raw': 'float64', 'smart_170_normalized': 'float64', 'smart_170_raw': 'float64', 'smart_171_normalized': 'float64', 'smart_171_raw': 'float64', 'smart_172_normalized': 'float64', 'smart_172_raw': 'float64', 'smart_173_normalized': 'float64', 'smart_173_raw': 'float64', 'smart_174_normalized': 'float64', 'smart_174_raw': 'float64', 'smart_175_normalized': 'float64', 'smart_175_raw': 'float64', 'smart_176_normalized': 'float64', 'smart_176_raw': 'float64', 'smart_177_normalized': 'float64', 'smart_177_raw': 'float64', 'smart_178_normalized': 'float64', 'smart_178_raw': 'float64', 'smart_179_normalized': 'float64', 'smart_179_raw': 'float64', 'smart_180_normalized': 'float64', 'smart_180_raw': 'float64', 'smart_181_normalized': 'float64', 'smart_181_raw': 'float64', 'smart_182_normalized': 'float64', 'smart_182_raw': 'float64', 'smart_183_normalized': 'float64', 'smart_183_raw': 'float64', 'smart_184_normalized': 'float64', 'smart_184_raw': 'float64', 'smart_187_normalized': 'float64', 'smart_187_raw': 'float64', 'smart_188_normalized': 'float64', 'smart_188_raw': 'float64', 'smart_189_normalized': 'float64', 'smart_189_raw': 'float64', 'smart_190_normalized': 'float64', 'smart_190_raw': 'float64', 'smart_191_normalized': 'float64', 'smart_191_raw': 'float64', 'smart_192_normalized': 'float64', 'smart_192_raw': 'float64', 'smart_193_normalized': 'float64', 'smart_193_raw': 'float64', 'smart_194_normalized': 'float64', 'smart_194_raw': 'float64', 'smart_195_normalized': 'float64', 'smart_195_raw': 'float64', 'smart_196_normalized': 'float64', 'smart_196_raw': 'float64', 'smart_197_normalized': 'float64', 'smart_197_raw': 'float64', 'smart_198_normalized': 'float64', 'smart_198_raw': 'float64', 'smart_199_normalized': 'float64', 'smart_199_raw': 'float64', 'smart_200_normalized': 'float64', 'smart_200_raw': 'float64', 'smart_201_normalized': 'float64', 'smart_201_raw': 'float64', 'smart_202_normalized': 'float64', 'smart_202_raw': 'float64', 'smart_206_normalized': 'float64', 'smart_206_raw': 'float64', 'smart_210_normalized': 'float64', 'smart_210_raw': 'float64', 'smart_218_normalized': 'float64', 'smart_218_raw': 'float64', 'smart_220_normalized': 'float64', 'smart_220_raw': 'float64', 'smart_222_normalized': 'float64', 'smart_222_raw': 'float64', 'smart_223_normalized': 'float64', 'smart_223_raw': 'float64', 'smart_224_normalized': 'float64', 'smart_224_raw': 'float64', 'smart_225_normalized': 'float64', 'smart_225_raw': 'float64', 'smart_226_normalized': 'float64', 'smart_226_raw': 'float64', 'smart_230_normalized': 'float64', 'smart_230_raw': 'float64', 'smart_231_normalized': 'float64', 'smart_231_raw': 'float64', 'smart_232_normalized': 'float64', 'smart_232_raw': 'float64', 'smart_233_normalized': 'float64', 'smart_233_raw': 'float64', 'smart_234_normalized': 'float64', 'smart_234_raw': 'float64', 'smart_235_normalized': 'float64', 'smart_235_raw': 'float64', 'smart_240_normalized': 'float64', 'smart_240_raw': 'float64', 'smart_241_normalized': 'float64', 'smart_241_raw': 'float64', 'smart_242_normalized': 'float64', 'smart_242_raw': 'float64', 'smart_244_normalized': 'float64', 'smart_244_raw': 'float64', 'smart_245_normalized': 'float64', 'smart_245_raw': 'float64', 'smart_246_normalized': 'float64', 'smart_246_raw': 'float64', 'smart_247_normalized': 'float64', 'smart_247_raw': 'float64', 'smart_248_normalized': 'float64', 'smart_248_raw': 'float64', 'smart_250_normalized': 'float64', 'smart_250_raw': 'float64', 'smart_251_normalized': 'float64', 'smart_251_raw': 'float64', 'smart_252_normalized': 'float64', 'smart_252_raw': 'float64', 'smart_254_normalized': 'float64', 'smart_254_raw': 'float64', 'smart_255_normalized': 'float64', 'smart_255_raw': 'float64'}

def clean_data_drives(df):
    """
    Clean and preprocess hard drive data for drive-level analysis.
    
    Removes all columns except 'failure' and 'serial_number', drops unnecessary metadata
    columns, and fills missing values with zeros. This function focuses on keeping only 
    the essential identifiers and failure status.
    
    Args:
        df (pd.DataFrame): Raw hard drive data DataFrame
        
    Returns:
        pd.DataFrame: Cleaned DataFrame containing only failure status and serial number
    """
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
    """
    Clean and preprocess hard drive data for SMART attribute analysis.
    
    Selects only relevant SMART attributes known to be predictive of hard drive failure
    based on research papers. Keeps specific SMART attributes (1,3,5,7,9,187,189,190,195,197),
    date and serial_number for time series analysis, and converts float columns to integers
    for efficiency.
    
    Args:
        df (pd.DataFrame): Raw hard drive data DataFrame with all SMART attributes
        
    Returns:
        pd.DataFrame: Cleaned DataFrame with only selected SMART attributes 
    """
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

class DriveDataLoader:
    """
    Centralized data loader that handles the train/test split at the drive level to prevent data leakage.
    
    This class loads all data once and creates separate train/test splits based on drive serial numbers,
    ensuring that no drive appears in both training and testing sets.
    """
    def __init__(self, root: str, train_ratio: float = 0.8, verbose: bool = False):
        """
        Initialize the data loader and perform the train/test split.
        
        Args:
            root (str): Root directory containing CSV files with hard drive data
            train_ratio (float): Ratio of drives to use for training (0.8 = 80% train, 20% test)
            verbose (bool): Whether to print detailed information during loading
        """
        self.dataset_path = root
        self.train_ratio = train_ratio
        self.verbose = verbose
        
        if not os.path.isdir(self.dataset_path):
            msg = f'Could not find the csv files in {self.dataset_path}.'
            raise FileNotFoundError(msg)
        
        # Load all data
        self._load_all_data()
        
        # Perform train/test split at drive level
        self._split_drives()
    
    def _load_all_data(self):
        """Load all CSV files and concatenate them."""
        data = []
        csv_files = []
        
        # Check if the path contains subfolders
        subfolders = [os.path.join(self.dataset_path, d) for d in os.listdir(self.dataset_path) 
                     if os.path.isdir(os.path.join(self.dataset_path, d))]
        
        if self.verbose:
            print(f'[DriveDataLoader] Found {len(subfolders)} subfolders in {self.dataset_path}.')
        
        if subfolders:
            # If there are subfolders, collect all CSVs from each subfolder
            for subfolder in subfolders:
                csv_files.extend([os.path.join(subfolder, f) for f in os.listdir(subfolder) 
                                if f.endswith(".csv")])
        else:
            # Otherwise, just collect CSVs from the root folder
            csv_files = [os.path.join(self.dataset_path, f) for f in os.listdir(self.dataset_path) 
                        if f.endswith(".csv")]

        for file_name in tqdm(csv_files, desc="Loading CSV files"):
            df = clean_data_smart(pd.read_csv(file_name, dtype=dtype_dict))
            data.append(df)
            if self.verbose:
                print(f'Loaded {file_name} with shape {df.shape}')
        
        self.all_data = pd.concat(data)
        
        # Sort to make consistent
        self.all_data.sort_values(by=['serial_number', 'date'], ascending=[True, True], inplace=True)
        
        if self.verbose:
            print(f'[DriveDataLoader] Loaded {len(self.all_data)} rows from all CSV files. DataFrame shape: {self.all_data.shape}')
    
    def _split_drives(self):
        """Split drives into train and test sets at the drive level."""
        # Get all unique drive serial numbers
        all_drives = list(self.all_data['serial_number'].unique())
        
        if self.verbose:
            print(f'[DriveDataLoader] Found {len(all_drives)} unique drives.')
        
        # Shuffle drives to randomize the split (with fixed seed for reproducibility)
        np.random.seed(42)
        np.random.shuffle(all_drives)
        
        # Calculate split point
        split_point = int(len(all_drives) * self.train_ratio)
        
        # Split drives
        self.train_drives = all_drives[:split_point]
        self.test_drives = all_drives[split_point:]
        
        if self.verbose:
            print(f'[DriveDataLoader] Split: {len(self.train_drives)} drives for training, {len(self.test_drives)} drives for testing.')
            print(f'[DriveDataLoader] Train ratio: {len(self.train_drives) / len(all_drives):.2%}')
    
    def get_train_data(self):
        """Get data for training drives only."""
        return self.all_data[self.all_data['serial_number'].isin(self.train_drives)]
    
    def get_test_data(self):
        """Get data for testing drives only."""
        return self.all_data[self.all_data['serial_number'].isin(self.test_drives)]

class _CustomDrives(Dataset):
    """
    Internal dataset class for loading and preprocessing hard drive SMART data for time series analysis.
    
    This class handles preprocessing of pre-split data, organizing the data by drive serial number,
    and preparing time-ordered sequences for training or testing. It splits data into input and target
    windows, allowing for time series prediction of future SMART values.
    
    Note: This is an internal implementation class used by CustomDrives.
    """    def __init__(self, 
                 data: pd.DataFrame,
                 input_len: int = 3,
                 label_len: int = 1,
                 verbose = False):
        """
        Initialize the dataset with pre-split data for time series prediction.
        
        Args:
            data (pd.DataFrame): Pre-split DataFrame containing SMART data for either training or testing
            input_len (int): Number of time steps to use as input (look-back window)
            label_len (int): Number of time steps to predict (forecast window)
            verbose (bool): Whether to print detailed information during loading
        """
        super().__init__()
        
        # Input len is the past window of time and label len is the future window 
        self.input_len = input_len
        self.label_len = label_len
        
        if verbose: 
            print(f'[_CustomDrives] Processing {len(data)} rows. DataFrame shape: {data.shape}')

        grouped = data.groupby('serial_number')

        if verbose: 
            print(f'[_CustomDrives] Grouped data by serial_number. Found {len(grouped.groups)} unique drives.')
        
        # Compute the size of each group and create a boolean mask for rows to keep
        group_sizes = grouped.size()
        groups_to_keep = group_sizes[group_sizes >= (input_len + label_len)].index
        self.data = data[data['serial_number'].isin(groups_to_keep)].groupby('serial_number')
        if verbose: 
            print(f'[_CustomDrives] After filtering by minimum sequence length (input_len + label_len = {input_len + label_len}), {len(self.data.groups)} serial numbers remain.')
        if len(self.data.groups) == 0:
            raise ValueError(f'No valid sequences found in the dataset with input_len={input_len} and label_len={label_len}. Please check your data or adjust the parameters.')
        
        # Calculate the length of the dataset
        self.len = len(self.data.groups)
        self.list_of_keys = list(self.data.groups.keys())
    
    def __len__(self):
        """
        Return the length of the dataset (number of drive sequences).
        
        Returns:
            int: Number of drive sequences in the dataset
        """
        return self.len
    
    def __getitem__(self, idx):
        """
        Get a single input-target sequence pair for a specific drive.
        
        Args:
            idx (int): Index of the drive to retrieve
            
        Returns:
            tuple: (train, label) where train contains past SMART data and 
                  label contains future SMART data for the same drive
        """
        values =  self.data.get_group(self.list_of_keys[idx])
        
        # We can return the drives that have n or more days
        train = values.iloc[:self.input_len]
        label = values.iloc[self.input_len:self.input_len + self.label_len]
        return train, label
    
class CustomDrives(Dataset):
    """
    Dataset class for hard drive SMART data time series forecasting.
    
    This class wraps the internal _CustomDrives implementation and prepares
    the data for PyTorch models by converting DataFrame rows to NumPy arrays
    and then to PyTorch tensors. It handles feature extraction and tensor
    conversion for both input and output sequences.
    """    def __init__(self, 
                 data_loader: Optional[DriveDataLoader] = None,
                 root: Optional[str] = None,
                 train: bool = False,
                 input_len: int = 3,
                 label_len: int = 1,
                 verbose = False):
        """
        Initialize the dataset with configuration for time series prediction.
        
        Args:
            data_loader (DriveDataLoader): Pre-initialized data loader with train/test split
            root (str): Root directory containing CSV files (used only if data_loader is None)
            train (bool): Whether this dataset is for training (True) or testing (False)
            input_len (int): Number of time steps to use as input (look-back window)
            label_len (int): Number of time steps to predict (forecast window)
            verbose (bool): Whether to print detailed information during loading
        """
        super().__init__()
        
        # Get the appropriate data based on train/test split
        if data_loader is not None:
            # Use pre-split data
            if train:
                data = data_loader.get_train_data()
            else:
                data = data_loader.get_test_data()
        else:
            # Fallback to old behavior (for backward compatibility)
            if root is None:
                raise ValueError("Either data_loader or root must be provided")
            # Create a temporary data loader
            temp_loader = DriveDataLoader(root=root, verbose=verbose)
            if train:
                data = temp_loader.get_train_data()
            else:
                data = temp_loader.get_test_data()
        
        self.dataset = _CustomDrives(data=data, 
                                     input_len=input_len,
                                     label_len=label_len,
                                     verbose=verbose)

    def __len__(self):
        """
        Return the length of the dataset (number of drive sequences).
        
        Returns:
            int: Number of drive sequences in the dataset
        """
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Get a single input-target tensor pair for model training/testing.
        
        Removes metadata columns and converts DataFrame data to PyTorch tensors.
        
        Args:
            idx (int): Index of the drive sequence to retrieve
            
        Returns:
            tuple: (train_tensor, label_tensor) where tensors contain numeric SMART
                  features for the input and output time windows
        """
        train, label = self.dataset[idx]
        train = train.drop(columns=['serial_number', 'date', 'failure']) # we don't need these columns for training
        label = label.drop(columns=['serial_number', 'date', 'failure']) 
        # now convert to torch tensors
        # return shape [input_len, num_features], [label_len, num_features]
        return torch.from_numpy(train.values).double(), torch.from_numpy(label.values).double()

class Net(nn.Module):
    """
    LSTM neural network architecture for time series prediction of SMART attributes.
    
    This model uses an LSTM (Long Short-Term Memory) network to learn time dependencies
    in hard drive SMART attributes, followed by a fully connected layer to predict
    future values of these attributes. The model predicts multiple days of values
    for all features simultaneously.
    """
    def __init__(self, n_neurons, features, days_to_predict=2):
        """
        Initialize the LSTM-based neural network.
        
        Args:
            n_neurons (int): Number of neurons in the LSTM hidden layer
            features (int): Number of input features (SMART attributes)
            days_to_predict (int): Number of future time steps to predict
        """
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size=features, hidden_size=n_neurons, batch_first=True)
        self.fc = nn.Linear(n_neurons, days_to_predict*features)
        self.days_to_predict = days_to_predict
        self.features = features

    def forward(self, x):
        """
        Forward pass through the network.
        
        Takes a batch of time series data, processes it through the LSTM,
        and outputs predictions for future time steps.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, sequence_length, features]
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, days_to_predict, features]
                        containing predictions for future time steps
        """
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the last output of the LSTM
        return out.view(-1, self.days_to_predict, self.features)

def save_best_model(model, model_path, metrics, save_whole_model=True):
    """
    Save model using joblib for complete model preservation.
    
    This function saves the model and its metrics. It can save either the complete model 
    (using joblib) or just the state_dict (using torch.save).
    
    Args:
        model (nn.Module): PyTorch model to save
        model_path (str): Path where the model should be saved
        metrics (dict): Dictionary containing model metrics like loss, accuracy, etc.
        save_whole_model (bool): Whether to save the complete model or just state_dict
    
    Returns:
        None
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save the model
    if save_whole_model:
        # Use joblib to save the entire model
        joblib_path = model_path.replace('.pth', '.joblib')
        joblib.dump(model, joblib_path)
        print(f"✅ Complete model saved to {joblib_path}")
    else:
        # Save only the state_dict
        torch.save(model.state_dict(), model_path)
        print(f"✅ Model state_dict saved to {model_path}")
    
    # Save the metrics alongside the model
    metrics_path = model_path.replace('.pth', '_metrics.json')
    
    # Add timestamp to metrics
    metrics['timestamp'] = datetime.now().isoformat()
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"✅ Model metrics saved to {metrics_path}")

def load_model(model_path, device=None, load_whole_model=True):
    """
    Load a saved model using either joblib or PyTorch.
    
    Args:
        model_path (str): Path to the saved model
        device (torch.device): Device to load the model to
        load_whole_model (bool): Whether to load complete model or state_dict
        
    Returns:
        tuple: (loaded_model, metrics_dict) if metrics file exists
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if load_whole_model:
        # Use joblib to load the entire model
        joblib_path = model_path.replace('.pth', '.joblib')
        if os.path.exists(joblib_path):
            model = joblib.load(joblib_path)
            model = model.to(device)
            print(f"✅ Complete model loaded from {joblib_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {joblib_path}")
    else:
        # Assuming this is the case where we need to instantiate model first
        raise ValueError("When load_whole_model=False, you must instantiate the model first")
      # Load the metrics if they exist
    metrics_path = model_path.replace('.pth', '_metrics.json')
    metrics = None
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            print(f"✅ Model metrics loaded from {metrics_path}")
    return model, metrics


def main():
    """
    Main function to execute the LSTM model training, evaluation, and CT analysis integration.
    
    This function handles the complete workflow:
    1. Dataset loading and preprocessing
    2. Model training with progress tracking
    3. Model evaluation and testing
    4. Prediction generation and visualization
    5. CT analysis integration for drive failure analysis
    """

    test_existing = False # Change to True to test existing model
    days_to_train = 3
    days_to_predict = 2
    look_back = days_to_train + days_to_predict
    path = "data/data_Q1_2025"
    verbose = True

    dataset_train = CustomDrives(root=path, 
                            train=True, 
                            input_len=days_to_train,
                            label_len=days_to_predict,
                            verbose=verbose)

    dataset_test = CustomDrives(root=path, 
                                train=False, 
                                input_len=days_to_train,
                                label_len=days_to_predict,
                                verbose=verbose)    

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
    num_epochs = 1

    # Move model to device
    model = model.to(device)
    
    train, labels = next(iter(train_loader))
    print(f'shape: {train.shape, labels.shape}')

    loss_curve = []
    minimum_loss = np.inf
      # Check if trained model exists
    model_path = 'models/LSTM/lstm_model.pth'
    joblib_path = model_path.replace('.pth', '.joblib')
    model_exists = os.path.exists(joblib_path) or os.path.exists(model_path)
    
    if model_exists and test_existing == True:
        try:
            # Try loading the full model with joblib first
            model, model_metrics = load_model(model_path, device, load_whole_model=True)
            if model_metrics:
                print(f"Previous best validation loss: {model_metrics.get('val_loss', 'N/A')}")
                minimum_loss = model_metrics.get('val_loss', minimum_loss)
            print("Trained model found! Loading and testing...")
        except (FileNotFoundError, ValueError) as e:
            # Fall back to loading state_dict if joblib file doesn't exist
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("Trained model state dict found! Loading and testing...")
        
        # Test the loaded model
        model.eval()
        test_loss = 0
        num_test_batches = 0
        
        print("Testing loaded model...")
        test_pbar = tqdm(test_loader, desc="Testing Model", leave=False)
        
        with torch.no_grad():
            for test_data, test_labels in test_pbar:
                test_data, test_labels = test_data.to(device), test_labels.to(device)
                predictions = model(test_data)
                loss = loss_function(predictions, test_labels)
                
                test_loss += loss.item()
                num_test_batches += 1
                
                test_pbar.set_postfix({'Test Loss': f'{loss.item():.6f}'})
        
        avg_test_loss = test_loss / num_test_batches if num_test_batches > 0 else 0
        print(f"Model Test Results - Average Loss: {avg_test_loss:.6f}")
        
        # Skip training and go directly to evaluation
        training_skipped = True
    else:
        print("No saved model found. Training from scratch...")
        training_skipped = False

    # Only train if no model exists
    if not training_skipped:        # Move model to device
        model = model.to(device)
        
        for epoch in tqdm(range(num_epochs), desc="Training Progress"):
            epoch_loss = 0
            num_batches = 0
            
            # Training loop with progress bar
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=False)
            for train_labels in train_pbar:
                model.zero_grad()
                train_data, labels = train_labels
                train_data, labels = train_data.to(device), labels.to(device)

                predictions = model(train_data)
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
                for test_data, test_labels in test_pbar:
                    test_data, test_labels = test_data.to(device), test_labels.to(device)
                    predictions = model(test_data)
                    loss = loss_function(predictions, test_labels)
                    
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
                
                # Create metrics dictionary with comprehensive information
                metrics = {
                    'val_loss': avg_val_loss,
                    'train_loss': avg_train_loss,
                    'epoch': epoch + 1,
                    'total_epochs': num_epochs,
                    'optimizer': optimizer.__class__.__name__,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'n_neurons': n_neurons,
                    'features': num_features,
                    'days_to_predict': days_to_predict,
                    'days_to_train': days_to_train,
                    'device': str(device)
                }
                
                # Save using our utility function
                save_best_model(model, model_path, metrics, save_whole_model=True)
                print(f"New best model saved with validation loss: {avg_val_loss:.6f}")
                
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
                plt.close()  # Close the figure to free memory    # Generate final plots and predictions
    if loss_curve:  # Only plot if we have training data
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        ax.plot(loss_curve, lw=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Loss")
        ax.set_title("Final Training Loss")
        
        # Save plots in model directory
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        plot_path = os.path.join(os.path.dirname(model_path), 'final_loss.png')
        plt.savefig(plot_path)
        print(f"✅ Final loss curve saved to {plot_path}")
        plt.close()

    # Generate predictions on test set
    model.eval()
    test_predictions = []
    test_targets = []
    
    print("Generating final predictions...")
    with torch.no_grad():
        for test_data, test_labels in tqdm(test_loader, desc="Generating predictions"):
            test_data, test_labels = test_data.to(device), test_labels.to(device)
            predictions = model(test_data)
            test_predictions.append(predictions.cpu().numpy())
            test_targets.append(test_labels.cpu().numpy())
    
    # Concatenate all predictions and targets
    test_predictions = np.concatenate(test_predictions, axis=0)
    test_targets = np.concatenate(test_targets, axis=0)
    
    # Plot predictions vs actual values for the first feature
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot first few samples for visualization
    num_samples_to_plot = min(50, len(test_predictions))
    x = np.arange(num_samples_to_plot)
    
    ax1.plot(x, test_predictions[:num_samples_to_plot, 0, 0], label='Predicted', alpha=0.7)
    ax1.plot(x, test_targets[:num_samples_to_plot, 0, 0], label='Actual', alpha=0.7)
    ax1.set_xlabel("Sample")
    ax1.set_ylabel("Feature 1 Value")
    ax1.set_title("Predictions vs Actual Values (Feature 1)")
    ax1.legend()
    
    # Plot prediction error
    error = test_predictions[:num_samples_to_plot, 0, 0] - test_targets[:num_samples_to_plot, 0, 0]
    ax2.plot(x, error, label='Prediction Error', color='red', alpha=0.7)
    ax2.set_xlabel("Sample")
    ax2.set_ylabel("Error")
    ax2.set_title("Prediction Error")
    ax2.legend()
    plt.tight_layout()
    
    # Save plots in model directory
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    predictions_path = os.path.join(os.path.dirname(model_path), 'test_predictions.png')
    plt.savefig(predictions_path)
    plt.show()
    
    # Save final predictions metrics
    final_metrics = {
        'test_mse': ((test_predictions - test_targets) ** 2).mean(),
        'test_mae': np.abs(test_predictions - test_targets).mean(),
        'test_samples': len(test_predictions),
        'timestamp': datetime.now().isoformat()
    }
      # Save prediction metrics
    predictions_metrics_path = os.path.join(os.path.dirname(model_path), 'prediction_metrics.json')
    with open(predictions_metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=4)
    print(f"✅ Final model evaluation completed. Results saved to {predictions_path}")
    print(f"✅ Prediction metrics saved to {predictions_metrics_path}")
    
    # Optional: Export predictions for CT analysis
    export_for_ct = True  # Set to True to generate CT-compatible features
    
    if export_for_ct and model_exists:
        print("\n🔗 Exporting LSTM predictions for CT analysis...")
        try:
            predictions_df, ct_features_df = export_for_ct_analysis(
                model_path=model_path,
                dataset_path=path,
                output_dir="ct_analysis"
            )
            
            if predictions_df is not None and ct_features_df is not None:
                print("✅ LSTM predictions exported successfully for CT analysis!")
                print(f"📁 Files saved in: ct_analysis/")
                print(f"📊 Generated {len(ct_features_df)} drive-level features")
                print("\n💡 To analyze these predictions with CT.py:")
                print("   1. Set use_lstm_predictions = True in CT.py main()")
                print("   2. Run CT.py to perform decision tree analysis on LSTM predictions")
            else:
                print("❌ Failed to export predictions for CT analysis")
                
        except Exception as e:
            print(f"❌ Error exporting for CT analysis: {e}")
    
    print("\n🎉 LSTM analysis pipeline completed!")

def generate_lstm_predictions(model_path, dataset_path, output_path="lstm_predictions.csv", device=None):
    """
    Generate LSTM predictions and save them in a format compatible with CT.py
    
    Args:
        model_path (str): Path to the saved LSTM model
        dataset_path (str): Path to the dataset for prediction
        output_path (str): Path to save the predictions CSV file
        device (torch.device): Device to run predictions on
        
    Returns:
        pd.DataFrame: DataFrame with LSTM predictions and metadata
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the trained model
    try:
        model, model_metrics = load_model(model_path, device, load_whole_model=True)
        print(f"✅ Model loaded successfully from {model_path}")
    except FileNotFoundError:
        print(f"❌ Model not found at {model_path}")
        return None
      # Load dataset for prediction
    days_to_train = model_metrics.get('days_to_train', 3) if model_metrics else 3
    days_to_predict = model_metrics.get('days_to_predict', 2) if model_metrics else 2
    
    dataset = CustomDrives(root=dataset_path, 
                          train=False, 
                          input_len=days_to_train,
                          label_len=days_to_predict,
                          verbose=True)
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Generate predictions
    model.eval()
    predictions_list = []
    
    print("Generating LSTM predictions...")
    with torch.no_grad():
        for idx, (input_data, target_data) in enumerate(tqdm(dataloader, desc="Making predictions")):
            input_data = input_data.to(device)
            
            # Get prediction from LSTM
            prediction = model(input_data)
            prediction_np = prediction.cpu().numpy()[0]  # Remove batch dimension
            
            # Get the actual drive data to extract metadata
            drive_data = dataset.dataset.data.get_group(dataset.dataset.list_of_keys[idx])
            
            # Create prediction entries for each predicted day
            for day_idx in range(days_to_predict):
                pred_entry = {
                    'serial_number': dataset.dataset.list_of_keys[idx],
                    'prediction_day': day_idx + 1,
                    'lstm_prediction_score': float(np.mean(prediction_np[day_idx])),  # Average across features
                    'lstm_max_feature': float(np.max(prediction_np[day_idx])),
                    'lstm_min_feature': float(np.min(prediction_np[day_idx])),
                    'lstm_std_feature': float(np.std(prediction_np[day_idx])),
                }
                
                # Add individual feature predictions
                feature_names = [f'smart_{i}_normalized' for i in [1, 3, 5, 7, 9, 187, 189, 190, 195, 197]] + ['smart_5_raw', 'smart_197_raw']
                for feat_idx, feat_name in enumerate(feature_names[:prediction_np.shape[1]]):
                    pred_entry[f'lstm_pred_{feat_name}'] = float(prediction_np[day_idx, feat_idx])
                
                predictions_list.append(pred_entry)
    
    # Convert to DataFrame and save
    predictions_df = pd.DataFrame(predictions_list)
    predictions_df.to_csv(output_path, index=False)
    print(f"✅ LSTM predictions saved to {output_path}")
    
    return predictions_df

def create_ct_compatible_features(lstm_predictions_df, threshold=0.5):
    """
    Convert LSTM predictions into features compatible with CT.py decision tree analysis
    
    Args:
        lstm_predictions_df (pd.DataFrame): DataFrame with LSTM predictions
        threshold (float): Threshold for converting predictions to binary features
        
    Returns:
        pd.DataFrame: DataFrame with features compatible with CT.py format
    """
    # Group by serial number to create drive-level features
    ct_features = []
    
    for serial_num, group in lstm_predictions_df.groupby('serial_number'):
        # Aggregate predictions across prediction days
        feature_dict = {
            'serial_number': serial_num,
            'lstm_avg_prediction': group['lstm_prediction_score'].mean(),
            'lstm_max_prediction': group['lstm_prediction_score'].max(),
            'lstm_min_prediction': group['lstm_prediction_score'].min(),
            'lstm_prediction_variance': group['lstm_prediction_score'].var(),
            'lstm_prediction_trend': group['lstm_prediction_score'].iloc[-1] - group['lstm_prediction_score'].iloc[0],
            'lstm_high_risk_days': (group['lstm_prediction_score'] > threshold).sum(),
            'lstm_anomaly_score': (group['lstm_std_feature'].mean()),
            'failure': 0  # Default to 0, will be updated with actual failure data if available
        }
        
        # Add aggregated SMART feature predictions
        smart_features = [col for col in group.columns if col.startswith('lstm_pred_smart_')]
        for feature in smart_features:
            feature_name = feature.replace('lstm_pred_', '')
            feature_dict[f'{feature_name}_lstm_avg'] = group[feature].mean()
            feature_dict[f'{feature_name}_lstm_max'] = group[feature].max()
            feature_dict[f'{feature_name}_lstm_trend'] = group[feature].iloc[-1] - group[feature].iloc[0]
        
        ct_features.append(feature_dict)
    
    ct_df = pd.DataFrame(ct_features)
    return ct_df

def export_for_ct_analysis(model_path, dataset_path, output_dir="ct_analysis", device=None):
    """
    Complete pipeline to export LSTM predictions for CT.py analysis
    
    Args:
        model_path (str): Path to the saved LSTM model
        dataset_path (str): Path to the dataset
        output_dir (str): Directory to save analysis files
        device (torch.device): Device to run predictions on
        
    Returns:
        tuple: (predictions_df, ct_features_df) DataFrames for analysis
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate LSTM predictions
    predictions_path = os.path.join(output_dir, "lstm_predictions.csv")
    predictions_df = generate_lstm_predictions(model_path, dataset_path, predictions_path, device)
    
    if predictions_df is None:
        return None, None
      # Create CT-compatible features
    ct_features_df = create_ct_compatible_features(predictions_df)
    ct_features_path = os.path.join(output_dir, "ct_features.csv")
    ct_features_df.to_csv(ct_features_path, index=False)
    
    print(f"✅ CT-compatible features saved to {ct_features_path}")
    print(f"📊 Generated {len(ct_features_df)} drive-level feature sets for CT analysis")
    
    return predictions_df, ct_features_df


if __name__ == '__main__':
    main()