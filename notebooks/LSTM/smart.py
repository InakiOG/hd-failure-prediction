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
import random
from itertools import product

def clean_data_smart(df, normalized_rows, raw_rows, columns_to_delete=['model','capacity_bytes','datacenter','cluster_id','vault_id','pod_id','pod_slot_num','is_legacy_format']):
    """
    Clean and preprocess hard drive data for SMART attribute analysis.
    
    Selects only specified SMART attributes based on the provided normalized_rows and raw_rows
    parameters. Keeps essential metadata columns (date, serial_number, failure) and removes
    unnecessary columns to optimize memory usage and model performance.
    
    Args:
        df (pd.DataFrame): Raw hard drive data DataFrame with all SMART attributes
        normalized_rows (list): List of SMART attribute IDs to include (normalized values)
        raw_rows (list): List of SMART attribute IDs to include (raw values)
        columns_to_delete (list): List of column names to remove from the dataset
        
    Returns:
        pd.DataFrame: Cleaned DataFrame with only selected SMART attributes and metadata
    """
    df.head()
    smart_allowed = ['date', 'serial_number', 'failure']
    for i in normalized_rows: 
        smart_allowed.append(f'smart_{i}_normalized')
    for i in raw_rows:
        smart_allowed.append(f'smart_{i}_raw')

    if len(normalized_rows) > 0 or len(raw_rows) > 0:
        for column in df.columns:
            if column not in smart_allowed:
                columns_to_delete.append(column)

    # Remove columns from columns_to_delete if they are not in the DataFrame
    columns_to_delete = [col for col in columns_to_delete if col in df.columns]
    df = df.drop(columns=columns_to_delete)
    df = df.fillna(0)
    float_columns   = df.select_dtypes(include=['float64']).columns
    df[float_columns] = df[float_columns].astype(int)
    

    return df

class DriveDataLoader:
    """
    Centralized data loader that handles train/test split at the drive level to prevent data leakage.
    
    This class loads all CSV files from the specified directory, applies data cleaning using
    the clean_data_smart function, and creates separate train/test splits based on drive 
    serial numbers. This ensures that no drive appears in both training and testing sets,
    preventing temporal data leakage that could artificially inflate model performance.
    
    The class supports both loading all available drives or selecting a specific number
    of drives for processing, with configurable minimum sequence length requirements.
    """
    
    def __init__(self, root: str, train_ratio: float = 0.8, min_sequence_length: int = 5, verbose: bool = False, num_drives = -1, normalized_rows = [], raw_rows = [], dtype_dict = {'date': 'str', 'serial_number': 'str', 'model': 'str', 'capacity_bytes': 'int32', 'failure': 'bool', 'datacenter': 'str', 'cluster_id': 'int8', 'vault_id': 'int16', 'pod_id': 'int16', 'pod_slot_num': 'float32', 'is_legacy_format': 'bool', 'smart_1_normalized': 'float64', 'smart_1_raw': 'float64', 'smart_2_normalized': 'float64', 'smart_2_raw': 'float64', 'smart_3_normalized': 'float64', 'smart_3_raw': 'float64', 'smart_4_normalized': 'float64', 'smart_4_raw': 'float64', 'smart_5_normalized': 'float64', 'smart_5_raw': 'float64', 'smart_7_normalized': 'float64', 'smart_7_raw': 'float64', 'smart_8_normalized': 'float64', 'smart_8_raw': 'float64', 'smart_9_normalized': 'float64', 'smart_9_raw': 'float64', 'smart_10_normalized': 'float64', 'smart_10_raw': 'float64', 'smart_11_normalized': 'float64', 'smart_11_raw': 'float64', 'smart_12_normalized': 'float64', 'smart_12_raw': 'float64', 'smart_13_normalized': 'float64', 'smart_13_raw': 'float64', 'smart_15_normalized': 'float64', 'smart_15_raw': 'float64', 'smart_16_normalized': 'float64', 'smart_16_raw': 'float64', 'smart_17_normalized': 'float64', 'smart_17_raw': 'float64', 'smart_18_normalized': 'float64', 'smart_18_raw': 'float64', 'smart_22_normalized': 'float64', 'smart_22_raw': 'float64', 'smart_23_normalized': 'float64', 'smart_23_raw': 'float64', 'smart_24_normalized': 'float64', 'smart_24_raw': 'float64', 'smart_27_normalized': 'float64', 'smart_27_raw': 'float64', 'smart_71_normalized': 'float64', 'smart_71_raw': 'float64', 'smart_82_normalized': 'float64', 'smart_82_raw': 'float64', 'smart_90_normalized': 'float64', 'smart_90_raw': 'float64', 'smart_160_normalized': 'float64', 'smart_160_raw': 'float64', 'smart_161_normalized': 'float64', 'smart_161_raw': 'float64', 'smart_163_normalized': 'float64', 'smart_163_raw': 'float64', 'smart_164_normalized': 'float64', 'smart_164_raw': 'float64', 'smart_165_normalized': 'float64', 'smart_165_raw': 'float64', 'smart_166_normalized': 'float64', 'smart_166_raw': 'float64', 'smart_167_normalized': 'float64', 'smart_167_raw': 'float64', 'smart_168_normalized': 'float64', 'smart_168_raw': 'float64', 'smart_169_normalized': 'float64', 'smart_169_raw': 'float64', 'smart_170_normalized': 'float64', 'smart_170_raw': 'float64', 'smart_171_normalized': 'float64', 'smart_171_raw': 'float64', 'smart_172_normalized': 'float64', 'smart_172_raw': 'float64', 'smart_173_normalized': 'float64', 'smart_173_raw': 'float64', 'smart_174_normalized': 'float64', 'smart_174_raw': 'float64', 'smart_175_normalized': 'float64', 'smart_175_raw': 'float64', 'smart_176_normalized': 'float64', 'smart_176_raw': 'float64', 'smart_177_normalized': 'float64', 'smart_177_raw': 'float64', 'smart_178_normalized': 'float64', 'smart_178_raw': 'float64', 'smart_179_normalized': 'float64', 'smart_179_raw': 'float64', 'smart_180_normalized': 'float64', 'smart_180_raw': 'float64', 'smart_181_normalized': 'float64', 'smart_181_raw': 'float64', 'smart_182_normalized': 'float64', 'smart_182_raw': 'float64', 'smart_183_normalized': 'float64', 'smart_183_raw': 'float64', 'smart_184_normalized': 'float64', 'smart_184_raw': 'float64', 'smart_187_normalized': 'float64', 'smart_187_raw': 'float64', 'smart_188_normalized': 'float64', 'smart_188_raw': 'float64', 'smart_189_normalized': 'float64', 'smart_189_raw': 'float64', 'smart_190_normalized': 'float64', 'smart_190_raw': 'float64', 'smart_191_normalized': 'float64', 'smart_191_raw': 'float64', 'smart_192_normalized': 'float64', 'smart_192_raw': 'float64', 'smart_193_normalized': 'float64', 'smart_193_raw': 'float64', 'smart_194_normalized': 'float64', 'smart_194_raw': 'float64', 'smart_195_normalized': 'float64', 'smart_195_raw': 'float64', 'smart_196_normalized': 'float64', 'smart_196_raw': 'float64', 'smart_197_normalized': 'float64', 'smart_197_raw': 'float64', 'smart_198_normalized': 'float64', 'smart_198_raw': 'float64', 'smart_199_normalized': 'float64', 'smart_199_raw': 'float64', 'smart_200_normalized': 'float64', 'smart_200_raw': 'float64', 'smart_201_normalized': 'float64', 'smart_201_raw': 'float64', 'smart_202_normalized': 'float64', 'smart_202_raw': 'float64', 'smart_206_normalized': 'float64', 'smart_206_raw': 'float64', 'smart_210_normalized': 'float64', 'smart_210_raw': 'float64', 'smart_218_normalized': 'float64', 'smart_218_raw': 'float64', 'smart_220_normalized': 'float64', 'smart_220_raw': 'float64', 'smart_222_normalized': 'float64', 'smart_222_raw': 'float64', 'smart_223_normalized': 'float64', 'smart_223_raw': 'float64', 'smart_224_normalized': 'float64', 'smart_224_raw': 'float64', 'smart_225_normalized': 'float64', 'smart_225_raw': 'float64', 'smart_226_normalized': 'float64', 'smart_226_raw': 'float64', 'smart_230_normalized': 'float64', 'smart_230_raw': 'float64', 'smart_231_normalized': 'float64', 'smart_231_raw': 'float64', 'smart_232_normalized': 'float64', 'smart_232_raw': 'float64', 'smart_233_normalized': 'float64', 'smart_233_raw': 'float64', 'smart_234_normalized': 'float64', 'smart_234_raw': 'float64', 'smart_235_normalized': 'float64', 'smart_235_raw': 'float64', 'smart_240_normalized': 'float64', 'smart_240_raw': 'float64', 'smart_241_normalized': 'float64', 'smart_241_raw': 'float64', 'smart_242_normalized': 'float64', 'smart_242_raw': 'float64', 'smart_244_normalized': 'float64', 'smart_244_raw': 'float64', 'smart_245_normalized': 'float64', 'smart_245_raw': 'float64', 'smart_246_normalized': 'float64', 'smart_246_raw': 'float64', 'smart_247_normalized': 'float64', 'smart_247_raw': 'float64', 'smart_248_normalized': 'float64', 'smart_248_raw': 'float64', 'smart_250_normalized': 'float64', 'smart_250_raw': 'float64', 'smart_251_normalized': 'float64', 'smart_251_raw': 'float64', 'smart_252_normalized': 'float64', 'smart_252_raw': 'float64', 'smart_254_normalized': 'float64', 'smart_254_raw': 'float64', 'smart_255_normalized': 'float64', 'smart_255_raw': 'float64'}, columns_to_delete = ['model','capacity_bytes','datacenter','cluster_id','vault_id','pod_id','pod_slot_num','is_legacy_format']):
        """
        Initialize the data loader and perform the train/test split.
        
        Args:
            root (str): Root directory containing CSV files with hard drive data
            train_ratio (float): Ratio of drives to use for training (0.8 = 80% train, 20% test)
            min_sequence_length (int): Minimum number of days required per drive to be included
            verbose (bool): Whether to print detailed information during loading process
            num_drives (int): Number of drives to randomly select (-1 for all drives)
            normalized_rows (list): List of SMART attribute IDs to include (normalized values)
            raw_rows (list): List of SMART attribute IDs to include (raw values)
            dtype_dict (dict): Data type specifications for efficient memory usage during CSV reading
            columns_to_delete (list): List of column names to remove during preprocessing
        """
        self.dataset_path = root
        self.train_ratio = train_ratio
        self.min_sequence_length = min_sequence_length
        self.verbose = verbose
        self.num_drives = num_drives
        self.normalized_rows = normalized_rows
        self.raw_rows = raw_rows
        self.dtype_dict = dtype_dict
        self.columns_to_delete = columns_to_delete
        
        if not os.path.isdir(self.dataset_path):
            msg = f'Could not find the csv files in {self.dataset_path}.'
            raise FileNotFoundError(msg)
        
        if num_drives > 0:
            self._get_drives(num_drives, min_sequence_length)
        else:
            # Load all data
            self._load_all_data()
            # Perform train/test split at drive level
            self._split_drives(min_sequence_length)
    
    def _get_drives(self, num_drives, min_sequence_length):
        """
        Select a specific number of drives that meet the minimum sequence length requirement.
        
        This method filters the loaded data to include only drives with sufficient data points,
        then randomly samples the requested number of drives for processing.
        
        Args:
            num_drives (int): Number of drives to randomly select
            min_sequence_length (int): Minimum number of days required per drive
            
        Raises:
            ValueError: If fewer drives meet the minimum sequence length than requested
        """
        if self.verbose:
            print(f'[DriveDataLoader] Loading drives with at least {min_sequence_length} days of data...')
        self._load_all_data()
        grouped = self.all_data.groupby('serial_number')
        valid_drives = [k for k, v in grouped.size().items() if v >= min_sequence_length]
        if len(valid_drives) < num_drives:
            raise ValueError(f"Requested {num_drives} drives, but only {len(valid_drives)} drives meet the minimum sequence length of {min_sequence_length}.")
        selected_drives = random.sample(valid_drives, num_drives)
        self.all_data = self.all_data[self.all_data['serial_number'].isin(selected_drives)]
        if self.verbose:
            print(f'[DriveDataLoader] Selected {len(selected_drives)} drives for processing.')
            print(f'[DriveDataLoader] Total rows in selected drives: {len(self.all_data)}')

    def _load_all_data(self):
        """
        Load all CSV files from the specified directory and concatenate them into a single DataFrame.
        
        This method supports both flat directory structures (CSV files directly in root) and
        nested structures (CSV files in subdirectories). It applies the clean_data_smart
        preprocessing function to each file before concatenation, and sorts the final
        dataset by serial_number and date for consistent ordering.
        
        The method uses the configured dtype_dict for efficient memory usage during CSV reading
        and applies the specified normalization parameters and column deletion rules.
        """
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
            df = clean_data_smart(pd.read_csv(file_name, dtype=self.dtype_dict), self.normalized_rows, self.raw_rows, self.columns_to_delete)
            data.append(df)
            if self.verbose:
                print(f'Loaded {file_name} with shape {df.shape}')
        print(f'[DriveDataLoader] Columns in data: {data[0].columns.tolist()}')

        self.all_data = pd.concat(data)
          # Sort to make consistent
        self.all_data.sort_values(by=['serial_number', 'date'], ascending=[True, True], inplace=True)
        
        if self.verbose:
            print(f'[DriveDataLoader] Loaded {len(self.all_data)} rows from all CSV files. DataFrame shape: {self.all_data.shape}')
    
    def _split_drives(self, min_sequence_length: int = 5):
        """
        Filter drives by minimum sequence length and split into train and test sets.
        
        This method performs drive-level splitting to prevent data leakage by ensuring
        that no drive appears in both training and testing sets. It first filters out
        drives that don't meet the minimum sequence length requirement, then randomly
        shuffles the remaining valid drives and splits them according to the configured
        train_ratio.
        
        The method provides detailed logging of the splitting process, including statistics
        about valid/invalid drives, sample drive information, and verification that no
        overlap exists between train and test sets.
        
        Args:
            min_sequence_length (int): Minimum number of days required per drive to be included
            
        Raises:
            ValueError: If no drives meet the minimum sequence length requirement
        """
        # Group by drive and check sequence lengths
        grouped = self.all_data.groupby('serial_number')
        group_sizes = grouped.size()
        
        if self.verbose:
            print(f'[DriveDataLoader] Found {len(grouped.groups)} unique drives.')
            print(f'[DriveDataLoader] Filtering drives with minimum {min_sequence_length} days...')
        
        # Filter drives that meet minimum sequence length requirement
        valid_drives = group_sizes[group_sizes >= min_sequence_length].index.tolist()
        invalid_drives = group_sizes[group_sizes < min_sequence_length]
        
        if self.verbose:
            print(f'[DriveDataLoader] Valid drives (>= {min_sequence_length} days): {len(valid_drives)}')
            print(f'[DriveDataLoader] Invalid drives (< {min_sequence_length} days): {len(invalid_drives)}')
            if len(invalid_drives) > 0:
                print(f'[DriveDataLoader] Sequence length distribution of invalid drives:')
                print(f'[DriveDataLoader]   {dict(invalid_drives.value_counts().sort_index())}')
        
        if len(valid_drives) == 0:
            raise ValueError(f'No drives found with minimum sequence length of {min_sequence_length} days. '
                           f'Please check your data or reduce the minimum sequence length requirement.')
        
        # Update the dataset to only include valid drives
        self.all_data = self.all_data[self.all_data['serial_number'].isin(valid_drives)]
        
        # Shuffle valid drives to randomize the split (with fixed seed for reproducibility)
        np.random.seed(42)
        np.random.shuffle(valid_drives)
          # Calculate split point
        split_point = int(len(valid_drives) * self.train_ratio)
        
        # Split drives
        self.train_drives = valid_drives[:split_point]
        self.test_drives = valid_drives[split_point:]
        
        if self.verbose:
            print(f'[DriveDataLoader] Split: {len(self.train_drives)} drives for training, {len(self.test_drives)} drives for testing.')
            print(f'[DriveDataLoader] Train ratio: {len(self.train_drives) / len(valid_drives):.2%}')
            print(f'[DriveDataLoader] Final dataset: {len(self.all_data)} rows from {len(valid_drives)} valid drives.')
            
            # Show sample drives from train and test sets to verify date ranges
            if len(self.train_drives) > 0:
                sample_train_drive = self.train_drives[0]
                train_drive_data = self.all_data[self.all_data['serial_number'] == sample_train_drive]
                train_dates = pd.to_datetime(train_drive_data['date'])
                print(f'[DriveDataLoader] Sample TRAIN drive {sample_train_drive}: {len(train_drive_data)} rows, dates {train_dates.min().date()} to {train_dates.max().date()}')
            
            if len(self.test_drives) > 0:
                sample_test_drive = self.test_drives[0]
                test_drive_data = self.all_data[self.all_data['serial_number'] == sample_test_drive]
                test_dates = pd.to_datetime(test_drive_data['date'])
                print(f'[DriveDataLoader] Sample TEST drive {sample_test_drive}: {len(test_drive_data)} rows, dates {test_dates.min().date()} to {test_dates.max().date()}')
            
            # Verify no overlap between train and test drives
            overlap = set(self.train_drives) & set(self.test_drives)
            if len(overlap) == 0:
                print(f'[DriveDataLoader] ✅ Verified: No drive overlap between train and test sets.')
            else:
                print(f'[DriveDataLoader] ⚠️  WARNING: Found {len(overlap)} drives in both train and test sets!')
            
            # Show sample drive groups to verify correct date ranges and separation
            print(f'\n[DriveDataLoader] 📊 Sample drive verification:')
            
            # Show one drive from training set
            if len(self.train_drives) > 0:
                sample_train_drive = self.train_drives[0]
                train_data_sample = self.all_data[self.all_data['serial_number'] == sample_train_drive]
                train_dates = sorted(train_data_sample['date'].unique())
                print(f'[DriveDataLoader] 🚂 TRAIN sample - Drive {sample_train_drive}:')
                print(f'[DriveDataLoader]   📅 Dates: {train_dates[0]} to {train_dates[-1]} ({len(train_dates)} days)')
                print(f'[DriveDataLoader]   📊 Rows: {len(train_data_sample)}')
            
            # Show one drive from test set
            if len(self.test_drives) > 0:
                sample_test_drive = self.test_drives[0]
                test_data_sample = self.all_data[self.all_data['serial_number'] == sample_test_drive]
                test_dates = sorted(test_data_sample['date'].unique())
                print(f'[DriveDataLoader] 🧪 TEST sample - Drive {sample_test_drive}:')
                print(f'[DriveDataLoader]   📅 Dates: {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} days)')
                print(f'[DriveDataLoader]   📊 Rows: {len(test_data_sample)}')
            
            # Verify no overlap between train and test drives
            train_set = set(self.train_drives)
            test_set = set(self.test_drives)
            overlap = train_set.intersection(test_set)
            if len(overlap) == 0:
                print(f'[DriveDataLoader] ✅ Verified: No drive overlap between train and test sets')
            else:
                print(f'[DriveDataLoader] ❌ WARNING: {len(overlap)} drives found in both train and test sets!')
    
    def get_train_data(self):
        """
        Get data for training drives only.
        
        Returns:
            pd.DataFrame: DataFrame containing only data from drives assigned to the training set
        """
        return self.all_data[self.all_data['serial_number'].isin(self.train_drives)]
    
    def get_test_data(self):
        """
        Get data for testing drives only.
        
        Returns:
            pd.DataFrame: DataFrame containing only data from drives assigned to the testing set
        """
        return self.all_data[self.all_data['serial_number'].isin(self.test_drives)]
    
    def get_all_data(self):
        """
        Get all loaded data regardless of train/test split.
        
        Returns:
            pd.DataFrame: Complete DataFrame containing all loaded and preprocessed drive data
        """
        return self.all_data

class _CustomDrives(Dataset):
    """
    Internal dataset class for loading and preprocessing hard drive SMART data for time series analysis.
    
    This class handles preprocessing of pre-split data, organizing the data by drive serial number,
    and preparing time-ordered sequences for training or testing. It creates sliding windows from
    each drive's time series data, splitting them into input sequences (past observations) and 
    target sequences (future values to predict).
    
    The class processes data that has already been filtered by DriveDataLoader to ensure drives
    meet minimum sequence length requirements and are properly assigned to train/test sets.
    
    Note: This is an internal implementation class used by CustomDrives and should not be
    instantiated directly.
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 input_len: int = 3,
                 label_len: int = 1,
                 verbose = False):
        """
        Initialize the dataset with pre-split data for time series prediction.
        
        This constructor processes pre-filtered data from DriveDataLoader and organizes it
        by drive serial number. Each drive's time series is prepared for creating input-target
        sequence pairs based on the specified window sizes.
        
        Args:
            data (pd.DataFrame): Pre-split DataFrame containing SMART data for either training or testing
            input_len (int): Number of time steps to use as input (look-back window)
            label_len (int): Number of time steps to predict (forecast window)
            verbose (bool): Whether to print detailed information during data processing
        """
        super().__init__()
          # Input len is the past window of time and label len is the future window 
        self.input_len = input_len
        self.label_len = label_len
        
        if verbose: 
            print(f'[_CustomDrives] Processing {len(data)} rows. DataFrame shape: {data.shape}')

        # Group by serial number - data is already filtered for valid drives
        self.data = data.groupby('serial_number')
        
        if verbose: 
            print(f'[_CustomDrives] Processing {len(self.data.groups)} drives (all pre-filtered for sufficient data).')
        
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
        
        This method retrieves the time series data for a drive at the specified index
        and splits it into input (historical) and target (future) sequences based on
        the configured window sizes. The input sequence represents past SMART attribute
        values, while the target sequence represents future values to be predicted.
        
        Args:
            idx (int): Index of the drive to retrieve
            
        Returns:
            tuple: (train, label) where:
                - train: DataFrame containing past SMART data (input_len time steps)
                - label: DataFrame containing future SMART data (label_len time steps)
        """
        values =  self.data.get_group(self.list_of_keys[idx])
        
        # We can return the drives that have n or more days
        train = values.iloc[:self.input_len]
        label = values.iloc[self.input_len:self.input_len + self.label_len]
        return train, label
    
class CustomDrives(Dataset):
    """
    PyTorch Dataset class for hard drive SMART data time series forecasting.
    
    This class serves as the main interface for loading hard drive data into PyTorch models.
    It wraps the internal _CustomDrives implementation and handles the conversion from
    pandas DataFrames to PyTorch tensors. The class supports both training and testing
    modes, working with pre-split data from DriveDataLoader to prevent data leakage.
    
    The dataset removes metadata columns (serial_number, date, failure) and converts
    numerical SMART attributes to PyTorch tensors with double precision for model training.
    It maintains the temporal structure of the data while providing efficient batch loading
    through PyTorch's DataLoader interface.
    """
    
    def __init__(self, 
                 data_loader: Optional[DriveDataLoader] = None,
                 root: Optional[str] = None,
                 train: bool = False,
                 input_len: int = 3,
                 label_len: int = 1,
                 verbose = False):
        """
        Initialize the dataset with configuration for time series prediction.
        
        This constructor sets up the dataset for either training or testing by obtaining
        the appropriate data split from the provided DriveDataLoader. It supports both
        new-style initialization (with data_loader) and legacy initialization (with root path).
        
        Args:
            data_loader (DriveDataLoader, optional): Pre-initialized data loader with train/test split
            root (str, optional): Root directory containing CSV files (used only if data_loader is None)
            train (bool): Whether this dataset is for training (True) or testing (False)
            input_len (int): Number of time steps to use as input (look-back window)
            label_len (int): Number of time steps to predict (forecast window)
            verbose (bool): Whether to print detailed information during initialization
            
        Raises:
            ValueError: If neither data_loader nor root is provided
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
        
        This method retrieves a drive sequence pair from the internal dataset, removes
        metadata columns that are not needed for model training (serial_number, date, failure),
        and converts the resulting numerical data to PyTorch tensors with double precision.
        
        Args:
            idx (int): Index of the drive sequence to retrieve
            
        Returns:
            tuple: (train_tensor, label_tensor) where:
                - train_tensor: PyTorch tensor of shape [input_len, num_features] containing historical SMART data
                - label_tensor: PyTorch tensor of shape [label_len, num_features] containing future SMART data to predict
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
    
    This model uses a Long Short-Term Memory (LSTM) network to learn temporal dependencies
    in hard drive SMART attributes, followed by a fully connected layer to predict future
    values. The architecture is designed to process sequences of SMART data and predict
    multiple future time steps for all features simultaneously.
    
    The model takes input sequences of shape [batch_size, sequence_length, num_features]
    and outputs predictions of shape [batch_size, days_to_predict, num_features], where
    each prediction represents the expected SMART attribute values for future time steps.
    """
    def __init__(self, n_neurons, features, days_to_predict=2):
        """
        Initialize the LSTM-based neural network architecture.
        
        Creates a single-layer LSTM followed by a fully connected layer for time series
        prediction. The model is configured to process multiple SMART attributes
        simultaneously and predict their future values.
        
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
        Forward pass through the LSTM network.
        
        Processes a batch of time series data through the LSTM layer and generates
        predictions for future time steps. The method uses the last hidden state
        from the LSTM to produce predictions for all future time steps and features.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, sequence_length, features]
                            containing historical SMART attribute values
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, days_to_predict, features]
                        containing predictions for future SMART attribute values
        """
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the last output of the LSTM
        return out.view(-1, self.days_to_predict, self.features)

def save_model(model, model_path, metrics, save_whole_model=True):
    """
    Save trained model and associated metrics to disk.
    
    This function provides comprehensive model persistence using either joblib for complete
    model serialization or PyTorch's native format for state dictionaries. It also saves
    training metrics alongside the model for tracking performance and configuration.
    
    Args:
        model (nn.Module): Trained PyTorch model to save
        model_path (str): Path where the model should be saved (with .pth extension)
        metrics (dict): Dictionary containing model metrics (loss, accuracy, hyperparameters, etc.)
        save_whole_model (bool): If True, save complete model using joblib; if False, save only state_dict
    
    Side Effects:
        - Creates directory structure if it doesn't exist
        - Saves model file (.joblib or .pth)
        - Saves metrics file (*_metrics.json)
        - Prints confirmation messages
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
    Load a previously saved model and its associated metrics from disk.
    
    This function supports loading models saved in either joblib format (complete model)
    or PyTorch state dictionary format. It automatically detects the appropriate device
    and loads associated metrics if available.
    
    Args:
        model_path (str): Path to the saved model file
        device (torch.device, optional): Device to load the model to. If None, auto-detects GPU/CPU
        load_whole_model (bool): If True, load complete model from joblib; if False, expects state_dict loading
        
    Returns:
        tuple: (loaded_model, metrics_dict) where:
            - loaded_model: The loaded PyTorch model ready for inference
            - metrics_dict: Dictionary containing saved metrics, or None if no metrics file exists
            
    Raises:
        FileNotFoundError: If the specified model file doesn't exist
        ValueError: If load_whole_model=False (requires pre-instantiated model)
    """
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
        # Assuming this is the case where we need to instantiate the model first
        raise ValueError("When load_whole_model=False, you must instantiate the model first")
      # Load the metrics if they exist
    metrics_path = model_path.replace('.pth', '_metrics.json')
    metrics = None
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            print(f"✅ Model metrics loaded from {metrics_path}")
    return model, metrics

def test_lstm_model(model, test_loader, device=None, loss_function=None, verbose=True):
    """
    Evaluate a trained LSTM model on test data and compute comprehensive metrics.
    
    This function performs model evaluation in inference mode, computing predictions
    on the test dataset and calculating loss metrics. It processes the entire test
    set and returns detailed results for analysis.
    
    Args:
        model (nn.Module): Trained LSTM model to evaluate
        test_loader (DataLoader): PyTorch DataLoader containing test data
        device (torch.device, optional): Device to run evaluation on. Auto-detects if None
        loss_function (callable, optional): Loss function to use. Defaults to nn.MSELoss()
        verbose (bool): Whether to print detailed progress information during evaluation
        
    Returns:
        dict: Comprehensive evaluation results containing:
            - 'predictions': numpy array of model predictions for all test samples
            - 'targets': numpy array of actual target values for all test samples  
            - 'metrics': dict with evaluation metrics including average loss
    """
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if loss_function is None:
        loss_function = nn.MSELoss()
    
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for test_data, test_labels in tqdm(test_loader, desc="Testing Model", leave=False):
            test_data, test_labels = test_data.to(device), test_labels.to(device)
            predictions = model(test_data)
            loss = loss_function(predictions, test_labels)
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(test_labels.cpu().numpy())
            
            total_loss += loss.item()
            num_batches += 1
            
            if verbose:
                tqdm.write(f"Batch Loss: {loss.item():.6f}")
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate additional metrics if needed
    metrics = {
        'average_loss': avg_loss,
        # Add more metrics here if needed
    }
    
    return {
        'predictions': all_predictions,
        'targets': all_targets,
        'metrics': metrics
    }

def load_data(root: str, 
              train_ratio: float = 0.8, 
              min_sequence_length: int = 5,
              input_len: int = 3,
              label_len: int = 1,
              normalized_rows: list = [],
              raw_rows: list = [],
              verbose: bool = False,
              batch_size: int = 3,
              dtype_dict: Optional[dict] = {'date': 'str', 'serial_number': 'str', 'model': 'str', 'capacity_bytes': 'int32', 'failure': 'bool', 'datacenter': 'str', 'cluster_id': 'int8', 'vault_id': 'int16', 'pod_id': 'int16', 'pod_slot_num': 'float32', 'is_legacy_format': 'bool', 'smart_1_normalized': 'float64', 'smart_1_raw': 'float64', 'smart_2_normalized': 'float64', 'smart_2_raw': 'float64', 'smart_3_normalized': 'float64', 'smart_3_raw': 'float64', 'smart_4_normalized': 'float64', 'smart_4_raw': 'float64', 'smart_5_normalized': 'float64', 'smart_5_raw': 'float64', 'smart_7_normalized': 'float64', 'smart_7_raw': 'float64', 'smart_8_normalized': 'float64', 'smart_8_raw': 'float64', 'smart_9_normalized': 'float64', 'smart_9_raw': 'float64', 'smart_10_normalized': 'float64', 'smart_10_raw': 'float64', 'smart_11_normalized': 'float64', 'smart_11_raw': 'float64', 'smart_12_normalized': 'float64', 'smart_12_raw': 'float64', 'smart_13_normalized': 'float64', 'smart_13_raw': 'float64', 'smart_15_normalized': 'float64', 'smart_15_raw': 'float64', 'smart_16_normalized': 'float64', 'smart_16_raw': 'float64', 'smart_17_normalized': 'float64', 'smart_17_raw': 'float64', 'smart_18_normalized': 'float64', 'smart_18_raw': 'float64', 'smart_22_normalized': 'float64', 'smart_22_raw': 'float64', 'smart_23_normalized': 'float64', 'smart_23_raw': 'float64', 'smart_24_normalized': 'float64', 'smart_24_raw': 'float64', 'smart_27_normalized': 'float64', 'smart_27_raw': 'float64', 'smart_71_normalized': 'float64', 'smart_71_raw': 'float64', 'smart_82_normalized': 'float64', 'smart_82_raw': 'float64', 'smart_90_normalized': 'float64', 'smart_90_raw': 'float64', 'smart_160_normalized': 'float64', 'smart_160_raw': 'float64', 'smart_161_normalized': 'float64', 'smart_161_raw': 'float64', 'smart_163_normalized': 'float64', 'smart_163_raw': 'float64', 'smart_164_normalized': 'float64', 'smart_164_raw': 'float64', 'smart_165_normalized': 'float64', 'smart_165_raw': 'float64', 'smart_166_normalized': 'float64', 'smart_166_raw': 'float64', 'smart_167_normalized': 'float64', 'smart_167_raw': 'float64', 'smart_168_normalized': 'float64', 'smart_168_raw': 'float64', 'smart_169_normalized': 'float64', 'smart_169_raw': 'float64', 'smart_170_normalized': 'float64', 'smart_170_raw': 'float64', 'smart_171_normalized': 'float64', 'smart_171_raw': 'float64', 'smart_172_normalized': 'float64', 'smart_172_raw': 'float64', 'smart_173_normalized': 'float64', 'smart_173_raw': 'float64', 'smart_174_normalized': 'float64', 'smart_174_raw': 'float64', 'smart_175_normalized': 'float64', 'smart_175_raw': 'float64', 'smart_176_normalized': 'float64', 'smart_176_raw': 'float64', 'smart_177_normalized': 'float64', 'smart_177_raw': 'float64', 'smart_178_normalized': 'float64', 'smart_178_raw': 'float64', 'smart_179_normalized': 'float64', 'smart_179_raw': 'float64', 'smart_180_normalized': 'float64', 'smart_180_raw': 'float64', 'smart_181_normalized': 'float64', 'smart_181_raw': 'float64', 'smart_182_normalized': 'float64', 'smart_182_raw': 'float64', 'smart_183_normalized': 'float64', 'smart_183_raw': 'float64', 'smart_184_normalized': 'float64', 'smart_184_raw': 'float64', 'smart_187_normalized': 'float64', 'smart_187_raw': 'float64', 'smart_188_normalized': 'float64', 'smart_188_raw': 'float64', 'smart_189_normalized': 'float64', 'smart_189_raw': 'float64', 'smart_190_normalized': 'float64', 'smart_190_raw': 'float64', 'smart_191_normalized': 'float64', 'smart_191_raw': 'float64', 'smart_192_normalized': 'float64', 'smart_192_raw': 'float64', 'smart_193_normalized': 'float64', 'smart_193_raw': 'float64', 'smart_194_normalized': 'float64', 'smart_194_raw': 'float64', 'smart_195_normalized': 'float64', 'smart_195_raw': 'float64', 'smart_196_normalized': 'float64', 'smart_196_raw': 'float64', 'smart_197_normalized': 'float64', 'smart_197_raw': 'float64', 'smart_198_normalized': 'float64', 'smart_198_raw': 'float64', 'smart_199_normalized': 'float64', 'smart_199_raw': 'float64', 'smart_200_normalized': 'float64', 'smart_200_raw': 'float64', 'smart_201_normalized': 'float64', 'smart_201_raw': 'float64', 'smart_202_normalized': 'float64', 'smart_202_raw': 'float64', 'smart_206_normalized': 'float64', 'smart_206_raw': 'float64', 'smart_210_normalized': 'float64', 'smart_210_raw': 'float64', 'smart_218_normalized': 'float64', 'smart_218_raw': 'float64', 'smart_220_normalized': 'float64', 'smart_220_raw': 'float64', 'smart_222_normalized': 'float64', 'smart_222_raw': 'float64', 'smart_223_normalized': 'float64', 'smart_223_raw': 'float64', 'smart_224_normalized': 'float64', 'smart_224_raw': 'float64', 'smart_225_normalized': 'float64', 'smart_225_raw': 'float64', 'smart_226_normalized': 'float64', 'smart_226_raw': 'float64', 'smart_230_normalized': 'float64', 'smart_230_raw': 'float64', 'smart_231_normalized': 'float64', 'smart_231_raw': 'float64', 'smart_232_normalized': 'float64', 'smart_232_raw': 'float64', 'smart_233_normalized': 'float64', 'smart_233_raw': 'float64', 'smart_234_normalized': 'float64', 'smart_234_raw': 'float64', 'smart_235_normalized': 'float64', 'smart_235_raw': 'float64', 'smart_240_normalized': 'float64', 'smart_240_raw': 'float64', 'smart_241_normalized': 'float64', 'smart_241_raw': 'float64', 'smart_242_normalized': 'float64', 'smart_242_raw': 'float64', 'smart_244_normalized': 'float64', 'smart_244_raw': 'float64', 'smart_245_normalized': 'float64', 'smart_245_raw': 'float64', 'smart_246_normalized': 'float64', 'smart_246_raw': 'float64', 'smart_247_normalized': 'float64', 'smart_247_raw': 'float64', 'smart_248_normalized': 'float64', 'smart_248_raw': 'float64', 'smart_250_normalized': 'float64', 'smart_250_raw': 'float64', 'smart_251_normalized': 'float64', 'smart_251_raw': 'float64', 'smart_252_normalized': 'float64', 'smart_252_raw': 'float64', 'smart_254_normalized': 'float64', 'smart_254_raw': 'float64', 'smart_255_normalized': 'float64', 'smart_255_raw': 'float64'}, 
              columns_to_delete: Optional[list] = ['model','capacity_bytes','datacenter','cluster_id','vault_id','pod_id','pod_slot_num','is_legacy_format']):
    """
    Load and preprocess hard drive data for LSTM training and testing.
    
    This function serves as the main entry point for data loading, creating a DriveDataLoader
    instance to handle train/test splitting at the drive level, then generating PyTorch
    datasets and data loaders ready for model training and evaluation.
    
    Args:
        root (str): Root directory containing CSV files with hard drive data
        train_ratio (float): Proportion of drives to allocate for training (0.8 = 80% train, 20% test)
        min_sequence_length (int): Minimum number of days required per drive to be included
        input_len (int): Number of time steps to use as input (look-back window)
        label_len (int): Number of time steps to predict (forecast window)
        normalized_rows (list): List of SMART attribute IDs to include (normalized values)
        raw_rows (list): List of SMART attribute IDs to include (raw values)
        verbose (bool): Whether to print detailed information during data loading
        batch_size (int): Batch size for PyTorch DataLoaders
        dtype_dict (dict, optional): Data type specifications for efficient CSV reading
        columns_to_delete (list, optional): List of column names to remove during preprocessing
        
    Returns:
        tuple: (train_loader, test_loader) where:
            - train_loader: PyTorch DataLoader for training data
            - test_loader: PyTorch DataLoader for testing data
    """
    data_loader = DriveDataLoader(root=root, 
                                 train_ratio=train_ratio, 
                                 min_sequence_length=min_sequence_length,
                                 normalized_rows=normalized_rows,
                                 raw_rows=raw_rows,
                                 verbose=verbose,
                                 dtype_dict=dtype_dict,
                                 columns_to_delete=columns_to_delete)
    # Create datasets using the pre-split data
    dataset_train = CustomDrives(data_loader=data_loader,
                                train=True, 
                                input_len=input_len,
                                label_len=label_len,
                                verbose=verbose)
    dataset_test = CustomDrives(data_loader=data_loader,
                               train=False, 
                               input_len=input_len,
                               label_len=label_len,
                               verbose=verbose)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

def train_model(features, n_neurons, model_path, days_to_predict, days_to_train, train_loader, test_loader, test_existing=False, learning_rate=0.001, num_epochs=1, device=None):
    """
    Train an LSTM neural network for hard drive failure prediction with comprehensive monitoring and persistence.
    
    This function handles the complete training pipeline, including model initialization, hyperparameter
    configuration, training loop execution, validation monitoring, and model persistence. It supports
    both training from scratch and loading existing models for evaluation.
    
    The training process implements early stopping based on validation loss, comprehensive progress
    tracking with tqdm, and automatic model saving when performance improves. The function also
    supports testing existing models without retraining.
    
    Args:
        features (int): Number of input features (SMART attributes) for the model
        n_neurons (int): Number of neurons in the LSTM hidden layer
        model_path (str): Path where the trained model should be saved/loaded
        days_to_predict (int): Number of future days the model should predict
        days_to_train (int): Number of historical days used as input for predictions
        train_loader (DataLoader): PyTorch DataLoader containing training data
        test_loader (DataLoader): PyTorch DataLoader containing validation/test data
        test_existing (bool): If True, load and evaluate existing model without retraining
        learning_rate (float): Learning rate for the Adam optimizer
        num_epochs (int): Maximum number of training epochs
        device (torch.device, optional): Device to run training on. Auto-detects if None
        
    Returns:
        tuple: (trained_model, model_existed) where:
            - trained_model: The trained PyTorch model ready for inference
            - model_existed: Boolean indicating if a pre-trained model was found and loaded
            
    Side Effects:
        - Saves trained model and metrics to disk
        - Generates and saves loss curves and prediction plots
        - Prints detailed training progress and evaluation results
    """
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Net(n_neurons, features, days_to_predict).double()
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Move model to device
    model = model.to(device)
    
    train, labels = next(iter(train_loader))
    print(f'shape: {train.shape, labels.shape}')

    loss_curve = []
    minimum_loss = np.inf

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
            # Print epoch summary
            if (epoch + 1) % max(1, num_epochs // 10) == 0 or epoch == num_epochs - 1:
                print(f"[{epoch + 1}/{num_epochs}] - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            
            # Save loss plot every 10 epochs
            # if (epoch + 1) % 10 == 0:
            #     fig, ax = plt.subplots(1, 1, figsize=(15, 5))
            #     ax.plot(loss_curve, lw=2)
            #     ax.set_xlabel("Epoch")
            #     ax.set_ylabel("Validation Loss")
            #     ax.set_title("Training Progress")
                
            #     # Save in model directory
            #     plot_dir = os.path.dirname(model_path)
            #     os.makedirs(plot_dir, exist_ok=True)
            #     plot_path = os.path.join(plot_dir, f'loss_epoch_{epoch+1}.png')
            #     plt.savefig(plot_path)
            #     plt.close()  # Close the figure to free memory
        metrics = {
            'val_loss': avg_val_loss,
            'train_loss': avg_train_loss,
            'epoch': epoch + 1,
            'total_epochs': num_epochs,
            'optimizer': optimizer.__class__.__name__,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'n_neurons': n_neurons,
            'features': features,
            'days_to_predict': days_to_predict,
            'days_to_train': days_to_train,
            'device': str(device)
        }
        save_model(model, model_path, metrics, save_whole_model=True)
        print(f"New best model saved with validation loss: {avg_val_loss:.6f}")
    # Generate final plots and predictions
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
        plt.close()  # Close the figure to free memory

    return model, model_exists

from itertools import product

def grid_search_lstm(root, train_ratio, min_sequence_length, days_to_train, days_to_predict, normalized_rows, raw_rows, verbose, num_features, device, param_grid=None, max_epochs=3):
    """
    Perform systematic hyperparameter optimization for LSTM models using grid search.
    
    This function explores different combinations of LSTM hyperparameters to find the
    optimal configuration for hard drive failure prediction. It trains multiple models
    with different parameter combinations and selects the best performing configuration
    based on validation loss.
    
    The grid search process loads data once and reuses it for all parameter combinations,
    ensuring fair comparison between different configurations. Each model is trained
    for a limited number of epochs to efficiently explore the parameter space.
    
    Args:
        root (str): Root directory containing training data
        train_ratio (float): Proportion of drives to use for training
        min_sequence_length (int): Minimum number of days required per drive
        days_to_train (int): Number of historical days used as input
        days_to_predict (int): Number of future days to predict
        normalized_rows (list): SMART attribute IDs to include (normalized values)
        raw_rows (list): SMART attribute IDs to include (raw values)
        verbose (bool): Whether to print detailed progress information
        num_features (int): Total number of input features
        device (torch.device): Device to run training on
        param_grid (dict, optional): Dictionary of parameter lists to search. Defaults to predefined grid
        max_epochs (int): Number of epochs to train each model configuration
        
    Returns:
        tuple: (best_parameters, best_validation_loss) where:
            - best_parameters: Dictionary containing the optimal hyperparameter combination
            - best_validation_loss: Lowest validation loss achieved during the search
            
    Notes:
        - Default parameter grid includes n_neurons [4, 8, 16] and learning_rate [0.001, 0.005]
        - Temporary models are saved as 'models/lstm_gridsearch.pth' during search
        - Best parameters exclude batch_size for compatibility with other functions
    """
    if param_grid is None:
        param_grid = {
            'n_neurons': [4, 8, 16],
            'learning_rate': [0.001, 0.005],
        }
    best_loss = float('inf')
    best_params = None

    train_loader, test_loader = load_data(
        root=root,
        train_ratio=train_ratio,
        min_sequence_length=min_sequence_length,
        input_len=days_to_train,
        label_len=days_to_predict,
        normalized_rows=normalized_rows,
        raw_rows=raw_rows,
        verbose=verbose,
    )
    for n_neurons, learning_rate in product(param_grid['n_neurons'], param_grid['learning_rate']):
        print(f"\nTesting n_neurons={n_neurons}, lr={learning_rate}")
        # Re-create loaders with new batch size

        model, _ = train_model(
            features=num_features,
            n_neurons=n_neurons,
            model_path='models/lstm_gridsearch.pth',
            days_to_predict=days_to_predict,
            days_to_train=days_to_train,
            train_loader=train_loader,
            test_loader=test_loader,
            test_existing=False,
            learning_rate=learning_rate,
            num_epochs=max_epochs,
            device=device
        )
        # Evaluate on validation set
        model.eval()
        val_loss = 0
        batches = 0
        with torch.no_grad():
            for val_data, val_labels in test_loader:
                val_data, val_labels = val_data.to(device), val_labels.to(device)
                predictions = model(val_data)
                loss = torch.nn.functional.mse_loss(predictions, val_labels)
                val_loss += loss.item()
                batches += 1
        avg_val_loss = val_loss / batches if batches > 0 else float('inf')
        print(f"Validation loss: {avg_val_loss:.6f}")
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_params = {'n_neurons': n_neurons, 'learning_rate': learning_rate}  # batch_size removed
    print(f"\nBest LSTM params: {best_params}, Validation loss: {best_loss:.6f}")
    return best_params, best_loss

def main():
    """
    Main execution function for LSTM-based hard drive failure prediction system.
    
    This function orchestrates the complete machine learning pipeline, including data loading,
    hyperparameter optimization, model training, evaluation, and result visualization. It
    demonstrates the full workflow from raw SMART data to trained model predictions.
    
    The pipeline includes:
    1. Device detection and configuration
    2. Data loading with train/test splitting at drive level
    3. Grid search for optimal hyperparameters
    4. Model training with progress monitoring
    5. Model evaluation and prediction generation
    6. Result visualization and metrics saving
    
    Configuration parameters are set at the beginning of the function and can be modified
    for different experimental setups. The function handles both training new models and
    testing existing saved models based on the test_existing flag.
    
    Side Effects:
        - Loads and preprocesses hard drive SMART data
        - Trains and saves LSTM models
        - Generates prediction plots and evaluation metrics
        - Saves results and model artifacts to disk
        - Prints comprehensive progress and performance information
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_existing = True # Change to True to test existing model
    days_to_train = 3
    days_to_predict = 2
    look_back = days_to_train + days_to_predict
    path = "data/data_test"
    verbose = True    # Create a single data loader that handles the train/test split properly
    print(f'Using device: {device}') 
    
    print("🔄 Loading and splitting data to prevent data leakage...")
    min_sequence_length = days_to_train + days_to_predict
    train_loader, test_loader = load_data(root=path,
                                         train_ratio=0.8, 
                                         min_sequence_length=min_sequence_length,
                                         input_len=days_to_train,
                                         label_len=days_to_predict,
                                         normalized_rows = [1, 3, 5, 7, 9, 187, 189, 190, 195, 197],
                                         raw_rows = [5, 197],
                                         verbose=verbose,
                                         batch_size=3)
    num_features = 12
    n_neurons = 4
    num_epochs = 1
    learning_rate = 0.001

    param_grid = {
        'n_neurons': [4, 8, 16, 32],
        'learning_rate': [0.001, 0.003, 0.005, 0.01],
        'batch_size': [2, 4, 8, 16]
    }

    best_params, best_loss = grid_search_lstm(
        root=path,
        train_ratio=0.8,
        min_sequence_length=min_sequence_length,
        days_to_train=days_to_train,
        days_to_predict=days_to_predict,
        normalized_rows=[1, 3, 5, 7, 9, 187, 189, 190, 195, 197],
        raw_rows=[5, 197],
        verbose=verbose,
        num_features=num_features,
        device=device,
        param_grid=None,
        max_epochs=num_epochs
    )
    if best_params is not None:
        n_neurons = best_params['n_neurons']
        learning_rate = best_params['learning_rate']
    else:
        raise RuntimeError("Grid search did not return any valid parameters. Please check your data and parameter grid.")
    # Check if trained model exists
    model_path = 'models/LSTM/lstm_model.pth'

    model, model_exists = train_model(features = num_features,
                n_neurons = n_neurons,
                model_path = model_path,
                days_to_predict = days_to_predict,
                days_to_train = days_to_train,
                train_loader = train_loader,
                test_loader = test_loader,
                test_existing = test_existing,
                learning_rate = learning_rate,
                num_epochs = num_epochs)

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


if __name__ == '__main__':
    main()