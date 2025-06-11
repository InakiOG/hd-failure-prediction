"""
Test script to verify verbose output showing sample drive groups
"""
import sys
import os

# Add the parent directory to path to import smart.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from notebooks.LSTM.smart import DriveDataLoader

def test_verbose_output():
    print("Testing verbose output with sample drive groups...")
      # Create data loader with verbose output enabled
    data_loader = DriveDataLoader(
        root="../../data/data_test",  # Use the smaller test dataset
        train_ratio=0.8,
        min_sequence_length=3,  # Use 3 days since we only have 6 days of data
        verbose=True
    )
    
    print(f"\nSummary:")
    print(f"- Total drives loaded: {len(data_loader.train_drives) + len(data_loader.test_drives)}")
    print(f"- Training drives: {len(data_loader.train_drives)}")
    print(f"- Testing drives: {len(data_loader.test_drives)}")
    
    # Get sample data from each set
    if len(data_loader.train_drives) > 0:
        train_data = data_loader.get_train_data()
        print(f"- Training data shape: {train_data.shape}")
    
    if len(data_loader.test_drives) > 0:
        test_data = data_loader.get_test_data()
        print(f"- Testing data shape: {test_data.shape}")

if __name__ == "__main__":
    test_verbose_output()
