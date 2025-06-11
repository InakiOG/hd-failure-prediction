#!/usr/bin/env python3
"""
Quick test script to verify the improved DriveDataLoader implementation.
This script tests the new sequence length filtering and train/test splitting logic.
"""

import sys
import os
sys.path.append('.')

from notebooks.LSTM.smart import DriveDataLoader

def test_data_loader():
    """Test the DriveDataLoader with different configurations."""
    
    print("🧪 Testing DriveDataLoader Implementation")
    print("=" * 50)
    
    # Test with different minimum sequence lengths
    test_configs = [
        {"min_seq": 5, "desc": "Standard (5 days minimum)"},
        {"min_seq": 10, "desc": "Stricter (10 days minimum)"},
        {"min_seq": 3, "desc": "Lenient (3 days minimum)"}
    ]
    
    data_path = "../../data/data_Q1_2025"
    
    for config in test_configs:
        print(f"\n📋 Testing: {config['desc']}")
        print(f"   Minimum sequence length: {config['min_seq']} days")
        print("-" * 40)
        
        try:
            # Create data loader
            loader = DriveDataLoader(
                root=data_path,
                train_ratio=0.8,
                min_sequence_length=config['min_seq'],
                verbose=True
            )
            
            # Get train and test data
            train_data = loader.get_train_data()
            test_data = loader.get_test_data()
            
            # Print summary statistics
            print(f"   ✅ Train data: {len(train_data)} rows, {train_data['serial_number'].nunique()} drives")
            print(f"   ✅ Test data: {len(test_data)} rows, {test_data['serial_number'].nunique()} drives")
            
            # Verify no overlap between train and test drives
            train_drives = set(train_data['serial_number'].unique())
            test_drives = set(test_data['serial_number'].unique())
            overlap = train_drives.intersection(test_drives)
            
            if len(overlap) == 0:
                print(f"   ✅ No data leakage: Train and test drives are completely separate")
            else:
                print(f"   ❌ Data leakage detected: {len(overlap)} drives appear in both sets!")
                
        except FileNotFoundError as e:
            print(f"   ⚠️  Data not found: {e}")
        except ValueError as e:
            print(f"   ⚠️  Configuration issue: {e}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n🎉 Data loader testing completed!")

if __name__ == "__main__":
    test_data_loader()
