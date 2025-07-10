import os
import pandas as pd
from tqdm import tqdm

# Folder containing CSV files
folder_path = 'data/all_data/all_data'  # Change this to your folder path

# Columns to load
usecols = ['serial_number', 'date', 'failure']

# Read all CSV files and concatenate into one DataFrame
all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
df = pd.concat(
    (pd.read_csv(f, usecols=usecols) for f in tqdm(all_files, desc="Reading CSVs")),
    ignore_index=True
)

# Ensure column names are consistent (strip whitespace, lower case)
df.columns = [col.strip().lower() for col in df.columns]

# Find serial numbers where failure == 1 (or True)
failed_serials = df[df['failure'] == 1]['serial_number'].unique()

# Group all rows for drives that have failed
failed_drives_df = df[df['serial_number'].isin(failed_serials)]

# Calculate the number of unique days each failed drive has data for
days_per_failed_drive = failed_drives_df.groupby('serial_number')['date'].nunique()
# Get the unique counts of days and the number of drives for each count
unique_days_counts = days_per_failed_drive.value_counts().sort_index()

# Print the result: number of days -> number of drives
print("Number of unique days for failed drives (non consecutive):")
for days, num_drives in unique_days_counts.items():
    print(f"{days} days: {num_drives} drives")

# Calculate the maximum number of consecutive days for each failed drive
def max_consecutive_days(dates):
    dates = pd.to_datetime(sorted(dates))
    diffs = dates.to_series().diff().dt.days.fillna(1)
    consecutive = (diffs != 1).cumsum()
    return consecutive.value_counts().max()

consecutive_days_per_drive = failed_drives_df.groupby('serial_number')['date'].apply(max_consecutive_days)
consecutive_days_counts = consecutive_days_per_drive.value_counts().sort_index()

print("\nNumber of maximum consecutive days for failed drives:")
for days, num_drives in consecutive_days_counts.items():
    print(f"{days} days: {num_drives} drives")