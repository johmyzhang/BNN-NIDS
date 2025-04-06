

## CAN Bus Dataset Merger
# This notebook reads and merges four CAN bus CSV datasets
# With handling for incomplete data rows (padding with zeros)

import pandas as pd
import os
import glob
from io import StringIO

# Define a function to process each file
def process_can_file(file_path):
    # Extract the attack type from the filename
    filename = os.path.basename(file_path)
    attack_type = filename.split('_')[0]

    # Read the file as text first to handle rows with varying column counts
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Process each line and ensure it has the correct number of fields
    processed_lines = []
    for line in lines:
        fields = line.strip().split(',')
        # Check if we have all fields (timestamp, CAN_ID, DLC, 8 data fields, label)
        if len(fields) < 12:
            # If fields are missing, pad with zeros for data fields
            fields_needed = 12 - len(fields)
            # The label should be the last field
            label = fields[-1]
            # Remove the label temporarily
            fields = fields[:-1]
            # Add zeros as needed
            fields.extend(['00'] * fields_needed)
            # Add the label back
            fields.append(label)
        # Add processed line
        processed_lines.append(','.join(fields))

    # Create a temporary file with properly formatted data
    temp_data = '\n'.join(processed_lines)

    # Read processed data with pandas
    df = pd.read_csv(StringIO(temp_data), header=None)

    df.columns = ['Timestamp', 'CAN_ID', 'DLC',
                  'DATA[0]', 'DATA[1]', 'DATA[2]', 'DATA[3]',
                  'DATA[4]', 'DATA[5]', 'DATA[6]', 'DATA[7]', 'Label']

    # Convert hexadecimal values to decimal
    df['CAN_ID'] = df['CAN_ID'].apply(lambda x: int(str(x), 16))

    # Convert data columns to decimal and handle any remaining issues
    for i in range(8):
        col = f'DATA[{i}]'
        df[col] = df[col].apply(lambda x: int(str(x), 16) if pd.notna(x) and str(x).strip() else 0)

    # Replace 'T' labels with specific attack type
    df['Label'] = df['Label'].apply(lambda x: attack_type if x == 'T' else 'R')

    # Drop unnecessary columns
    df = df.drop(['Timestamp', 'DLC'], axis=1)

    # Add source file information
    df['Source_File'] = filename

    return df

# Use this in a Jupyter notebook
# Create a dataframe to hold all merged data
merged_df = pd.DataFrame()

# Path to the directory containing the CSV files
data_dir = '/Users/yizhang/PycharmProjects'  # Change this to the directory containing your CSV files

# Find all dataset CSV files
file_paths = glob.glob(os.path.join(data_dir, '*_dataset.csv'))

# Process each file and append to merged dataset
for file_path in file_paths:
    print(f"Processing {file_path}...")
    try:
        processed_df = process_can_file(file_path)
        merged_df = pd.concat([merged_df, processed_df], ignore_index=True)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Display summary information
print("\nMerged Dataset Summary:")
print(f"Total rows: {len(merged_df)}")
print(f"Files processed: {merged_df['Source_File'].nunique()}")
print(f"Unique labels: {merged_df['Label'].unique()}")

# Save the merged dataset
output_path = os.path.join(data_dir, 'merged_can_dataset.csv')
merged_df.to_csv(output_path, index=False)
print(f"\nMerged dataset saved to {output_path}")

# Display a sample of the merged data
print("\nSample rows from merged dataset:")
merged_df.head()

# Check for rows with zero values (padded data)
padded_count = ((merged_df.iloc[:, 2:10] == 0).sum(axis=1) > 0).sum()
print(f"\nRows with padded data fields (containing zeros): {padded_count}")