import pandas as pd
from sklearn.model_selection import train_test_split
import os


raw_data_file = 'data.csv'
filename = 'dataset.csv'

print("Creating dataset.")
print("="*20)
raw_data = pd.read_csv("data.csv")

print(f"Total number of classes with occurences from raw data: \n {raw_data['direction'].value_counts()}\n\n")
print("="*20)


print("Making a balanced dataset..")
print("="*20)

balanced_df = (
    raw_data.groupby('direction')
    .apply(lambda x: x.sample(n=2000, random_state=42))
    .reset_index(drop=True)
)

print(f"Total number of classes with occurences from processed data: \n {balanced_df['direction'].value_counts()}\n\n")
print("="*20)

print("shuffling dataset..")
balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)


balanced_df['depth_image'] = balanced_df['depth_image'].apply(os.path.basename)

balanced_df = balanced_df.drop(columns=['rgb_image', "time"])

print(f"Saving in the file {filename}")
print("="*20)


balanced_df.to_csv(filename, index=False)

unique_dirs = sorted(balanced_df['direction'].unique())
direction_to_idx = {label: idx for idx, label in enumerate(unique_dirs)}
balanced_df['label'] = balanced_df['direction'].map(direction_to_idx)

# Step 1: Train/test+val split (70/30) with stratification
train_df, temp_df = train_test_split(
    balanced_df, test_size=0.3, stratify=balanced_df['label'], random_state=42
)

# Step 2: Val/test split (15/15) from remaining 30%
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42
)

# Step 3: Save splits to CSV
split_dir = 'split'
os.makedirs(split_dir, exist_ok=True)

train_df.to_csv(os.path.join(split_dir, "train.csv"), index=False)
val_df.to_csv(os.path.join(split_dir, "val.csv"), index=False)
test_df.to_csv(os.path.join(split_dir, "test.csv"), index=False)

# Optional: Show balance check
print("Class distribution:")
print("Train:", train_df['direction'].value_counts(normalize=True))
print("Val:  ", val_df['direction'].value_counts(normalize=True))
print("Test: ", test_df['direction'].value_counts(normalize=True))