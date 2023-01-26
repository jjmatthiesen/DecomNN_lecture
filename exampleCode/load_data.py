from torch.utils.data import Dataset
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

"""
DATALOADER EXAMPLE
For a custom dataset

Based on: 
https://towardsdatascience.com/how-to-use-datasets-and-dataloader-in-pytorch-for-custom-text-data-270eed7f7c00

"""


class CustomTextDataset(Dataset):
    def __init__(self, data, labels):
        self.labels = labels
        self.data = data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        text = self.data[idx]
        sample = {"Text": text, "Class": label}
        return sample


train_data_raw = ['Happy', 'Amazing', 'Sad', 'Unhapy', 'Glum', 'Great']
train_labels_raw = [1, 1, 0, 0, 0, 1]

# create Pandas DataFrame
data_labels_df = pd.DataFrame({'Text': train_data_raw, 'Labels': train_labels_raw})
# define data set object
train_data = CustomTextDataset(data_labels_df['Text'], data_labels_df['Labels'])


#########################
# How to see data
#########################


# print data:
print(train_data.data)

# print labels
print(train_data.labels)

# get one set [data, labels] by index
print(train_data.__getitem__(0))

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=2, shuffle=True)

# Display image and label.
print('\nFirst iteration of dataset: ', next(iter(train_loader)), '\n')

# Print how many items are in the data set
print('Length of data set: ', len(train_loader), '\n')

# Print entire data set
print('Entire data set: ', list(train_loader), '\n')
