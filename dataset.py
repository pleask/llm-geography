from abc import ABC, abstractmethod
import os

import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

from generate_dataset import load_raw_data


def _normalise_column(df, column_name):
    scaler = StandardScaler()
    df[column_name] = scaler.fit_transform(df[[column_name]])


class CityDistanceDataset(Dataset):
    def __init__(self, data_dir):
        self.df = pd.read_csv(os.path.join(data_dir, "distances.csv"))
        self._normalise_columns()

    def _normalise_columns(self):
        _normalise_column(self.df, "Distance")

    def __getitem__(self, idx):
        x =  f'{self.df.loc[idx, "City A"]}, {self.df.loc[idx, "City B"]}'
        y = self.df.loc[idx, "Distance"]
        return x, y

    def __len__(self):
        return len(self.df) 


class CoordinateDataset(Dataset):
    def __init__(self, data_dir):
        self.df = pd.read_csv(os.path.join(data_dir, "cities.csv"))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return row['City'], (row['Latitude'], row['Longitude'])

    def __len__(self):
        return len(self.df)

    def get_city_coordinates(self, city):
        row = self.df[self.df['City'] == city]
        return row['Latitude'].item(), row['Longitude'].item()