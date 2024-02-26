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


def get_city_to_int(data_dir):
    cities_df = pd.read_csv(os.path.join(data_dir, "cities.csv"))
    return {cities_df.loc[i]['City']: i for i in range(len(cities_df))}


class MiddleCityDataset(Dataset):
    def __init__(self, data_dir):
        self.cities_to_int = get_city_to_int(data_dir)

    def __getitem__(self, idx):
        x = [
            self.city_to_int[self.df.loc[idx, "City A"]],
            self.city_to_int[self.df.loc[idx, "City B"]],
        ]
        y = self.city_to_int[self.df.loc[idx, "Middle City"]],
        return x, y

    def _normalise_columns(self):
        pass

    def __len__(self):
        return len(self.df)


class CityDistanceDataset(Dataset):
    def __init__(self, data_dir, return_strings=False):
        self.return_strings = return_strings
        self.city_to_int = get_city_to_int(data_dir)
        self.city_count = len(self.city_to_int)
        self.df = pd.read_csv(os.path.join(data_dir, "distances.csv"))

    def _normalise_columns(self):
        _normalise_column(self.df, "Distance")

    def __getitem__(self, idx):
        if self.return_strings:
            x = f'{self.df.loc[idx, "City A"]} {self.df.loc[idx, "City B"]}'.lower()
        else:
            x = [
                self.city_to_int[self.df.loc[idx, "City A"]],
                self.city_to_int[self.df.loc[idx, "City B"]],
            ]
        y = self.df.loc[idx, "Distance"]
        return x, y

    def __len__(self):
        return len(self.df) 


class CoordinateDistanceDataset(Dataset):
    """
    X values are the normalised coordinates of the city, and the y value is the
    distance between the two cities.
    """
    def __init__(self, data_dir):
        cities_df = pd.read_csv(os.path.join(data_dir, "cities.csv"))
        self.city_count = len(cities_df)
        distances_df = pd.read_csv(os.path.join(data_dir, "distances.csv"))

        cities_df.columns = ['City A', 'City A Latitude', 'City A Longitude']
        merged_df = pd.merge(distances_df, cities_df, on='City A')
        cities_df.columns = ['City B', 'City B Latitude', 'City B Longitude']
        merged_df = pd.merge(merged_df, cities_df, on='City B')

    def _normalise_columns(self):
        for column_name in [
            "City A Latitude",
            "City A Longitude",
            "City B Latitude",
            "City B Longitude",
            "Distance",
        ]:
            _normalise_column(self.df, column_name)

    def __getitem__(self, idx):
        x = [
            torch.tensor([
                self.df.loc[idx, "City A Latitude"],
                self.df.loc[idx, "City A Longitude"],
            ], dtype=torch.float32),
            torch.tensor([
                self.df.loc[idx, "City B Latitude"],
                self.df.loc[idx, "City B Longitude"],
            ], dtype=torch.float32),
        ]
        y = self.df.loc[idx, "Distance"]
        return x, y
    
    def __len__(self):
        return len(self.df)


class CoordinateDataset(Dataset):
    def __init__(self, data_dir):
        self.df = pd.read_csv(os.path.join(data_dir, "cities.csv"))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        latitude = row["Latitude"]
        longitude = row["Longitude"]
        return idx, (latitude, longitude)

    def __len__(self):
        return len(self.df)