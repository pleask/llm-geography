from abc import ABC, abstractmethod

import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

from generate_dataset import load_raw_data


def _normalise_column(df, column_name):
    scaler = StandardScaler()
    df[column_name] = scaler.fit_transform(df[[column_name]])


class BaseDataset(ABC, Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self._normalise_columns()

    def __len__(self):
        return len(self.df)

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def _normalise_columns(self):
        pass


def get_unique_cities(*sequences):
    cities = set()
    for sequence in sequences:
        sequence.unique()
        cities |= set(sequence)
    return sorted(list(cities)), {city: i for i, city in enumerate(cities)}


class MiddleCityDataset(BaseDataset):
    def __init__(self, csv_file, city_count=-1):
        super().__init__(csv_file)
        _, self.city_to_int = get_unique_cities(self.df["City A"], self.df["City B"], self.df['Middle City'])

    def __getitem__(self, idx):
        x = [
            self.city_to_int[self.df.loc[idx, "City A"]],
            self.city_to_int[self.df.loc[idx, "City B"]],
        ]
        y = self.city_to_int[self.df.loc[idx, "Middle City"]],
        return x, y

    def _normalise_columns(self):
        pass


class CityDistanceDataset(BaseDataset):
    def __init__(self, csv_file):
        super().__init__(csv_file)
        _, self.city_to_int = get_unique_cities(self.df["City A"], self.df["City B"])

    def _normalise_columns(self):
        _normalise_column(self.df, "Distance")

    def __getitem__(self, idx):
        x = [
            self.city_to_int[self.df.loc[idx, "City A"]],
            self.city_to_int[self.df.loc[idx, "City B"]],
        ]
        y = self.df.loc[idx, "Distance"]
        return x, y


class CoordinateDistanceDataset(BaseDataset):
    """
    X values are the normalised coordinates of the city, and the y value is the
    distance between the two cities.
    """
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


class CoordinateDataset(BaseDataset):
    def __init__(self, csv_file):
        super().__init__(csv_file)
        self.cities, self.city_to_int = get_unique_cities(self.df["City A"], self.df["City B"])

    def __getitem__(self, idx):
        city_name = self.cities[idx]
        try:
            latitude = self.df.loc[self.df["City A"] == city_name, "City A Latitude"].values[0]
            longitude = self.df.loc[self.df["City A"] == city_name, "City A Longitude"].values[0]
        except IndexError:
            latitude = self.df.loc[self.df["City B"] == city_name, "City B Latitude"].values[0]
            longitude = self.df.loc[self.df["City B"] == city_name, "City B Longitude"].values[0]

        return idx, [latitude, longitude]

    def _normalise_columns(self):
        # Don't normalise the columns for the linear probe.
        pass

    def __len__(self):
        return len(self.city_to_int)