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
    def __init__(self, csv_file, city_count=-1):
        self.df = pd.read_csv(csv_file)
        if city_count > 0:
            self.df = self.df[:city_count]

        self._normalise_columns()

    def __len__(self):
        return len(self.df)

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def _normalise_columns(self):
        pass


class CityDistanceDataset(BaseDataset):
    def __init__(self, csv_file, city_count=-1):
        super().__init__(csv_file, city_count)
        cities_a = self.df["City A"].unique()
        cities_b = self.df["City B"].unique()
        combined_cities = list(set(cities_a) | set(cities_b))
        combined_cities = sorted(combined_cities)
        self.city_to_int = {city: i for i, city in enumerate(combined_cities)}

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


# TODO: Switch to using the other dataset for consistency (ie. city count)
class CoordinateDataset(Dataset):
    def __init__(self, city_count=-1):
        self.df = load_raw_data()
        if city_count > 0:
            self.df = self.df[:city_count]

        _normalise_column(self.df, "Latitude")
        _normalise_column(self.df, "Longitude")

        cities = self.df.index.values.tolist()
        cities = sorted(cities)
        print(cities[8])
        print(cities[384])
        print(cities[784])
        quit()
        self.city_to_int = {city: i for i, city in enumerate(cities)}

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = self.city_to_int[row.name],
        y = [
            row['Latitude'],
            row['Longitude']
        ]
        return x, y

    def __len__(self):
        return len(self.df)


class CoordinateDataset(BaseDataset):
    def __init__(self, csv_file, city_count=-1):
        super().__init__(csv_file, city_count)
        cities_a = self.df["City A"].unique()
        cities_b = self.df["City B"].unique()
        self.cities = list(set(cities_a) | set(cities_b))
        self.cities = sorted(self.cities)
        self.city_to_int = {city: i for i, city in enumerate(self.cities)}

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