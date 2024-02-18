import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
import multiprocessing
import os

from geographiclib.geodesic import Geodesic
from geopy.distance import great_circle
import pandas as pd
from tqdm import tqdm


def get_distance_to_line_position(point, line, position):
    p = line.Position(position, Geodesic.STANDARD | Geodesic.LONG_UNROLL)
    distance_to_line = Geodesic.WGS84.InverseLine(
        point['Latitude'], point['Longitude'], p["lat2"], p["lon2"]
    )
    return distance_to_line.s13


def binary_search_lowest_distance(point, geodesic_line):
    low = 0
    high = geodesic_line.s13
    precision = 1e-6

    while high - low > precision:
        mid = (low + high) / 2
        mid_distance = get_distance_to_line_position(point, geodesic_line, mid)
        left_distance = get_distance_to_line_position(
            point, geodesic_line, mid - precision
        )
        right_distance = get_distance_to_line_position(
            point, geodesic_line, mid + precision
        )

        if left_distance < mid_distance:
            high = mid
        elif right_distance < mid_distance:
            low = mid
        else:
            return mid

    return (low + high) / 2


def distance_point_to_geodesic(point, geodesic_line):
    position = binary_search_lowest_distance(point, geodesic_line)
    return get_distance_to_line_position(point, geodesic_line, position)


def get_city_distances(city_coordinates, city_A, city_B):
    geodesic_line = Geodesic.WGS84.InverseLine(
        city_A['Latitude'], city_A['Longitude'], city_B['Latitude'], city_B['Longitude']
    )
    distances = []

    for coordinates in city_coordinates:
        dist = distance_point_to_geodesic(coordinates, geodesic_line)
        distances.append(dist)

    return distances


def get_nearest_city(data, city_A, city_B):
    city_names = data.index.values.tolist()
    city_names.remove(city_A)
    city_names.remove(city_B)
    city_coordinates = [get_coordinates(data, city) for city in city_names]
    distances = get_city_distances(
        city_coordinates, get_coordinates(data, city_A), get_coordinates(data, city_B)
    )

    paired = sorted(zip(distances, city_names))
    nearest_city = paired[0][1]

    return nearest_city


def get_coordinates(data, city):
    d = data.loc[city, ["Latitude", "Longitude"]]
    if isinstance(d, pd.DataFrame):
        return d.iloc[0]
    return d


def get_distance_between_cities(city_A_coords, city_B_coords, wgs84=False):
    if wgs84:
        geodesic_line = Geodesic.WGS84.InverseLine(
            city_A_coords["Latitude"],
            city_A_coords["Longitude"],
            city_B_coords["Latitude"],
            city_B_coords["Longitude"],
        )
        return geodesic_line.s13 / 1000
    else:
        city_A = (city_A_coords["Latitude"], city_A_coords["Longitude"])
        city_B = (city_B_coords["Latitude"], city_B_coords["Longitude"])
        return great_circle(city_A, city_B).kilometers


def get_row(data, city_A, city_B, get_middle_city=False, wgs84=False):
    city_A_coords = get_coordinates(data, city_A)
    city_B_coords = get_coordinates(data, city_B)
    distance_A_B = get_distance_between_cities(city_A_coords, city_B_coords, wgs84=wgs84)

    row = {
        "City A": city_A,
        "City A Latitude": city_A_coords["Latitude"],
        "City A Longitude": city_A_coords["Longitude"],
        "City B": city_B,
        "City B Latitude": city_B_coords["Latitude"],
        "City B Longitude": city_B_coords["Longitude"],
        "Distance": distance_A_B,
    }

    if get_middle_city:
        middle_city = get_nearest_city(data, city_A, city_B)
        row["Middle City"] = middle_city
        middle_city_coords = get_coordinates(data, middle_city)
        row["Middle City Latitude"] = middle_city_coords["Latitude"]
        row["Middle City Longitude"] = middle_city_coords["Longitude"]

    return pd.Series(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--capitals_only", action="store_true", default=False)
    parser.add_argument("--get_middle_city", action="store_true", default=False)
    parser.add_argument("--wgs84", action="store_true", default=False)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--city_count", type=int, default=-1)
    parser.add_argument("--skip", type=int, default=0, help="Determines how many combinations to skip.")
    parser.add_argument("--batch", type=int, default=-1, help="Number of combinations to process in this batch.")
    parser.add_argument("--no_mp", action="store_true", default=False, help="Don't use multiprocessing, useful for debugging or running on slurm.")
    args = parser.parse_args()

    assert not (args.get_middle_city and args.wgs84), 'Cannot use wgs84 with get_middle_city'

    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)

    get_raw_data_path = lambda file_name: os.path.join(current_directory, "raw_data", file_name)

    if args.capitals_only:
        data = pd.read_csv(
            get_raw_data_path("capital_cities.csv"),
            usecols=["Capital City", "Latitude", "Longitude"],
        )
        data = data.rename(columns={"Capital City": "City"})
    else:
        data = pd.read_csv(
            get_raw_data_path("all_cities.csv"),
            usecols=["city", "lat", "lng"]
        )
        data = data.rename(
            columns={"city": "City", "lat": "Latitude", "lng": "Longitude"}
        )

    data["Latitude"] = data["Latitude"].astype("float32")
    data["Longitude"] = data["Longitude"].astype("float32")
    data.set_index("City", inplace=True)

    cities = data.index.values.tolist()
    if args.city_count > 0:
        cities = cities[:args.city_count]
    pairs = combinations(cities, 2)

    # Skip and batch logic for parallel processing.
    if args.skip > 0:
        for _ in range(args.skip):
            next(pairs)
    if args.batch > 0:
        pairs = [next(pairs) for _ in range(args.batch)]
    else:
        pairs = list(pairs)

    if len(pairs) == 0:
        print("No pairs to process")
        exit(0)

    columns = [
        "City A",
        "City A Latitude",
        "City A Longitude",
        "City B",
        "City B Latitude",
        "City B Longitude",
        "Distance",
    ]
    if args.get_middle_city:
        columns.extend(
            [
                "Middle City",
                "Middle City Latitude",
                "Middle City Longitude",
            ]
        )

    assert not os.path.exists(args.output_file), "Output file already exists"

    # Create an empty dataframe and write the columns to the output file
    df = pd.DataFrame(columns=columns)
    df.to_csv(args.output_file, index=False)

    lock = multiprocessing.Lock()
    def _job(pair):
        row = get_row(data, pair[0], pair[1], get_middle_city=args.get_middle_city)
        row_df = pd.DataFrame([row], columns=columns)
        with lock:
            row_df.to_csv(args.output_file, mode="a", index=False, header=False)
        
    if args.no_mp:
        for pair in tqdm(pairs):
            _job(pair)
    else:
        with ProcessPoolExecutor(max_workers=24) as executor:
            executor.map(_job, pairs)