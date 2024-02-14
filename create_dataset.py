from itertools import combinations
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from geopy import distance
from geographiclib.geodesic import Geodesic


def get_distance_to_line_position(point, line, position):
    p = line.Position(position, Geodesic.STANDARD | Geodesic.LONG_UNROLL)
    distance_to_line = Geodesic.WGS84.InverseLine(point[0], point[1], p['lat2'], p['lon2'])
    return distance_to_line.s13

def binary_search_lowest_distance(point, geodesic_line):
    low = 0
    high = geodesic_line.s13
    precision = 1e-6

    while high - low > precision:
        mid = (low + high) / 2
        mid_distance = get_distance_to_line_position(point, geodesic_line, mid)
        left_distance = get_distance_to_line_position(point, geodesic_line, mid - precision)
        right_distance = get_distance_to_line_position(point, geodesic_line, mid + precision)

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
    geodesic_line = Geodesic.WGS84.InverseLine(city_A[0], city_A[1], city_B[0], city_B[1])
    distances = []

    for coordinates in city_coordinates:
        dist = distance_point_to_geodesic(coordinates, geodesic_line)
        distances.append(dist)

    return distances

def get_city_coordinates(city):
    return tuple(df[df['Capital City'] == city][['Latitude', 'Longitude']].values[0])

def get_nearest_city(city_A, city_B):
    city_names = df['Capital City'].tolist()
    city_names.remove(city_A)
    city_names.remove(city_B)
    city_coordinates = [get_city_coordinates(city) for city in city_names]
    distances = get_city_distances(city_coordinates, get_city_coordinates(city_A), get_city_coordinates(city_B))

    paired = sorted(zip(distances, city_names))
    nearest_city = paired[0][1]

    return nearest_city

if __name__ == "__main__":
    url = "https://gist.githubusercontent.com/ofou/df09a6834a8421b4f376c875194915c9/raw/355eb56e164ddc3cd1a9467c524422cb674e71a9/country-capital-lat-long-population.csv"
    df = pd.read_csv(url)

    WORKERS = 24
    RUN_TIME = 3

    cities = df['Capital City'][:10].tolist()
    pairs = list(combinations(cities, 2))
    print(f'''
    Getting nearest cities for {len(pairs)} city pairs.
    Each pair takes around {RUN_TIME} seconds to process, and you are using {WORKERS} workers.

    This will take around {(len(pairs) // WORKERS + 1)* RUN_TIME } seconds to complete.
    ''')

    def get_nearest_city_pair(pair):
        city_A, city_B = pair
        return (city_A, city_B, get_nearest_city(city_A, city_B))

    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        nearest_cities = list(executor.map(get_nearest_city_pair, pairs))

    nearest_cities_df = pd.DataFrame(nearest_cities, columns=['City A', 'City B', 'Nearest City'])
    nearest_cities_df.to_csv('nearest_cities.csv', index=False)