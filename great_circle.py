import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader

from model import Transformer
from dataset import CityDistanceDataset, CoordinateDataset


def get_activation(layer):
    def hook(module, input, output):
        module.activation = output.detach()
    return hook


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='Path to the model', required=True)
    parser.add_argument('--dataset', type=str, help='Path to the dataset CSV file')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    city_distance_dataset = CityDistanceDataset(args.dataset)
    coordinate_dataset = CoordinateDataset(args.dataset, normalise=False)

    city_latitudes = [coordinate_dataset[city][1][0] for city in range(len(coordinate_dataset))]
    sine_latitudes = []
    for i in range(len(city_distance_dataset)):
        city = city_distance_dataset[i][0][0]
        latitude = city_latitudes[city]
        sine_latitude = math.sin(math.radians(latitude))
        sine_latitudes.append(sine_latitude)

    # TODO: Load the parameters too
    # TODO: Why 987 and not 1000?
    model = Transformer(d_model=32, nhead=2, d_hid=128, nlayers=1, dropout=0.0, ntoken=987).to(device)
    model.load_state_dict(torch.load(args.model_path))

    hook = model.transformer.transformer_encoder.register_forward_hook(get_activation('encoder'))

    dataloader = DataLoader(city_distance_dataset, batch_size=4096, shuffle=False)
    activations = []
    for i, (x, y) in enumerate(dataloader):
        x = torch.stack([x[0], x[1]], dim=1).to(device)
        model(x)
        activations.append(model.transformer.transformer_encoder.activation)
    activations = torch.cat(activations)
    activations = activations.reshape(len(activations), -1)

    X_train, X_test, y_train, y_test = train_test_split(activations.detach().cpu().numpy(), sine_latitudes, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Mean squared error:", np.mean((y_pred - y_test) ** 2))

    print("Example predictions:")
    for i in range(5):
        prediction = model.predict([X_test[i]])
        print(f"Prediction {i+1}: {prediction}, Actual: {y_test[i]}")

    # Plot predictions against true values
    plt.scatter(y_test, y_pred)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Predictions vs True Values')
    plt.savefig('predictions_plot.png')
    plt.show()