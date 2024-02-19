import argparse

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

from model import Transformer
from dataset import CoordinateDataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='Path to the model', required=True)
    parser.add_argument('--dataset', type=str, help='Path to the dataset CSV file')
    args = parser.parse_args()

    coordinate_dataset = CoordinateDataset(args.dataset, city_count=1000)
    coordinates = [d[1] for d in coordinate_dataset]

    # TODO: Load the parameters too
    # TODO: Why 987 and not 1000?
    model = Transformer(d_model=32, nhead=2, d_hid=128, nlayers=1, dropout=0.0, ntoken=987)
    model.load_state_dict(torch.load(args.model_path))

    embeddings = None
    for param in model.embedding.parameters():
        embeddings = param.detach().numpy()
        break

    X_train, X_test, y_train, y_test = train_test_split(embeddings, coordinates, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Mean squared error:", np.mean((y_pred - y_test) ** 2))

    print("Example predictions:")
    for i in range(5):
        prediction = model.predict([X_test[i]])
        print(f"Prediction {i+1}: {prediction}, Actual: {y_test[i]}")
