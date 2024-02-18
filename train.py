from abc import abstractmethod, ABC
import argparse
import math 

import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn, Tensor
from torch.optim import Adam
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader, random_split 
from tqdm import tqdm
import wandb


class BaseDataset(ABC, Dataset):
    def __init__(self, csv_file, city_count=-1):
        self.df = pd.read_csv(csv_file)
        if city_count > 0:
            self.df = self.df[:city_count]

        self._normalise_columns()

    def _normalise_column(self, column_name):
        scaler = StandardScaler()
        self.df[column_name] = scaler.fit_transform(self.df[[column_name]])

    def __len__(self):
        return len(self.df)

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def _normalise_columns(self):
        pass


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
            self._normalise_column(column_name)


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


class CityDistanceDataset(BaseDataset):
    def __init__(self, csv_file, city_count=-1):
        super().__init__(csv_file, city_count)
        cities_a = self.df["City A"].unique()
        cities_b = self.df["City B"].unique()
        combined_cities = list(set(cities_a) | set(cities_b))
        self.city_to_int = {city: i for i, city in enumerate(combined_cities)}

    def _normalise_columns(self):
        self._normalise_column("Distance")

    def __getitem__(self, idx):
        x = [
            self.city_to_int[self.df.loc[idx, "City A"]],
            self.city_to_int[self.df.loc[idx, "City B"]],
        ]
        y = self.df.loc[idx, "Distance"]
        return x, y



class RegressionHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)

        self.init_weights()

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)

    def init_weights(self) -> None:
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = torch.mean(output, dim=1)
        return output


class RegressionTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.5,
        ntoken=-1
    ):
        super().__init__()

        if ntoken == -1:
            self.embedding = nn.Linear(2, d_model)
        else:
            self.embedding = nn.Embedding(ntoken, d_model)

        self.transformer = Transformer(d_model, nhead, d_hid, nlayers, dropout)
        self.regression_head = RegressionHead(d_model)
        self.d_model = d_model

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor) -> Tensor:
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.transformer(src)
        output = self.regression_head(src)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


if __name__ == "__main__":
    wandb.init(project="geo")

    parser = argparse.ArgumentParser(description='Train model on geography dataset')
    parser.add_argument('--dataset', type=str, help='Path to the dataset CSV file')
    parser.add_argument('--use_coordinates', type=str, help='Whether or not to use the coordinates, otherwise just uses city names as covariates.')
    args = parser.parse_args()

    if args.use_coordinates:
        dataset = CoordinateDistanceDataset(args.dataset)
    else:
        dataset = CityDistanceDataset(args.dataset)

    train_count = int(len(dataset) * 0.8)
    val_count = len(dataset) - train_count
    train_dataset, val_dataset = random_split(dataset, [train_count, val_count])

    EPOCHS = 500
    BATCH_SIZE = 2**10

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Train dataset size:", len(train_dataset))
    print("Validation dataset size:", len(val_dataset))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D_MODEL = 32
    NHEAD = 2
    D_HID = 128
    NLAYERS = 1
    DROPOUT = 0.
    LR = 1e-4
    L1_LAMBDA = 0. #1e-5

    wandb.config.update(
        {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "d_model": D_MODEL,
            "nhead": NHEAD,
            "d_hid": D_HID,
            "nlayers": NLAYERS,
            "dropout": DROPOUT,
            "lr": LR,
            "l1_lambda": L1_LAMBDA,
        }
    )

    model = RegressionTransformer(
        d_model=D_MODEL,
        nhead=NHEAD,
        d_hid=D_HID,
        nlayers=NLAYERS,
        dropout=DROPOUT,
        ntoken=len(dataset.city_to_int)
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters in the model:", total_params)

    loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LR)

    LOG_EXAMPLES = True
    def _print_examples(outputs, y, train=True):
        if not LOG_EXAMPLES:
            return
        if train:
            print('Train Examples')
        else:
            print('Val Examples')
        for _o, _y in zip(outputs[:3], y[:3]):
            print("{:.3f}".format(_o.item()), "{:.3f}".format(_y.item()))

    def _process_batch(x, y):
        if args.coordinates:
            x = [x[0].to(device), x[1].to(torch.float32).to(device)]
        else:
            x = torch.stack([x[0], x[1]], dim=1).to(device)
        y = y.to(torch.float32).to(device).unsqueeze(1)
        return x, y

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        total_loss = 0
        total_loss_with_l1 = 0
        for i, (x, y) in enumerate(train_dataloader):
            x, y = _process_batch(x, y)
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs, y)

            # Log the loss without the l1 norm added for better comparison between model sizes
            total_loss += loss.item()

            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss += L1_LAMBDA * l1_norm

            total_loss_with_l1 += loss.item()

            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_dataloader)
        avg_train_loss_with_l1 = total_loss_with_l1 / len(train_dataloader)
        wandb.log({"train_loss": avg_train_loss, "train_loss_with_l1": avg_train_loss_with_l1})

        _print_examples(outputs, y, train=True)

        model.eval()
        with torch.no_grad():
            total_loss = 0
            for i, (x, y) in enumerate(val_dataloader):
                x, y = _process_batch(x, y)
                outputs = model(x)
                loss = loss_fn(outputs, y)
                total_loss += loss.item()

            avg_val_loss = total_loss / len(val_dataloader)

            _print_examples(outputs, y, train=False)
        wandb.log({"val_loss": avg_val_loss})

        tqdm.write(
            f"Epoch {epoch}: Validation loss {avg_val_loss:.3f}, train loss {avg_train_loss:.3f}, train loss with l1 {avg_train_loss_with_l1:.3f}"
        )