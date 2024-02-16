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


class CoordinateDistanceDataset(Dataset):
    """
    X values are the normalised coordinates of the city, and the y value is the
    distance between the two cities.
    """

    def __init__(self, csv_file, city_count=-1):
        self.df = pd.read_csv(csv_file)
        if city_count > 0:
            self.df = self.df[:city_count]

        for column_name in [
            "City A Latitude",
            "City A Longitude",
            "City B Latitude",
            "City B Longitude",
            "Distance",
        ]:
            self._normalise_column(column_name)

    def _normalise_column(self, column_name):
        scaler = StandardScaler()
        self.df[column_name] = scaler.fit_transform(self.df[[column_name]])

    def __len__(self):
        return len(self.df)

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


class TransformerModel(nn.Module):
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
        self.embedding = nn.Linear(2, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, 1)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = src.to(torch.float32)
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = torch.mean(output, dim=1)
        output = self.linear(output)
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
    dataset = CoordinateDistanceDataset("data/output.csv")
    print(f'Using {len(dataset)} samples')
    train_count = int(len(dataset) * 0.8)
    val_count = len(dataset) - train_count
    train_dataset, val_dataset = random_split(dataset, [train_count, val_count])

    EPOCHS = 50
    BATCH_SIZE = 4096

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    D_MODEL = 256
    NHEAD = 2
    D_HID = 1024
    NLAYERS = 1
    DROPOUT = 0.
    LR = 1e-6
    L1_LAMBDA = 0.00

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

    model = TransformerModel(
        d_model=D_MODEL,
        nhead=NHEAD,
        d_hid=D_HID,
        nlayers=NLAYERS,
        dropout=DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters in the model:", total_params)

    loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LR)

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        total_loss = 0
        for i, (x, y) in enumerate(train_dataloader):
            x = torch.stack([x[0], x[1]], dim=1).to(torch.float32).to(device)
            y = y.to(torch.float32).to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs, y)

            # Log the loss without the l1 norm added for better comparison between model sizes
            total_loss += loss.item()

            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss += L1_LAMBDA * l1_norm

            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_dataloader)
        wandb.log({"train_loss": avg_train_loss})

        model.eval()
        with torch.no_grad():
            total_loss = 0
            for i, (x, y) in enumerate(val_dataloader):
                x = torch.stack([x[0].to(torch.int32), x[1].to(torch.int32)], dim=1).to(device)
                y = y.to(torch.float32).to(device).unsqueeze(1)
                outputs = model(x)
                loss = loss_fn(outputs, y)
                total_loss += loss.item()

            avg_val_loss = total_loss / len(val_dataloader)
        wandb.log({"val_loss": avg_val_loss})
        wandb.log({'example': torch.abs(y[0] - outputs[0])})

        tqdm.write(
            f"Epoch {epoch}: Validation loss {avg_val_loss:.3f}, train loss {avg_train_loss:.3f}"
        )

    model.eval()
    print("TRAIN")
    for i, (x, y) in enumerate(train_dataloader):
        x = torch.stack([x[0].to(torch.int32), x[1].to(torch.int32)], dim=1).to(device)
        y = y.to(torch.float32).to(device).unsqueeze(1)
        outputs = model(x)
        break
    for (p, y) in zip(outputs[:10], y[:10]):
        print("{:.3f}".format(p.item()), "{:.3f}".format(y.item()), "{:.3f}".format(torch.abs(p - y).item()))

    print("VAL")
    for i, (x, y) in enumerate(val_dataloader):
        x = torch.stack([x[0].to(torch.int32), x[1].to(torch.int32)], dim=1).to(device)
        y = y.to(torch.float32).to(device).unsqueeze(1)
        outputs = model(x)
        break
    for (p, y) in zip(outputs[:10], y[:10]):
        print("{:.3f}".format(p.item()), "{:.3f}".format(y.item()), "{:.3f}".format(torch.abs(p - y).item()))