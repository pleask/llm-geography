import argparse
import os

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split 
from tqdm import tqdm
import wandb
from dataset import CityDistanceDataset, CoordinateDistanceDataset, MiddleCityDataset

from model import Transformer

# TODO: Lots of if statements for middle_city - find a better way to handle this
if __name__ == "__main__":
    wandb.init(project="geo")

    parser = argparse.ArgumentParser(description='Train model on geography dataset')
    parser.add_argument('--dataset', type=str, help='Path to the dataset CSV file')
    parser.add_argument('--coordinate_distance', action='store_true', default=False, help='Use the coordinates as the covariates and the distance as the target.')
    parser.add_argument('--city_distance', action='store_true', default=False, help='Use the city names as the covariates and the distance as the target.')
    parser.add_argument('--middle_city', action='store_true', default=False, help='Use the city names as the covariates and the middle city as the target.')
    parser.add_argument('--model_path', type=str, help='Where to save the model.', required=True)
    parser.add_argument("--print_examples", action="store_true", default=False, help="Print out examples of the model's predictions.")

    args = parser.parse_args()

    assert args.coordinate_distance or args.city_distance or args.middle_city, "Please specify the type of dataset you are using."

    if args.coordinate_distance:
        dataset = CoordinateDistanceDataset(args.dataset)
    elif args.city_distance:
        dataset = CityDistanceDataset(args.dataset)
    elif args.middle_city:
        dataset = MiddleCityDataset(args.dataset)

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
    LR = 1e-3
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

    model = Transformer(
        d_model=D_MODEL,
        nhead=NHEAD,
        d_hid=D_HID,
        nlayers=NLAYERS,
        dropout=DROPOUT,
        ntoken=dataset.city_count,
        regressor=not args.middle_city,
    ).to(device)

    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path))
        print("Loaded model parameters from existing model at", args.model_path)

    total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters in the model:", total_params)

    if args.middle_city:
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LR)

    def _print_examples(outputs, y, train=True):
        if not args.print_examples:
            return
        if train:
            print('Train Examples')
        else:
            print('Val Examples')
        for _o, _y in zip(outputs[:3], y[:3]):
            if args.middle_city:
                print(torch.argmax(_o).item(), _y.item())
            else:
                print("{:.3f}".format(_o.item()), "{:.3f}".format(_y.item()))

    def _process_batch(x, y):
        if args.coordinate_distance:
            x = [x[0].to(device), x[1].to(torch.float32).to(device)]
        else:
            x = torch.stack([x[0], x[1]], dim=1).to(device)
        
        # TODO: Why is this necessary?
        if args.middle_city:
            y = y[0].to(device)
        else:
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

        torch.save(model.state_dict(), args.model_path)
