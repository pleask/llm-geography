import os
import argparse

import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split 
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tokenizers import Tokenizer, CharBPETokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm
import wandb

from dataset import CityDistanceDataset, CoordinateDataset
from model import Transformer


class _CityTokenizerReturnValue:
    def __init__(self, ids):
        self.ids = ids
        self.attention_mask = [1 for _ in ids]


class CityTokenizer:
    """
    Super hacky implementation of a tokenizer that only tokenizes city names.
    """
    def __init__(self, cities):
        self.cities = cities
        self.city_to_int = {city:i for i, city in enumerate(cities)}

    def encode(self, city):
        cities = city.split(', ')
        return _CityTokenizerReturnValue([self.city_to_int[cities[0]], self.city_to_int[cities[1]]])
    
    def encode_batch(self, cities):
        return [self.encode(city) for city in cities]

    def save(self, _):
        # TODO: Make this act more like a hf tokenizer
        pass

    def get_vocab(self):
        return self.city_to_int.keys()


def _get_tokenizer(cities, tokenizer_path, tokenize=False):
    """
    If the tokenizer already exists, load it. Otherwise, train a new one.

    If tokenize is False, a dummy tokenizer is used that uses each city name as a token. Otherwise, a WordPiece tokenizer is trained.
    """
    if not tokenize:
        return CityTokenizer(cities)

    if os.path.exists(tokenizer_path):
        tokenizer = Tokenizer.from_file(tokenizer_path)
        return tokenizer

    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.train_from_iterator(cities, trainer)
    tokenizer.save(tokenizer_path)
    return tokenizer


def tokenize_batch(tokenizer, x, device='cpu'):
    output = tokenizer.encode_batch(x)
    input_ids = [torch.tensor(enc.ids) for enc in output]
    attention_mask = [torch.tensor(enc.attention_mask) for enc in output]
    input_ids = pad_sequence(input_ids, batch_first=True).to(device)
    attention_mask = pad_sequence(attention_mask, batch_first=True).to(device)
    return input_ids, attention_mask


def get_or_train_coordinate_model(data_dir, model_path, continue_training=False, tokenize=False, epochs=500):
    cities = [s[0] for s in CoordinateDataset(data_dir)]

    text = cities
    if tokenize:
        df = pd.read_csv('raw_data/all_cities.csv')
        text = df['city'].tolist()
    tokenizer = _get_tokenizer(text, model_path + '.tokenizer', tokenize=tokenize)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    distance_dataset = CityDistanceDataset(data_dir)

    train_count = int(len(distance_dataset) * 0.8)
    val_count = len(distance_dataset) - train_count
    train_dataset, val_dataset = random_split(distance_dataset, [train_count, val_count])

    BATCH_SIZE = 2**10
    LR = 1e-3
    if not tokenize:
        D_MODEL=32
        NHEAD=4
        D_HID=32
        NLAYERS=1
    else:
        D_MODEL=128
        NHEAD=4
        D_HID=128
        NLAYERS=2

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = Transformer(
        d_model=D_MODEL,
        nhead=NHEAD,
        d_hid=D_HID,
        nlayers=NLAYERS,
        ntoken=len(tokenizer.get_vocab()),
        regressor=True,
    ).to(device)

    if os.path.exists(model_path):
        params = torch.load(model_path)
        model.load_state_dict(params)
        print("Loaded model parameters from existing model at", model_path)
    
    if not continue_training:
        return model, tokenizer

    tags = ['tokenised'] if tokenize else []
    wandb.init(project='geo', tags=tags, config={
        'epochs': epochs,
        'batch_size': BATCH_SIZE,
        'lr': LR,
        'd_model': D_MODEL,
        'nhead': NHEAD,
        'd_hid': D_HID,
        'nlayers': NLAYERS,
    })

    loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LR)
    log_interval = 100

    for epoch in tqdm(range(epochs), desc='Epochs'):
        model.train()
        train_loss = 0
        for batch, (x, y) in enumerate(tqdm(train_dataloader, desc='Training batches', leave=False)):
            input_ids, attention_mask = tokenize_batch(tokenizer, x, device)
            y = y.to(torch.float32).to(device)
            y_pred = model(input_ids, attention_mask).squeeze(1)
            loss = loss_fn(y_pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
            if batch % log_interval == 0:
                wandb.log({'train_loss': loss.item()})
        train_loss /= len(train_dataloader)

        model.eval()
        val_loss = 0
        for x, y in tqdm(val_dataloader, desc='Validation batches', leave=False):
            input_ids, attention_mask = tokenize_batch(tokenizer, x, device)
            y = y.to(torch.float32).to(device)
            y_pred = model(input_ids, attention_mask).squeeze(1)
            val_loss += loss_fn(y_pred, y).item()
        val_loss /= len(val_dataloader)

        wandb.log({'train_loss': train_loss, 'val_loss': val_loss})
        tqdm.write(f"Epoch {epoch + 1}/{epochs} | Train loss: {train_loss:,.4f} | Val loss: {val_loss:,.4f}")

        torch.save(model.state_dict(), model_path)

    return model, tokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or load a coordinate model.')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the data.')
    parser.add_argument('--model_path', type=str, required=True, help='Model location.')
    parser.add_argument('--continue_training', type=bool, default=False, help='Continue training from the last checkpoint.')
    parser.add_argument('--tokenize', type=bool, default=False, help='Use a word piece tokenizer rather than the city names.')
    parser.add_argument('--epochs', type=int, help='Epochs to train for.')
    args = parser.parse_args()

    get_or_train_coordinate_model(args.data_dir, args.model_path, continue_training=args.continue_training, epochs=args.epochs, tokenize=args.tokenize)