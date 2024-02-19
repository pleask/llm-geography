import math

import torch
from torch import Tensor, nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class ClassificationHead(nn.Module):
    def __init__(self, d_model: int, ntoken: int):
        super().__init__()
        self.linear = nn.Linear(d_model, ntoken)

        self.init_weights()

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)

    def init_weights(self) -> None:
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)


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


class TransformerInner(nn.Module):
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


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.5,
        ntoken=-1,
        regressor=True,
    ):
        super().__init__()

        if ntoken == -1:
            self.embedding = nn.Linear(2, d_model)
        else:
            self.embedding = nn.Embedding(ntoken, d_model)

        self.transformer = TransformerInner(d_model, nhead, d_hid, nlayers, dropout)
        if regressor:
            self.regression_head = RegressionHead(d_model)
        else:
            self.regression_head = ClassificationHead(d_model, ntoken)

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