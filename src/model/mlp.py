import torch.nn as nn
import time
import torch
import logging

logger = logging.getLogger(__name__)


class mlp(nn.Module):
    def __init__(
        self, in_features, out_features, n_layer, batch_norm=False, activation="relu"
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_layer = n_layer
        self.batch_norm = batch_norm
        self.feature_decrement = (in_features - out_features) // (n_layer - 1)
        self.activation = nn.GELU if activation == "gelu" else nn.ReLU
        self.build_mlp()

    def build_mlp(self):
        layers = []
        current_in_features = self.in_features
        for i in range(self.n_layer):
            current_out_features = self.in_features - (i * self.feature_decrement)
            if i == self.n_layer - 1:
                current_out_features = self.out_features
            layers.append(nn.Linear(current_in_features, current_out_features))
            if self.batch_norm and i != self.n_layer - 1:
                layers.append(nn.BatchNorm1d(current_out_features))

                layers.append(self.activation())  # Adding a ReLU activation function
            current_in_features = current_out_features
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.cat(x, dim=-1).reshape((x[0].shape[0], -1))
        return self.model(x).reshape((x.shape[0], 2, -1))
