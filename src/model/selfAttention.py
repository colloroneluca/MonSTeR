import torch.nn as nn
import time
import torch


class selfAttention(nn.Module):
    def __init__(self, in_features, out_features, n_heads, n_layer, batch_norm=False):
        super(selfAttention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.n_layer = n_layer
        self.feature_decrement = (in_features - out_features) // (n_layer - 1)
        self.batch_norm = batch_norm

        self.attention_layers = self.build_attention_layers()
        self.fc = nn.Linear(
            int(in_features / 2), int(out_features / 2)
        )  # Adjusting for the final output shape

    def build_attention_layers(self):
        layers = []
        current_in_features = int(self.in_features / 4)

        for i in range(self.n_layer):
            current_out_features = int(self.in_features / 4) - (
                i * int(self.feature_decrement / 4)
            )
            if i == self.n_layer - 1:
                current_out_features = int(self.out_features / 2)

            layers.append(
                nn.MultiheadAttention(
                    embed_dim=current_in_features,
                    num_heads=self.n_heads,
                    batch_first=True,
                )
            )

            if self.batch_norm and i != self.n_layer - 1:
                layers.append(nn.BatchNorm1d(current_in_features))

            layers.append(nn.ReLU())

            current_in_features = current_out_features

        return nn.ModuleList(layers)

    def forward(self, x):
        # x is of shape (batch_size, 4, in_features/4)
        x = torch.cat(x, dim=1)

        cls_tok = torch.zeros(x.shape)[:, 0:2, :].to(x.device)
        x = torch.cat((cls_tok, x), dim=1)
        for i, layer in enumerate(self.attention_layers):
            if isinstance(layer, nn.MultiheadAttention):
                attn_output, _ = layer(x, x, x)
                x = attn_output
            elif isinstance(layer, nn.BatchNorm1d):
                x = layer(x.transpose(1, 2)).transpose(1, 2)
            elif isinstance(layer, nn.ReLU):
                x = layer(x)

        x = x[:, :2, ...]
        return x
