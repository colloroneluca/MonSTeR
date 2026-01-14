from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

from einops import repeat


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=False) -> None:
        super().__init__()
        self.batch_first = batch_first

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if self.batch_first:
            x = x + self.pe.permute(1, 0, 2)[:, : x.shape[1], :]
        else:
            x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)
    
import torch.nn.functional as F
class AttentionUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionUpsample, self).__init__()
        seq_trans_decoder_layer = nn.TransformerDecoderLayer(
            d_model=256,
            nhead=8,
            activation='gelu',
            batch_first=True,
        )
        self.seqTransDecoder = nn.TransformerDecoder(
            seq_trans_decoder_layer, num_layers=2
        )
        self.sequence_pos_encoding = PositionalEncoding(
            256, dropout=0.1, batch_first=True
        )
        self.gelu = torch.nn.GELU()

    def forward(self, x, tgt_key_padding_mask):
        x = self.gelu(x) #it seems that the transformer before this fuction has no activation at the end 
        time_queries = torch.zeros(x.shape[0], 256, x.shape[2], device=x.device)
        time_queries = self.sequence_pos_encoding(time_queries)

        output = self.seqTransDecoder(
            tgt=time_queries, memory=x,
        )
        
        return output

class ACTORStyleTextEncoder(nn.Module):
    # Similar to ACTOR but "action agnostic" and more general
    def __init__(
        self,
        nfeats: int,
        vae: bool,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
        operator: bool =  False,
        ntok: int = None,
        latent_dim_token = 256,
    ) -> None:
        super().__init__()
        
        self.nfeats = nfeats
        #self.projection = nn.Linear(nfeats, latent_dim)
        self.tok_projection = nn.Linear(latent_dim, latent_dim_token)
        self.vae = vae
        if ntok is not None:
            self.nbtokens = ntok
        else:
            self.nbtokens = 2 if vae else 1
        self.tokens = nn.Parameter(torch.randn(self.nbtokens, latent_dim))

        self.sequence_pos_encoding = PositionalEncoding(
            latent_dim, dropout=dropout, batch_first=True
        )

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )

        self.seqTransEncoder = nn.TransformerEncoder(
            seq_trans_encoder_layer, num_layers=num_layers
        )
        
            
    def forward(self, x_dict: Dict) -> Tensor: #text
        x = x_dict["x"] #([16, 102, 768])
        mask = x_dict["mask"]

        #x = self.projection(x) #torch.Size([16, 102, 768])

        device = x.device
        bs = len(x)

        tokens = repeat(self.tokens, "nbtoken dim -> bs nbtoken dim", bs=bs) #torch.Size([16, 2, 256])
        xseq = torch.cat((tokens, x), 1) #torch.Size([16, 104, 256])

        token_mask = torch.ones((bs, self.nbtokens), dtype=bool, device=device) #
        aug_mask = torch.cat((token_mask, mask), 1)

        # add positional encoding
        xseq = self.sequence_pos_encoding(xseq) #torch.Size([16, 104, 256])
        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask) #torch.Size([16, 104, 256])
        cls_tokens =self.tok_projection(final[:, : self.nbtokens])
        sent_tokens = final[:, self.nbtokens:]
        return cls_tokens, sent_tokens , None #torch.Size([16, 2, 256])
    

class ACTORStyleMotionEncoder(nn.Module):
    # Similar to ACTOR but "action agnostic" and more general
    def __init__(
        self,
        nfeats: int,
        vae: bool,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
        operator: bool =  False,
        ntok: int = None,
        latent_dim_token = 256,
    ) -> None:
        super().__init__()
        self.nfeats = nfeats
        self.projection = nn.Linear(nfeats, 768) #harcoded
        self.tok_projection = nn.Linear(latent_dim, latent_dim_token)
        self.vae = vae
        if ntok is not None:
            self.nbtokens = ntok
        else:
            self.nbtokens = 2 if vae else 1
        self.tokens = nn.Parameter(torch.randn(self.nbtokens, latent_dim))

        self.sequence_pos_encoding = PositionalEncoding(
            latent_dim, dropout=dropout, batch_first=True
        )

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )

        self.seqTransEncoder = nn.TransformerEncoder(
            seq_trans_encoder_layer, num_layers=num_layers
        )
        
            
    def forward(self, x_dict: Dict) -> Tensor: #text
        x = x_dict["x"] #([16, 102, 768])
        x = self.projection(x)
        mask = x_dict["mask"]

        #x = self.projection(x) #torch.Size([16, 102, 768])

        device = x.device
        bs = len(x)

        tokens = repeat(self.tokens, "nbtoken dim -> bs nbtoken dim", bs=bs) #torch.Size([16, 2, 256])
        xseq = torch.cat((tokens, x), 1) #torch.Size([16, 104, 256])

        token_mask = torch.ones((bs, self.nbtokens), dtype=bool, device=device) #
        aug_mask = torch.cat((token_mask, mask), 1)

        # add positional encoding
        xseq = self.sequence_pos_encoding(xseq) #torch.Size([16, 104, 256])
        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask) #torch.Size([16, 104, 256])
        cls_tokens =self.tok_projection(final[:, : self.nbtokens])
        sent_tokens = final[:, self.nbtokens:]
        return cls_tokens, sent_tokens , None #torch.Size([16, 2, 256])


class ACTORStyleDecoder(nn.Module):
    # Similar to ACTOR Decoder

    def __init__(
        self,
        nfeats: int,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        output_feats = nfeats
        self.nfeats = nfeats

        self.sequence_pos_encoding = PositionalEncoding(
            latent_dim, dropout, batch_first=True
        )

        seq_trans_decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )

        self.seqTransDecoder = nn.TransformerDecoder(
            seq_trans_decoder_layer, num_layers=num_layers
        )

        self.final_layer = nn.Linear(latent_dim, output_feats)

    def forward(self, z_dict: Dict) -> Tensor:
        z = z_dict["z"]
        mask = z_dict["mask"]

        latent_dim = z.shape[1]
        bs, nframes = mask.shape

        z = z[:, None]  # sequence of 1 element for the memory

        # Construct time queries
        time_queries = torch.zeros(bs, nframes, latent_dim, device=z.device)
        time_queries = self.sequence_pos_encoding(time_queries)

        # Pass through the transformer decoder
        # with the latent vector for memory
        output = self.seqTransDecoder(
            tgt=time_queries, memory=z, tgt_key_padding_mask=~mask
        )

        output = self.final_layer(output)
        # zero for padded area
        output[~mask] = 0
        return output
