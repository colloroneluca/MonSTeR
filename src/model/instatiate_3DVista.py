"""
Module for instantiating and managing 3DVista scene encoders.

This module provides classes and utilities for encoding 3D scene information
using the 3DVista framework, including point cloud processing, spatial reasoning,
and attention-based feature fusion.

WARNING: Hardcoded Paths in 3DVista
====================================
This module contains hardcoded paths to the 3DVista external component. While we are
aware of this design choice, it was implemented this way due to the structure of the
original 3DVista codebase and has proven to be the most practical approach for importing
and utilizing the necessary components.

Note: Given that this implementation closely follows the original 3DVista design patterns,
we may not be able to support significant rewrites or refactoring of this code, as such
modifications fall outside the scope of MonSTeR's core implementation and may introduce
compatibility issues with the external 3DVista framework.
"""

import os
import sys
import json
import copy
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import einops
from einops import repeat

three_d_vista_path = os.path.abspath("./src/external_comp/ThreeDVista")
sys.path.append(three_d_vista_path)
params_path = os.path.join(
    three_d_vista_path,
    "data/scanfamily/annotations/meta_data/MonSTeR_scene_enc_params.json",
)
from model.vision.point_encoder import PointTokenizeEncoder
from model.vision.unified_encoder import UnifiedSpatialCrossEncoderV2


params = json.load(open(params_path, "r"))
pretrained = params["pretrained"]
unified_enc_params = params["unified_enc_params"]
point_enc_params = params["point_enc_params"]
hidden_size = point_enc_params["hidden_size"]
mixup_strategy = point_enc_params["mixup_strategy"]
mixup_stage1 = point_enc_params["mixup_stage1"]
mixup_stage2 = point_enc_params["mixup_stage2"]
pairwise_rel_type = point_enc_params["pairwise_rel_type"]
spatial_dim = point_enc_params["spatial_dim"]
num_attention_heads = point_enc_params["num_attention_heads"]
num_layers = point_enc_params["num_layers"]
dim_loc = point_enc_params["dim_loc"]
projection_dim = params["projection_dim"]
n_feats = params["n_feats"]


def get_mlp_head(input_size, hidden_size, output_size, dropout=0):
    return nn.Sequential(
        nn.Linear(input_size, hidden_size // 2),
        nn.ReLU(),
        nn.LayerNorm(hidden_size // 2, eps=1e-12),
        nn.Dropout(dropout),
        nn.Linear(hidden_size // 2, output_size),
    )


def calc_pairwise_locs(
    obj_centers,
    obj_whls,
    eps=1e-10,
    pairwise_rel_type="center",
    spatial_dist_norm=True,
    spatial_dim=5,
):
    if pairwise_rel_type == "mlp":
        obj_locs = torch.cat([obj_centers, obj_whls], 2)
        pairwise_locs = torch.cat(
            [
                einops.repeat(obj_locs, "b l d -> b l x d", x=obj_locs.size(1)),
                einops.repeat(obj_locs, "b l d -> b x l d", x=obj_locs.size(1)),
            ],
            dim=3,
        )
        return pairwise_locs

    pairwise_locs = einops.repeat(obj_centers, "b l d -> b l 1 d") - einops.repeat(
        obj_centers, "b l d -> b 1 l d"
    )
    pairwise_dists = torch.sqrt(torch.sum(pairwise_locs**2, 3) + eps)  # (b, l, l)
    if spatial_dist_norm:
        max_dists = torch.max(pairwise_dists.view(pairwise_dists.size(0), -1), dim=1)[0]
        norm_pairwise_dists = pairwise_dists / einops.repeat(max_dists, "b -> b 1 1")
    else:
        norm_pairwise_dists = pairwise_dists

    if spatial_dim == 1:
        return norm_pairwise_dists.unsqueeze(3)

    pairwise_dists_2d = torch.sqrt(torch.sum(pairwise_locs[..., :2] ** 2, 3) + eps)
    if pairwise_rel_type == "center":
        pairwise_locs = torch.stack(
            [
                norm_pairwise_dists,
                pairwise_locs[..., 2] / pairwise_dists,
                pairwise_dists_2d / pairwise_dists,
                pairwise_locs[..., 1] / pairwise_dists_2d,
                pairwise_locs[..., 0] / pairwise_dists_2d,
            ],
            dim=3,
        )
    elif pairwise_rel_type == "vertical_bottom":
        bottom_centers = torch.clone(obj_centers)
        bottom_centers[:, :, 2] -= obj_whls[:, :, 2]
        bottom_pairwise_locs = einops.repeat(
            bottom_centers, "b l d -> b l 1 d"
        ) - einops.repeat(bottom_centers, "b l d -> b 1 l d")
        bottom_pairwise_dists = torch.sqrt(
            torch.sum(bottom_pairwise_locs**2, 3) + eps
        )  # (b, l, l)
        bottom_pairwise_dists_2d = torch.sqrt(
            torch.sum(bottom_pairwise_locs[..., :2] ** 2, 3) + eps
        )
        pairwise_locs = torch.stack(
            [
                norm_pairwise_dists,
                bottom_pairwise_locs[..., 2] / bottom_pairwise_dists,
                bottom_pairwise_dists_2d / bottom_pairwise_dists,
                pairwise_locs[..., 1] / pairwise_dists_2d,
                pairwise_locs[..., 0] / pairwise_dists_2d,
            ],
            dim=3,
        )

    if spatial_dim == 4:
        pairwise_locs = pairwise_locs[..., 1:]
    return pairwise_locs


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


class SceneDecoder(nn.Module):
    def __init__(
        self, d_model=768, out_dim=1, nhead=2, dim_feedforward=2048, dropout=0.1
    ):
        super(SceneDecoder, self).__init__()
        self.num_points = 4000
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=2)
        self.projection = nn.Linear(d_model, self.num_points)

    def forward(self, fused, scene_embeds, fused_mask, scene_embeds_mask):
        """
        Apply the self-attention encoder layer to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, embedding_dim)

        Returns:
            torch.Tensor: Output tensor after self-attention encoding

        """
        # V, K = memory
        scene_embeds_mask = ~scene_embeds_mask
        fused_mask = ~fused_mask
        output = self.decoder(
            memory=fused,
            memory_key_padding_mask=fused_mask,
            tgt=scene_embeds,
            tgt_key_padding_mask=scene_embeds_mask,
        )
        return self.projection(output)


class SelfAttentionEncoder(nn.Module):
    def __init__(
        self, d_model=768, out_dim=256, nhead=2, dim_feedforward=2048, dropout=0.1
    ):
        super(SelfAttentionEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.sequence_pos_encoding = PositionalEncoding(
            d_model, dropout=dropout, batch_first=True
        )
        self.projection = nn.Linear(d_model, out_dim)
        self.token = nn.Parameter(torch.randn(2, 768))

    def forward(self, obj_embeds, obj_mask):
        """
        Apply the self-attention encoder layer to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, embedding_dim)

        Returns:
            torch.Tensor: Output tensor after self-attention encoding

        """
        bs = len(obj_embeds)
        tokens = repeat(self.token, "nbtoken dim -> bs nbtoken dim", bs=bs)
        scene = torch.cat((tokens, obj_embeds), 1)
        token_mask = torch.ones((bs, 2), dtype=bool, device=obj_embeds.device)
        scene_mask = ~torch.cat((token_mask, obj_mask), 1)
        scene = self.sequence_pos_encoding(scene)
        output = self.encoder(src=scene, src_key_padding_mask=scene_mask)
        return self.projection(output[:, :2, :])


class CrossAttentionEncoder(nn.Module):
    def __init__(
        self, d_model=768, out_dim=256, nhead=2, dim_feedforward=2048, dropout=0.1
    ):
        super(CrossAttentionEncoder, self).__init__()
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=2)
        self.sequence_pos_encoding = PositionalEncoding(
            d_model, dropout=dropout, batch_first=True
        )
        self.projection = nn.Linear(d_model, out_dim)
        self.token = nn.Parameter(torch.randn(2, 768))

    def forward(self, memory, memory_mask, tgt, tgt_mask):
        """
        Apply the self-attention encoder layer to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, embedding_dim)

        Returns:
            torch.Tensor: Output tensor after self-attention encoding
        """
        bs = len(memory)
        tokens = repeat(self.token, "nbtoken dim -> bs nbtoken dim", bs=bs)
        tgt = torch.cat((tokens, tgt), 1)
        token_mask = torch.ones((bs, 2), dtype=bool, device=memory.device)
        tgt_mask = ~torch.cat((token_mask, tgt_mask), 1)
        memory_mask = ~memory_mask
        tgt = self.sequence_pos_encoding(tgt)
        memory = self.sequence_pos_encoding(memory)
        output = self.decoder(
            tgt=tgt,
            tgt_key_padding_mask=tgt_mask,
            memory=memory,
            memory_key_padding_mask=memory_mask,
        )
        return self.projection(output[:, :2, :])


class ThreeDVistaScene(nn.Module):
    def __init__(self):
        super(ThreeDVistaScene, self).__init__()
        cwd = os.getcwd()
        os.chdir(three_d_vista_path)
        from ..external_comp.ThreeDVista.model.vision.basic_modules import (
            get_mixup_function,
        )
        from ..external_comp.ThreeDVista.dataset.path_config import SCAN_FAMILY_BASE

        self.int2cat = json.load(
            open(
                os.path.join(
                    SCAN_FAMILY_BASE,
                    "annotations/meta_data/scannetv2_raw_categories.json",
                ),
                "r",
            )
        )
        self.cat2int = {w: i for i, w in enumerate(self.int2cat)}
        self.cat2vec = json.load(
            open(
                os.path.join(
                    SCAN_FAMILY_BASE, "annotations/meta_data/cat2glove42b.json"
                ),
                "r",
            )
        )
        self.point_tokenize_encoder = PointTokenizeEncoder(**point_enc_params)
        self.point_feature_extractor = (
            self.point_tokenize_encoder.point_feature_extractor
        )
        self.point_cls_head = self.point_tokenize_encoder.point_cls_head
        self.sem_cls_embed_layer = self.point_tokenize_encoder.sem_cls_embed_layer
        self.sem_mask_embeddings = self.point_tokenize_encoder.sem_mask_embeddings
        self.pairwise_rel_type = pairwise_rel_type
        self.spatial_dim = spatial_dim
        self.spatial_encoder = self.point_tokenize_encoder.spatial_encoder
        self.self_attention = SelfAttentionEncoder()
        self.nfeats = n_feats
        self.dropout = nn.Dropout(0.1)
        self.counter = 0
        pth = torch.load("../../../" + pretrained)
        self.point_tokenize_encoder.load_state_dict(pth["point_encoder"])
        os.chdir(cwd)

    def forward(self, x):
        if "train_dl_len" not in x.keys():
            # this for eval
            cur_step = 1
            max_steps = 1
        else:
            train_dl_len = x["train_dl_len"]
            epoch = x["epoch"]
            cur_step = epoch * train_dl_len + self.counter
            max_steps = x["total_steps"]
        obj_pcds = x["obj_fts"].float()
        obj_labels = x["obj_labels"]
        obj_masks = x["obj_masks"]
        obj_sem_masks = x["obj_sem_masks"]
        obj_locs = x["obj_locs"].float()
        batch_size, num_objs, _, _ = obj_pcds.size()
        obj_embeds = self.point_feature_extractor(
            einops.rearrange(obj_pcds, "b o p d -> (b o) p d")
        )
        obj_embeds = einops.rearrange(obj_embeds, "(b o) d -> b o d", b=batch_size)
        obj_embeds = self.dropout(obj_embeds)
        obj_sem_cls = self.point_cls_head(obj_embeds)
        if int(obj_labels[0, 0]) != -1:
            obj_sem_cls_mix = self.point_tokenize_encoder.mixup_function(
                obj_sem_cls, obj_labels, cur_step, max_steps
            )
        else:
            obj_sem_cls_mix = obj_sem_cls.clone()
        obj_sem_cls_mix = torch.argmax(obj_sem_cls_mix, dim=2)
        obj_sem_cls_embeds = torch.Tensor(
            [
                self.cat2vec[self.int2cat[int(i)]]
                for i in obj_sem_cls_mix.view(batch_size * num_objs)
            ]
        )
        obj_sem_cls_embeds = obj_sem_cls_embeds.view(batch_size, num_objs, 300).cuda()
        obj_sem_cls_embeds = self.sem_cls_embed_layer(obj_sem_cls_embeds)
        obj_embeds = obj_embeds + obj_sem_cls_embeds
        # get semantic mask embeds
        obj_embeds = obj_embeds.masked_fill(
            obj_sem_masks.unsqueeze(2).logical_not(), 0.0
        )
        obj_sem_mask_embeds = self.sem_mask_embeddings(
            torch.zeros((batch_size, num_objs)).long().cuda()
        ) * obj_sem_masks.logical_not().unsqueeze(2)
        obj_embeds = obj_embeds + obj_sem_mask_embeds
        # spatial reasoning
        pairwise_locs = calc_pairwise_locs(
            obj_locs[:, :, :3],
            obj_locs[:, :, 3:],
            pairwise_rel_type=self.pairwise_rel_type,
            spatial_dist_norm=True,
            spatial_dim=self.spatial_dim,
        )
        for i, pc_layer in enumerate(self.spatial_encoder):
            query_pos = self.point_tokenize_encoder.loc_layers[0](obj_locs)
            obj_embeds = obj_embeds + query_pos
            obj_embeds, self_attn_matrices = pc_layer(
                obj_embeds, pairwise_locs, tgt_key_padding_mask=obj_masks.logical_not()
            )
        cls = self.self_attention(obj_embeds, obj_masks)
        return cls, obj_embeds, {"crop": None}


class ThreeDVistaUnified(nn.Module):
    def __init__(self):
        super(ThreeDVistaUnified, self).__init__()
        cwd = os.getcwd()
        os.chdir(three_d_vista_path)
        from ..external_comp.ThreeDVista.model.vision.basic_modules import (
            get_mixup_function,
        )
        from ..external_comp.ThreeDVista.dataset.path_config import SCAN_FAMILY_BASE

        self.unified_encoder = UnifiedSpatialCrossEncoderV2(**unified_enc_params)
        self.cross_attention = CrossAttentionEncoder()
        self.counter = 0
        pth = torch.load("../../../" + pretrained)
        self.unified_encoder.load_state_dict(pth["unified_encoder"])
        self.project = nn.Linear(projection_dim[0], projection_dim[1])
        os.chdir(cwd)

    def forward(
        self, memory_basic_features, memory_masks, tgt_embeds, tgt_locs, tgt_masks
    ):
        # Define the forward pass
        memory_fuse_feature, tgt_fuse_feature = self.unified_encoder(
            memory_basic_features, memory_masks * 1, tgt_embeds, tgt_locs, tgt_masks
        )
        fused = self.cross_attention(
            memory_fuse_feature, memory_masks, tgt_fuse_feature, tgt_masks
        )
        fused_mask = torch.ones(
            (fused.shape[0], fused.shape[1]), dtype=torch.bool, device=fused.device
        )
        return fused, None, {"crop": None}
