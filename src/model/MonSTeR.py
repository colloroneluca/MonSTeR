import logging

logger = logging.getLogger(__name__)

from typing import Dict, Optional
import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from src.model.temos import TEMOS
from src.model.losses import (
    InfoNCE_with_filtering,
    compute_reconstruction_losses,
    compute_KL_losses,
    compute_latent_losses,
    compute_contrastive_losses,
)
from src.model.utils import get_dataset_pair
from src.model.metrics import all_3D_contrastive_metrics
from src.model.instatiate_3DVista import ThreeDVistaScene
from src.model.instatiate_3DVista import ThreeDVistaUnified


def transpose(x):
    return x.permute(*torch.arange(x.ndim - 1, -1, -1))


def mv_mult(m, v):
    return torch.matmul(m, v.unsqueeze(-1)).squeeze(-1)


def get_sim_matrix(x, y):
    x_logits = torch.nn.functional.normalize(x, dim=-1)
    y_logits = torch.nn.functional.normalize(y, dim=-1)
    sim_matrix = x_logits @ transpose(y_logits)
    return sim_matrix


# Scores are between 0 and 1
def get_score_matrix(x, y):
    sim_matrix = get_sim_matrix(x, y)
    scores = sim_matrix / 2 + 0.5
    return scores


def compute_cross_modal(
    m1, m2, m1_lat, m2_lat, normalization_fn, cross_modal_eval="mean", use_latents=False
):
    m1_2_m2 = None  

    if use_latents:
        m1_2_m2 = get_sim_matrix(m1_lat, m2_lat).cpu().numpy()
    else:
        match cross_modal_eval:
            case "mean":
                m1_2_m2 = (m1 + m2) / 2
            case "max":
                m1_2_m2 = np.maximum(m1, m2)
            case "norm-max":
                m1_2_m2 = np.maximum(normalization_fn(m1), normalization_fn(m2))
            case "norm-mean":
                m1_2_m2 = (normalization_fn(m1) + normalization_fn(m2)) / 2
            case "sum_lat_infer":
                m1_2_m2 = get_sim_matrix(m1_lat, m2_lat).cpu().numpy()

    return m1_2_m2


class MonSTeR(TEMOS):
    r"""TMR: Text-to-Motion Retrieval
    Using Contrastive 3D Human Motion Synthesis
    Find more information about the model on the following website:
    https://mathis.petrovich.fr/tmr

    Args:
        motion_encoder: a module to encode the input motion features in the latent space (required).
        text_encoder: a module to encode the text embeddings in the latent space (required).
        motion_decoder: a module to decode the latent vector into motion features (required).
        vae: a boolean to make the model probabilistic (required).
        fact: a scaling factor for sampling the VAE (optional).
        sample_mean: sample the mean vector instead of random sampling (optional).
        lmd: dictionary of losses weights (optional).
        lr: learninig rate for the optimizer (optional).
        temperature: temperature of the softmax in the contrastive loss (optional).
        threshold_selfsim: threshold used to filter wrong negatives for the contrastive loss (optional).
        threshold_selfsim_metrics: threshold used to filter wrong negatives for the metrics (optional).
    """

    def __init__(
        self,
        motion_encoder: nn.Module,
        text_encoder: nn.Module,
        motion_decoder: nn.Module,
        vae: bool,
        fact: Optional[float] = None,
        sample_mean: Optional[bool] = False,
        lmd: Dict = {"recons": 1.0, "latent": 1.0e-5, "kl": 1.0e-5, "contrastive": 0.1},
        lr: float = 1e-4,
        temperature: float = 0.7,
        threshold_selfsim: float = 0.80,
        threshold_selfsim_metrics: float = 0.95,
        use_scene: bool = False,
        cross_mod_eval: str = "mean",  
        bridge="all",
        train_dl_len=None,
        total_steps=None,
        baseline=False,
        no_single=False,
        **kwargs
    ) -> None:
        # Initialize module like TEMOS

        self.baseline = baseline
        self.no_single = no_single

        logger.info(f"doing no single = {self.no_single}")

        super().__init__(
            motion_encoder=motion_encoder,
            text_encoder=text_encoder,
            motion_decoder=motion_decoder,
            ts_encoder=ThreeDVistaUnified(),
            tm_encoder=ThreeDVistaUnified(),
            ms_encoder=ThreeDVistaUnified(),
            vae=vae,
            fact=fact,
            sample_mean=sample_mean,
            lmd=lmd,
            lr=lr,
            use_scene=use_scene,
        )

        self.eval_dev = self.device
        self.cross_mod_eval = cross_mod_eval
        self.train_dl_len = train_dl_len
        self.total_steps = total_steps

        # Ensure that scene encoder's parameters require grad
        self.scene_encoder = ThreeDVistaScene()
        for n, p in self.scene_encoder.named_parameters():
            if not p.requires_grad:
                p.requires_grad = True

        self.contrastive_loss_fn = InfoNCE_with_filtering(
            temperature=temperature,
            threshold_selfsim=threshold_selfsim,
        )

        self.threshold_selfsim_metrics = threshold_selfsim_metrics
        self.normalization_fn = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
        self.bridge = bridge
        self.gelu = torch.nn.GELU()
        self.validation_step_t_latents = []
        self.validation_step_m_latents = []
        self.validation_step_sent_emb = []
        self.validation_step_s_latents = []
        self.validation_step_ts_latents = []
        self.validation_step_tm_latents = []
        self.validation_step_ms_latents = []
        self.all_batches = []

    # NOTE: we inherit TEMOS training step, where we call the compute_scene_loss method.
    # When calling the forward, instead, the model will call TEMOS's forward function, selecting
    # the appropriate encoder based on the input dimensionality.
    # To avoid this (optional), we could override the training_step and forward as well.

    def compute_scene_loss(
        self,
        batch: Dict,
        return_all=False,
        phase="None",
        sample_mean=False,
        return_motions=False,
    ) -> Dict | tuple:

        motion_x_dict = batch["motion_x_dict"]
        text_x_dict = batch["text_x_dict"]

        bs = motion_x_dict["x"].shape[0]
        mask = motion_x_dict["mask"]
        ref_motions = motion_x_dict["x"]
        scene_x_dict = batch["scene_x_dict"]
        obj_masks = scene_x_dict["obj_masks"]
        obj_locs = scene_x_dict["obj_locs"].float()
        sent_emb = batch["sent_emb"]

        rot_labels = None
        t_dists, m_dists, s_dists = (None,) * 3
        ts_dists, tm_dists, ms_dists = (None,) * 3
        t_motions, m_motions, s_motions = (None,) * 3
        ts_motions, tm_motions, ms_motions = (None,) * 3
        (
            t_latents,
            m_latents,
            s_latents,
            ts_latents,
            tm_latents,
            ms_latents,
            abs_m_latents,
        ) = (None,) * 7

        t_tok_embeds, s_tok_embeds, m_tok_embeds, abs_m_tok_embeds = (None,) * 4

        ms, st, tm = (None,) * 3

        # Latents are coming from reparametrization trick, so they are stocastic if sample mean is false
        # TEXT
        t_motions, t_latents, t_dists, t_encoded, t_tok_embeds, _ = self(
            text_x_dict, mask=mask, return_all=True, sample_mean=sample_mean
        )

        # MOTION
        m_motions, m_latents, m_dists, m_encoded, m_tok_embeds, _ = self(
            motion_x_dict, mask=mask, return_all=True, sample_mean=sample_mean
        )

        # SCENE
        s_motions, s_latents, s_dists, s_encoded, s_tok_embeds, scene_info = self(
            scene_x_dict, mask=mask, return_all=True, sample_mean=sample_mean
        )

        # self.debug_counter += 1

        # CROSS-MODAL (i.e. standard MonSTeR)
        # Otherwise computes w/o cross-modal encodings
        if not self.baseline:

            # TEXT + SCENE
            data_dict = {
                "memory_basic_features": t_tok_embeds,
                "memory_masks": text_x_dict["mask"],
                "tgt_embeds": s_tok_embeds,
                "tgt_locs": obj_locs,
                "tgt_masks": obj_masks,
            }
            ts_motions, ts_latents, ts_dists, ts_encoded, ts_tok_embeds, scene_info = (
                self(
                    data_dict,
                    mask=mask,
                    return_all=True,
                    sample_mean=sample_mean,
                    joint_mod="ts",
                )
            )

            # TEXT + MOTION
            data_dict = {
                "memory_basic_features": t_tok_embeds,
                "memory_masks": text_x_dict["mask"],
                "tgt_embeds": m_tok_embeds,
                "tgt_locs": None,
                "tgt_masks": mask,
            }
            tm_motions, tm_latents, tm_dists, tm_encoded, tm_tok_embeds, _ = self(
                data_dict,
                mask=mask,
                return_all=True,
                sample_mean=sample_mean,
                joint_mod="tm",
            )

            # MOTION + SCENE
            data_dict = {
                "memory_basic_features": m_tok_embeds,
                "memory_masks": mask,
                "tgt_embeds": s_tok_embeds,
                "tgt_locs": obj_locs,
                "tgt_masks": obj_masks,
            }
            ms_motions, ms_latents, ms_dists, ms_encoded, ms_tok_embeds, _ = self(
                data_dict,
                mask=mask,
                return_all=True,
                sample_mean=sample_mean,
                joint_mod="ms",
            )

        losses = {}

        # Reconstruction losses
        # With baseline = True, computes recons for t,m,s
        # With baseline = False, computes recons for ts
        losses["recons"] = compute_reconstruction_losses(
            t_motions,
            s_motions,
            m_motions,
            ts_motions,
            ref_motions,
            recon_loss_fn=self.reconstruction_loss_fn,
            baseline=self.baseline,
        )

        # KL Divergence losses
        # Only computed when VAE is True
        # With baseline = True, computes KL for ...
        # With baseline = False, computes KL for ...
        # With no_single = True, skips single modality KLs
        if self.vae:
            losses["kl"] = compute_KL_losses(
                t_dists,
                m_dists,
                s_dists,
                ts_dists,
                kl_loss_fn=self.kl_loss_fn,
                baseline=self.baseline,
            )

        # Latent losses
        # With baseline = True, computes L1 smooth loss for ...
        # With baseline = False, computes L1 smooth loss for ...
        losses["latent"] = compute_latent_losses(
            t_latents,
            m_latents,
            s_latents,
            ts_latents,
            tm_latents,
            ms_latents,
            latent_loss_fn=self.latent_loss_fn,
            bridge=self.bridge,
            baseline=self.baseline,
            no_single=self.no_single,
        )

        # Contrastive losses (aka INFO-NCE losses)
        # With baseline = True, computes contrastive for single modalities but not crossmodal
        # With baseline = False, computes contrastive also for crossmodal
        # With no_single skips computation of single modality contrastive losses but allows
        # crossmodal contrastive losses, i.e. to test MonSTeR without single modality alignment
        losses["contrastive"] = compute_contrastive_losses(
            t_latents,
            m_latents,
            s_latents,
            ts_latents,
            tm_latents,
            ms_latents,
            sent_emb,
            contrastive_loss_fn=self.contrastive_loss_fn,
            rot_labels=rot_labels,
            no_single=self.no_single,
            baseline=self.baseline,
            bridge=self.bridge,
        )

        losses["loss"] = sum(
            self.lmd[x] * val for x, val in losses.items() if x in self.lmd
        )

        if return_all:
            return (
                losses,
                t_latents,
                m_latents,
                s_latents,
                ts_latents,
                tm_latents,
                ms_latents,
            )
        else:
            return losses

    def test_step(self, batch: Dict, batch_idx: int) -> Tensor:
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self) -> None:
        self.on_validation_epoch_end()

    def validation_step(self, batch: Dict, batch_idx: int) -> Tensor:

        self.eval()
        bs = len(batch["motion_x_dict"]["x"])

        losses, t_latents, m_latents, s_latents, ts_latents, tm_latents, ms_latents = (
            self.compute_scene_loss(batch, return_all=True, sample_mean=True)
        )

        self.validation_step_s_latents.append(
            s_latents.to(self.eval_dev) if s_latents is not None else None
        )

        self.validation_step_ts_latents.append(
            ts_latents.to(self.eval_dev) if ts_latents is not None else None
        )

        self.validation_step_tm_latents.append(
            tm_latents.to(self.eval_dev) if tm_latents is not None else None
        )

        self.validation_step_ms_latents.append(
            ms_latents.to(self.eval_dev) if ms_latents is not None else None
        )

        self.validation_step_t_latents.append(t_latents.to(self.eval_dev))
        self.validation_step_m_latents.append(m_latents.to(self.eval_dev))
        self.validation_step_sent_emb.append(batch["sent_emb"].to(self.eval_dev))

        # We gather all data so we can later compute the mask for evaluation
        # in the same way as MoPa's paper
        self.all_batches.append(batch)

        for loss_name in sorted(losses):
            loss_val = losses[loss_name]
            self.log(
                f"val_{loss_name}",
                loss_val,
                on_epoch=True,
                on_step=True,
                batch_size=bs,
                sync_dist=True,
            )
        return losses["loss"]

    def on_validation_epoch_end(self):

        self.compute_all_metrics(
            self.validation_step_t_latents,
            self.validation_step_m_latents,
            self.validation_step_s_latents,
            self.validation_step_sent_emb,
            self.validation_step_ts_latents,
            self.validation_step_tm_latents,
            self.validation_step_ms_latents,
            batches=self.all_batches,
        )

        self.validation_step_t_latents.clear()
        self.validation_step_m_latents.clear()
        self.validation_step_sent_emb.clear()
        self.validation_step_s_latents.clear()
        self.validation_step_ts_latents.clear()
        self.validation_step_tm_latents.clear()
        self.validation_step_ms_latents.clear()
        self.all_batches.clear()

    def log_loss_dict(self, losses: dict, bs=None, on_epoch=False):
        for loss_name in sorted(losses):
            loss_val = losses[loss_name]
            self.log(
                f"train_{loss_name}",
                loss_val,
                on_epoch=on_epoch,
                on_step=True,
                batch_size=bs,
                sync_dist=True,
            )

    def compute_all_metrics(
        self,
        validation_step_t_latents,
        validation_step_m_latents,
        validation_step_s_latents,
        validation_step_sent_emb,
        validation_step_ts_latents,
        validation_step_tm_latents,
        validation_step_ms_latents,
        batches=None,
        return_sims=False,
        return_dataset_pair=False,
        print_metrics=True,
    ):

        # Compute dataset pairs for filtering in metrics
        dataset_pair = get_dataset_pair(batches)

        modalities = [
            "t2m/m2t",
            "s2m/m2s",
            "s2t/t2s",
            "st2m/m2st",
            "tm2s/s2tm",
            "sm2t/t2sm",
        ]

        # Turn latents into tensors (this is necessary as they move between CPU-GPU during metrics computation)
        # Maybe this can be avoided
        t_latents = torch.cat(validation_step_t_latents)
        m_latents = torch.cat(validation_step_m_latents)
        s_latents = torch.cat(validation_step_s_latents)
        sent_emb = torch.cat(validation_step_sent_emb)

        # Compute single modality similarity matrices
        tm = get_sim_matrix(t_latents, m_latents).cpu().numpy()
        sim_matrices = [tm]
        sm = get_sim_matrix(s_latents, m_latents).cpu().numpy()
        sim_matrices.append(sm)
        st = get_sim_matrix(s_latents, t_latents).cpu().numpy()
        sim_matrices.append(st)
        ts_latents = torch.cat(validation_step_ts_latents)
        tm_latents = torch.cat(validation_step_tm_latents)
        ms_latents = torch.cat(validation_step_ms_latents)

        # For the baseline, we compute cross-modal similarities as the mean of the single modalities
        # For MonSTeR (and variants), we use the appropriate cross-modal method
        if self.baseline:
            st2m = (sm + tm) / 2
            tm2s = (st.T + sm.T) / 2
            sm2t = (st + tm.T) / 2
        else:
            st2m = compute_cross_modal(
                None,
                None,
                ts_latents,
                m_latents,
                cross_modal_eval=self.cross_mod_eval,
                normalization_fn=self.normalization_fn,
                use_latents=True,
            )
            tm2s = compute_cross_modal(
                None,
                None,
                tm_latents,
                s_latents,
                cross_modal_eval=self.cross_mod_eval,
                normalization_fn=self.normalization_fn,
                use_latents=True,
            )
            sm2t = compute_cross_modal(
                None,
                None,
                ms_latents,
                t_latents,
                cross_modal_eval=self.cross_mod_eval,
                normalization_fn=self.normalization_fn,
                use_latents=True,
            )  # for je sm/ms are already comuted with abs if needed

        sim_matrices.append(st2m)
        sim_matrices.append(tm2s)
        sim_matrices.append(sm2t)

        if print_metrics:
            all_contrastive = []

            for i in range(2):
                for m, sim in zip(modalities, sim_matrices):
                    all_contrastive.append(
                        all_3D_contrastive_metrics(
                            sims=sim,  # in this case the sim is a list with [m1m2_2_m3, m3_2_m1m2]
                            dataset_pair=dataset_pair,
                            emb=sent_emb.cpu().numpy(),
                            threshold=(
                                self.threshold_selfsim_metrics if i == 0 else None
                            ),
                            modality=m,
                        )
                    )
            for contrastive_metrics in all_contrastive:
                for loss_name in sorted(contrastive_metrics):
                    loss_val = contrastive_metrics[loss_name]
                    self.log(
                        f"{loss_name}",
                        loss_val,
                        sync_dist=True,
                    )
            self.log(
                f"{len}",
                len(sim),
                sync_dist=True,
            )

        # Depending on the flag returns the requested values
        # Useful for retrieval at test time
        if return_sims and return_dataset_pair:
            return modalities, sim_matrices, dataset_pair
        if return_sims:
            return modalities, sim_matrices
        if return_dataset_pair:
            return modalities, dataset_pair


