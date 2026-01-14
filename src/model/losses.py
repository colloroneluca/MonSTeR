import torch
import torch.nn.functional as F


def compute_reconstruction_losses(
    t_motions,
    s_motions,
    m_motions,
    ts_motions,
    ref_motions,
    recon_loss_fn,
    baseline=False,
) -> int:

    losses = {}

    if baseline:
        losses = {
            "rec_t2m": recon_loss_fn(t_motions, ref_motions),
            "rec_s2m": recon_loss_fn(s_motions, ref_motions),
            "rec_m2m": recon_loss_fn(m_motions, ref_motions),
        }
    else:
        losses["rec_ts2m"] = recon_loss_fn(ts_motions, ref_motions)

    return sum(losses.values())


def compute_KL_losses(
    t_dists,
    m_dists,
    s_dists,
    ts_dists,
    kl_loss_fn,
    baseline=False,
) -> int:

    # Create a centred normal distribution to compare with
    # logvar = 0 -> std = 1
    kl_losses = {}
    ref_mus = torch.zeros_like(m_dists[0])
    ref_logvar = torch.zeros_like(m_dists[1])
    ref_dists = (ref_mus, ref_logvar)

    if baseline:
        kl_losses = {
            "kl_t2m": kl_loss_fn(t_dists, m_dists),
            "kl_t2s": kl_loss_fn(t_dists, s_dists),
            "kl_m2t": kl_loss_fn(m_dists, t_dists),
            "kl_m2s": kl_loss_fn(m_dists, s_dists),
            "kl_s2t": kl_loss_fn(s_dists, t_dists),
            "kl_s2m": kl_loss_fn(s_dists, m_dists),
            "kl_m2ref": kl_loss_fn(m_dists, ref_dists),
            "kl_t2ref": kl_loss_fn(t_dists, ref_dists),
            "kl_s2ref": kl_loss_fn(s_dists, ref_dists),
        }
    else:
        kl_losses["kl_ts2ref"] = kl_loss_fn(ts_dists, ref_dists)

    return sum(kl_losses.values())


def compute_latent_losses(
    t_latents,
    m_latents,
    s_latents,
    ts_latents,
    tm_latents,
    ms_latents,
    latent_loss_fn,
    bridge="all",
    baseline=False,
    no_single=False,
):
    lat_losses = {}

    if baseline or not no_single:

        lat_losses = {
            "lat_t2m": latent_loss_fn(t_latents, m_latents),
            "lat_s2m": latent_loss_fn(s_latents, m_latents),
            "lat_s2t": latent_loss_fn(s_latents, t_latents),
        }

        lat_losses = {
            k: v
            for k, v in lat_losses.items()
            if (bridge in k.split("_")[-1] or bridge == "all")
        }

    if not baseline:
        lat_losses["lat_ts2m"] = latent_loss_fn(ts_latents, m_latents)
        lat_losses["lat_tm2s"] = latent_loss_fn(tm_latents, s_latents)
        lat_losses["lat_ms2t"] = latent_loss_fn(ms_latents, t_latents)

    # Latent manifold loss
    return sum(lat_losses.values())


def compute_contrastive_losses(
    t_latents,
    m_latents,
    s_latents,
    ts_latents,
    tm_latents,
    ms_latents,
    sent_emb,
    contrastive_loss_fn,
    rot_labels=None,
    no_single=False,
    baseline=False,
    bridge="all",
):

    contr_losses = {}

    if not no_single:
        contr_losses = {
            "contr_t2m": contrastive_loss_fn(t_latents, m_latents, sent_emb),
            "contr_s2m": contrastive_loss_fn(
                s_latents, m_latents, sent_emb, rot_labels=rot_labels
            ),
            "contr_s2t": contrastive_loss_fn(s_latents, t_latents, sent_emb),
        }
        contr_losses = {
            k: v
            for k, v in contr_losses.items()
            if (bridge in k.split("_")[-1] or bridge == "all")
        }

    if not baseline:
        contr_losses["contr_ts2m"] = contrastive_loss_fn(
            ts_latents, m_latents, sent_emb, rot_labels=rot_labels
        )
        contr_losses["contr_tm2s"] = contrastive_loss_fn(
            tm_latents, s_latents, sent_emb, rot_labels=rot_labels
        )
        contr_losses["contr_ms2t"] = contrastive_loss_fn(
            ms_latents, t_latents, sent_emb, rot_labels=rot_labels
        )

    return sum(contr_losses.values())


# For reference
# https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
# https://pytorch.org/docs/stable/_modules/torch/distributions/kl.html#kl_divergence
class KLLoss:
    def __call__(self, q, p):
        mu_q, logvar_q = q
        mu_p, logvar_p = p

        log_var_ratio = logvar_q - logvar_p
        t1 = (mu_p - mu_q).pow(2) / logvar_p.exp()
        div = 0.5 * (log_var_ratio.exp() + t1 - 1 - log_var_ratio)
        return div.mean()

    def __repr__(self):
        return "KLLoss()"


class InfoNCE_with_filtering:
    def __init__(self, temperature=0.7, threshold_selfsim=0.8):
        self.temperature = temperature
        self.threshold_selfsim = threshold_selfsim
        self.all_diags = None
        self.labels = None


    def get_sim_matrix(self, x, y):
        x_logits = torch.nn.functional.normalize(x, dim=-1)
        y_logits = torch.nn.functional.normalize(y, dim=-1)
        sim_matrix = x_logits @ y_logits.T
        return sim_matrix

    def __call__(self, t, m, sent_emb=None, s=None, rot_labels=None):

        bs, device = len(t), t.device
        sim_matrix = self.get_sim_matrix(t, m) / self.temperature

        if sent_emb is not None and self.threshold_selfsim:
            # put the threshold value between -1 and 1
            real_threshold_selfsim = 2 * self.threshold_selfsim - 1
            # Filtering too close values
            # mask them by putting -inf in the sim_matrix
            selfsim = sent_emb @ sent_emb.T
            selfsim_nodiag = selfsim - selfsim.diag().diag()
            idx = torch.where(selfsim_nodiag > real_threshold_selfsim)
            sim_matrix[idx] = -torch.inf

        labels = torch.arange(bs, device=device)

        total_loss = (
                F.cross_entropy(sim_matrix, labels)
                + F.cross_entropy(sim_matrix.T, labels)
            ) / 2

        return total_loss

    def __repr__(self):
        return f"Constrastive(temp={self.temp})"
