import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


def build_and_multiply(t, m, s):
    bs = len(t)
    sim = (t @ m.T).unsqueeze(0).repeat(bs, 1, 1)
    sim += (s @ m.T).unsqueeze(1).repeat(1, bs, 1)
    sim += (t @ s.T).unsqueeze(2).repeat(1, 1, bs)
    return sim / 3


def compute_and_intersection(dict_a, dict_b):
    """Computes the intersection of values between two dictionaries.

    Args:
      dict_a: The first dictionary.
      dict_b: The second dictionary.

    Returns:
      A new dictionary containing the intersection of values for each key.
    """

    intersection_dict = {}
    for key in dict_a:
        if key in dict_b:
            intersection_dict[key] = list(set(dict_a[key]) & set(dict_b[key]))
    return intersection_dict


def select_dataset_pair(dataset_pair, modality):
    all_dataset_pairs = []
    mod = modality.split("/")
    for i in reversed(range(2)):
        dataset_pairs = []
        retrived_m = modality.split("/")[0].split("2")[i]
        for char in retrived_m:
            if char == "m":
                dataset_pairs.append(dataset_pair["motion_pairs"])
            elif char == "t":
                dataset_pairs.append(dataset_pair["sent_pairs"])
            elif char == "s":
                dataset_pairs.append(dataset_pair["scene_pairs"])
            else:
                print("something went wrong in select dataset pairs")
        if len(dataset_pairs) > 1:
            all_dataset_pairs.append(
                compute_and_intersection(dataset_pairs[0], dataset_pairs[1])
            )
        else:
            all_dataset_pairs.append(dataset_pairs[0])
    return all_dataset_pairs, mod


def all_3D_contrastive_metrics(
    sims,
    dataset_pair=None,
    emb=None,
    threshold=None,
    rounding=2,
    return_cols=False,
    modality="t2m/m2t",
    compute_unpaired=False,
    prefix=None,
):

    text_selfsim = None
    assert dataset_pair is not None
    if type(sims) is not list:
        sims = [sims]
    if type(emb) is list:
        emb = [e.cpu().numpy() for e in emb]
        emb = np.concatenate(emb)

    if emb is not None:
        # below we build a text sim matrix to be used to mask the sim_matrices like st2m
        text_selfsim_matrix = emb @ emb.T
        emb = torch.tensor(emb)
    else:
        text_selfsim_matrix = None

    # If the mod has 2 conditioning we can use the old function (the matrix as 1s on diag and is nxn)
    dataset_pair, modalities = select_dataset_pair(dataset_pair, modality)

    if threshold:
        real_threshold = 2 * threshold - 1
        thr_pairs = np.where(text_selfsim_matrix > real_threshold)
        thr_pairs_dict = {k: [] for k in range(len(text_selfsim_matrix))}
        for k, v in zip(thr_pairs[0], thr_pairs[1]):
            thr_pairs_dict[k].append(v)
        joined_0 = {
            k: list(set(thr_pairs_dict[k] + dataset_pair[0][k])) for k in thr_pairs_dict
        }
        joined_1 = {
            k: list(set(thr_pairs_dict[k] + dataset_pair[1][k])) for k in thr_pairs_dict
        }
        dataset_pair = [joined_0, joined_1]

    # dataset_pair is formed like [m2_mask, m1_mask]
    m1_2_m2_m, m1_2_m2_cols = contrastive_metrics_MoPa(
        sims=sims[0],
        text_selfsim=text_selfsim_matrix,
        threshold=threshold,
        dataset_pair=dataset_pair[0],
        modality=modalities[0],
        return_cols=True,
        rounding=rounding,
        prefix=prefix,
    )

    # If the mod has 2 conditioning we can use the old function (the matrix as 1s on diag and is nxn)
    m2_2_m1_m, m2_2_m1_cols = contrastive_metrics_MoPa(
        sims[0].T,
        text_selfsim=text_selfsim_matrix,
        threshold=threshold,
        dataset_pair=dataset_pair[1],
        modality=modalities[1],
        return_cols=True,
        rounding=rounding,
        prefix=prefix,
    )

    m1_2_m2_m.update(m2_2_m1_m)

    return m1_2_m2_m


def contrastive_metrics_MoPa(
    sims,  # bs x bs
    text_selfsim=None,
    dataset_pair=None,  # dict
    verbose=True,
    threshold=None,
    return_cols=False,
    rounding=2,
    modality=None,
    break_ties="averaging",
    prefix=None,
):

    rank_dict = {}
    # Text->Motion
    ranks = np.zeros(sims.shape[0])  # tensor n x n
    for index, score in enumerate(sims):
        inds = np.argsort(score)[::-1]  # [::-1] used to reverse order

        # Score
        rank = 1e20

        # If the text is matched in a better rank position (because the captions might be duplicated)
        # then we pick the better ranks
        for i in dataset_pair[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    if prefix is None:
        prefix = "Thr" if threshold else "All"
    for k in [1, 2, 3, 5, 10]:
        # Compute metrics
        r = 100.0 * len(np.where(ranks < k)[0]) / len(ranks)
        rank_dict[f"{prefix}/rank{k}/{modality}"] = r

    return rank_dict, None


