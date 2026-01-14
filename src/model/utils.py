"""
Utility functions for MonSTeR and other models.
"""

import torch
import numpy as np


def get_dataset_pair(batches, verbose=False):
    """
    Computes the dataset pairs for a given batch of elements.
    Use verbose flag to print info on matching pairs for debugging.s
    """

    bs = 0
    for batch in batches:
        bs += batch["motion_x_dict"]["x"].shape[0]

    # TEXT PAIRS
    sent_emb = torch.cat([batch["sent_emb"] for batch in batches], dim=0).to("cpu")
    sent_pairs = get_indices_dicts(sent_emb, bs)

    # MOTION PAIRS
    # We consider all motions to be different: its is unlikely that two motions appear identical and identically oriented in space.
    motion_pairs = {i: [i] for i in range(bs)}

    if verbose:
        for k, v in motion_pairs.items():
            if len(v) > 1:
                print("Found a motion pair!", k, v)

    # SCENE PAIRS
    ids = []
    for batch in batches:
        ids.extend(batch["scene_x_dict"]["ids"])

    if "/" not in ids[0]:
        # Trumans scenes are separated just by a dash
        ids_motion = []
        ids_temp = []

        for i in ids:
            # Change the amount of info to create unique ids for each scene
            # the comparison is either between the barebone scene (just the scene name)
            # or the scene with objects placed in a certain position (i.e. scenename_2023-10-11@19-10-0_100_200)
            # NOTE: modifying this will change the dataset pairs and thus the results!!
            scene, action_lab = i.split("_actlabel_")
            action, start, end = action_lab.split("_")
            ids_temp.append(scene + "_" + action)
            ids_motion.append(action_lab)
        ids = ids_temp
    else:
        # Humanise scenes have dash
        ids_motion = [id.split("/")[1][10:] for id in ids]

    scene_pairs = get_indices_dicts(np.array(ids), from_ids=True)

    return {
        "motion_pairs": motion_pairs,
        "sent_pairs": sent_pairs,
        "scene_pairs": scene_pairs,
    }


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


def select_dataset_pair_modality(dataset_pair, modality):
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


def get_indices_dicts(x, bs=None, from_ids=False):
    # This method is painfully slow but ensures that
    # elements are the EXACT same by comparing tensors one by one.

    pairs = {}
    if not from_ids:

        for i in range(x.size(0)):
            tensor_i = x[i]

            for j in range(x.size(0)):
                tensor_j = x[j]
                if torch.equal(tensor_i, tensor_j):
                    if i in pairs.keys():
                        pairs[i].append(j)
                    else:
                        pairs[i] = [j]
    else:
        tmp = {}
        # Collect the indices for each string
        for i, string in enumerate(x):
            if string not in tmp:
                tmp[string] = []
            tmp[string].append(i)

        # Convert the dictionary to the required format
        for indices in tmp.values():
            for idx in indices:
                pairs[idx] = indices

    return pairs
