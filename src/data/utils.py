import re


def get_synonyms(word):
    """Fetch synonyms for a word from WordNet."""
    # synonyms = set()
    # for syn in wordnet.synsets(word):
    #     for lemma in syn.lemmas():
    #         synonyms.add(
    #             lemma.name().replace("_", " ")
    #         )  # Add the synonym and replace underscores
    # return list(synonyms)
    exit("Nlkt WordNet not available")


import json, orjson
import os
import codecs as cs

import numpy as np
import torch


###################################### HUMANISE loader utils ######################################


def check_input(pointcloud):
    mask_over_5000 = (np.abs(pointcloud) > 500).any(axis=1)

    # Create a mask for rows that have ANY NaN value
    mask_nans = np.isnan(pointcloud).any(axis=1)

    # Combine the masks: we want to remove rows that match either condition
    mask_combined = mask_over_5000 | mask_nans

    # Filter to keep only rows that do NOT meet those conditions
    pointcloud_filtered = pointcloud[~mask_combined]
    dropped_indices = np.where(mask_combined)[0]
    # Save the filtered DataFrame back to data['scene_cut']
    return pointcloud_filtered, dropped_indices


def read_split(path, split):
    split_file = os.path.join(path, "splits", split + ".txt")
    id_list = []
    with cs.open(split_file, "r") as f:
        for line in f.readlines():
            id_list.append(line.strip())
    return id_list


def load_annotations(path, name="annotations.json"):
    json_path = os.path.join(path, name)
    with open(json_path, "rb") as ff:
        return orjson.loads(ff.read())


###################################### MISC utils ###################################################


def pad_tensors(tensors, lens=None, pad=0):
    try:
        assert tensors.shape[0] <= lens
    except:
        print(tensors.shape[0], lens)
        print(tensors.shape)
    if tensors.shape[0] == lens:
        return tensors
    shape = list(tensors.shape)
    shape[0] = lens - shape[0]
    res = torch.ones(shape, dtype=tensors.dtype) * pad
    res = torch.cat((tensors, res), dim=0)
    return res


# NOTE: unused function
def write_json(data, path):
    with open(path, "w") as ff:
        ff.write(json.dumps(data, indent=4))


# NOTE: unused function
def list_files(directory):
    file_list = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            file_list.append(os.path.join(root, filename))
    return file_list
