from tqdm import tqdm


import logging
logger = logging.getLogger(__name__)

import numpy as np
import torch
from torch.utils.data import Dataset

from .collate import collate_text_motion
from src.data.load_scenes import load_scene_humanise
from src.data.utils import read_split, load_annotations


class TextMotionDataset(Dataset):
    def __init__(
        self,
        path: str,
        motion_loader,
        text_to_sent_emb,
        text_to_token_emb,
        split: str = "train",
        min_seconds: float = 2.0,
        max_seconds: float = 10.0,
        preload: bool = True,
        scenes_path: str = None,
        debug: bool = False,
        use_color: bool = False,
        transform: bool = False,
        rotate: bool = False 
    ):

        self.collate_fn = collate_text_motion
        self.split = split
        self.keyids = read_split(path, split)
        self.scenes_path = scenes_path
        self.text_to_sent_emb = text_to_sent_emb
        self.text_to_token_emb = text_to_token_emb
        self.motion_loader = motion_loader
        logger.info(f"Using motion loader from {self.motion_loader.base_dir}")
        self.min_seconds = min_seconds
        self.max_seconds = max_seconds
        self.debug = debug
        self.scene_loader = load_scene_humanise(
            scenes_path=scenes_path,
            split=split,
            use_color=use_color,
            max_points=20000,
            transform=transform,
        )
        self.annotations = load_annotations(path)
        replicated_keyids = []

        for keyid in self.keyids:
            num_indices = 1
            for i in range(num_indices):
                replicated_keyids.append(
                    keyid + "__" + str(i)
                )  # adds trailing __0. At test this means it takes the first sample in annotations. At train this is basically ignored as it samples a random idx for the sample

        self.keyids = replicated_keyids

        ################################################
        # Uncomment these lines to load only elements with at least N samples. Useful to export all'dataset's latents or similar
        # ind = 9 
        # self.keyids = [keyid for keyid in self.keyids if len(self.annotations[keyid[:-3]]['annotations'])>ind]
        # print(f'keeping those with at least {ind+1} samples. Len = {len(self.keyids)}')
        ################################################

        if self.debug:
            logger.info("You are in Debug mode, loading just 100 samples")
            self.annotations = {
                k: v for i, (k, v) in enumerate(self.annotations.items()) if i < 100
            }
        self.keyids = [keyid for keyid in self.keyids if keyid[:-3] in self.annotations]
        self.is_training = split == "train"
        self.nfeats = 66 #self.motion_loader.nfeats
        if split != "train" and preload:
            data = []
            for dt in tqdm(self, desc="Preloading the dataset"):
                data.append(dt)
            self.data = data

    def __len__(self):
        return len(self.keyids)

    def load_keyid(self, keyid):
        annotations = self.annotations[keyid[:-3]]
        index = int(keyid[-1])
        if self.is_training:
            index = np.random.randint(len(annotations["annotations"]))

        annotation = annotations["annotations"][index]
        text = annotation["recaptioning"]

        text_x_dict = self.text_to_token_emb(text)
        sent_emb = self.text_to_sent_emb(text)
        
        data_scene, id = self.scene_loader(
            annotations, index, return_for_goal_dist=False, split=self.split
        )
        
        joints = data_scene["joints"][:, :22, :]
        joints = joints.reshape(len(joints), -1)
        motion_x_dict = {}
        motion_x_dict["x"] = torch.FloatTensor(joints)
        motion_x_dict["length"] = len(joints)
        keyid = keyid[:-1] + str(index)

        output = {
            "motion_x_dict": motion_x_dict,
            "text_x_dict": text_x_dict,
            "text": text,
            "keyid": keyid,
            "keyidx": keyid[:-3]
            + "_"
            + str(
                index
            ), 
            "sent_emb": sent_emb,
        }

        output["scene_x_dict"] = data_scene
        output["scene_x_dict"]["ids"] = str(id)
        
        return output

    def __getitem__(self, index):
        if not hasattr(self, "data") or self.split == "train":
            keyid = self.keyids[index]
            return self.load_keyid(keyid)
        else:
            return self.data[index]

   
    



