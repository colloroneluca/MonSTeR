from pathlib import Path
import pickle as pkl
import numpy as np
import torch
import pandas as pd
import logging
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)
from tqdm import tqdm

from src.data.utils import check_input, pad_tensors

MAPPING_PATH = "src/external_comp/ThreeDVista/vista_int2class.pkl"


class load_scene_humanise:
    def __init__(
        self,
        scenes_path,
        split,
        use_color,
        max_points=20000,
        num_points=4000,
        num_objs=50,
        **kwargs
    ):
        self.max_points = max_points
        self.scenes_path = scenes_path
        self.split = split
        self.use_color = use_color
        self.map_humanise_class_to_vista_class = (
            self.build_map_humanise_class_to_vista_class()
        )
        self.num_points = num_points
        self.num_objs = num_objs


    def build_map_humanise_class_to_vista_class(self):
        classes = [
            "floor",
            "wall",
            "cabinet",
            "bed",
            "chair",
            "sofa",
            "table",
            "door",
            "window",
            "bookshelf",
            "picture",
            "counter",
            "desk",
            "curtain",
            "refrigerator",
            "bathtub",
            "shower curtain",
            "toilet",
            "sink",
            "otherprop",
        ]
        with open(MAPPING_PATH, "rb") as f:
            int2class = pkl.load(f)
        class2int = {class_name: key for key, class_name in int2class.items()}
        index_to_key = {}
        # Iterate over the classes list with indices
        for index, class_name in enumerate(classes):
            # Check if the class name exists in class2int
            if class_name == "otherprop":
                class_name = "object"
            if class_name in class2int:
                # Map the index to the corresponding key in int2class
                index_to_key[index] = class2int[class_name]
            else:
                # Handle classes not found in int2class
                print(f"Warning: Class '{class_name}' not found in int2class.")
                index_to_key[index] = None  # Or handle as you see fit
        return index_to_key

    def get_path(self, annotations, index):
        path = annotations["path"]
        scene = annotations["annotations"][index]["scene"].split("_")[0]
        action, folder, _ = path.split("/")
        id0, id1 = folder.split("_")
        folder = f"{'0' * (8-len(id0))}" + str(id0) + "_" + id1
        path = (
            Path(self.scenes_path)
            / Path(action)
            / Path(scene + "_" + folder)
            / Path(str(index) + ".pkl")
        )
        relative = Path(action) / Path(scene + "_" + folder) / Path(str(index) + ".pkl")
        return path, relative, action, folder, scene

    def get_objs(self, pointcloud, crop_mask, path, crop_scene=False, motion=None):

        # check no corrupted values is loaded in data - can happen
        pointcloud, dropped = check_input(pointcloud)

        if len(dropped) > 0:
            all_indices = np.arange(len(crop_mask))
            not_dropped = np.delete(all_indices, dropped)
            crop_mask = crop_mask[not_dropped]
            print("dropped", len(dropped))

        # identify unique objects in the pointcloud
        unique_instances = np.unique(pointcloud[:, -2])
        instance_pointclouds = {}

        for instance_id in unique_instances:
            # Create a mask for the current instance
            instance_mask = pointcloud[:, -2] == instance_id

            # Extract points belonging to the current instance
            assert pointcloud.shape[-1] > 6
            # if you got this assert it means that you miss either color of instance/seg
            instance_points = pointcloud[instance_mask, :6]  # x, y, z coordinates
            instance_crop_mask = crop_mask[..., np.newaxis][instance_mask].squeeze()

            # Extract the class ID (assuming it's the same for all points in the instance)
            instance_class = pointcloud[instance_mask, -1][0]

            try:
                if len(instance_points) > 0:
                    instance_class = self.map_humanise_class_to_vista_class[
                        instance_class
                    ]
                    instance_pointclouds[instance_id] = {
                        "points": instance_points,
                        "class": instance_class,
                        "crop_mask": instance_crop_mask,
                    }
            except:
                print(
                    f"Warning: Class ID '{instance_class}' not found in map_humanise_class_to_vista_class. Returning 'object' id 5"
                )
                instance_pointclouds[instance_id] = {
                    "points": np.random.normal(0, 10, (5000, 6)),
                    "class": 5,
                    "crop_mask": np.random.choice([True, False], size=(5000,)),
                }

        return instance_pointclouds


    def all_objs_info(self, objs):
        obj_fts = []
        obj_locs = []
        obj_boxes = []
        obj_crop_mask_list = []
        obj_pcds = [obj["points"] for k, obj in objs.items()]
        obj_labels = [obj["class"] for k, obj in objs.items()]
        obj_crop_masks = [obj["crop_mask"] for k, obj in objs.items()]

        for obj_pcd, obj_crop_mask in zip(obj_pcds, obj_crop_masks):
            # build locs
            if len(obj_pcd > 100):
                obj_center = obj_pcd[:, :3].mean(0)
                obj_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
                obj_locs.append(np.concatenate([obj_center, obj_size], 0))
                # build box
                obj_box_center = (obj_pcd[:, :3].max(0) + obj_pcd[:, :3].min(0)) / 2
                obj_box_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
                obj_boxes.append(np.concatenate([obj_box_center, obj_box_size], 0))
                if self.split == "train":
                    # subsample
                    pcd_idxs = np.random.choice(
                        len(obj_pcd),
                        size=self.num_points,
                        replace=(len(obj_pcd) < self.num_points),
                    )  # random here

                else:  # at test we just take points evenly spaced to avoid any randomness
                    n_total = len(obj_pcd)
                    if n_total >= self.num_points:
                        # Take self.num_points indices spread uniformly across 0 … n_total-1
                        pcd_idxs = np.linspace(
                            0, n_total - 1, self.num_points, dtype=int
                        )
                    else:
                        # Not enough points: cycle through the original indices until we reach the target length
                        pcd_idxs = np.resize(np.arange(n_total), self.num_points)
                obj_pcd = obj_pcd[pcd_idxs]  # keep only the selected points
                ##############################
                try:
                    obj_crop_mask = obj_crop_mask[pcd_idxs]
                except:
                    obj_crop_mask = np.random.choice([True, False], size=pcd_idxs.shape)
                    print("Something went wrong. Initializing random object crop")
                obj_crop_mask_list.append(obj_crop_mask)
                # normalize
                # We normalize here to be in conformity with 3d vista. However, the alignment between motion and scene is not lost
                # as scene encoder will do spatial attention using obj_locs which are computed before normalization.
                obj_pcd[:, :3] = obj_pcd[:, :3] - obj_pcd[:, :3].mean(0)
                max_dist = np.max(np.sqrt(np.sum(obj_pcd[:, :3] ** 2, 1)))
                if max_dist < 1e-6:  # take care of tiny point-clouds, i.e., padding
                    max_dist = 1
                obj_pcd[:, :3] = obj_pcd[:, :3] / max_dist

                obj_fts.append(obj_pcd)
    
        # convert to torch
        obj_fts = torch.from_numpy(np.stack(obj_fts, 0))
        obj_locs = torch.from_numpy(np.array(obj_locs))
        obj_boxes = torch.from_numpy(np.array(obj_boxes))
        obj_crop_mask = torch.from_numpy(np.array(obj_crop_mask_list))
        obj_labels = (
            torch.LongTensor(obj_labels)
            if obj_labels[0] is not None
            else -torch.ones((obj_fts.shape[0]))
        )
        data_dict = {}

        N = obj_fts.shape[0]

        if N > self.num_objs:
            # Randomly select 'num_objs' indices without replacement

            if self.split == "train":
                indices = torch.randperm(N)[: self.num_objs]
                ## Subsample the tensor along the first dimension
                ####################
                # Evenly spaced indices (0, 2, 4, … when N=4000 and num_objs=2000)
            else:
                step = N / self.num_objs  # fractional step
                indices = (
                    (torch.arange(self.num_objs, dtype=torch.float32) * step)
                    .floor()
                    .long()
                )  # → int64
            ####################
            data_dict["obj_fts"] = obj_fts[indices]
            data_dict["obj_locs"] = obj_locs[indices]
            data_dict["obj_boxes"] = obj_boxes[indices]
            data_dict["obj_labels"] = obj_labels[indices]
            data_dict["obj_crop_mask"] = obj_crop_mask[indices]
            data_dict["obj_masks"] = torch.arange(self.num_objs) < len(
                data_dict["obj_locs"]
            )  # O
            data_dict["obj_sem_masks"] = torch.arange(self.num_objs) < len(
                data_dict["obj_locs"]
            )
        else:
            data_dict["obj_fts"] = pad_tensors(
                obj_fts, lens=self.num_objs, pad=1.0
            ).float()  # O, 1024, 6
            data_dict["obj_crop_mask"] = pad_tensors(
                obj_crop_mask, lens=self.num_objs, pad=-100.0
            ).float()
            data_dict["obj_masks"] = torch.arange(self.num_objs) < len(obj_locs)  # O
            data_dict["obj_sem_masks"] = torch.arange(self.num_objs) < len(obj_locs)
            data_dict["obj_locs"] = pad_tensors(
                obj_locs, lens=self.num_objs, pad=0.0
            ).float()  # O, 3
            data_dict["obj_boxes"] = pad_tensors(
                obj_boxes, lens=self.num_objs, pad=0.0
            ).float()  # O, 3
            data_dict["obj_labels"] = pad_tensors(
                obj_labels, lens=self.num_objs, pad=-100
            ).long()  # O

        # NOTE: returns data_dict = {
        #     "obj_fts": obj_fts, # N, 6
        #     "obj_locs": obj_locs, # N, 6
        #     "obj_labels": obj_labels, # N,
        #     "obj_boxes": obj_boxes, # N, 6
        # }
        return data_dict


    def __call__(
        self,
        annotations,
        index,
        return_for_goal_dist=False,
        split="val", 
        return_objs=False,
        all_motions=None,
    ):
        path, relative, action, folder, scene_name = self.get_path(annotations, index)

        with open(path, "rb") as f:
            data = pkl.load(f)
            scene = data["scene_cut"]
            crop_mask = data["idx_crop"]  # unused

        all_objs = []
        if all_motions is not None:
            for motion in all_motions:
                objs = self.get_objs(
                    scene, crop_mask, path, motion=motion["motion_x_dict"]["x"]
                )  # obj_dict
                obj_info = self.all_objs_info(objs)
                all_objs.append(obj_info)
        else:
            objs = self.get_objs(
                scene, crop_mask, path, motion=data["joints"]
            )  # obj_dict
            obj_info = self.all_objs_info(objs)  # this obj info can be written in cache

        if len(all_objs) > 0:
            obj_info = {"all_objs": all_objs, "joints": data["joints"]}
        else:
            obj_info["joints"] = data["joints"]

        if return_objs:  # delete probably
            obj_info["obj_list"] = data["obj_list"]

        return (
            obj_info,
            relative,
        )
