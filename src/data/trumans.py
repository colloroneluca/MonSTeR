import os
from os.path import join as pjoin
import pickle as pkl
import json, orjson

from tqdm import tqdm

import numpy as np
import torch
from torch.utils import data
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

import open3d as o3d
import trimesh

from src.data.utils import pad_tensors

""" 
Previously used loading utils, mostly operating on meshes. Kept here for backward compatibility if needed
for future reference.


# def normalize(self, data):
#     shape_orig = data.shape
#     data = data.reshape((-1, 3))
#     # data = (data - self.mean) / self.std
#     data = -1.0 + 2.0 * (data - self.min) / (self.max - self.min)
#     data = data.reshape(shape_orig)

#     return data

#  def convert_action_dict(action_dict):
#         # each entry is
#         # structured as follows:
#         # START_END action_label_1
#         # START END action_label_2
#         # and so on...

#         new_action_dict = {}
#         for key, value in action_dict.items():
#             start, _, end = key.partition("_")
#             new_action_dict[int(start), int(end)] = value

#         return new_action_dict

# def __transform_mesh__(mesh, target_location, angles):
#     rotation = R.from_euler("xyz", angles, degrees=False).as_matrix()

#     # Embed the rotation matrix in a 4x4 transformation matrix
#     rotation_transform = np.eye(4)
#     rotation_transform[:3, :3] = rotation

#     # Translate the mesh from the origin to the target location
#     translation_to_target = np.eye(4)
#     translation_to_target[:3, 3] = target_location

#     # Combine the transformations: translate to origin -> rotate -> translate to target
#     transform_matrix = (
#         translation_to_target @ rotation_transform  # @ translation_to_origin
#     )

#     # Apply the transformation to the mesh
#     mesh.apply_transform(transform_matrix)

#     return mesh
"""

def __load_obj__(filepath):
    # load the object file
    mesh = trimesh.load(filepath)
    return mesh


""" DATA loading utils, most of them operate on pointclouds for faster processing/loading.
Additionally, most of them use our cached pointclouds converted from meshes for speed. """


def __load_obj_files_from_folder__(folder_path):
    obj_dict = {}

    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".obj"):
            # Get the file name without extension
            file_name_no_ext = os.path.splitext(file_name)[0]
            # Load the .obj file
            obj_path = os.path.join(folder_path, file_name)
            loaded_obj = __load_obj__(obj_path)
            obj_dict[file_name_no_ext] = loaded_obj
    return obj_dict


def __load_object_info_file__(object_file, start_frame):
    # each file contains the position of each frame in the object
    # we only use the initial position
    object_info = {}

    try:
        object_dict = np.load(object_file, allow_pickle=True).item()
        for key, value in object_dict.items():
            # each value is a dictionary containing rotation and position info
            # which are the lists described before
            object_info[key] = {
                "rotation": value["rotation"][start_frame],
                "location": value["location"][start_frame],
            }
    except:
        print(f"Error loading info file {object_file}, returning no object info")
    return object_info


def __transform_pc__(pc, target_location, angles):
    rotation = R.from_euler("xyz", angles, degrees=False).as_matrix()

    # Embed the rotation matrix in a 4x4 transformation matrix
    rotation_transform = np.eye(4)
    rotation_transform[:3, :3] = rotation

    # Translate the mesh from the origin to the target location
    translation_to_target = np.eye(4)
    translation_to_target[:3, 3] = target_location

    # Combine the transformations: translate to origin -> rotate -> translate to target
    transform_matrix = (
        translation_to_target @ rotation_transform  # @ translation_to_origin
    )

    pc = trimesh.PointCloud(pc)
    pc = pc.apply_transform(transform_matrix)

    # Apply the transformation to the mesh
    return np.array(pc.vertices)


def __trumansclass2vista__(trumans_class: str) -> int:
    # NOTE: this class is an ugly mapping
    match trumans_class:
        case "book_left_01" | "book_right_01":
            return 31
        case "bottle_03" | "bottle_04" | "bottle_06":
            return 42
        case "cabinet_base_01" | "cabinet_base_02" | "cabinet_base_03":
            return 15
        case "cabinet_door_01" | "cabinet_door_02" | "cabinet_door_03":
            return 359
        case "cup_01":
            return 52
        case "door_door_02":
            return 4
        case "drawer_base_01" | "drawer_base_02" | "drawer_base_03" | "drawer_base_04":
            return 217
        case (
            "drawer_drawer_01"
            | "drawer_drawer_02"
            | "drawer_drawer_03"
            | "drawer_drawer_04"
        ):
            return 217
        case "fridge_base_01" | "fridge_door_01":
            return 74
        case "handbag_01":
            return 39
        case "keyboard_02":
            return 41
        case "laptop_base_01" | "laptop_base_02":
            return 64
        case "laptop_screen_01" | "laptop_screen_02":
            return 362
        case "microwave_base_1" | "microwave_base_2":
            return 58
        case "microwave_door_1" | "microwave_door_2":
            return 58
        case "monitor_01":
            return 14
        case "mouse_01":
            return 103
        case "movable_chair_base_01" | "movable_chair_base_02":
            return 1
        case "movable_chair_seat_01" | "movable_chair_seat_02":
            return 1
        case "oven_base_01" | "oven_door_01":
            return 157
        case "pen_01":
            return 539  # actually pen holder, but no close match
        case "phone_01":
            return 51
        case "static_chair_03":
            return 1
        case "vase_03":
            return 258
        case "whiteboard_01":
            return 29
        case _:
            return -101


def __get_obj_dict__(object_folder, object_info, n_points=2000, obj_dict=None):
    """
    Given a list of object info,
    loads each object, applies rototranslation
    and finally returns a pointcloud.
    """

    added_objs = 0
    incorrectly_resized = ["movable_chair_base_02"]
    objs_dict = {}

    for obj_name, info in object_info.items():
        # load object mesh

        rotation = info["rotation"]
        location = info["location"]

        if obj_name in incorrectly_resized:
            # NOTE: for now, skip incorrectly sized/positioned objects mentioned above
            continue

        # Use pcs converted to np for faster loading
        object_pc_path = os.path.join(object_folder, obj_name + ".npy")
        if not os.path.exists(object_pc_path):
            obj_mesh = __load_obj__(os.path.join(object_folder, obj_name + ".obj"))
            # save object pc
            point_cloud_np = np.array(obj_mesh.sample(n_points))
            np.save(object_pc_path, point_cloud_np)
            point_cloud = point_cloud_np
        else:
            point_cloud = np.load(object_pc_path)

        # apply rotation and translation on pc
        point_cloud = __transform_pc__(point_cloud, location, rotation)

        # pad with 0's using torch to replace color
        zeros = np.zeros((n_points, 3))
        point_cloud = np.concatenate((point_cloud, zeros), axis=1)

        # add to dict
        # NOTE: added_objs acts as instance counter
        objs_dict[added_objs] = {
            "points": point_cloud,
            "class": __trumansclass2vista__(obj_name),
        }
        added_objs += 1
    return objs_dict


def __all_objs_info__(objs, num_points=4000, num_objs=20, split="train"):
    obj_fts = []
    obj_locs = []
    obj_boxes = []
    obj_pcds = [obj["points"] for k, obj in objs.items()]
    obj_labels = [obj["class"] for k, obj in objs.items()]

    for obj_pcd in obj_pcds:
        # build locs
        # if rot_matrix is not None:
        #     obj_pcd[:, :3] = np.matmul(obj_pcd[:, :3], rot_matrix.transpose())
        if len(obj_pcd > 100):
            obj_center = obj_pcd[:, :3].mean(0)
            obj_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
            obj_locs.append(np.concatenate([obj_center, obj_size], 0))
            # build box
            obj_box_center = (obj_pcd[:, :3].max(0) + obj_pcd[:, :3].min(0)) / 2
            obj_box_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
            obj_boxes.append(np.concatenate([obj_box_center, obj_box_size], 0))
            # subsample

            if split == "train":
                # subsample
                pcd_idxs = np.random.choice(
                    len(obj_pcd),
                    size=num_points,
                    replace=(len(obj_pcd) < num_points),
                )  # random here

            else:  # at test we just take points evenly spaced to avoid any randomness
                n_total = len(obj_pcd)
                if n_total >= num_points:
                    # Take self.num_points indices spread uniformly across 0 … n_total-1
                    pcd_idxs = np.linspace(0, n_total - 1, num_points, dtype=int)
                else:
                    # Not enough points: cycle through the original indices until we reach the target length
                    pcd_idxs = np.resize(np.arange(n_total), num_points)
            obj_pcd = obj_pcd[pcd_idxs]  # keep only the selected points
            ###################################
            # normalize
            obj_pcd[:, :3] = obj_pcd[:, :3] - obj_pcd[:, :3].mean(0)
            max_dist = np.max(np.sqrt(np.sum(obj_pcd[:, :3] ** 2, 1)))
            if max_dist < 1e-6:  # take care of tiny point-clouds, i.e., padding
                max_dist = 1
            obj_pcd[:, :3] = obj_pcd[:, :3] / max_dist
            obj_fts.append(obj_pcd)

    # HERE
    # convert to torch
    obj_fts = torch.from_numpy(np.stack(obj_fts, 0))
    obj_locs = torch.from_numpy(np.array(obj_locs))
    obj_boxes = torch.from_numpy(np.array(obj_boxes))
    obj_labels = (
        torch.LongTensor(obj_labels)
        if obj_labels[0] is not None
        else -torch.ones((obj_fts.shape[0]))
    )
    data_dict = {}

    N = obj_fts.shape[0]

    if N > num_objs:
        if split == "train":
            indices = torch.randperm(N)[:num_objs]
            ## Subsample the tensor along the first dimension
            ####################
            # Evenly spaced indices (0, 2, 4, … when N=4000 and num_objs=2000)
        else:
            step = N / num_objs  # fractional step
            indices = (
                (torch.arange(num_objs, dtype=torch.float32) * step).floor().long()
            )
        # Subsample the tensor along the first dimension
        data_dict["obj_fts"] = obj_fts[indices]
        data_dict["obj_locs"] = obj_locs[indices]
        data_dict["obj_boxes"] = obj_boxes[indices]
        data_dict["obj_labels"] = obj_labels[indices]
        data_dict["obj_masks"] = torch.arange(num_objs) < len(
            data_dict["obj_locs"]
        )  # O
        data_dict["obj_sem_masks"] = torch.arange(num_objs) < len(data_dict["obj_locs"])
    else:
        data_dict["obj_fts"] = pad_tensors(
            obj_fts, lens=num_objs, pad=1.0
        ).float()  # O, 1024, 6
        data_dict["obj_masks"] = torch.arange(num_objs) < len(obj_locs)  # O
        data_dict["obj_sem_masks"] = torch.arange(num_objs) < len(obj_locs)
        data_dict["obj_locs"] = pad_tensors(
            obj_locs, lens=num_objs, pad=0.0
        ).float()  # O, 3
        data_dict["obj_boxes"] = pad_tensors(
            obj_boxes, lens=num_objs, pad=0.0
        ).float()  # O, 3
        data_dict["obj_labels"] = pad_tensors(
            obj_labels, lens=num_objs, pad=-100
        ).long()  # O

    # data_dict = {
    #     "obj_fts": obj_fts, # N, 6
    #     "obj_locs": obj_locs, # N, 6
    #     "obj_labels": obj_labels, # N,
    #     "obj_boxes": obj_boxes, # N, 6
    # }
    return data_dict


# NOTE: unused??
def find_class(data):
    find_class = {}
    for obj in data["objects"]:
        find_class[obj["id"]] = obj["classTitle"]
    return find_class


def rotation_matrix_from_euler_angles(rx, ry, rz, degrees=True):
    if degrees:
        rx = np.deg2rad(rx)
        ry = np.deg2rad(ry)
        rz = np.deg2rad(rz)
    # Create rotation matrices for each axis
    Rx = np.array(
        [[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]]
    )
    Ry = np.array(
        [[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]]
    )
    Rz = np.array(
        [[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]]
    )
    # Assuming 'zyx' rotation order (intrinsic rotations)
    R = Rz @ Ry @ Rx
    return R


def __segment_pointcloud__(pointcloud_file, json_file, debug=False):
    label_cache_dir = "datasets/Trumans/Scene_mesh/labels_cache"
    os.makedirs(label_cache_dir, exist_ok=True)
    outdir = json_file.split(".")[0].split("/")[-1]
    full_out_dir = os.path.join("datasets/Trumans/Scene_mesh/labels_cache", outdir)
    os.makedirs(full_out_dir, exist_ok=True)
    pkl_name = os.path.join(full_out_dir, "objects.pkl")
    if os.path.exists(pkl_name):
        with open(pkl_name, "rb") as f:
            segmented_dict = pkl.load(f)
    else:
        print("Caching", pkl_name)
        segmented_dict = {}
        # Load point cloud data in PLY format
        pointcloud_file = pointcloud_file.replace(" ", "")
        # pcd = o3d.io.read_point_cloud(pointcloud_file)
        pcd = o3d.io.read_triangle_mesh(pointcloud_file)
        points = np.asarray(pcd.vertices)
        segmented_pcd = o3d.geometry.PointCloud()
        segmented_pcd.points = o3d.utility.Vector3dVector(points)
        pcd = segmented_pcd
        # Load JSON data
        all_indices = []
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
        except:
            print("Not found ", json_file)
            return {}
        # Process each figure
        find_class = {}
        for obj in data["objects"]:
            find_class[obj["id"]] = obj["classTitle"]
        for i, figure in enumerate(data["figures"]):
            id = figure["objectId"]
            name = find_class[id]
            if name == "Room":
                print("Skipping room")
                continue
            if name == "Fridge":
                name = "refrigerator"
            geometry = figure["geometry"]
            position = geometry["position"]
            rotation = geometry["rotation"]
            dimensions = geometry["dimensions"]
            # Extract position, rotation, dimensions
            px, py, pz = position["x"], position["y"], position["z"]
            rx, ry, rz = rotation["x"], rotation["y"], rotation["z"]
            dx, dy, dz = dimensions["x"], dimensions["y"], dimensions["z"]
            # Compute rotation matrix
            R = rotation_matrix_from_euler_angles(rx, ry, rz, degrees=True)
            # Create OrientedBoundingBox
            center = np.array([px, py, pz])
            extent = np.array([dx, dy, dz])
            obb = o3d.geometry.OrientedBoundingBox(center, R, extent)
            # Get indices of points within the bounding box
            indices = obb.get_point_indices_within_bounding_box(pcd.points)
            # Extract the points within the bounding box
            all_indices.append(indices)
            segmented_points = points[indices]
            segmented_dict[i] = {}
            segmented_dict[i]["points"] = np.concatenate(
                (segmented_points, np.zeros_like(segmented_points)), axis=-1
            )
            with open("src/external_comp/ThreeDVista/vista_int2class.pkl", "rb") as f:
                vista_classes = pkl.load(f)
            name2class = {v: k for k, v in vista_classes.items()}
            segmented_dict[i]["class"] = name2class[name.lower()]
            # Save the segmented points to a PLY file
            if debug:
                segmented_pcd = o3d.geometry.PointCloud()
                segmented_pcd.points = o3d.utility.Vector3dVector(segmented_points)
                # If the original point cloud has colors, include them
                if pcd.colors:
                    colors = np.asarray(pcd.colors)[indices]
                    segmented_pcd.colors = o3d.utility.Vector3dVector(colors)

                os.makedirs(outdir, exist_ok=True)
                output_file = f"{name}.ply"
                o3d.io.write_point_cloud(
                    os.path.join(outdir, output_file), segmented_pcd
                )
                # meshes.append(segmented_pcd.points)
                print(f"Saved {len(segmented_points)} points to {output_file}")
        all_indices = np.unique(np.concatenate(all_indices, axis=0))
        not_indices = np.setdiff1d(
            np.arange(len(points)), all_indices
        )  # [ind for ind in np.arange(len(points)) if ind not in all_indices]
        points_left = points[not_indices]
        segmented_dict[i + 1] = {}
        segmented_dict[i + 1]["points"] = np.concatenate(
            (points_left, np.zeros_like(points_left)), axis=-1
        )
        segmented_dict[i + 1]["class"] = 5  # (object)
        with open(pkl_name, "wb") as f:
            pkl.dump(segmented_dict, f)
    return segmented_dict


class TrumansDataset(data.Dataset):
    def __init__(
        self,
        cfg,
        mean,
        std,
        split_path,
        eval_mode=False,
        patch_size=16,
        fps=None,
        text_to_token_emb=None,
        text_to_sent_emb=None,
        # relative: bool,
        debug=False,
        **kwargs,
    ):
        """
        Modified version from https://github.com/jnnan/trumans_utils/tree/main/
        Loads motion data, scene data and textual description of the action/scene.
        """

        # generic params
        self.max_motion_length = cfg.dataset.max_motion_length
        self.padding = cfg.preprocess.padding

        self.patch_size = patch_size
        self.eval_mode = eval_mode
        self.split = kwargs["split"]

        self.debug = debug

        # useful folders
        self.object_folder = cfg.dataset.object_folder
        self.obj_dict = __load_obj_files_from_folder__(
            os.path.join(self.object_folder, "Object_mesh")
        )
        self.scene_mesh_folder = cfg.dataset.scene_mesh_folder
        action_filepath = cfg.dataset.action_filepath
        with open(action_filepath, "rb") as ff:
            self.action_json = orjson.loads(ff.read())

        # TODO: add in config
        self.all_actions_path = os.path.join(
            cfg.dataset.all_actions_path, "annotations.json"
        )
        print("Using captions in ", self.all_actions_path, "\n \n \n")
        with open(self.all_actions_path, "rb") as ff:
            self.all_action_json = orjson.loads(ff.read())

        self.joints_filepath = cfg.dataset.joints_filepath
        self.global_orient_filepath = cfg.dataset.global_orient_filepath
        self.translation_filepath = cfg.dataset.translation_filepath
        self.text_to_sent_emb = text_to_sent_emb
        self.text_to_token_emb = text_to_token_emb

        # split file
        split_file = os.path.join(split_path, kwargs["split"] + ".txt")
        if "train" in split_file:
            self.split = "train"
            if self.debug:
                split_file = os.path.join(split_path, "train_debug.txt")
        elif "val" in split_file:
            self.split = "val"
        elif "test" in split_file:
            self.split = "test"
        else:
            raise ValueError("Unknown split file")

        # load the split file and all the entries
        self.split_entries = []
        with open(split_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                self.split_entries.append(line.strip())

        # MOTION data (only one that fits in memory)
        # kinematic chain should be the same as humanise
        self.kinematic_chain = [
            [0, 2, 5, 8, 11],
            [0, 1, 4, 7, 10],
            [0, 3, 6, 9, 12, 15],
            [9, 14, 17, 19, 21],
            [9, 13, 16, 18, 20],
        ]

        # (3792068, 24, 3) array with joints position in the 24,3 format
        print("Loading motion joints data")
        self.joints = np.load(self.joints_filepath)
        self.global_orient = np.load(self.global_orient_filepath)
        self.trasl = np.load(self.translation_filepath)

        self.norm = np.array([-1.7961988, -0.10060275, -1.3675903])
        self.min = self.norm[0].astype(np.float32)
        self.max = self.norm[1].astype(np.float32)

        # max scene points
        self.max_points = cfg.dataset.max_points

    def __len__(self):
        # use old mapping to retrieve correct length
        return len(self.split_entries)

    def __getitem__(self, item):

        data_name_with_frames = self.split_entries[item]
        data_name, frames = data_name_with_frames.split("original")

        # skip first underscore and cast
        original_starting_frame, original_ending_frame = frames[1:].split("_")
        original_starting_frame = int(original_starting_frame)
        original_ending_frame = int(original_ending_frame)

        scene_name, action_frame_name = data_name[:-1].split("action")
        # remove underscore
        scene_filename = scene_name[:-1] + ".obj"
        action_name, frames = action_frame_name.split("frame")
        # remove underscores
        action_name = action_name[1:-1]
        frames = frames[1:]

        start_frame, end_frame = frames.split("_")
        start_frame = int(start_frame)
        end_frame = int(end_frame)

        object_info_dict = __load_object_info_file__(
            pjoin(self.object_folder, "Object_pose", action_name + ".npy"), start_frame
        )

        # STEP 2: return list of all objects, correctly rotated and sampled into pc
        object_mesh_folder = pjoin(self.object_folder, "Object_mesh")
        obj_dict = __get_obj_dict__(
            object_mesh_folder, object_info_dict, n_points=4000, obj_dict=self.obj_dict
        )

        json_folder = "datasets/Trumans/Scene_mesh/labels"
        json_file = "prova_Trumans_" + scene_name[:-1] + ".pcd.json"
        mesh_path = pjoin(self.scene_mesh_folder, scene_filename)
        seg_dict = __segment_pointcloud__(
            mesh_path, os.path.join(json_folder, json_file)
        )
        obj_num = len(obj_dict)
        seg_dict = {k + obj_num: v for k, v in seg_dict.items()}
        obj_dict.update(seg_dict)

        motion = self.joints[
            original_starting_frame + start_frame : original_starting_frame + end_frame
        ]

        # load motion data
        m_length = motion.shape[0]
        motion = motion[..., :22, :3]

        # Sampling for very long motions, to avoid memory issues
        if motion.shape[0] > 300:
            motion = motion[::2]

        obj_info = __all_objs_info__(obj_dict, split=self.split)
        caption = self.all_action_json[data_name_with_frames + "\n"]["annotations"][0][
            "text"
        ]

        caption = {
            "text": caption,
        }
        sent_emb = self.text_to_sent_emb(caption["text"])

        text_x_dict = self.text_to_token_emb(caption["text"])
        text_x_dict["length"] = int(text_x_dict["length"])

        feats_guo = torch.tensor(motion).reshape(len(motion), -1).float()
        output = {
            "motion_x_dict": {"x": feats_guo, "length": feats_guo.shape[0]},
            "text_x_dict": text_x_dict,
            "text": caption,
            "keyid": item,
            "sent_emb": sent_emb,
        }

        output["scene_x_dict"] = obj_info
        output["scene_x_dict"]["ids"] = str(
            scene_filename + "_actlabel_" + action_name + "_" + frames,
        )
        output["motion_x_dict"]["rot_label"] = False

        return output

