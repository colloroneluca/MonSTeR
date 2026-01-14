import torch

from typing import List, Dict, Optional
from torch import Tensor
from torch.utils.data import default_collate


def length_to_mask(length, device: torch.device = None) -> Tensor:
    if device is None:
        device = "cpu"

    if isinstance(length, list):
        length = torch.tensor(length, device=device)

    max_len = max(length)
    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len
    ) < length.unsqueeze(1)
    return mask


def collate_tensor_with_padding(batch: List[Tensor]) -> Tensor:
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate_x_dict(lst_x_dict: List, *, device: Optional[str] = None) -> Dict:

    x = collate_tensor_with_padding([x_dict["x"] for x_dict in lst_x_dict])
    if device is not None:
        x = x.to(device)
    length = [x_dict["length"] for x_dict in lst_x_dict]
    mask = length_to_mask(length, device=x.device)
    if "id" in lst_x_dict[0].keys():
        ids = [lst_x_dict[i]["id"] for i in range(len(lst_x_dict))]
        batch = {"x": x, "length": length, "mask": mask, "ids": ids}
    else:
        batch = {"x": x, "length": length, "mask": mask}
    if "rot_label" in lst_x_dict[0].keys():
        rot_labels = [[x_dict["rot_label"] for x_dict in lst_x_dict]]
        batch["rot_labels"] = rot_labels
    if "crop_mask" in lst_x_dict[0].keys():
        crops = [crop["crop_mask"].unsqueeze(0) for crop in lst_x_dict]
        batch["crop_mask"] = torch.concat(crops)
    return batch


def collate_text_motion(lst_elements: List, *, device: Optional[str] = None) -> Dict:
    one_el = lst_elements[0]
    keys = one_el.keys()

    x_dict_keys = [key for key in keys if "x_dict" in key]
    other_keys = [key for key in keys if "x_dict" not in key]

    batch = {key: default_collate([x[key] for x in lst_elements]) for key in other_keys}

    for key, val in batch.items():
        if isinstance(val, torch.Tensor) and device is not None:
            batch[key] = val.to(device)
    # if 'scene_x_dict' in keys:
    try:
        for key in x_dict_keys:
            if "scene" in key:
                if "all_objs" in lst_elements[0][key].keys():
                    batch[key] = {}
                    batch[key]["obj_fts"] = torch.concatenate(
                        [
                            x["obj_fts"].unsqueeze(0)
                            for x in lst_elements[0][key]["all_objs"]
                        ]
                    )
                    batch[key]["obj_locs"] = torch.concatenate(
                        [
                            x["obj_locs"].unsqueeze(0)
                            for x in lst_elements[0][key]["all_objs"]
                        ]
                    )
                    if "point_set" in lst_elements[0]["scene_x_dict"].keys():
                        batch[key]["point_set"] = [
                            x["point_set"] for x in lst_elements[0][key]["all_objs"]
                        ]
                    batch[key]["obj_boxes"] = torch.concatenate(
                        [
                            x["obj_boxes"].unsqueeze(0)
                            for x in lst_elements[0][key]["all_objs"]
                        ]
                    )
                    batch[key]["obj_labels"] = torch.concatenate(
                        [
                            x["obj_labels"].unsqueeze(0)
                            for x in lst_elements[0][key]["all_objs"]
                        ]
                    )
                    batch[key]["obj_sem_masks"] = torch.concatenate(
                        [
                            x["obj_sem_masks"].unsqueeze(0)
                            for x in lst_elements[0][key]["all_objs"]
                        ]
                    )
                    batch[key]["obj_masks"] = torch.concatenate(
                        [
                            x["obj_masks"].unsqueeze(0)
                            for x in lst_elements[0][key]["all_objs"]
                        ]
                    )
                    batch[key]["ids"] = [x[key]["ids"] for x in lst_elements]
                else:
                    batch[key] = {}
                    batch[key]["obj_fts"] = torch.concatenate(
                        [x[key]["obj_fts"].unsqueeze(0) for x in lst_elements]
                    )
                    batch[key]["obj_locs"] = torch.concatenate(
                        [x[key]["obj_locs"].unsqueeze(0) for x in lst_elements]
                    )
                    if "point_set" in lst_elements[0]["scene_x_dict"].keys():
                        batch[key]["point_set"] = [
                            x[key]["point_set"] for x in lst_elements
                        ]
                    batch[key]["obj_boxes"] = torch.concatenate(
                        [x[key]["obj_boxes"].unsqueeze(0) for x in lst_elements]
                    )
                    batch[key]["obj_labels"] = torch.concatenate(
                        [x[key]["obj_labels"].unsqueeze(0) for x in lst_elements]
                    )
                    batch[key]["obj_sem_masks"] = torch.concatenate(
                        [x[key]["obj_sem_masks"].unsqueeze(0) for x in lst_elements]
                    )
                    batch[key]["obj_masks"] = torch.concatenate(
                        [x[key]["obj_masks"].unsqueeze(0) for x in lst_elements]
                    )
                    batch[key]["ids"] = [x[key]["ids"] for x in lst_elements]
            else:
                batch[key] = collate_x_dict(
                    [x[key] for x in lst_elements], device=device
                )
    except:
        print()

    return batch
