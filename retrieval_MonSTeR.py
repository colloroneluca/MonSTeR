import os
import glob
from collections import defaultdict
import random
import wandb
import csv
import yaml
from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate

from tqdm import tqdm
import warnings
import logging

logger = logging.getLogger(__name__)

import torch
import pytorch_lightning as pl
import numpy as np

from src.config import read_config
from src.load import load_model_from_cfg
from src.model.metrics import all_3D_contrastive_metrics
from src.data.collate import collate_text_motion

warnings.filterwarnings("ignore", category=FutureWarning, message=".*resume_download.*")


### Config UTILS


class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                # Recursively convert nested dictionaries
                value = Config(value)
            elif isinstance(value, list):
                # If the value is a list, process each item
                value = [
                    Config(item) if isinstance(item, dict) else item for item in value
                ]
            setattr(self, key, value)


def write_csv(metrics, name):

    csv_file = f"{name}.csv"

    # Check if the file already exists
    file_exists = os.path.isfile(csv_file)

    # Write the data to the CSV file, appending if the file already exists
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        columns = list(metrics.keys())
        data = list(metrics.values())
        # If the file doesn't exist, write the header
        if not file_exists:
            writer.writerow(columns)

        # Write the row(s) of data
        # for row in data:
        writer.writerow(data)
        print("wrote CSV")


def log_table(run, metrics, protocol, filter_metrics=None):

    s2m = list(
        {k: v for k, v in metrics.items() if k.split("/")[-1] == "s2m"}.values()
    )[
        3
    ]  # taking the rank 5
    t2m = list(
        {k: v for k, v in metrics.items() if k.split("/")[-1] == "t2m"}.values()
    )[3]
    st2m = list(
        {k: v for k, v in metrics.items() if k.split("/")[-1] == "st2m"}.values()
    )[3]
    result = st2m - max([s2m, t2m])
    if filter_metrics:
        metrics = {
            k: v
            for k, v in metrics.items()
            if k.split("/")[-1] in filter_metrics and k.split("/")[-2] == "rank5"
        }
    protocol = list(list(metrics.items())[0][0].split("/"))[0]
    metrics.update({f"{protocol}/rank5/st2m-best": result})
    table = wandb.Table(
        columns=["Run_Name"] + ["T/" + k for k, v in metrics.items()],
        data=[[wandb.run.name] + [format(v, ".2f") for k, v in metrics.items()]],
    )
    run.log({f"{'Plain_' if filter_metrics else ''}Test_{protocol}": table})
    metrics.update({"Run_Name": wandb.run.name})
    write_csv(metrics, name=f"{'Plain_' if filter_metrics else ''}Test_{protocol}")
    logger.info(f"Logged Table {'Plain_' if filter_metrics else ''}Test_{protocol}")


def pop_key(cfg, key):
    attribute_name, _, sub_attribute_name = key.partition(".")
    if sub_attribute_name:
        attribute = getattr(cfg, attribute_name, None)
        if attribute is not None:
            delattr(attribute, sub_attribute_name)
            return getattr(attribute, sub_attribute_name, None)
    else:
        popped = getattr(cfg, attribute_name, None)
        if popped is not None:
            delattr(cfg, attribute_name)
            return popped
    return None


def save_metric(path, metrics):
    # Sort the metrics dictionary by key
    sorted_metrics = dict(sorted(metrics.items()))

    # Group metrics by the prefix before the first '/'
    grouped_metrics = defaultdict(dict)
    for key, value in sorted_metrics.items():
        prefix = key.split("/")[0]
        grouped_metrics[prefix][key] = value

    # Manually construct the YAML string with blank lines between different groups
    yaml_string = ""
    for prefix, items in grouped_metrics.items():
        if yaml_string:
            yaml_string += "\n"
        yaml_string += yaml.dump(
            items, sort_keys=True, default_flow_style=False
        ).strip()
        yaml_string += "\n"

    # Write the formatted YAML string to the file
    with open(path, "w") as f:
        f.write(yaml_string)


### UTILS for ckpt loading


def find_files_with_string(directory, search_string):
    matching_files = []

    for root, dirs, files in os.walk(directory):

        has_files_dir = "files" in dirs

        # case 1: match in PATH (run dir)
        if search_string in root and has_files_dir:
            matching_files.append(root)
            continue

        # case 2: match in FILENAME (fallback)
        for f in files:
            if search_string in f and has_files_dir:
                matching_files.append(os.path.join(root, f))
                break

    assert len(matching_files) == 1, f"Expected 1 match, got {matching_files}"
    return os.path.join(matching_files[0], "files")


def get_epoch_number(checkpoint):
    # Split by 'epoch=' and take the part after it
    # Split further by '.ckpt' to get only the number
    return int(checkpoint.split("epoch=")[1].split(".ckpt")[0])


def find_ckpt(run_dir, model_name="MonSTeR"):
    run_id = run_dir.split("/")[-2].split("-")[-1]
    dir_name = os.path.dirname(run_dir)
    base_dir = os.path.split(os.path.split(dir_name)[0])[0]
    ckpt_folder = os.path.join(
        base_dir, f"{model_name}/{run_id}/checkpoints/best*.ckpt"
    )
    all_ckpt_path = glob.glob(ckpt_folder)
    ckpt_path = max(all_ckpt_path, key=get_epoch_number)
    print(
        "Logging model from ",
        ckpt_path,
        " which was selected between: \n",
        all_ckpt_path,
    )
    return ckpt_path, run_id


def move_tensors_to_device(data, device):
    """
    Recursively move tensors in a nested dictionary to the specified device.

    Args:
        data (dict): A dictionary that may contain other dictionaries, lists, tuples, and tensors.
        device: The device to which tensors should be moved.

    Returns:
        The updated data structure with tensors moved to the specified device.
    """
    if isinstance(data, dict):
        return {k: move_tensors_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_tensors_to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_tensors_to_device(item, device) for item in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data


def compute_sim_matrix(
    model,
    dataset,
    keyids,
    batch_size=256,
    protocol=None,
    dl=None,
    return_motions=False,
    return_latents=False,
):
    device = model.device
    model.eval()
    with torch.inference_mode():
        all_data_splitted = dl
        # by batch (can be too costly on cuda device otherwise)
        validation_step_t_latents = []
        validation_step_m_latents = []
        validation_step_sent_emb = []
        validation_step_s_latents = []
        validation_step_ts_latents = []
        validation_step_tm_latents = []
        validation_step_ms_latents = []
        all_batches = []
        all_motions = []
        motions = None
        eval_dev = "cpu"
        for batch in tqdm(all_data_splitted, leave=False):
            batch = move_tensors_to_device(batch, device=device)
            if return_motions:
                (
                    losses,
                    t_latents,
                    m_latents,
                    s_latents,
                    ts_latents,
                    tm_latents,
                    ms_latents,
                    ma_latents,
                    matrices,
                    t_projected,
                    m_projected,
                    s_projected,
                    motions,
                ) = model.compute_scene_loss(
                    batch,
                    return_all=True,
                    sample_mean=True,
                    return_motions=return_motions,
                )
            else:
                (
                    losses,
                    t_latents,
                    m_latents,
                    s_latents,
                    ts_latents,
                    tm_latents,
                    ms_latents,
                ) = model.compute_scene_loss(
                    batch,
                    return_all=True,
                    sample_mean=True,
                    return_motions=return_motions,
                )
            validation_step_s_latents.append(
                s_latents.to(model.eval_dev) if s_latents is not None else None
            )
            validation_step_ts_latents.append(
                ts_latents.to(model.eval_dev) if ts_latents is not None else None
            )
            validation_step_tm_latents.append(
                tm_latents.to(model.eval_dev) if tm_latents is not None else None
            )
            validation_step_ms_latents.append(
                ms_latents.to(model.eval_dev) if ms_latents is not None else None
            )
            validation_step_t_latents.append(t_latents.to(model.eval_dev))
            validation_step_m_latents.append(m_latents.to(model.eval_dev))
            validation_step_sent_emb.append(batch["sent_emb"].to(model.eval_dev))
            all_motions.append(motions)
            all_batches.append(batch)

        modalities, sim_matrices, dataset_pair = model.compute_all_metrics(
            validation_step_t_latents,
            validation_step_m_latents,
            validation_step_s_latents,
            validation_step_sent_emb,
            validation_step_ts_latents,
            validation_step_tm_latents,
            validation_step_ms_latents,
            batches=all_batches,
            return_sims=True,
            return_dataset_pair=True,
        )
    returned = {
        "sim_matrix": sim_matrices,
        "sent_emb": validation_step_sent_emb,
        "modalities": modalities,
        "dataset_pair": dataset_pair,
        "motions": all_motions,
        "latents": (
            {"tm": validation_step_tm_latents, "ms": validation_step_ms_latents}
            if return_latents
            else None
        ),
    }
    return returned


@hydra.main(version_base=None, config_path="configs", config_name="retrieval")
def retrieval(newcfg: DictConfig) -> None:

    run_dir = find_files_with_string(newcfg.outputs, newcfg.id)
    protocol = newcfg.protocol
    threshold_val = newcfg.threshold
    device = newcfg.device
    ckpt_name, run_id = find_ckpt(run_dir)
    os.environ["WANDB_MODE"] = "offline"

    assert protocol in ["all", "all_no_nsim", "normal", "nsim", "guo", "vanilla"]
    if protocol == "all":
        protocols = ["normal", "nsim", "guo"]
    elif protocol == "all_no_nsim":
        protocols = ["normal", "guo"]
    elif protocol == "vanilla":
        protocols = ["normal"]
    else:
        protocols = [protocol]

    save_dir = os.path.join(run_dir, "contrastive_metrics")
    os.makedirs(save_dir, exist_ok=True)
    cfg = read_config(run_dir)
    pl.seed_everything(cfg.seed, workers=True)

    # Fix seed
    SEED = cfg.seed
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # required for CUDA determinism
    random.seed(SEED)
    np.random.seed(SEED)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    logger.info("Loading the model")
    model = load_model_from_cfg(
        cfg, ckpt_name, eval_mode=True, device=device, is_path=True
    )
    model.eval()
    datasets = {}
    results = {}
    cfg.dataloader_val["batch_size"] = (
        32  # Hardcoded to always be in conformity with how t2m (guo et al) works
    )
    for protocol in protocols:
        ########Data########
        if protocol not in datasets:
            if protocol in ["normal", "threshold", "guo"]:
                dataset = instantiate(
                    cfg.data, split="test", debug=False, preload=False
                )
                retr_dataloader = instantiate(
                    cfg.dataloader_val,
                    dataset=dataset,
                    pin_memory=True,
                    collate_fn=collate_text_motion,
                    shuffle=False,
                )
                datasets.update(
                    {key: dataset for key in ["normal", "threshold", "guo"]}
                )
        dataset = datasets[protocol]

        ########Retrieval########
        # Compute sim_matrix for each protocol
        if protocol not in results:
            if protocol in ["normal", "threshold"]:

                res = compute_sim_matrix(
                    model,
                    dataset,
                    None,
                    batch_size=cfg.dataloader_val["batch_size"],
                    dl=retr_dataloader,
                    return_motions=False,
                )
                results.update({key: res for key in ["normal", "threshold"]})
            elif protocol == "guo":
                results["guo"] = []
                for batch in retr_dataloader:
                    guo_res = compute_sim_matrix(
                        model,
                        None,
                        None,
                        batch_size=cfg.dataloader_val["batch_size"],
                        dl=[batch],
                        protocol="guo",
                    )
                    results["guo"].append(guo_res)

        ########Scoring########
        result = results[protocol]
        # Compute the metrics
        if protocol == "guo":
            all_metrics = {}
            protocol_name = protocol
            for x in result:
                for m, sim in zip(x["modalities"], x["sim_matrix"]):
                    metrics = all_3D_contrastive_metrics(
                        sim,
                        emb=None,
                        threshold=None,
                        modality=m,
                        dataset_pair=x["dataset_pair"],
                        prefix="Guo",
                    )
                    for metric, value in metrics.items():
                        if metric in all_metrics.keys():
                            all_metrics[metric].append(value)
                        else:
                            all_metrics[metric] = [value]
            avg_metrics = {}
            for metric in all_metrics.keys():
                avg_metrics[metric] = (
                    np.floor(np.array(all_metrics[metric]).mean() * 100) / 100
                ).item()
            metrics = avg_metrics
            protocol_name = protocol
        else:
            all_metrics = {}
            protocol_name = protocol
            if protocol == "threshold":
                emb = result["sent_emb"]
                threshold = threshold_val
                protocol_name = protocol + f"_{threshold}"
            else:
                emb, threshold = None, None

            for m, sim in zip(result["modalities"], result["sim_matrix"]):
                metrics = all_3D_contrastive_metrics(
                    sim,
                    emb=emb,
                    threshold=threshold,
                    modality=m,
                    dataset_pair=result["dataset_pair"],
                    prefix="Goal" if protocol_name == "goal" else None,
                )
                all_metrics.update(metrics)
                print(metrics, "\n --- \n")
            metrics = all_metrics

        metric_name = f"{protocol_name}.yaml"
        path = os.path.join(save_dir, metric_name)
        save_metric(path, metrics)
        logger.info(f"Testing done, metrics saved in:\n{path}")


if __name__ == "__main__":
    retrieval()
