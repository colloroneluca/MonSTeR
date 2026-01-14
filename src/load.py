import os
from omegaconf import DictConfig
import logging
import hydra
import torch

from src.config import read_config

logger = logging.getLogger(__name__)


def extract_ckpt(run_dir, ckpt_name="last"):
    """Split the lightning checkpoint into
    separate state_dict modules for faster loading"""
    import torch

    ckpt_path = os.path.join(run_dir, f"logs/checkpoints/{ckpt_name}.ckpt")

    extracted_path = os.path.join(run_dir, f"{ckpt_name}_weights")
    os.makedirs(extracted_path, exist_ok=True)

    new_path_template = os.path.join(extracted_path, "{}.pt")
    ckpt_dict = torch.load(ckpt_path)
    state_dict = ckpt_dict["state_dict"]
    module_names = list(set([x.split(".")[0] for x in state_dict.keys()]))

    # should be ['motion_encoder', 'text_encoder', 'motion_decoder'] for example
    for module_name in module_names:
        path = new_path_template.format(module_name)
        sub_state_dict = {
            ".".join(x.split(".")[1:]): y.cpu()
            for x, y in state_dict.items()
            if x.split(".")[0] == module_name
        }
        torch.save(sub_state_dict, path)


def load_model(run_dir, **params):
    # Load last config
    cfg = read_config(run_dir)
    cfg.run_dir = run_dir
    return load_model_from_cfg(cfg, **params)


def load_model_from_cfg(
    cfg, ckpt_name="last", device="cpu", eval_mode=True, is_path=False, model=None
):
    # import src.prepare  # noqa
    import torch

    run_dir = cfg.run_dir
    if model is None:
        model = hydra.utils.instantiate(cfg.model)

    # Loading modules one by one
    # motion_encoder / text_encoder / text_decoder

    if is_path:
        pt_path = ckpt_name
        ckpt_dict = torch.load(pt_path)
        state_dict = ckpt_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)
    else:

        pt_path = os.path.join(run_dir, f"{ckpt_name}_weights")

        if not os.path.exists(pt_path):
            logger.info("The extracted model is not found. Split into submodules..")
            extract_ckpt(run_dir, ckpt_name)

        for fname in os.listdir(pt_path):
            module_name, ext = os.path.splitext(fname)

            if ext != ".pt":
                continue

            module = getattr(model, module_name, None)
            if module is None:
                continue

            module_path = os.path.join(pt_path, fname)
            state_dict = torch.load(module_path)
            module.load_state_dict(state_dict)
            logger.info(f"    {module_name} loaded")

    logger.info("Loading previous checkpoint done")
    model = model.to(device)
    logger.info(f"Put the model on {device}")
    if eval_mode:
        model = model.eval()
        logger.info("Put the model in eval mode")
    return model


def load_partial_state_dict(model, checkpoint_path):
    """
    Loads weights from a checkpoint into the model, copying only the parameters
    with matching dimensions and discarding the rest. Returns the percentage
    of parameters successfully loaded.

    Args:
        model (torch.nn.Module): The model to load parameters into.
        checkpoint_path (str): Path to the checkpoint file.

    Returns:
        float: Percentage of parameters successfully loaded.
    """
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Extract the state_dict from the checkpoint
    if "state_dict" in checkpoint:
        checkpoint_state_dict = checkpoint["state_dict"]
    else:
        checkpoint_state_dict = checkpoint

    # Get the model's current state_dict
    model_state_dict = model.state_dict()

    # Keep track of total parameters and loaded parameters
    total_params = 0
    loaded_params = 0

    # Create a new state_dict to load into the model
    new_state_dict = {}

    # Iterate over all parameters in the model
    for name, param in model_state_dict.items():
        total_params += param.numel()  # Total number of elements

        if name in checkpoint_state_dict:
            checkpoint_param = checkpoint_state_dict[name]
            if param.size() == checkpoint_param.size():
                # Sizes match, load the parameter
                new_state_dict[name] = checkpoint_param
                loaded_params += param.numel()
            else:
                # Size mismatch, keep the original parameter
                new_state_dict[name] = param
                # print(f"Size mismatch for '{name}': "
                #       f"checkpoint param size {checkpoint_param.size()}, "
                #       f"model param size {param.size()}. Skipping.")
        else:
            # Parameter not found in checkpoint
            new_state_dict[name] = param
            print(f"Parameter '{name}' not found in checkpoint. Skipping.")

    # Load the new state_dict into the model
    model.load_state_dict(new_state_dict)

    # Calculate the percentage of parameters loaded
    percentage_loaded = (loaded_params / total_params) * 100
    print(f"Loaded {percentage_loaded:.2f}% of the model parameters.")

    return percentage_loaded


def load_pretrained(model, cfg, ckpt_name):
    pt_path = os.path.join(ckpt_name)

    if not os.path.exists(pt_path):
        logger.info("The extracted model is not found. Split into submodules..")
        extract_ckpt(cfg.run_dir, ckpt_name)
    if ".ckpt" in ckpt_name:
        state_dict = torch.load(ckpt_name)["state_dict"]
        try:
            model.load_state_dict(state_dict, strict=False)
        except:
            load_partial_state_dict(module, module_path)
    else:
        for fname in os.listdir(pt_path):
            module_name, ext = os.path.splitext(fname)

            if ext != ".pt":
                continue

            module = getattr(model, module_name, None)
            if module is None:
                continue

            module_path = os.path.join(pt_path, fname)
            state_dict = torch.load(module_path)
            try:
                module.load_state_dict(state_dict, strict=False)
            except:
                load_partial_state_dict(module, module_path)
            logger.info(f"    {module_name} loaded")

        logger.info("Loading previous checkpoint done")
    return model


@hydra.main(version_base=None, config_path="../configs", config_name="load_model")
def hydra_load_model(cfg: DictConfig) -> None:
    run_dir = cfg.run_dir
    ckpt_name = cfg.ckpt
    device = cfg.device
    eval_mode = cfg.eval_mode
    return load_model(run_dir, ckpt_name, device, eval_mode)


if __name__ == "__main__":
    hydra_load_model()
