import os
import json
from omegaconf import DictConfig, OmegaConf


def save_config(cfg: DictConfig, path=None) -> str:
    if path is None:
        path = os.path.join(cfg.run_dir, "config.json")
    else:
        path = os.path.join(path, "config.json")
    config = OmegaConf.to_container(cfg, resolve=True)
    with open(path, "w") as f:
        string = json.dumps(config, indent=4)
        f.write(string)
    return path


def read_config(run_dir: str, return_json=False) -> DictConfig:
    path = os.path.join(run_dir, "config.json")
    with open(path, "r") as f:
        config = json.load(f)
    if return_json:
        return config
    cfg = OmegaConf.create(config)
    cfg.run_dir = run_dir
    return cfg


def custom_read_config(run_dir: str, return_json=False, cfg=None) -> DictConfig:
    path = os.path.join(run_dir, "config.json")
    args = cfg
    with open(path, "r") as f:
        config = json.load(f)

    if return_json:
        return config
    cfg = OmegaConf.create(config)
    cfg.run_dir = run_dir
    return cfg, args


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
