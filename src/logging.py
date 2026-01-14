import logging
import tqdm
import wandb
import json


def log_config(config_path):
    with open(config_path, "r") as json_file:
        config = json.load(json_file)
    wandb.config.update(config, allow_val_change=True)


def get_run_name(cfg):
    try:
        dataset = cfg.data.cfg.dataset["dataset_name"]
    except:
        print("Could not get dataset name from config, setting to 'MonSTeR'")
        dataset = "MonSTeR"
    name = f"{dataset+'|'}" + f"{'Baseline' if cfg.model.baseline else 'NoBaseline'}"
    return name


# from https://stackoverflow.com/questions/38543506/change-logging-print-function-to-tqdm-write-so-logging-doesnt-interfere-wit
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)
