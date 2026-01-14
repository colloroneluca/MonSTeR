# MonSTeR: A Unified Model for Motion, Scene, Text Retrieval

![MonSTeR Logo](https://drive.google.com/thumbnail?id=1KB5n4C6nNkx8JgWdmVUnfohDo7AwQAvB)  
**Authors:**  
Luca Collorone¹*, Matteo Gioia¹*, Massimiliano Pappa¹, Paolo Leoni¹, Giovanni Ficarra¹³, Or Litany², Indro Spinelli¹, Fabio Galasso¹  
(*equal contribution; author indices as in the paper)

## Contents
- [Abstract - What's MonSTeR?](#abstract---whats-monster)
- [Code](#code)
    - [Environment](#environment)
    - [Folder structure](#folder-structure)
    - [Datasets and checkpoints](#datasets-and-checkpoints)
- [Run the experiments](#run-the-experiments)
- [Citation](#citation)




## Abstract - What's MonSTeR?

Intention drives human movement in complex environments, but such movement can only happen if the surrounding context supports it. Despite the intuitive nature of this mechanism, existing research has not yet provided tools to evaluate the alignment between skeletal movement (motion), intention (text), and the surrounding context (scene).

In this work, we introduce **MonSTeR**, the first **MOtioN-Scene-TExt Retrieval** model. Inspired by the modeling of higher-order relations, MonSTeR constructs a unified latent space by leveraging unimodal and cross-modal representations. This allows MonSTeR to capture the intricate dependencies between modalities, enabling flexible but robust retrieval across various tasks.

Our results show that MonSTeR outperforms trimodal models that rely solely on unimodal representations. Furthermore, we validate the alignment of our retrieval scores with human preferences through a dedicated user study. We demonstrate the versatility of MonSTeR’s latent space on zero-shot **in-Scene Object Placement** and **Motion Captioning**.

## Code

### Environment 

Set up a working environment with the steps below. Only **conda install** is officially supported via issues.

1. Create and activate the env, then add CUDA tooling: `conda create -n MonSTeR python=3.10.14`, `conda activate MonSTeR`, `conda install nvidia/label/cuda-11.8.0::cuda-nvcc`.
2. Install PyTorch (CUDA 11.8 wheels): `pip install torch==2.0.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118`.
3. Install Python deps: `pip install -r requirements.txt`.
4. Build PointNet2 ops: `cd src/external_comp/ThreeDVista/model/vision/pointnet2`.
5. `python setup.py install` (if CUDA mismatch occurs, try `export CUDA_HOME=[YOUR_CONDA_PATH]/envs/MonSTeR/`).


### Folder structure

Before training or testing, ensure your workspace roughly matches the layout below:

```text
MonSTeR
├── configs/
├── datasets/
├── logs/
├── outputs/
├── src/
│   ├── callback/
│   ├── data/
│   ├── external_comp/
│   ├── logger/ 
│   ├── model/
│   ├── config.py
│   ├── load.py
│   └── logging.py
│
├── stats/
├── average_rank_metrics.py
├── compute_metric.py
├── README.md
├── requirements.txt
├── retrieval_MonSTeR.py
└── train_MonSTeR.py
```

### Datasets and checkpoints

The checkpoints and data can be downloaded at this link: TBA.

- Place checkpoints under `outputs/`.
- Place datasets under `datasets/`.

Please **do not** install additional 3DVista packages in this environment; they can break the setup.

## Run the experiments

### Train your own model

```python
python -m train_MonSTeR data=[DATA] model.no_single=False
```

Replace [DATA] with either humanise.yaml or trumans.yaml. 
If you use model.no_single=True you will train the "w/o single" ablation. 

- Run configurations auto-save in the corresponding Weights & Biases run folder. Example: `outputs/MonSTeR_humanise.yaml/wandb/[RUN_TIMESTAMP]/files/config.json`.
- Checkpoints for a run are under the matching run directory. Example: `outputs/MonSTeR_humanise.yaml/MonSTeR/[RUN_ID]/checkpoints/best_st2m-epoch=0.ckpt`.

### Inference 
```python
python -m retrieval_MonSTeR id=[YOUR_RUN_ID] data=[DATA]
```

To average metrics use:
```python
python average_rank_metrics.py --input_file outputs/MonSTeR_humanise.yaml/wandb/[RUN_TIMESTAMP]/files/contrastive_metrics/normal.yaml
```

To test pretrained checkpoints replace RUN_ID with 'hltiyn94' for HUMANISE+ and 'qrn0h1wr' for TRUMANS+.
## Citation

Please cite our paper if you use MonSTeR in your research:

```
@InProceedings{Collorone_2025_ICCV,
    author    = {Collorone, Luca and Gioia, Matteo and Pappa, Massimiliano and Leoni, Paolo and Ficarra, Giovanni and Litany, Or and Spinelli, Indro and Galasso, Fabio},
    title     = {MonSTeR: a Unified Model for Motion, Scene, Text Retrieval},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {10940-10949}
}
```

