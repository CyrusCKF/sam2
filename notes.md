# Sam2 fine tune

## Installation

1. Create new venv `python3.10 -m venv .venv`
2. Activate venv `.\.venv\Scripts\activate` or `source .venv/bin/activate`
3. Install pytorch `pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu124`
4. Install dependencies `pip install -e ".[dev]"`

## Commands

- Train sam in floor\sam2 `python training/train.py -c configs/sam2.1_training/sam2.1_hiera_t_floorplan_finetune.yaml  --use-cluster 0 --num-gpus 1`
  - View tensorboard `tensorboard --bind_all --logdir ./sam2_logs/`

## Stuff

May modify the hyper parameters in `sam2\sam2\configs\sam2.1_training\sam2.1_hiera_t_floorplan_finetune.yaml`
