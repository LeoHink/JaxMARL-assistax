import os
import os.path as osp
import numpy as np
import jax.numpy as jnp
from collections import defaultdict
from itertools import product
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--base-dir",
    help="The base directory for the sweep",
    default=".",
)
parser.add_argument(
    "--agent-names",
    help="The agent names. Assumes directory structure is in same order as given",
    nargs="+",
    default=["human", "robot"],
)
parser.add_argument(
    "-o",
    "--output-dir",
    help="The directory to save the resulting npy files to. Metrics will be placed in the metrics subdirectory, and hyperparameters placed in the hparams subdirectory.",
)
args = parser.parse_args()

# TODO parse these from command line?
DB = {
    "range": [0.6, 1.0], # range restriction
    "strength": [0.6, 1.0], # joint strength
    "tremor": [0.0, 0.4], # tremor magnitude
    "seed": [0], # seed
}
N_SEEDS = 3

human_tags = [
    "human_r{}_s{}_t{}_{}.{}".format(*cfg)
    for cfg in product(DB["range"],
                       DB["strength"],
                       DB["tremor"],
                       DB["seed"],
                       range(N_SEEDS))
]
robot_tags = [
    "robot_r{}_s{}_t{}_{}.{}".format(*cfg)
    for cfg in product(DB["range"],
                       DB["strength"],
                       DB["tremor"],
                       DB["seed"],
                       range(N_SEEDS))
]
# loop over human
hxps = []
for hcfg in product(DB["range"], DB["strength"], DB["tremor"], DB["seed"]):
    rxps = []
    for rcfg in product(DB["range"], DB["strength"], DB["tremor"], DB["seed"]):
        xp = jnp.load(osp.join(
            args.base_dir,
            "human_r{}_s{}_t{}_{}".format(*hcfg),
            "robot_r{}_s{}_t{}_{}".format(*rcfg),
            "xreturns.npy"
        ))
        rxps.append(xp)
    hxps.append(jnp.concat(rxps, axis=1))
xp_matrix = jnp.concat(hxps, axis=0)

os.makedirs(args.output_dir, exist_ok=True)
np.save(osp.join(args.output_dir, f"xp_matrix.npy"), xp_matrix)
with open(osp.join(args.output_dir, f"human_tags.txt"), "w") as f:
    f.write("\n".join(human_tags))
with open(osp.join(args.output_dir, f"robot_tags.txt"), "w") as f:
    f.writelines("\n".join(robot_tags))
