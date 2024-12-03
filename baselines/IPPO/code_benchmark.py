import os
import time
from tqdm import tqdm
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from flax.traverse_util import flatten_dict
import safetensors.flax
import optax
import distrax
import jaxmarl
from jaxmarl.wrappers.baselines import get_space_dim, LogEnvState
from jaxmarl.wrappers.baselines import LogWrapper
import hydra
from omegaconf import OmegaConf
from typing import Sequence, NamedTuple, Any, Dict



def _tree_take(pytree, indices, axis=None):
    return jax.tree.map(lambda x: x.take(indices, axis=axis), pytree)

def _tree_shape(pytree):
    return jax.tree.map(lambda x: x.shape, pytree)

def _unstack_tree(pytree):
    leaves, treedef = jax.tree_util.tree_flatten(pytree)
    unstacked_leaves = zip(*leaves)
    return [jax.tree_util.tree_unflatten(treedef, leaves)
            for leaves in unstacked_leaves]

def _stack_tree(pytree_list, axis=0):
    return jax.tree.map(
        lambda *leaf: jnp.stack(leaf, axis=axis),
        *pytree_list
    )

def _concat_tree(pytree_list, axis=0):
    return jax.tree.map(
        lambda *leaf: jnp.concat(leaf, axis=axis),
        *pytree_list
    )

def _tree_split(pytree, n, axis=0):
    leaves, treedef = jax.tree.flatten(pytree)
    split_leaves = zip(
        *jax.tree.map(lambda x: jnp.array_split(x,n,axis), leaves)
    )
    return [
        jax.tree.unflatten(treedef, leaves)
        for leaves in split_leaves
    ]

def _take_episode(pipeline_states, dones, time_idx=-1, eval_idx=0):
    episodes = _tree_take(pipeline_states, eval_idx, axis=1)
    dones = dones.take(eval_idx, axis=1)
    return [
        state
        for state, done in zip(_unstack_tree(episodes), dones)
        if not (done)
    ]

def _compute_episode_returns(eval_info, time_axis=-2):
    done_arr = eval_info.done["__all__"]
    first_timestep = [slice(None) for _ in range(done_arr.ndim)]
    first_timestep[time_axis] = 0
    episode_done = jnp.cumsum(done_arr, axis=time_axis, dtype=bool)
    episode_done = jnp.roll(episode_done, 1, axis=time_axis)
    episode_done = episode_done.at[tuple(first_timestep)].set(False)
    undiscounted_returns = jax.tree.map(
        lambda r: (r*(1-episode_done)).sum(axis=time_axis),
        eval_info.reward
    )
    return undiscounted_returns



@hydra.main(version_base=None, config_path="config", config_name="ippo_mabrax_benchmark")
def main(config):
    config = OmegaConf.to_container(config, resolve=True)

    # IMPORT FUNCTIONS BASED ON ARCHITECTURE
    match (config["network"]["recurrent"], config["network"]["agent_param_sharing"]):
        case (False, False):
            from ippo_ff_nps_mabrax import make_train
        case (False, True):
            from ippo_ff_ps_mabrax import make_train
        case (True, False):
            from ippo_rnn_nps_mabrax import make_train
        case (True, True):
            from ippo_rnn_ps_mabrax import make_train

    rng = jax.random.PRNGKey(config["SEED"])
    train_rng, eval_rng = jax.random.split(rng)
    train_rngs = jax.random.split(train_rng, config["NUM_SEEDS"])    
    save_train_state = config.get("SAVE_TRAIN_STATE", False)
    with jax.disable_jit(config["DISABLE_JIT"]):
        train_jit = jax.jit(
            make_train(config, save_train_state=save_train_state),
            device=jax.devices()[config["DEVICE"]]
        )
        # first run (includes JIT)
        start_time_j = time.time()
        print(f"JIT+Compute start: {time.ctime()}")
        out = jax.vmap(train_jit, in_axes=(0, None, None, None))(
            train_rngs,
            config["LR"], config["ENT_COEF"], config["CLIP_EPS"]
        )
        start_time_c = time.time()
        print(f"Compute start: {time.ctime()}")
        out = jax.vmap(train_jit, in_axes=(0, None, None, None))(
            train_rngs,
            config["LR"], config["ENT_COEF"], config["CLIP_EPS"]
        )
        end_time = time.time()
        print(f"Compute end: {time.ctime()}")
        compute_time = end_time - start_time_c
        jit_time = (start_time_c-start_time_j) - compute_time
        sps = config['TOTAL_TIMESTEPS']/compute_time
        mem_stats = jax.devices()[0].memory_stats()
        mem_util = mem_stats["peak_bytes_in_use"]/mem_stats["bytes_limit"]
        print(f"[[[ JIT: {jit_time:.2} --- TIME: {compute_time:.2} --- SPS: {int(sps)} --- MEM: {mem_util:.4}]]]")
        env_name = config["ENV_NAME"]
        os.makedirs(f"mabrax/{env_name}", exist_ok=True)
        with open(f"mabrax/{env_name}/benchmark.csv", mode="a", encoding="utf-8") as f:
            print(",".join((
                str(int(config["NUM_ENVS"])),
                str(int(config["TOTAL_TIMESTEPS"])),
                str(int(config["NUM_STEPS"])),
                str(int(config["UPDATE_EPOCHS"])),
                str(int(config["NUM_MINIBATCHES"])),
                str(int(sps)),
                str(int(jit_time)),
                str(int(compute_time)),
                f"{mem_util:.6f}",
                str(save_train_state)
                )),
                file=f
            )


if __name__ == "__main__":
    main()
