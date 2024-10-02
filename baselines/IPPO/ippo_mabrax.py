import os
import time
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
    unstacked_leaves = list(zip(*leaves))
    return [jax.tree_util.tree_unflatten(treedef, leaves)
            for leaves in unstacked_leaves]

def _take_episode(pipeline_states, dones, time_idx=-1, eval_idx=0):
    episodes = _tree_take(pipeline_states, eval_idx, axis=1)
    dones = dones.take(eval_idx, axis=1)
    return [
        state
        for state, done in zip(_unstack_tree(episodes), dones)
        if not (done)
    ]

def _compute_episode_returns(eval_info, time_axis=-2):
    episode_done = jnp.cumsum(eval_info.done["__all__"], axis=time_axis, dtype=bool)
    episode_rewards = eval_info.reward["__all__"] * (1-episode_done)
    undiscounted_returns = episode_rewards.sum(axis=time_axis)
    return undiscounted_returns



@hydra.main(version_base=None, config_path="config", config_name="ippo_mabrax")
def main(config):
    config = OmegaConf.to_container(config)

    # IMPORT FUNCTIONS BASED ON ARCHITECTURE
    match (config["network"]["recurrent"], config["network"]["agent_param_sharing"]):
        case (False, False):
            from ippo_ff_nps_mabrax import make_train as make_train
            from ippo_ff_nps_mabrax import make_evaluation as make_evaluation
        case (False, True):
            from ippo_ff_ps_mabrax import make_train as make_train
            from ippo_ff_ps_mabrax import make_evaluation as make_evaluation
        case (True, False):
            from ippo_rnn_nps_mabrax import make_train as make_train
            from ippo_rnn_nps_mabrax import make_evaluation as make_evaluation
        case (True, True):
            from ippo_rnn_ps_mabrax import make_train as make_train
            from ippo_rnn_ps_mabrax import make_evaluation as make_evaluation

    rng = jax.random.PRNGKey(config["SEED"])
    train_rng, eval_rng = jax.random.split(rng)
    train_rngs = jax.random.split(train_rng, config["NUM_SEEDS"])    
    with jax.disable_jit(config["DISABLE_JIT"]):
        train_jit = jax.jit(
            make_train(config),
            device=jax.devices()[config["DEVICE"]]
        )
        # first run (includes JIT)
        out = jax.vmap(train_jit, in_axes=(0, None, None, None))(
            train_rngs,
            config["LR"], config["ENT_COEF"], config["CLIP_EPS"]
        )

        # SAVE TRAIN METRICS
        EXCLUDED_METRICS = ["train_state"]
        jnp.save("metrics.npy", {
            key: val
            for key, val in out["metrics"].items()
            if key not in EXCLUDED_METRICS
            },
            allow_pickle=True
        )

        # SAVE PARAMS
        all_train_states = out["metrics"]["train_state"]
        final_train_state = out["runner_state"].train_state
        safetensors.flax.save_file(
            flatten_dict(all_train_states.params, sep='/'),
            "all_params.safetensors"
        )
        safetensors.flax.save_file(
            flatten_dict(final_train_state.params, sep='/'),
            "final_params.safetensors"
        )

        # RUN EVALUATION
        eval_env, run_eval = make_evaluation(config)
        eval_jit = jax.jit(
            run_eval,
            static_argnames=["log_env_state"],
        )
        eval_all = jax.vmap(
            eval_jit, in_axes=(None, 0, None),
        )(eval_rng, _tree_take(all_train_states, 0, axis=0), False)

        # COMPUTE RETURNS
        first_episode_returns = _compute_episode_returns(eval_all)
        mean_episode_returns = first_episode_returns.mean(axis=-1)

        # SAVE RETURNS
        jnp.save("returns.npy", mean_episode_returns)

        # RENDER
        # Run episodes for render (saving env_state at each timestep)
        eval_final = eval_jit(eval_rng, _tree_take(final_train_state, 0, axis=0), True)
        first_episode_done = jnp.cumsum(eval_final.done["__all__"], axis=0, dtype=bool)
        first_episode_rewards = eval_final.reward["__all__"] * (1-first_episode_done)
        first_episode_returns = first_episode_rewards.sum(axis=0)
        episode_argsort = jnp.argsort(first_episode_returns, axis=-1)
        worst_idx = episode_argsort.take(0,axis=-1)
        best_idx = episode_argsort.take(-1, axis=-1)
        median_idx = episode_argsort.take(episode_argsort.shape[-1]//2, axis=-1)

        from brax.io import html
        worst_episode = _take_episode(
            eval_final.env_state.env_state.pipeline_state, first_episode_done,
            time_idx=-1, eval_idx=worst_idx,
        )
        median_episode = _take_episode(
            eval_final.env_state.env_state.pipeline_state, first_episode_done,
            time_idx=-1, eval_idx=median_idx,
        )
        best_episode = _take_episode(
            eval_final.env_state.env_state.pipeline_state, first_episode_done,
            time_idx=-1, eval_idx=best_idx,
        )
        html.save("final_worst.html", eval_env.sys, worst_episode)
        html.save("final_median.html", eval_env.sys, median_episode)
        html.save("final_best.html", eval_env.sys, best_episode)


if __name__ == "__main__":
    main()
