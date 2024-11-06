import os
import time
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from flax.traverse_util import flatten_dict, unflatten_dict
import safetensors.flax
import optax
import distrax
import jaxmarl
from jaxmarl.wrappers.baselines import get_space_dim, LogEnvState
from jaxmarl.wrappers.baselines import LogWrapper
import hydra
from omegaconf import OmegaConf
from typing import Sequence, NamedTuple, Any, Dict, Callable



@struct.dataclass
class EvalNetworkState:
    apply_fn: Callable = struct.field(pytree_node=False)
    params: Dict

def _tree_take(pytree, indices, axis=None):
    return jax.tree.map(lambda x: x.take(indices, axis=axis), pytree)

def _tree_shape(pytree):
    return jax.tree.map(lambda x: x.shape, pytree)

def _unstack_tree(pytree):
    leaves, treedef = jax.tree_util.tree_flatten(pytree)
    unstacked_leaves = list(zip(*leaves))
    return [jax.tree_util.tree_unflatten(treedef, leaves)
            for leaves in unstacked_leaves]

def _stack_tree(pytree_list, axis=0):
    return jax.tree.map(
        lambda *leaf: jnp.stack(leaf, axis=axis),
        *pytree_list
    )

def _take_episode(pipeline_states, dones, time_idx=None, eval_idx=0):
    if time_idx is not None:
        pipeline_states = _tree_take(pipeline_states, time_idx, axis=0)
        dones = dones.take(time_idx, axis=0)
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


@hydra.main(version_base=None, config_path="config", config_name="ippo_mabrax_render")
def main(config):
    config = OmegaConf.to_container(config, resolve=True)

    # IMPORT FUNCTIONS BASED ON ARCHITECTURE
    match (config["network"]["recurrent"], config["network"]["agent_param_sharing"]):
        case (False, False):
            from ippo_ff_nps_mabrax import make_train, make_evaluation, EvalInfoLogConfig
            from ippo_ff_nps_mabrax import MultiActorCritic as NetworkArch
        case (False, True):
            from ippo_ff_ps_mabrax import make_train, make_evaluation, EvalInfoLogConfig
            from ippo_ff_ps_mabrax import ActorCritic as NetworkArch
        case (True, False):
            from ippo_rnn_nps_mabrax import make_train, make_evaluation, EvalInfoLogConfig
            from ippo_rnn_nps_mabrax import MultiActorCriticRNN as NetworkArch
        case (True, True):
            from ippo_rnn_ps_mabrax import make_train, make_evaluation, EvalInfoLogConfig
            from ippo_rnn_ps_mabrax import ActorCriticRNN as NetworkArch
        case _:
            raise Exception

    rng = jax.random.PRNGKey(config["SEED"])
    rng, eval_rng = jax.random.split(rng)
    with jax.disable_jit(config["DISABLE_JIT"]):
        env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
        # LOAD PARAMS
        agent_params = {
            agent_name: unflatten_dict(safetensors.flax.load_file(path), sep='/')
            for agent_name, path in config["crossplay"]["paths"].items()
        }
        eval_env, run_eval = make_evaluation(config)
        eval_log_config = EvalInfoLogConfig(
            env_state=True,
            done=True,
            action=False,
            value=False,
            reward=True,
            log_prob=False,
            obs=False,
            info=False,
            avail_actions=False,
        )
        eval_jit = jax.jit(
            run_eval,
            static_argnames=["log_eval_info"],
        )
        network = NetworkArch(config=config)
        robot = _tree_take(
            agent_params["robot"],
            config["crossplay"]["robot_seed"],
            axis=0
        )
        human = _tree_take(
            agent_params["human"],
            config["crossplay"]["human_seed"],
            axis=0
        )
        team_network_state = EvalNetworkState(
            apply_fn=network.apply,
            params=_stack_tree([robot, human]),
        )
        xeval = eval_jit(eval_rng, team_network_state, eval_log_config)
        first_episode_done = jnp.cumsum(xeval.done["__all__"], axis=0, dtype=bool)
        first_episode_rewards = xeval.reward["__all__"] * (1-first_episode_done)
        first_episode_returns = first_episode_rewards.sum(axis=0)
        episode_argsort = jnp.argsort(first_episode_returns, axis=-1)
        worst_idx = episode_argsort.take(0,axis=-1)
        best_idx = episode_argsort.take(-1, axis=-1)
        median_idx = episode_argsort.take(episode_argsort.shape[-1]//2, axis=-1)

        from brax.io import html
        worst_episode = _take_episode(
            xeval.env_state.env_state.pipeline_state,
            first_episode_done,
            eval_idx=worst_idx,
        )
        median_episode = _take_episode(
            xeval.env_state.env_state.pipeline_state,
            first_episode_done,
            eval_idx=median_idx,
        )
        best_episode = _take_episode(
            xeval.env_state.env_state.pipeline_state,
            first_episode_done,
            eval_idx=best_idx,
        )
        html.save("final_worst.html", eval_env.sys, worst_episode)
        html.save("final_median.html", eval_env.sys, median_episode)
        html.save("final_best.html", eval_env.sys, best_episode)


if __name__ == "__main__":
    main()
