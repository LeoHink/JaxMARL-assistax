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
from jaxmarl.wrappers.aht import ZooManager, LoadAgentWrapper
import hydra
from omegaconf import OmegaConf
import pandas as pd
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

def _take_episode(pipeline_states, dones, time_idx=-1, eval_idx=0):
    # TODO make sure this works for the NPS code.
    episodes = _tree_take(pipeline_states, time_idx, axis=0)
    episodes = _tree_take(episodes, eval_idx, axis=1)
    dones = dones.take(time_idx, axis=0)
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

 
@hydra.main(version_base=None, config_path="config", config_name="ippo_mabrax")
def main(config):
    config = OmegaConf.to_container(config, resolve=True)

    # IMPORT FUNCTIONS BASED ON ARCHITECTURE
    
    # Dictionary to hold functions per algorithm (this is disgusting so needs to be refactored)
    alg_funcs = {}

    if "IPPO" in config["crossplay"]["robot_algos"]:
        match (config["network"]["recurrent"], config["network"]["agent_param_sharing"]):
            case (False, False):
                from IPPO.ippo_ff_nps_mabrax import (
                    make_train as ippo_make_train,
                    make_evaluation as ippo_make_evaluation,
                    EvalInfoLogConfig as ippo_EvalInfoLogConfig,
                    MultiActorCritic as ippo_NetworkArch,
                )
                alg_funcs["IPPO"] = {
                    "make_train": ippo_make_train,
                    "make_evaluation": ippo_make_evaluation,
                    "EvalInfoLogConfig": ippo_EvalInfoLogConfig,
                    "NetworkArch": ippo_NetworkArch,
                }
            case (False, True):
                from IPPO.ippo_ff_ps_mabrax import (
                    make_train as ippo_make_train,
                    make_evaluation as ippo_make_evaluation,
                    EvalInfoLogConfig as ippo_EvalInfoLogConfig,
                    ActorCritic as ippo_NetworkArch,
                )
                alg_funcs["IPPO"] = {
                    "make_train": ippo_make_train,
                    "make_evaluation": ippo_make_evaluation,
                    "EvalInfoLogConfig": ippo_EvalInfoLogConfig,
                    "NetworkArch": ippo_NetworkArch,
                }
            case (True, False):
                from IPPO.ippo_rnn_nps_mabrax import (
                    make_train as ippo_make_train,
                    make_evaluation as ippo_make_evaluation,
                    EvalInfoLogConfig as ippo_EvalInfoLogConfig,
                    MultiActorCriticRNN as ippo_NetworkArch,
                )
                alg_funcs["IPPO"] = {
                    "make_train": ippo_make_train,
                    "make_evaluation": ippo_make_evaluation,
                    "EvalInfoLogConfig": ippo_EvalInfoLogConfig,
                    "NetworkArch": ippo_NetworkArch,
                }
            case (True, True):
                from IPPO.ippo_rnn_ps_mabrax import (
                    make_train as ippo_make_train,
                    make_evaluation as ippo_make_evaluation,
                    EvalInfoLogConfig as ippo_EvalInfoLogConfig,
                    ActorCriticRNN as ippo_NetworkArch,
                )
                alg_funcs["IPPO"] = {
                    "make_train": ippo_make_train,
                    "make_evaluation": ippo_make_evaluation,
                    "EvalInfoLogConfig": ippo_EvalInfoLogConfig,
                    "NetworkArch": ippo_NetworkArch,
                }
            case _:
                raise Exception("Invalid network configuration for IPPO")

    if "MAPPO" in config["crossplay"]["robot_algos"]:
        match (config["network"]["recurrent"], config["network"]["agent_param_sharing"]):
            case (False, False):
                from MAPPO.mappo_ff_nps_mabrax import (
                    make_train as mappo_make_train,
                    make_evaluation as mappo_make_evaluation,
                    EvalInfoLogConfig as mappo_EvalInfoLogConfig,
                    MultiActor as mappo_NetworkArch,
                )
                alg_funcs["MAPPO"] = {
                    "make_train": mappo_make_train,
                    "make_evaluation": mappo_make_evaluation,
                    "EvalInfoLogConfig": mappo_EvalInfoLogConfig,
                    "NetworkArch": mappo_NetworkArch,
                }
            case (False, True):
                from MAPPO.mappo_ff_ps_mabrax import (
                    make_train as mappo_make_train,
                    make_evaluation as mappo_make_evaluation,
                    EvalInfoLogConfig as mappo_EvalInfoLogConfig,
                    Actor as mappo_NetworkArch,
                )
                alg_funcs["MAPPO"] = {
                    "make_train": mappo_make_train,
                    "make_evaluation": mappo_make_evaluation,
                    "EvalInfoLogConfig": mappo_EvalInfoLogConfig,
                    "NetworkArch": mappo_NetworkArch,
                }
            case (True, False):
                from MAPPO.mappo_rnn_nps_mabrax import (
                    make_train as mappo_make_train,
                    make_evaluation as mappo_make_evaluation,
                    EvalInfoLogConfig as mappo_EvalInfoLogConfig,
                    MultiActorRNN as mappo_NetworkArch,
                )
                alg_funcs["MAPPO"] = {
                    "make_train": mappo_make_train,
                    "make_evaluation": mappo_make_evaluation,
                    "EvalInfoLogConfig": mappo_EvalInfoLogConfig,
                    "NetworkArch": mappo_NetworkArch,
                }
            case (True, True):
                from MAPPO.mappo_rnn_ps_mabrax import (
                    make_train as mappo_make_train,
                    make_evaluation as mappo_make_evaluation,
                    EvalInfoLogConfig as mappo_EvalInfoLogConfig,
                    ActorRNN as mappo_NetworkArch,
                )
                alg_funcs["MAPPO"] = {
                    "make_train": mappo_make_train,
                    "make_evaluation": mappo_make_evaluation,
                    "EvalInfoLogConfig": mappo_EvalInfoLogConfig,
                    "NetworkArch": mappo_NetworkArch,
                }
            case _:
                raise Exception("Invalid network configuration for MAPPO")

    if "MASAC" in config["crossplay"]["robot_algos"]:
        from MASAC.masac_ff_nps_mabrax import (
            make_train as masac_make_train,
            make_evaluation as masac_make_evaluation,
            EvalInfoLogConfig as masac_EvalInfoLogConfig,
            MultiSACActor as masac_NetworkArch,
        )
        alg_funcs["MASAC"] = {
            "make_train": masac_make_train,
            "make_evaluation": masac_make_evaluation,
            "EvalInfoLogConfig": masac_EvalInfoLogConfig,
            "NetworkArch": masac_NetworkArch,
        }


    rng = jax.random.PRNGKey(config["SEED"])
    rng, eval_rng = jax.random.split(rng)
    with jax.disable_jit(config["DISABLE_JIT"]):
        zoo = ZooManager(config["ZOO_PATH"])
        scenario = config["ENV_NAME"]

        human_index_filtered = zoo.index.query(f'scenario == "{scenario}"'
                                 ).query('scenario_agent_id == "human"')
        
        robo_filtered = {}
        for alg in config["crossplay"]["robot_algos"]:
            robo_filtered[alg] = zoo.index.query(f'algorithm == "{alg}"'
                                         ).query(f'scenario == "{scenario}"'
                                         ).query('scenario_agent_id == "robot"')
            

        
        env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
        # LOAD PARAMS
        
        agent_params = {
            agent_name: unflatten_dict(safetensors.flax.load_file(path), sep='/')
            for agent_name, path in config["crossplay"]["paths"].items()
        }
        eval_env, run_eval = make_evaluation(config)
        eval_log_config = EvalInfoLogConfig(
            env_state=False,
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
        def cross_evaluate(rng, params):
            spliced_params = jax.tree.map(
                lambda *p: jnp.stack(p, axis=0),
                *(params[a] for a in env.agents)
            )
            # network & train state
            eval_network_state = EvalNetworkState(
                apply_fn=network.apply,
                params=spliced_params,
            )
            return eval_jit(rng, eval_network_state, eval_log_config)

        xeval = jax.vmap(
            jax.vmap(
                cross_evaluate,
                in_axes=(None, {"robot": 0, "human": None}),
            ),
            in_axes=(None, {"robot": None, "human": 0}),
        )(eval_rng, agent_params)

        first_episode_returns = _compute_episode_returns(xeval)
        mean_episode_returns = first_episode_returns["__all__"].mean(axis=-1)
        jnp.save("xreturns.npy", mean_episode_returns)


if __name__ == "__main__":
    main()
