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
from jaxmarl.wrappers.aht import ZooManager, LoadAgentWrapper
import hydra
from omegaconf import OmegaConf
import pandas as pd
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



@hydra.main(version_base=None, config_path="config", config_name="ippo_mabrax_aht")
def main(config):
    config = OmegaConf.to_container(config, resolve=True)

    # IMPORT FUNCTIONS BASED ON ARCHITECTURE
    match (config["network"]["recurrent"], config["network"]["agent_param_sharing"]):
        case (False, False):
            from ippo_ff_nps_mabrax import make_train, make_evaluation, EvalInfoLogConfig
        case (False, True):
            from ippo_ff_ps_mabrax import make_train, make_evaluation, EvalInfoLogConfig
        case (True, False):
            from ippo_rnn_nps_mabrax import make_train, make_evaluation, EvalInfoLogConfig
        case (True, True):
            from ippo_rnn_ps_mabrax import make_train, make_evaluation, EvalInfoLogConfig

    rng = jax.random.PRNGKey(config["SEED"])
    train_rng, eval_rng = jax.random.split(rng)
    train_rngs = jax.random.split(train_rng, config["NUM_SEEDS"])    
    print(f"Starting training with {config['TOTAL_TIMESTEPS']} timesteps \n num envs: {config['NUM_ENVS']} \n num seeds: {config['NUM_SEEDS']} \n for env: {config['ENV_NAME']}")
    with jax.disable_jit(config["DISABLE_JIT"]):
        zoo = ZooManager(config["ZOO_PATH"])
        alg = config["ALGORITHM"]
        env_name = config["ENV_NAME"]
        index_filtered = zoo.index.query(f'algorithm == "{alg}"').query(f'scenario == "{env_name}"').query('scenario_agent_id == "human"')
        train_set = index_filtered.sample(frac=0.5)
        breakpoint()
        test_set = index_filtered.drop(train_set.index)
        env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
        train_jit = jax.jit(
            make_train(
                config,
                save_train_state=True,
                load_zoo={"human": list(train_set.agent_uuid)}
            ),
            device=jax.devices()[config["DEVICE"]]
        )
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
        env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
        all_train_states = out["metrics"]["train_state"]
        final_train_state = out["runner_state"].train_state
        safetensors.flax.save_file(
            flatten_dict(all_train_states.params, sep='/'),
            "all_params.safetensors"
        )
        if config["network"]["agent_param_sharing"]:
            safetensors.flax.save_file(
                flatten_dict(final_train_state.params, sep='/'),
                "final_params.safetensors"
            )
        else:
            # split by agent
            split_params = _unstack_tree(
                jax.tree.map(lambda x: x.swapaxes(0,1), final_train_state.params)
            )
            for agent, params in zip(env.agents, split_params):
                safetensors.flax.save_file(
                    flatten_dict(params, sep='/'),
                    f"{agent}.safetensors",
                )

        # RUN EVALUATION
        # Assume the first 2 dimensions are batch dims
        batch_dims = jax.tree.leaves(_tree_shape(all_train_states.params))[:2]
        n_sequential_evals = int(jnp.ceil(
            config["NUM_EVAL_EPISODES"] * jnp.prod(jnp.array(batch_dims))
            / config["GPU_ENV_CAPACITY"]
        ))
        def _flatten_and_split_trainstate(trainstate):
            # We define this operation and JIT it for memory reasons
            flat_trainstate = jax.tree.map(
                lambda x: x.reshape((x.shape[0]*x.shape[1],*x.shape[2:])),
                trainstate
            )
            return _tree_split(flat_trainstate, n_sequential_evals)
        split_trainstate = jax.jit(_flatten_and_split_trainstate)(all_train_states)
        breakpoint()
        eval_train_env, run_eval_train = make_evaluation(config, load_zoo={"human": list(train_set.agent_uuid)})
        eval_test_env, run_eval_test = make_evaluation(config, load_zoo={"human": list(test_set.agent_uuid)})
                
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
        eval_train_jit = jax.jit(run_eval_train, static_argnames=["log_eval_info"])
        eval_train_vmap = jax.vmap(eval_train_jit, in_axes=(None, 0, None))
        eval_test_jit = jax.jit(run_eval_test, static_argnames=["log_eval_info"])
        eval_test_vmap = jax.vmap(eval_test_jit, in_axes=(None, 0, None))
        evals_train = _concat_tree([
            eval_train_vmap(eval_rng, ts, eval_log_config)
            for ts in tqdm(split_trainstate, desc="Evaluation batches")
        ])
        evals_train = jax.tree.map(
            lambda x: x.reshape((*batch_dims, *x.shape[1:])),
            evals_train
        )
        evals_test = _concat_tree([
            eval_test_vmap(eval_rng, ts, eval_log_config)
            for ts in tqdm(split_trainstate, desc="Evaluation batches")
        ])
        evals_test = jax.tree.map(
            lambda x: x.reshape((*batch_dims, *x.shape[1:])),
            evals_test
        )

        # COMPUTE RETURNS
        train_first_episode_returns = _compute_episode_returns(evals_train)
        train_first_episode_returns = train_first_episode_returns["__all__"]
        train_mean_episode_returns = train_first_episode_returns.mean(axis=-1)
        test_first_episode_returns = _compute_episode_returns(evals_test)
        test_first_episode_returns = test_first_episode_returns["__all__"]
        test_mean_episode_returns = test_first_episode_returns.mean(axis=-1)

        # SAVE RETURNS
        jnp.save("train_returns.npy", train_mean_episode_returns)
        jnp.save("test_returns.npy", test_mean_episode_returns)



if __name__ == "__main__":
    main()
