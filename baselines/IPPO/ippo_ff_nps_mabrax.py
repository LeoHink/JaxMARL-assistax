""" 
Based on the PureJaxRL Implementation of PPO
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import optax
import distrax
import jaxmarl
from jaxmarl.wrappers.baselines import get_space_dim, LogEnvState
from jaxmarl.wrappers.baselines import LogWrapper, LogCrossplayWrapper
# from jaxmarl.wrappers.aht import ZooManager, LoadAgentWrapper
from jaxmarl.wrappers.aht_all import ZooManager, LoadAgentWrapper, LoadEvalAgentWrapper
import hydra
from omegaconf import OmegaConf
from typing import Sequence, NamedTuple, Any, Dict, Optional
import os 
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="0.95"

import functools


@functools.partial(
    nn.vmap,
    in_axes=0, out_axes=0,
    variable_axes={"params": 0},
    split_rngs={"params": True},
    axis_name="agents",
)
class MultiActorCritic(nn.Module):
    config: Dict

    @nn.compact
    def __call__(self, x):
        if self.config["network"]["activation"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        obs, done, avail_actions = x

        actor_mean = nn.Dense(
            self.config["network"]["actor_hidden_dim"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.config["network"]["actor_hidden_dim"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.config["ACT_DIM"],
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )(actor_mean)
        actor_log_std = self.param(
            "log_std",
            nn.initializers.zeros,
            (self.config["ACT_DIM"],)
        )
        pi = (actor_mean, jnp.exp(actor_log_std))

        critic = nn.Dense(
            self.config["network"]["critic_hidden_dim"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        critic = activation(critic)
        critic = nn.Dense(
            self.config["network"]["critic_hidden_dim"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0)
        )(critic)

        return pi, jnp.squeeze(critic, axis=-1)

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray

class RunnerState(NamedTuple):
    train_state: TrainState
    env_state: LogEnvState
    last_obs: Dict[str, jnp.ndarray]
    last_done: jnp.ndarray
    update_step: int
    rng: jnp.ndarray
    ag_idx: Optional[int] = None # This is the index of the agent we are currently running in the parallel envs. This is only used for crossplay. It is a dict with keys being the agent names and values being the index of the agent in the parallel envs.

class UpdateState(NamedTuple):
    train_state: TrainState
    traj_batch: Transition
    advantages: jnp.ndarray
    targets: jnp.ndarray
    rng: jnp.ndarray

class UpdateBatch(NamedTuple):
    traj_batch: Transition
    advantages: jnp.ndarray
    targets: jnp.ndarray

class EvalInfo(NamedTuple):
    env_state: Optional[LogEnvState]
    done: Optional[jnp.ndarray]
    action: Optional[jnp.ndarray]
    value: Optional[jnp.ndarray]
    reward: Optional[jnp.ndarray]
    log_prob: Optional[jnp.ndarray]
    obs: Optional[jnp.ndarray]
    info: Optional[jnp.ndarray]
    avail_actions: Optional[jnp.ndarray]
    ag_idx: Optional[jnp.ndarray]

@struct.dataclass
class EvalInfoLogConfig:
    env_state: bool = True
    done: bool = True
    action: bool = True
    value: bool = True
    reward: bool = True
    log_prob: bool = True
    obs: bool = True
    info: bool = True
    avail_actions: bool = True

def batchify(qty: Dict[str, jnp.ndarray], agents: Sequence[str]) -> jnp.ndarray:
    """Convert dict of arrays to batched array."""
    return jnp.stack(tuple(qty[a] for a in agents))

def unbatchify(qty: jnp.ndarray, agents: Sequence[str]) -> Dict[str, jnp.ndarray]:
    """Convert batched array to dict of arrays."""
    # N.B. assumes the leading dimension is the agent dimension
    return dict(zip(agents, qty))

def make_train(config, save_train_state=False, load_zoo=False):
    if load_zoo:
        zoo = ZooManager(config["ZOO_PATH"])
        env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
        env = LoadAgentWrapper.load_from_zoo(env, zoo, load_zoo)
    else:
        env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["OBS_DIM"] = get_space_dim(env.observation_space(env.agents[0]))
    config["ACT_DIM"] = get_space_dim(env.action_space(env.agents[0]))
    env = LogWrapper(env, replace_info=True)
    print(f"Num updates: {config['NUM_UPDATES']}")
    print(f"Env: {config['ENV_NAME']}")
    def linear_schedule(initial_lr):
        def _linear_schedule(count):
            frac = (
                1.0
                - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
                / config["NUM_UPDATES"]
            )
            return initial_lr * frac
        return _linear_schedule

    def train(rng, lr, ent_coef, clip_eps):

        # INIT NETWORK
        network = MultiActorCritic(config=config)
        rng, network_rng = jax.random.split(rng)
        init_x = (
            jnp.zeros( # obs
                (env.num_agents, 1, config["OBS_DIM"])
            ),
            jnp.zeros( # done
                (env.num_agents, 1)
            ),
            jnp.zeros( # avail_actions
                (env.num_agents, 1, config["ACT_DIM"])
            ),
        )
        network_params = network.init(network_rng, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule(lr), eps=config["ADAM_EPS"]),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(lr, eps=config["ADAM_EPS"])
            )
        if config["SCALE_CLIP_EPS"]:
            clip_eps /= env.num_agents
        if config["RATIO_CLIP_EPS"]:
            clip_eps_min = 1.0 - clip_eps
            clip_eps_max = 1.0/(1.0 - clip_eps)
        else:
            clip_eps_min = 1.0 - clip_eps
            clip_eps_max = 1.0 + clip_eps
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset)(reset_rng)
        init_dones = jnp.zeros((env.num_agents, config["NUM_ENVS"]), dtype=bool)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                rng = runner_state.rng
                obs_batch = batchify(runner_state.last_obs, env.agents) # I'll need to change something here for global obs probably stack global obs twice
                avail_actions = jax.vmap(env.get_avail_actions)(runner_state.env_state.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents)
                )
                ac_in = (
                    obs_batch,
                    runner_state.last_done,
                    avail_actions
                )
                # SELECT ACTION
                (actor_mean, actor_std), value = runner_state.train_state.apply_fn(
                    runner_state.train_state.params,
                    ac_in,
                )
                actor_std = jnp.expand_dims(actor_std, axis=1)
                pi = distrax.MultivariateNormalDiag(actor_mean, actor_std)
                rng, act_rng = jax.random.split(rng)
                action, log_prob = pi.sample_and_log_prob(seed=act_rng)
                
                env_act = unbatchify(action, env.agents)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step)( # for pixel use env_state.env_state.frames instead of obs
                    rng_step, runner_state.env_state, env_act,
                )
                
                done_batch = batchify(done, env.agents)
                info = jax.tree_util.tree_map(lambda x: x.swapaxes(0,1), info)
                transition = Transition(
                    done=done_batch,
                    action=action,
                    value=value,
                    reward=batchify(reward, env.agents),
                    log_prob=log_prob,
                    obs=obs_batch,
                    info=info,
                    avail_actions=avail_actions,
                )
                runner_state = RunnerState(
                    train_state=runner_state.train_state,
                    env_state=env_state,
                    last_obs=obsv,
                    last_done=done_batch,
                    update_step=runner_state.update_step,
                    rng=rng,
                )
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            last_obs_batch = batchify(runner_state.last_obs, env.agents)
            ac_in = (
                last_obs_batch,
                runner_state.last_done,
                jnp.ones((env.num_agents, config["NUM_ENVS"], config["ACT_DIM"]), dtype=jnp.uint8),
            )
            _, last_val = runner_state.train_state.apply_fn(
                runner_state.train_state.params,
                ac_in,
            )

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=config["ADVANTAGE_UNROLL_DEPTH"],
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        ac_in = (
                            traj_batch.obs,
                            traj_batch.done,
                            traj_batch.avail_actions,
                        )
                        (actor_mean, actor_std), value = train_state.apply_fn(
                            params,
                            ac_in,
                        )
                        actor_std = jnp.expand_dims(actor_std, axis=1)
                        pi = distrax.MultivariateNormalDiag(actor_mean, actor_std)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_losses = jnp.square(value - targets)
                        value_loss = 0.5 * value_losses.mean(axis=-1)

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        gae = (
                            (gae - gae.mean(axis=-1, keepdims=True))
                            / (gae.std(axis=-1, keepdims=True) + 1e-8)
                        )
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                clip_eps_min,
                                clip_eps_max,
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean(axis=-1)
                        entropy = pi.entropy().mean(axis=-1)
                        # debug metrics
                        approx_kl = ((ratio - 1) - logratio).mean(axis=-1)
                        clip_frac_min = jnp.mean(ratio < clip_eps_min, axis=-1)
                        clip_frac_max = jnp.mean(ratio > clip_eps_max, axis=-1)
                        # ---
                        total_loss = (
                            loss_actor.sum()
                            + config["VF_COEF"] * value_loss.sum()
                            - ent_coef * entropy.sum()
                        )
                        return total_loss, (
                            value_loss,
                            loss_actor,
                            entropy,
                            approx_kl,
                            clip_frac_min,
                            clip_frac_max,
                        )

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, batch_info.traj_batch, batch_info.advantages, batch_info.targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    loss_info = {
                        "total_loss": total_loss[0],
                        "actor_loss": total_loss[1][1],
                        "critic_loss": total_loss[1][0],
                        "entropy": total_loss[1][2],
                        "approx_kl": total_loss[1][3],
                        "clip_frac_min": total_loss[1][4],
                        "clip_frac_max": total_loss[1][5],
                    }
                    return train_state, loss_info

                rng = update_state.rng

                batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
                minibatch_size = batch_size // config["NUM_MINIBATCHES"]
                assert (
                    batch_size % minibatch_size == 0
                ), "unable to equally partition into minibatches"
                batch = UpdateBatch(
                    traj_batch=update_state.traj_batch,
                    advantages=update_state.advantages,
                    targets=update_state.targets,
                )
                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, batch_size)
                batch = jax.tree_util.tree_map(
                    lambda x: x.swapaxes(1,2),
                    batch
                ) # swap axes to (step, env, agent, ...)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size, *x.shape[2:]), order="F"),
                    batch
                ) # reshape axes to (step*env, agent, ...)
                # order="F" preserves the agent and ... dimension
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0),
                    batch
                ) # shuffle: maintains axes (step*env, agent, ...)
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, (config["NUM_MINIBATCHES"], -1, *x.shape[1:])
                    ),
                    shuffled_batch
                ) # split into minibatches. axes (n_mini, minibatch_size, agent, ...)
                minibatches = jax.tree_util.tree_map(
                    lambda x: x.swapaxes(1,2),
                    minibatches
                ) # swap axes to (n_mini, agent, minibatch_size, ...)
                train_state, loss_info = jax.lax.scan(
                    _update_minbatch, update_state.train_state, minibatches
                )
                update_state = UpdateState(
                    train_state=train_state,
                    traj_batch=update_state.traj_batch,
                    advantages=update_state.advantages,
                    targets=update_state.targets,
                    rng=rng,
                )
                return update_state, loss_info

            runner_rng, update_rng = jax.random.split(runner_state.rng)
            update_state = UpdateState(
                train_state=runner_state.train_state,
                traj_batch=traj_batch,
                advantages=advantages,
                targets=targets,
                rng=update_rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            update_step = runner_state.update_step + 1
            metric = traj_batch.info
            metric = jax.tree_util.tree_map(lambda x: x.mean(axis=(0,2)), metric)
            loss_info = jax.tree_util.tree_map(lambda x: x.mean(axis=(0,1)), loss_info)
            metric = {
                **metric,
                **loss_info,
                "update_step": update_step,
                "env_step": update_step * config["NUM_STEPS"] * config["NUM_ENVS"],
            }
            if save_train_state:
                metric.update({"train_state": update_state.train_state})
            runner_state = RunnerState(
                train_state=update_state.train_state,
                env_state=runner_state.env_state,
                last_obs=runner_state.last_obs,
                last_done=runner_state.last_done,
                update_step=update_step,
                rng=runner_rng,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = RunnerState(
            train_state=train_state,
            env_state=env_state,
            last_obs=obsv,
            last_done=init_dones,
            update_step=0,
            rng=_rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


def make_evaluation(config, load_zoo=False, crossplay=False):
    if load_zoo:
        zoo = ZooManager(config["ZOO_PATH"])
        env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
        if crossplay:
            env = LoadEvalAgentWrapper.load_from_zoo(env, zoo, load_zoo)
        else:
            env = LoadAgentWrapper.load_from_zoo(env, zoo, load_zoo)
    else:
        env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    config["OBS_DIM"] = int(get_space_dim(env.observation_space(env.agents[0])))
    config["ACT_DIM"] = int(get_space_dim(env.action_space(env.agents[0])))
    if crossplay:
        env = LogCrossplayWrapper(env, replace_info=True, crossplay_info=crossplay) # this is stupid we do not need crossplay infor to be determined by crossplay
    else:
        env = LogWrapper(env, replace_info=True)
    max_steps = env.episode_length

    def run_evaluation(rngs, train_state, log_eval_info=EvalInfoLogConfig()):
        
        if crossplay:
            rng_reset, rng_env = jax.random.split(rngs[0])
        else:
            rng_reset, rng_env = jax.random.split(rngs)
        
        
        rngs_reset = jax.random.split(rng_reset, config["NUM_EVAL_EPISODES"])
        init_dones = jnp.zeros((env.num_agents, config["NUM_EVAL_EPISODES"],), dtype=bool)
        if crossplay:
            init_obsv, init_env_state = jax.vmap(env.reset, in_axes=(0, None))(rngs_reset, None) # init ag_idx with None
            init_runner_state = RunnerState(
                train_state=train_state,
                env_state=init_env_state,
                last_obs=init_obsv,
                last_done=init_dones,
                update_step=0,
                rng=rng_env,
                ag_idx=init_env_state.env_state.ag_idx # Init with None to start running epiodes
            )

        else:
            init_obsv, init_env_state = jax.vmap(env.reset)(rngs_reset)
            init_runner_state = RunnerState(
                train_state=train_state,
                env_state=init_env_state,
                last_obs=init_obsv,
                last_done=init_dones,
                update_step=0,
                rng=rng_env,
                # ag_idx=env_state.env_state.ag_idx['human'] # This is the wrong spot to be incrementing
            )
        

        def _run_episode(runner_state, episode_rng):
            
            
            rng_reset, rng_env = jax.random.split(episode_rng)
            rngs_reset = jax.random.split(rng_reset, config["NUM_EVAL_EPISODES"])
            init_dones = jnp.zeros((env.num_agents, config["NUM_EVAL_EPISODES"],), dtype=bool)
            if crossplay:
                obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(rngs_reset, runner_state.ag_idx) # I think this would skip the 0 index so I probably also want to init this with None
                
                runner_state = RunnerState(
                    train_state=runner_state.train_state,
                    env_state=env_state,
                    last_obs=obsv,
                    last_done=init_dones,
                    update_step=runner_state.update_step,
                    rng=rng_env,
                    ag_idx=env_state.env_state.ag_idx # This is dict {'human': ag_idx} will be of dimension len(parallel envs)
                )

            else:
                obsv, env_state = jax.vmap(env.reset)(rngs_reset)
                
                runner_state = RunnerState(
                    train_state=runner_state.train_state,
                    env_state=env_state,
                    last_obs=obsv,
                    last_done=init_dones,
                    update_step=runner_state.update_step,
                    rng=rng_env,
                    # ag_idx=env_state.env_state.ag_idx['human'] # This is the wrong spot to be incrementing
                )

            def _env_step(runner_state, unused):
                rng = runner_state.rng
                obs_batch = batchify(runner_state.last_obs, env.agents)
                avail_actions = jax.vmap(env.get_avail_actions)(runner_state.env_state.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents)
                )
                ac_in = (
                    obs_batch,
                    runner_state.last_done,
                    avail_actions
                )
                # SELECT ACTION
                (actor_mean, actor_std), value = runner_state.train_state.apply_fn(
                    runner_state.train_state.params,
                    ac_in,
                )
                
                actor_std = jnp.expand_dims(actor_std, axis=1)
                pi = distrax.MultivariateNormalDiag(actor_mean, actor_std)
                rng, act_rng = jax.random.split(rng)
                action, log_prob = pi.sample_and_log_prob(seed=act_rng)
                env_act = unbatchify(action, env.agents)
                
                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_EVAL_EPISODES"])
                obsv, env_state, reward, done, info = jax.vmap(env.step)(
                    rng_step, runner_state.env_state, env_act,
                )
                done_batch = batchify(done, env.agents)
                info = jax.tree_util.tree_map(lambda x: x.swapaxes(0,1), info)
                eval_info = EvalInfo(
                    env_state=(env_state if log_eval_info.env_state else None),
                    done=(done if log_eval_info.done else None),
                    action=(action if log_eval_info.action else None),
                    value=(value if log_eval_info.value else None),
                    reward=(reward if log_eval_info.reward else None),
                    log_prob=(log_prob if log_eval_info.log_prob else None),
                    obs=(obs_batch if log_eval_info.obs else None),
                    info=(info if log_eval_info.info else None),
                    avail_actions=(avail_actions if log_eval_info.avail_actions else None),
                    ag_idx=(runner_state.ag_idx if crossplay else None)
                )
                runner_state = RunnerState(
                    train_state=runner_state.train_state,
                    env_state=env_state,
                    last_obs=obsv,
                    last_done=done_batch,
                    update_step=runner_state.update_step,
                    rng=rng,
                    ag_idx=(runner_state.ag_idx if crossplay else None), # This is dict {'human': ag_idx} will )
                )
                return runner_state, eval_info
            
            runner_state, episode_eval_info = jax.lax.scan(
                _env_step, runner_state, None, max_steps
            )

            return runner_state, episode_eval_info

        if crossplay:
            runner_state, all_episode_eval_infos = jax.lax.scan(
                _run_episode, init_runner_state, rngs
        )
        else:
            runner_state, all_episode_eval_infos = _run_episode(init_runner_state, rngs)
        
        return all_episode_eval_infos
    
    return env, run_evaluation

# def make_evaluation(config, load_zoo=False, crossplay=False):
#     if load_zoo:
#         zoo = ZooManager(config["ZOO_PATH"])
#         env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
#         if crossplay:
#             env = LoadEvalAgentWrapper.load_from_zoo(env, zoo, load_zoo)
#         else:
#             env = LoadAgentWrapper.load_from_zoo(env, zoo, load_zoo)
#     else:
#         env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
#     config["OBS_DIM"] = int(get_space_dim(env.observation_space(env.agents[0])))
#     config["ACT_DIM"] = int(get_space_dim(env.action_space(env.agents[0])))
#     env = LogWrapper(env, replace_info=True, crossplay_info=True)
#     max_steps = env.episode_length

#     def run_evaluation(rng, train_state, log_eval_info=EvalInfoLogConfig(), num_episodes=1):

#         def _run_episode(episode_rng):

#             rng_reset, rng_env = jax.random.split(episode_rng) # Note the refers to the number we vmap across
#             rngs_reset = jax.random.split(rng_reset, config["NUM_EVAL_EPISODES"])
#             obsv, env_state = jax.vmap(env.reset)(rngs_reset)
#             init_dones = jnp.zeros((env.num_agents, config["NUM_EVAL_EPISODES"],), dtype=bool) # Note the refers to the number we vmap across

#             runner_state = RunnerState(
#                 train_state=train_state,
#                 env_state=env_state,
#                 last_obs=obsv,
#                 last_done=init_dones,
#                 update_step=0,
#                 rng=rng_env,
#             )

#             def _env_step(runner_state, unused):
#                 rng = runner_state.rng
#                 obs_batch = batchify(runner_state.last_obs, env.agents)
#                 avail_actions = jax.vmap(env.get_avail_actions)(runner_state.env_state.env_state)
#                 avail_actions = jax.lax.stop_gradient(
#                     batchify(avail_actions, env.agents)
#                 )
#                 ac_in = (
#                     obs_batch,
#                     runner_state.last_done,
#                     avail_actions
#                 )
#                 # SELECT ACTION
#                 (actor_mean, actor_std), value = runner_state.train_state.apply_fn(
#                     runner_state.train_state.params,
#                     ac_in,
#                 )
#                 actor_std = jnp.expand_dims(actor_std, axis=1)
#                 pi = distrax.MultivariateNormalDiag(actor_mean, actor_std)
#                 rng, act_rng = jax.random.split(rng)
#                 action, log_prob = pi.sample_and_log_prob(seed=act_rng)
#                 env_act = unbatchify(action, env.agents)

#                 # STEP ENV
#                 rng, _rng = jax.random.split(rng)
#                 rng_step = jax.random.split(_rng, config["NUM_EVAL_EPISODES"])
#                 obsv, env_state, reward, done, info = jax.vmap(env.step)(
#                     rng_step, runner_state.env_state, env_act,
#                 )
#                 
#                 done_batch = batchify(done, env.agents)
#                 info = jax.tree_util.tree_map(lambda x: x.swapaxes(0,1), info) # why do we do this actually? 
#                 eval_info = EvalInfo(
#                     env_state=(env_state if log_eval_info.env_state else None),
#                     done=(done if log_eval_info.done else None),
#                     action=(action if log_eval_info.action else None),
#                     value=(value if log_eval_info.value else None),
#                     reward=(reward if log_eval_info.reward else None),
#                     log_prob=(log_prob if log_eval_info.log_prob else None),
#                     obs=(obs_batch if log_eval_info.obs else None),
#                     info=(info if log_eval_info.info else None),
#                     avail_actions=(avail_actions if log_eval_info.avail_actions else None),
#                 )
#                 runner_state = RunnerState(
#                     train_state=runner_state.train_state,
#                     env_state=env_state,
#                     last_obs=obsv,
#                     last_done=done_batch,
#                     update_step=runner_state.update_step,
#                     rng=rng,
#                 )
#                 return runner_state, eval_info

#             _, episode_eval_info = jax.lax.scan(
#                 _env_step, runner_state, None, max_steps
#             )

#             return episode_eval_info
        
#         _, all_episode_eval_infos = jax.lax.scan(
#             lambda carry, rng: (carry, _run_episode(rng)),
#             None,
#             rng
#         )
#         return all_episode_eval_infos
        
#     return env, run_evaluation

@hydra.main(version_base=None, config_path="config", config_name="ippo_ff_mabrax")
def main(config):
    config_key = hash(config) % 2**62
    sweep_config = config.SWEEP
    config = OmegaConf.to_container(config)
    rng = jax.random.PRNGKey(config["SEED"])
    hparam_rng, run_rng = jax.random.split(rng, 2)
    # generate hyperparams
    NUM_HPARAM_CONFIGS = sweep_config.num_configs
    lr_rng, ent_coef_rng, clip_eps_rng = jax.random.split(hparam_rng, 3)

    if sweep_config.get("lr", False):
        lrs = 10**jax.random.uniform(
            lr_rng,
            shape=(NUM_HPARAM_CONFIGS,),
            minval=sweep_config.lr.min,
            maxval=sweep_config.lr.max,
        )
        lr_axis = 0
    else:
        lrs = config["LR"]
        lr_axis = None

    if sweep_config.get("ent_coef", False):
        ent_coefs = 10**jax.random.uniform(
            ent_coef_rng,
            shape=(NUM_HPARAM_CONFIGS,),
            minval=sweep_config.ent_coef.min,
            maxval=sweep_config.ent_coef.max,
        )
        ent_coef_axis = 0
    else:
        ent_coefs = config["ENT_COEF"]
        ent_coef_axis = None

    if sweep_config.get("clip_eps", False):
        clip_epss = 10**jax.random.uniform(
            clip_eps_rng,
            shape=(NUM_HPARAM_CONFIGS,),
            minval=sweep_config.clip_eps.min,
            maxval=sweep_config.clip_eps.max,
        )
        clip_eps_axis = 0
    else:
        clip_epss = config["CLIP_EPS"]
        clip_eps_axis = None

    run_rngs = jax.random.split(run_rng, config["NUM_SEEDS"])
    with jax.disable_jit(config["DISABLE_JIT"]):
        train_jit = jax.jit(
            make_train(config),
            device=jax.devices()[config["DEVICE"]]
        )
        out = jax.vmap(
            jax.vmap(
                train_jit,
                in_axes=(0, None, None, None),
            ),
            in_axes=(None, lr_axis, ent_coef_axis, clip_eps_axis)
        )(run_rngs, lrs, ent_coefs, clip_epss)
    jnp.save(f"metrics_{config_key}.npy", out["metrics"], allow_pickle=True)
    jnp.save(f"hparams_{config_key}.npy", {
        "lr": lrs,
        "ent_coef": ent_coefs,
        "clip_eps": clip_epss,
        "ratio_clip_eps": config["RATIO_CLIP_EPS"],
        "num_steps": config["NUM_STEPS"],
        "num_envs": config["NUM_ENVS"],
        "update_epochs": config["UPDATE_EPOCHS"],
        "num_minibatches": config["NUM_MINIBATCHES"],
        }
    )


if __name__ == "__main__":
    main()
