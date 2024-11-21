""" 
Based on the PureJaxRL Implementation of PPO
"""

import functools
import jax
import jax.numpy as jnp
import flax.linen as nn
import jax.numpy as jnp
from flax import struct
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import optax
import distrax
import jaxmarl
from jaxmarl.wrappers.baselines import get_space_dim, LogEnvState
from jaxmarl.wrappers.baselines import LogWrapper
import hydra
from omegaconf import OmegaConf
from typing import Sequence, NamedTuple, Any, Dict, Optional


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            # assume resets comes in with shape (n_step,)
            jnp.expand_dims(resets,-1),
            self.initialize_carry(rnn_state.shape),
            rnn_state
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(hidden_shape):
        hidden_size = hidden_shape[-1]
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), hidden_shape)

class ActorRNN(nn.Module):
    config: Dict

    @nn.compact
    def __call__(self, hstate, x):
        if self.config["network"]["activation"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        obs, done, avail_actions = x

        embedding = nn.Dense(
            self.config["network"]["embedding_dim"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        embedding = activation(embedding)

        rnn_in = (embedding, done)
        hstate, embedding = ScannedRNN()(hstate, rnn_in)

        actor_mean = nn.Dense(
            self.config["network"]["gru_hidden_dim"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0)
        )(embedding)
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
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_log_std))
        return hstate, pi

class CriticRNN(nn.Module):
    config: Dict

    @nn.compact
    def __call__(self, hstate, x):
        if self.config["network"]["activation"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        obs, done = x

        embedding = nn.Dense(
            self.config["network"]["embedding_dim"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        embedding = activation(embedding)

        rnn_in = (embedding, done)
        hstate, embedding = ScannedRNN()(hstate, rnn_in)
        critic = nn.Dense(
            self.config["network"]["gru_hidden_dim"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0)
        )(critic)

        return hstate, jnp.squeeze(critic, axis=-1)

class Transition(NamedTuple):
    done: jnp.ndarray
    all_done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    global_obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray

class ActorCriticTrainState(NamedTuple):
    actor: TrainState
    critic: TrainState

class ActorCriticHiddenState(NamedTuple):
    actor: jnp.ndarray
    critic: jnp.ndarray

class RunnerState(NamedTuple):
    train_state: ActorCriticTrainState
    env_state: LogEnvState
    last_obs: Dict[str, jnp.ndarray]
    last_done: jnp.ndarray
    last_all_done: jnp.ndarray
    hstate: ActorCriticHiddenState
    update_step: int
    rng: jnp.ndarray

class UpdateState(NamedTuple):
    train_state: ActorCriticTrainState
    init_hstate: ActorCriticHiddenState
    traj_batch: Transition
    advantages: jnp.ndarray
    targets: jnp.ndarray
    rng: jnp.ndarray

class UpdateBatch(NamedTuple):
    init_hstate: ActorCriticHiddenState
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
    return jnp.concatenate(tuple(qty[a] for a in agents))

def unbatchify(qty: jnp.ndarray, agents: Sequence[str]) -> Dict[str, jnp.ndarray]:
    """Convert batched array to dict of arrays."""
    # N.B. assumes the leading dimension is the agent dimension
    return dict(zip(agents, jnp.split(qty, len(agents))))

def make_train(config, save_train_state=False):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["OBS_DIM"] = get_space_dim(env.observation_space(env.agents[0]))
    config["ACT_DIM"] = get_space_dim(env.action_space(env.agents[0]))
    config["GOBS_DIM"] = get_space_dim(env.observation_space("global"))
    env = LogWrapper(env, replace_info=True)

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
        actor_network = ActorRNN(config=config)
        critic_network = CriticRNN(config=config)
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)
        init_x_actor = (
            jnp.zeros( # obs
                (1, env.num_agents*config["NUM_ENVS"], config["OBS_DIM"])
            ),
            jnp.zeros( # done
                (1, env.num_agents*config["NUM_ENVS"])
            ),
            jnp.zeros( # avail_actions
                (1, env.num_agents*config["NUM_ENVS"], config["ACT_DIM"])
            ),
        )
        init_hstate_actor = jnp.zeros(
            (env.num_agents*config["NUM_ENVS"], config["network"]["gru_hidden_dim"])
        )
        init_x_critic = (
            jnp.zeros( # obs
                (1, env.num_agents*config["NUM_ENVS"], config["GOBS_DIM"])
            ),
            jnp.zeros( # done
                (1, env.num_agents*config["NUM_ENVS"])
            ),
        )
        init_hstate_critic = jnp.zeros(
            (env.num_agents*config["NUM_ENVS"], config["network"]["gru_hidden_dim"])
        )
        actor_network_params = actor_network.init(actor_rng, init_hstate_actor, init_x_actor)
        critic_network_params = critic_network.init(critic_rng, init_hstate_critic, init_x_critic)
        if config["ANNEAL_LR"]:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule(lr), eps=config["ADAM_EPS"]),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule(lr), eps=config["ADAM_EPS"]),
            )
        else:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(lr, eps=config["ADAM_EPS"])
            )
            critic_tx = optax.chain(
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
        actor_train_state = TrainState.create(
            apply_fn=actor_network.apply,
            params=actor_network_params,
            tx=actor_tx,
        )
        critic_train_state = TrainState.create(
            apply_fn=critic_network.apply,
            params=critic_network_params,
            tx=critic_tx,
        )
        train_state = ActorCriticTrainState(
            actor=actor_train_state,
            critic=critic_train_state,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset)(reset_rng)
        init_dones = jnp.zeros((env.num_agents*config["NUM_ENVS"],), dtype=bool)
        init_all_dones = jnp.zeros((env.num_agents*config["NUM_ENVS"],), dtype=bool)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                rng = runner_state.rng
                obs_batch = batchify(runner_state.last_obs, env.agents)
                global_obs = jnp.tile(runner_state.last_obs["global"], (env.num_agents, 1))
                avail_actions = jax.vmap(env.get_avail_actions)(runner_state.env_state.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents)
                )
                actor_in = (
                    # add time dimension to pass to RNN
                    jnp.expand_dims(obs_batch, 0),
                    jnp.expand_dims(runner_state.last_done, 0),
                    jnp.expand_dims(avail_actions, 0),
                )
                critic_in = (
                    # add time dimension to pass to RNN
                    jnp.expand_dims(global_obs, 0),
                    jnp.expand_dims(runner_state.last_all_done, 0),
                )
                # SELECT ACTION
                actor_hstate, pi = runner_state.train_state.actor.apply_fn(
                    runner_state.train_state.actor.params,
                    runner_state.hstate.actor, actor_in,
                )
                rng, act_rng = jax.random.split(rng)
                action, log_prob = pi.sample_and_log_prob(seed=act_rng)
                # remove time dimension
                action = action.squeeze(0)
                log_prob = log_prob.squeeze(0)
                env_act = unbatchify(action, env.agents)

                # COMPUTE VALUE
                critic_hstate, value = runner_state.train_state.critic.apply_fn(
                    runner_state.train_state.critic.params,
                    runner_state.hstate.critic, critic_in,
                )
                value = value.squeeze(0) # remove time dimension

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step)(
                    rng_step, runner_state.env_state, env_act,
                )
                done_batch = batchify(done, env.agents)
                all_done = jnp.tile(done["__all__"], env.num_agents)
                info = jax.tree_util.tree_map(jnp.concatenate, info)
                transition = Transition(
                    done=done_batch,
                    action=action,
                    all_done=all_done,
                    value=value,
                    reward=batchify(reward, env.agents),
                    log_prob=log_prob,
                    obs=obs_batch,
                    global_obs=global_obs,
                    info=info,
                    avail_actions=avail_actions,
                )
                runner_state = RunnerState(
                    train_state=runner_state.train_state,
                    env_state=env_state,
                    last_obs=obsv,
                    last_done=done_batch,
                    last_all_done=all_done,
                    hstate=ActorCriticHiddenState(actor=actor_hstate, critic=critic_hstate),
                    update_step=runner_state.update_step,
                    rng=rng,
                )
                return runner_state, transition

            init_hstate = runner_state.hstate
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            last_global_obs = jnp.tile(runner_state.last_obs["global"], (env.num_agents, 1))
            critic_in = (
                # add time dimension to pass to RNN
                jnp.expand_dims(last_global_obs, 0),
                jnp.expand_dims(runner_state.last_done, 0),
            )
            _, last_val = runner_state.train_state.critic.apply_fn(
                runner_state.train_state.critic.params,
                runner_state.hstate.critic, critic_in,
            )
            last_val = last_val.squeeze(0)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.all_done,
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
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _actor_loss_fn(actor_params, init_actor_hstate, traj_batch, gae):
                        # RERUN NETWORK
                        actor_in = (
                            traj_batch.obs,
                            traj_batch.done,
                            traj_batch.avail_actions,
                        )
                        _, pi = train_state.actor.apply_fn(
                            actor_params,
                            init_actor_hstate.squeeze(0), # remove step dim
                            actor_in,
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        gae = (
                            (gae - gae.mean(axis=(-2,-1), keepdims=True))
                            / (gae.std(axis=(-2,-1), keepdims=True) + 1e-8)
                        )
                        pg_loss1 = ratio * gae
                        pg_loss2 = (
                            jnp.clip(
                                ratio,
                                clip_eps_min,
                                clip_eps_max,
                            )
                            * gae
                        )
                        pg_loss = -jnp.minimum(pg_loss1, pg_loss2)
                        pg_loss = pg_loss.mean()
                        entropy = pi.entropy().mean()
                        # debug metrics
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac_min = jnp.mean(ratio < clip_eps_min)
                        clip_frac_max = jnp.mean(ratio > clip_eps_max)
                        # ---
                        actor_loss = (
                            pg_loss.sum()
                            - ent_coef * entropy.sum()
                        )
                        return actor_loss, (
                            pg_loss,
                            entropy,
                            approx_kl,
                            clip_frac_min,
                            clip_frac_max,
                        )

                    def _critic_loss_fn(critic_params, init_critic_hstate, traj_batch, targets):
                        # RERUN NETWORK
                        critic_in = (
                            traj_batch.global_obs,
                            traj_batch.all_done,
                        )
                        _, value = train_state.critic.apply_fn(
                            critic_params,
                            init_critic_hstate.squeeze(0), # remove step dim
                            critic_in,
                        )
                        # CALCULATE VALUE LOSS
                        value_losses = jnp.square(value - targets)
                        value_loss = 0.5 * value_losses.mean()
                        critic_loss =  config["VF_COEF"] * value_loss
                        return critic_loss, (value_loss,)

                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    actor_loss, actor_grads = actor_grad_fn(
                        train_state.actor.params,
                        batch_info.init_hstate.actor,
                        batch_info.traj_batch,
                        batch_info.advantages,
                    )
                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                    critic_loss, critic_grads = critic_grad_fn(
                        train_state.critic.params,
                        batch_info.init_hstate.critic,
                        batch_info.traj_batch,
                        batch_info.targets,
                    )

                    train_state = ActorCriticTrainState(
                        actor = train_state.actor.apply_gradients(grads=actor_grads),
                        critic = train_state.critic.apply_gradients(grads=critic_grads),
                    )
                    loss_info = {
                        "total_loss": actor_loss[0] + critic_loss[0],
                        "actor_loss": actor_loss[1][0],
                        "critic_loss": critic_loss[1][0],
                        "entropy": actor_loss[1][1],
                        "approx_kl": actor_loss[1][2],
                        "clip_frac_min": actor_loss[1][3],
                        "clip_frac_max": actor_loss[1][4],
                    }
                    return train_state, loss_info

                rng = update_state.rng

                batch_size = config["NUM_ENVS"] * env.num_agents
                minibatch_size = batch_size // config["NUM_MINIBATCHES"]
                assert (
                    batch_size % minibatch_size == 0
                ), "unable to equally partition into minibatches"
                batch = UpdateBatch(
                    init_hstate=jax.tree.map( # add step dim
                        lambda x: jnp.expand_dims(x, 0),
                        update_state.init_hstate,
                    ),
                    traj_batch=update_state.traj_batch,
                    advantages=update_state.advantages,
                    targets=update_state.targets,
                )
                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, batch_size)
                # initial axes: (step, agent*env, ...)
                batch = jax.tree_util.tree_map(
                    lambda x: x.swapaxes(0,1),
                    batch
                ) # swap axes to (agent*env, step, ...)
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0),
                    batch
                ) # shuffle: maintains axes (agent*env, step ...)
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, (config["NUM_MINIBATCHES"], -1, *x.shape[1:])
                    ),
                    shuffled_batch
                ) # split into minibatches. axes (n_mini, minibatch_size, step, ...)
                minibatches = jax.tree_util.tree_map(
                    lambda x: x.swapaxes(1,2),
                    minibatches
                ) # swap axes to (n_mini, step, minibatch_size, ...)
                train_state, loss_info = jax.lax.scan(
                    _update_minbatch, update_state.train_state, minibatches
                )
                update_state = UpdateState(
                    train_state=train_state,
                    init_hstate=update_state.init_hstate,
                    traj_batch=update_state.traj_batch,
                    advantages=update_state.advantages,
                    targets=update_state.targets,
                    rng=rng,
                )
                return update_state, loss_info

            runner_rng, update_rng = jax.random.split(runner_state.rng)
            update_state = UpdateState(
                train_state=runner_state.train_state,
                init_hstate=init_hstate,
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
            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)
            loss_info = jax.tree_util.tree_map(lambda x: x.mean(), loss_info)
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
                last_all_done=runner_state.last_all_done,
                hstate=runner_state.hstate,
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
            last_all_done=init_all_dones,
            hstate=ActorCriticHiddenState(
                actor=init_hstate_actor,
                critic=init_hstate_critic,
            ),
            update_step=0,
            rng=_rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train

def make_evaluation(config):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    config["OBS_DIM"] = get_space_dim(env.observation_space(env.agents[0]))
    config["ACT_DIM"] = get_space_dim(env.action_space(env.agents[0]))
    config["GOBS_DIM"] = get_space_dim(env.observation_space("global"))
    env = LogWrapper(env, replace_info=True)
    max_steps = env.episode_length

    def run_evaluation(rng, train_state, log_eval_info=EvalInfoLogConfig()):
        rng_reset, rng_env = jax.random.split(rng)
        rngs_reset = jax.random.split(rng_reset, config["NUM_EVAL_EPISODES"])
        obsv, env_state = jax.vmap(env.reset)(rngs_reset)
        init_dones = jnp.zeros((env.num_agents*config["NUM_EVAL_EPISODES"],), dtype=bool)
        init_all_dones = jnp.zeros((env.num_agents*config["NUM_EVAL_EPISODES"],), dtype=bool)
        init_hstate_actor = jnp.zeros(
            (env.num_agents*config["NUM_EVAL_EPISODES"], config["network"]["gru_hidden_dim"])
        )
        init_hstate_critic = jnp.zeros(
            (env.num_agents*config["NUM_EVAL_EPISODES"], config["network"]["gru_hidden_dim"])
        )
        runner_state = RunnerState(
            train_state=train_state,
            env_state=env_state,
            last_obs=obsv,
            last_done=init_dones,
            last_all_done=init_all_dones,
            hstate=ActorCriticHiddenState(actor=init_hstate_actor, critic=init_hstate_critic),
            update_step=0,
            rng=rng_env,
        )

        def _env_step(runner_state, unused):
            rng = runner_state.rng
            obs_batch = batchify(runner_state.last_obs, env.agents)
            global_obs = jnp.tile(runner_state.last_obs["global"], (env.num_agents, 1))
            avail_actions = jax.vmap(env.get_avail_actions)(runner_state.env_state.env_state)
            avail_actions = jax.lax.stop_gradient(
                batchify(avail_actions, env.agents)
            )
            actor_in = (
                # add time dimension to pass to RNN
                jnp.expand_dims(obs_batch, 0),
                jnp.expand_dims(runner_state.last_done, 0),
                jnp.expand_dims(avail_actions, 0),
            )
            critic_in = (
                # add time dimension to pass to RNN
                jnp.expand_dims(global_obs, 0),
                jnp.expand_dims(runner_state.last_all_done, 0),
            )

            # SELECT ACTION
            actor_hstate, pi = runner_state.train_state.actor.apply_fn(
                runner_state.train_state.actor.params,
                runner_state.hstate.actor, actor_in,
            )
            rng, act_rng = jax.random.split(rng)
            action, log_prob = pi.sample_and_log_prob(seed=act_rng)
            # remove time dimension
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
            env_act = unbatchify(action, env.agents)

            # COMPUTE VALUE
            critic_hstate, value = runner_state.train_state.critic.apply_fn(
                runner_state.train_state.critic.params,
                runner_state.hstate.critic, critic_in,
            )
            value = value.squeeze(0) # remove time dimension

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["NUM_EVAL_EPISODES"])
            obsv, env_state, reward, done, info = jax.vmap(env.step)(
                rng_step, runner_state.env_state, env_act,
            )
            done_batch = batchify(done, env.agents)
            all_done = jnp.tile(done["__all__"], env.num_agents)
            info = jax.tree_util.tree_map(jnp.concatenate, info)
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
            )
            runner_state = RunnerState(
                train_state=runner_state.train_state,
                env_state=env_state,
                last_obs=obsv,
                last_done=done_batch,
                last_all_done=all_done,
                hstate=ActorCriticHiddenState(actor=actor_hstate, critic=critic_hstate),
                update_step=runner_state.update_step,
                rng=rng,
            )
            return runner_state, eval_info

        _, eval_info = jax.lax.scan(
            _env_step, runner_state, None, max_steps
        )

        return eval_info
    return env, run_evaluation

@hydra.main(version_base=None, config_path="config", config_name="mappo_mabrax")
def main(config):
    config = OmegaConf.to_container(config)
    rng = jax.random.PRNGKey(config["SEED"])
    hparam_rng, run_rng = jax.random.split(rng, 2)


    with jax.disable_jit(config["DISABLE_JIT"]):
        train_jit = jax.jit(
            make_train(config),
            device=jax.devices()[config["DEVICE"]]
        )
        out = train_jit(run_rng, config["LR"], config["ENT_COEF"], config["CLIP_EPS"])
        breakpoint()


if __name__ == "__main__":
    main()
