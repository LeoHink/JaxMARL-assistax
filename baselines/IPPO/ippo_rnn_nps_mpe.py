""" 
Based on the PureJaxRL Implementation of PPO
"""

import functools
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import optax
import distrax
import jaxmarl
from jaxmarl.wrappers.baselines import get_space_dim, LogEnvState
from jaxmarl.wrappers.baselines import MPELogWrapper as LogWrapper 
import hydra
from omegaconf import OmegaConf
from typing import Sequence, NamedTuple, Any, Dict

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

class ActorCriticRNN(nn.Module):
    config: Dict

    @nn.compact
    def __call__(self, hstate, x):
        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        obs, done, avail_actions = x

        embedding = nn.Dense(
            self.config["EMBEDDING_DIM"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        embedding = activation(embedding)

        rnn_in = (embedding, done)
        hstate, embedding = ScannedRNN()(hstate, rnn_in)

        actor_logits = nn.Dense(
            self.config["GRU_HIDDEN_DIM"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0)
        )(embedding)
        actor_logits = activation(actor_logits)
        actor_logits = nn.Dense(
            self.config["ACT_DIM"],
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )(actor_logits)
        pi = distrax.Categorical(logits=actor_logits)

        critic = nn.Dense(
            self.config["GRU_HIDDEN_DIM"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0)
        )(critic)

        return hstate, pi, jnp.squeeze(critic, axis=-1)

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
    hstate: jnp.ndarray
    update_step: int
    rng: jnp.ndarray

class UpdateState(NamedTuple):
    train_state: TrainState
    init_hstate: jnp.ndarray
    traj_batch: Transition
    advantages: jnp.ndarray
    targets: jnp.ndarray
    rng: jnp.ndarray

class UpdateBatch(NamedTuple):
    init_hstate: jnp.ndarray
    traj_batch: Transition
    advantages: jnp.ndarray
    targets: jnp.ndarray

def batchify(qty: Dict[str, jnp.ndarray], agents: Sequence[str]) -> jnp.ndarray:
    """Convert dict of arrays to batched array."""
    return jnp.stack(tuple(qty[a] for a in agents))

def unbatchify(qty: jnp.ndarray, agents: Sequence[str]) -> Dict[str, jnp.ndarray]:
    """Convert batched array to dict of arrays."""
    # N.B. assumes the leading dimension is the agent dimension
    return dict(zip(agents, qty))

def make_train(config):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["OBS_DIM"] = get_space_dim(env.observation_space(env.agents[0]))
    config["ACT_DIM"] = get_space_dim(env.action_space(env.agents[0]))
    env = LogWrapper(env)

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
        MultiActorCriticRNN = nn.vmap(
            ActorCriticRNN,
            in_axes=0, out_axes=0,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=env.num_agents,
            axis_name="agents",
        )
        network = MultiActorCriticRNN(config=config)
        rng, network_rng = jax.random.split(rng)
        init_x = (
            jnp.zeros( # obs
                (env.num_agents, 1, config["NUM_ENVS"], config["OBS_DIM"])
            ),
            jnp.zeros( # done
                (env.num_agents, 1, config["NUM_ENVS"])
            ),
            jnp.zeros( # avail_actions
                (env.num_agents, 1, config["NUM_ENVS"], config["ACT_DIM"])
            ),
        )
        init_hstate = jnp.zeros(
            (env.num_agents, config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        )
        network_params = network.init(network_rng, init_hstate, init_x)
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
                obs_batch = batchify(runner_state.last_obs, env.agents)
                avail_actions = jax.vmap(env.get_avail_actions)(runner_state.env_state.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents)
                )
                ac_in = (
                    # add time dimension to pass to RNN
                    jnp.expand_dims(obs_batch, 1),
                    jnp.expand_dims(runner_state.last_done, 1),
                    jnp.expand_dims(avail_actions, 1),
                )
                # SELECT ACTION
                hstate, pi, value = runner_state.train_state.apply_fn(
                    runner_state.train_state.params,
                    runner_state.hstate, ac_in,
                )
                rng, act_rng = jax.random.split(rng)
                action, log_prob = pi.sample_and_log_prob(seed=act_rng)
                # remove time dimension
                action = action.squeeze(1)
                log_prob = log_prob.squeeze(1)
                value = value.squeeze(1)
                env_act = unbatchify(action, env.agents)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step)(
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
                    hstate=hstate,
                    update_step=runner_state.update_step,
                    rng=rng,
                )
                return runner_state, transition

            init_hstate = runner_state.hstate
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            last_obs_batch = batchify(runner_state.last_obs, env.agents)
            ac_in = (
                # add time dimension to pass to RNN
                jnp.expand_dims(last_obs_batch, 1),
                jnp.expand_dims(runner_state.last_done, 1),
                jnp.ones(
                    (env.num_agents, 1, config["NUM_ENVS"], config["ACT_DIM"]),
                    dtype=jnp.uint8
                ),
            )
            _, _, last_val = runner_state.train_state.apply_fn(
                runner_state.train_state.params,
                runner_state.hstate, ac_in,
            )
            last_val = last_val.squeeze(1)

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
                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        ac_in = (
                            traj_batch.obs,
                            traj_batch.done,
                            traj_batch.avail_actions,
                        )
                        _, pi, value = train_state.apply_fn(
                            params,
                            init_hstate.squeeze(1), # remove step dim
                            ac_in,
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_losses = jnp.square(value - targets)
                        value_loss = 0.5 * value_losses.mean(axis=(-2,-1))

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        gae = (
                            (gae - gae.mean(axis=(-2,-1), keepdims=True))
                            / (gae.std(axis=(-2,-1), keepdims=True) + 1e-8)
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
                        loss_actor = loss_actor.mean(axis=(-2,-1))
                        entropy = pi.entropy().mean(axis=(-2,-1))
                        # debug metrics
                        approx_kl = ((ratio - 1) - logratio).mean(axis=(-2,-1))
                        clip_frac_min = jnp.mean(ratio < clip_eps_min, axis=(-2,-1))
                        clip_frac_max = jnp.mean(ratio > clip_eps_max, axis=(-2,-1))
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
                        train_state.params,
                        batch_info.init_hstate,
                        batch_info.traj_batch,
                        batch_info.advantages,
                        batch_info.targets
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

                batch_size = config["NUM_ENVS"]
                minibatch_size = batch_size // config["NUM_MINIBATCHES"]
                assert (
                    batch_size % minibatch_size == 0
                ), "unable to equally partition into minibatches"
                batch = UpdateBatch(
                    init_hstate=jnp.expand_dims(update_state.init_hstate, 0), # add step dimension
                    traj_batch=update_state.traj_batch,
                    advantages=update_state.advantages,
                    targets=update_state.targets,
                )
                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, batch_size)
                # batch: (step, agent, env, ...)
                batch = jax.tree_util.tree_map(
                    lambda x: x.swapaxes(0,2),
                    batch
                ) # swap axes to (env, agent, step, ...)
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0),
                    batch
                ) # shuffle: maintains (env, agent, step, ...)
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, (config["NUM_MINIBATCHES"], -1, *x.shape[1:])
                    ),
                    shuffled_batch
                ) # split into minibatches: (n_mini, minibatch_size, agent, step, ...)
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.moveaxis(x, 1, 3),
                    minibatches
                ) # swap axes to (n_mini, agent, step, minibatch_size, ...)
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
            metric = jax.tree_util.tree_map(lambda x: x.mean(axis=(0,2)), metric)
            loss_info = jax.tree_util.tree_map(lambda x: x.mean(axis=(0,1)), loss_info)
            metric = {
                **metric,
                **loss_info,
                "update_step": update_step,
                "env_step": update_step * config["NUM_STEPS"] * config["NUM_ENVS"],
            }
            runner_state = RunnerState(
                train_state=update_state.train_state,
                env_state=runner_state.env_state,
                last_obs=runner_state.last_obs,
                last_done=runner_state.last_done,
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
            hstate=init_hstate,
            update_step=0,
            rng=_rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


@hydra.main(version_base=None, config_path="config", config_name="ippo_rnn_mpe")
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
        clip_epss = config["ENT_COEF"]
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
