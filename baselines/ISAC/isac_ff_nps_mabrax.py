import os
# os.environ["XLA_FLAGS"] = (
#     "--xla_gpu_enable_triton_softmax_fusion=true "
#     "--xla_gpu_triton_gemm_any=true "
#     "--xla_dump_to=xla_dump "
# )
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="0.95"
import jax
import jax.numpy as jnp
import flax.linen as nn
from tqdm import tqdm
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from flax import struct
import optax
import distrax
import sys
import numpy as np
import jaxmarl
from jaxmarl.wrappers.baselines import get_space_dim, LogEnvState
from jaxmarl.wrappers.baselines import LogWrapper
import hydra
from omegaconf import OmegaConf
from typing import Sequence, NamedTuple, TypeAlias, Any, Dict
import wandb
import os
import functools
from functools import partial
import tensorflow_probability.substrates.jax.distributions as tfd
from flax.core.scope import FrozenVariableDict
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState
import flashbax as fbx
import safetensors.flax
from flax.traverse_util import flatten_dict

os.environ['WANDB_DIR'] = './wandb'
os.environ['WANDB_CACHE_DIR'] = './wandb'
os.environ['WANDB_CONFIG_DIR'] = './wandb'
os.environ['WANDB_DATA_DIR'] = './wandb'

# Helper functions remain the same
def _tree_take(pytree, indices, axis=None):
    return jax.tree_map(lambda x: x.take(indices, axis=axis), pytree)

def _tree_shape(pytree):
    return jax.tree_map(lambda x: x.shape, pytree)

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

def batchify(qty: Dict[str, jnp.ndarray], agents: Sequence[str]) -> jnp.ndarray:
    """Convert dict of arrays to batched array."""
    return jnp.stack(tuple(qty[a] for a in agents))

def unbatchify(qty: jnp.ndarray, agents: Sequence[str]) -> Dict[str, jnp.ndarray]:
    """Convert batched array to dict of arrays."""
    return dict(zip(agents, qty))

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
    
@functools.partial(
    nn.vmap,
    in_axes=0, out_axes=0,
    variable_axes={"params": 0},
    split_rngs={"params": True},
    axis_name="agents",
)
class MultiSACActor(nn.Module):
    config: Dict
    
    @nn.compact
    def __call__(self, x):
        if self.config["network"]["activation"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
            
        obs, done, avail_actions = x
        # actor Network
        actor_hidden = nn.Dense(
            self.config["network"]["actor_hidden_dim"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0)
        )(obs)
        actor_hidden = activation(actor_hidden)
        actor_hidden = nn.Dense(
            self.config["network"]["actor_hidden_dim"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0)
        )(actor_hidden)
        actor_hidden = activation(actor_hidden)
        
        # output mean
        actor_mean = nn.Dense(
            self.config["ACT_DIM"],
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )(actor_hidden)
        
        # log std
        log_std = self.param(
            "log_std",
            nn.initializers.zeros,
            (self.config["ACT_DIM"],)
        )
        actor_log_std = jnp.broadcast_to(log_std, actor_mean.shape)

        return actor_mean, jnp.exp(actor_log_std)

@functools.partial(
    nn.vmap,
    in_axes=0, out_axes=0,
    variable_axes={"params": 0},
    split_rngs={"params": True},
    axis_name="agents",
)
class MultiSACQNetwork(nn.Module):
    config: Dict
    
    @nn.compact
    def __call__(self, x, action):
        if self.config["network"]["activation"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh


        obs, done, avail_actions = x
        x = jnp.concatenate([obs, action], axis=-1)
        
        x = nn.Dense(
            self.config["network"]["critic_hidden_dim"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        x = activation(x)
        x = nn.Dense(
            self.config["network"]["critic_hidden_dim"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        x = activation(x)
        x = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0)
        )(x)
        
        return jnp.squeeze(x, axis=-1)

class QVals(NamedTuple):
    q1: FrozenVariableDict
    q2: FrozenVariableDict


class QValsAndTarget(NamedTuple):
    online: QVals
    targets: QVals


class SacParams(NamedTuple):
    actor: FrozenVariableDict
    q: QValsAndTarget
    log_alpha: jnp.ndarray


class OptStates(NamedTuple):
    actor: optax.OptState
    q: optax.OptState
    alpha: optax.OptState


class Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    next_obs: jnp.ndarray

class UpdateState(NamedTuple):
    train_state: TrainState
    traj_batch: Transition
    q1_target: jnp.ndarray
    q2_target: jnp.ndarray
    rng: jnp.ndarray


# class SACTrainStates(NamedTuple):
#     """wrapper to manage multiple train states for SAC."""
#     actor: TrainState
#     q1: TrainState
#     q2: TrainState
#     q1_target: Dict
#     q2_target: Dict
#     log_alpha: jnp.ndarray
#     alpha_opt_state: optax.OptState

#     @classmethod
#     def create(
#         cls,
#         *,
#         actor_fn,
#         q_fn,
#         actor_params,
#         q1_params,
#         q2_params,
#         actor_tx,
#         q_tx,
#         alpha_tx,
#         init_log_alpha: float = 0.0,
#     ):
#         """Creates new train states with step=0 and initialized optimizers."""
#         actor_state = TrainState.create(
#             apply_fn=actor_fn,
#             params=actor_params,
#             tx=actor_tx,
#         )

#         q1_state = TrainState.create(
#             apply_fn=q_fn,
#             params=q1_params,
#             tx=q_tx,
#         )

#         q2_state = TrainState.create(
#             apply_fn=q_fn,
#             params=q2_params,
#             tx=q_tx,
#         )

#         # initialize target networks
#         q1_target = q2_params

#         q2_target = q2_params

#         log_alpha = jnp.array(init_log_alpha)
#         alpha_opt_state = alpha_tx.init(log_alpha) 

#         return cls(
#             actor=actor_state,
#             q1=q1_state,
#             q2=q2_state,
#             q1_target=q1_target,
#             q2_target=q2_target,
#             log_alpha=log_alpha,
#             alpha_opt_state=alpha_opt_state,
#         )
    
#     def apply_gradients(
#         self,
#         *,
#         actor_grads,
#         q1_grads,
#         q2_grads,
#         tau,
#         new_log_alpha,
#         new_alpha_opt_state,
#     ):
#         """Updates all train states with their respective gradients."""
#         new_actor = self.actor.apply_gradients(grads=actor_grads)
#         new_q1 = self.q1.apply_gradients(grads=q1_grads)
#         new_q2 = self.q2.apply_gradients(grads=q2_grads)

#         new_q1_target = optax.incremental_update(new_q1.params, self.q1_target, tau)
#         new_q2_target = optax.incremental_update(new_q2.params, self.q2_target, tau)

#         return self._replace(
#             actor=new_actor,
#             q1=new_q1,
#             q2=new_q2,
#             q1_target=new_q1_target,
#             q2_target=new_q2_target,
#             log_alpha=new_log_alpha,
#             alpha_opt_state=new_alpha_opt_state,    
#         )

#     @property
#     def alpha(self):
#         """Convenience property to get current temperature parameter."""
#         return jnp.exp(self.log_alpha)

class SACTrainStates(NamedTuple):
    actor: TrainState
    q1: TrainState
    q2: TrainState
    q1_target: Dict
    q2_target: Dict
    log_alpha: jnp.ndarray
    alpha_opt_state: optax.OptState


BufferState: TypeAlias = TrajectoryBufferState[Transition]
class RunnerState(NamedTuple):
    train_states: SACTrainStates
    env_state: LogEnvState
    last_obs: Dict[str, jnp.ndarray]
    last_done: jnp.ndarray
    t: int
    buffer_state: BufferState
    rng: jnp.ndarray

class EvalState(NamedTuple):
    train_states: SACTrainStates
    env_state: LogEnvState
    last_obs: Dict[str, jnp.ndarray]
    last_done: jnp.ndarray
    update_step: int
    rng: jnp.ndarray

class EvalInfo(NamedTuple):
    env_state: LogEnvState
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray


def make_train(config, save_train_state=True): #TODO: implement the save_train_state thing
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["ROLLOUT_LENGTH"] // config["NUM_ENVS"]
    )
    config["SCAN_STEPS"] = config["NUM_UPDATES"] // config["NUM_CHECKPOINTS"]
    config["EXPLORE_SCAN_STEPS"] = config["EXPLORE_STEPS"] // config["NUM_ENVS"]
    print(f"NUM_UPDATES: {config['NUM_UPDATES']} \n SCAN_STEPS: {config['SCAN_STEPS']} \n EXPLORE_STEPS: {config['EXPLORE_STEPS']} \n NUM_CHECKPOINTS: {config['NUM_CHECKPOINTS']}")
    config["OBS_DIM"] = get_space_dim(env.observation_space(env.agents[0]))
    config["ACT_DIM"] = get_space_dim(env.action_space(env.agents[0]))
    env = LogWrapper(env, replace_info=True)
    

    def train(rng, p_lr, q_lr, alpha_lr): 

        
        actor = MultiSACActor(config=config)
        q = MultiSACQNetwork(config=config)

        rng, actor_rng, q1_rng, q2_rng = jax.random.split(rng, num=4)

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
        actor_params = actor.init(actor_rng, init_x)
        dummy_action = jnp.zeros((env.num_agents, 1, config["ACT_DIM"]))
        q1_params = q.init(q1_rng, init_x, dummy_action)
        q2_params = q.init(q2_rng, init_x, dummy_action)

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset)(reset_rng)
        init_dones = jnp.zeros((env.num_agents, config["NUM_ENVS"]), dtype=bool)

        action_space = env.action_space(env.agents[0])
        action = jnp.zeros((env.num_agents, config["NUM_ENVS"], action_space.shape[0]))

        init_transition = Transition(
            obs=batchify(obsv, env.agents),
            action=action,
            reward=jnp.zeros((env.num_agents, config["NUM_ENVS"]), dtype=float),
            done=init_dones,
            next_obs=batchify(obsv, env.agents)
        )

        rb = fbx.make_item_buffer(
            max_length=config["BUFFER_SIZE"]//config["NUM_ENVS"],
            min_length=config["EXPLORE_STEPS"]//config["NUM_ENVS"],
            sample_batch_size=int(config["BATCH_SIZE"]),
            add_batches=True,
        )
                                                                                                                                                
        # buffer_state = rb.init(_tree_take(init_transition, 0, axis=1))
        buffer_state = rb.init(jax.tree.map(lambda x: jnp.moveaxis(x,1,0), init_transition))

        target_entropy = -config["TARGET_ENTROPY_SCALE"] * config["ACT_DIM"]

        if config["AUTOTUNE"]:
            log_alpha = jnp.zeros_like(target_entropy)
        else: # TODO: catually implement the non autotune case
            log_alpha = jnp.log(config["INIT_ALPHA"])
            log_alpha = jnp.broadcast_to(log_alpha, target_entropy.shape)

    
        grad_clip = optax.clip_by_global_norm(config["MAX_GRAD_NORM"])

        actor_opt = optax.chain(grad_clip, optax.adam(config["POLICY_LR"]))

        q1_opt = optax.chain(grad_clip, optax.adam(config["Q_LR"]))
        # Testing with 2 separate optimizers for each q function 
        q2_opt = optax.chain(grad_clip, optax.adam(config["Q_LR"]))
        
        alpha_opt = optax.chain(grad_clip, optax.adam(config["ALPHA_LR"]))
        alpha_opt_state = alpha_opt.init(log_alpha)

        tanh_bijector = distrax.Block(distrax.Tanh(), ndims=1)
        
        actor_train_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor_params,
            tx=actor_opt,
        )
        q1_train_state = TrainState.create(
            apply_fn=q.apply,
            params=q1_params,
            tx=q1_opt,
        )

        q2_train_state = TrainState.create(
            apply_fn=q.apply,
            params=q2_params,
            tx=q2_opt,
        )
        
        train_states = SACTrainStates(
            actor=actor_train_state,
            q1=q1_train_state,
            q2=q2_train_state,
            q1_target=q1_params,
            q2_target=q2_params,
            log_alpha=log_alpha,
            alpha_opt_state=alpha_opt_state,
        )

        runner_state = RunnerState(
            train_states=train_states,
            env_state=env_state,
            last_obs=obsv,
            last_done=init_dones,
            t=0,
            buffer_state=buffer_state,
            rng=_rng,
        )
        
        breakpoint()
        # TODO: implement an explore function here which does random exploration to fill replay buffer at beginnning of training
        
        def _explore(runner_state, unused):
            
            breakpoint()
            rng, explore_rng = jax.random.split(runner_state.rng)
            avail_actions = jax.vmap(env.get_avail_actions)(runner_state.env_state.env_state)
            
            avail_actions_shape = batchify(avail_actions, env.agents).shape
            action = jax.random.uniform(explore_rng, avail_actions_shape, minval=-1, maxval=1)
            env_act = unbatchify(action, env.agents)
            rng_step = jax.random.split(explore_rng, config["NUM_ENVS"])
           
            obsv, env_state, reward, done, info = jax.vmap(env.step)(
                    rng_step, runner_state.env_state, env_act,
                )
            
            t = runner_state.t + config["NUM_ENVS"]
            
            obs_batch = batchify(runner_state.last_obs, env.agents)
            done_batch = batchify(done, env.agents)

            # maybe I should include info in the transition?
            transition = Transition(
                    obs = obs_batch,
                    action = action,
                    reward = batchify(reward, env.agents),
                    done = done_batch,
                    next_obs = batchify(obsv, env.agents),
                )

            runner_state = RunnerState(
                train_states=runner_state.train_states,
                env_state=env_state,
                last_obs = obsv,
                last_done=done_batch,
                t = t,
                buffer_state=runner_state.buffer_state,
                rng=rng
            )
            breakpoint()

            return runner_state, transition 
        
        breakpoint()
        explore_runner_state, explore_traj_batch = jax.lax.scan(
                _explore, runner_state, None, config["EXPLORE_SCAN_STEPS"]
            )

        explore_buffer_state = rb.add(
            runner_state.buffer_state,
            jax.tree.map(lambda x: jnp.moveaxis(x,2,1), explore_traj_batch) # move batch axis to start
        )

        breakpoint()
        
        # changed this to reflect the explore info gathered for training
        explore_runner_state = RunnerState(
            train_states=explore_runner_state.train_states,
            env_state=explore_runner_state.env_state,
            last_obs=explore_runner_state.last_obs,
            last_done=explore_runner_state.last_done,
            t=runner_state.t,
            buffer_state= explore_buffer_state,
            rng=runner_state.rng
        )
        
        def _checkpoint_step(runner_state, unused):
            """ Used to reduce amount of parameters we save during training. """

            def _update_step(runner_state, unused):
                """ The SAC update"""
            
                def _env_step(runner_state, unused):
                    """ Step the environment """
                    # breakpoint()
                    rng = runner_state.rng
                    obs_batch = batchify(runner_state.last_obs, env.agents)
                    avail_actions = jax.vmap(env.get_avail_actions)(runner_state.env_state.env_state)
                    avail_actions = jax.lax.stop_gradient(
                        batchify(avail_actions, env.agents)
                    )
                    ac_in = (obs_batch, runner_state.last_done, avail_actions)

                    # SELECT ACTION
                    
                    rng, action_rng = jax.random.split(rng)
                    (actor_mean, actor_std) = runner_state.train_states.actor.apply_fn(
                        runner_state.train_states.actor.params, 
                        ac_in
                        )
                    
                    # pi_normal = distrax.Normal(actor_mean, actor_std)
                    # pi_tanh_normal = distrax.Transformed(pi_normal, bijector=distrax.Tanh())
                    # pi_tanh = distrax.Independent(pi_tanh_normal, 1)
                    # action, log_prob = pi_tanh.sample_and_log_prob(seed=action_rng)
                    pi = distrax.MultivariateNormalDiag(actor_mean, actor_std)
                    pi_tanh = distrax.Transformed(pi, bijector=tanh_bijector)
                    action, log_prob = pi_tanh.sample_and_log_prob(seed=action_rng)
                    env_act = unbatchify(action, env.agents)

                    #STEP ENV
                    rng, _rng = jax.random.split(rng)
                    rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                    obsv, env_state, reward, done, info = jax.vmap(env.step)(
                        rng_step, runner_state.env_state, env_act,
                    )
                    done_batch = batchify(done, env.agents)
                    info = jax.tree_util.tree_map(lambda x: x.swapaxes(0,1), info)
                    # q1_value = runner_state.train_states.q1.apply_fn(
                    #     runner_state.train_states.q1.params,
                    #     ac_in, action)
                    # q2_value = runner_state.train_states.q2.apply_fn(
                    #     runner_state.train_states.q2.params,
                    #     ac_in, action)

                    transition = Transition(
                        obs = obs_batch,
                        action = action,
                        reward = batchify(reward, env.agents),
                        done = done_batch,
                        next_obs = batchify(obsv, env.agents),
                    )

                    # buffer_state = rb.add(runner_state.buffer_state, transition)

                    t = runner_state.t + 1

                    runner_state = RunnerState(
                        train_states=runner_state.train_states,
                        env_state=env_state,
                        last_obs=obsv,
                        last_done=done_batch,
                        t=t,
                        buffer_state=runner_state.buffer_state,
                        rng=rng,
                    )

                    return runner_state, transition # I probably don't need to return transition with the rb
                
                runner_state, traj_batch  = jax.lax.scan(
                    _env_step, runner_state, None, config["ROLLOUT_LENGTH"]
                )
                # this traj batch is shape (rollout_length, num_agents, num_envs, dim) needs to be reshaped to (rollout_length, num_envs, num_agents, dim)

                # breakpoint()
                # if I want to add the entire trajectory I need to initialize somewhat differently to give it (batch_size, trajectory_length) maybe when initializing
                new_buffer_state = rb.add(
                    runner_state.buffer_state,
                    jax.tree.map(lambda x: jnp.moveaxis(x,2,1), traj_batch) # move batch axis to start
                )
                breakpoint()
                # buffer_state = rb.add(runner_state.buffer_state, transitions)
                def _update_networks(carry, rng): 

                    rng, batch_sample_rng, q_sample_rng, actor_update_rng = jax.random.split(rng, 4)
                    train_state, buffer_state = carry
                    batch = rb.sample(buffer_state, batch_sample_rng).experience
                    breakpoint()
                    def reshape_fn(x):
                        if len(x.shape) == 4:  # obs, action, next_obs
                            return x.transpose(2, 0, 1, 3).reshape((x.shape[2], x.shape[0] * x.shape[1], x.shape[3]))
                        else:  # reward, done
                            return x.transpose(2, 0, 1).reshape((x.shape[2], x.shape[0] * x.shape[1]))

                    batch = jax.tree_util.tree_map(reshape_fn, batch)
                                        
                    # batch = jax.tree_util.tree_map(
                    #     lambda x: x.transpose(1, 0, 2, 3).reshape(
                    #         (x.shape[1], x.shape[0] * x.shape[2], x.shape[3])
                    #     ),
                    #     batch
                    # )

                    breakpoint()

                    #UPDATE Q_NETWORKS
                    def q_loss_fn(q1_online_params, q2_online_params, obs, dones, action, target_q, avail_actions):
                        
                        # compute current Q-values
                        # current_q1 = batched_q(
                        #     q1_online_params, 
                        #     (obs, dones, avail_actions), action
                        # )
                        # current_q2 = batched_q(
                        #     q2_online_params, 
                        #     (obs, dones, avail_actions), action
                        # )
                        current_q1 = q.apply(
                            q1_online_params, 
                            (obs, dones, avail_actions), action
                        )
                        current_q2 = q.apply(
                            q2_online_params, 
                            (obs, dones, avail_actions), action
                        )
                    
                        # MSE loss for both Q-networks
                        q1_loss = jnp.mean(jnp.square(current_q1 - target_q))
                        q2_loss = jnp.mean(jnp.square(current_q2 - target_q))

                        breakpoint()
                        return q1_loss + q2_loss, (q1_loss, q2_loss)
                    
                    # loss for the actor
                    def actor_loss_fn(actor_params, q1_params, q2_params, obs, dones, alpha, rng, avail_actions):

                        next_ac_in = (obs, dones, avail_actions)
                        # actor_mean, actor_std = batched_actor(
                        #     actor_params, 
                        #     next_ac_in
                        # )

                        # for consistency with ippo maybe I should do train_state.actor.apply 
                        
                        # actor_mean, actor_std = actor.apply(
                        #     actor_params, 
                        #     next_ac_in
                        # )

                        actor_mean, actor_std = train_state.actor.apply_fn(
                            actor_params, 
                            next_ac_in
                        )


                        breakpoint()
                        pi = distrax.MultivariateNormalDiag(actor_mean, actor_std)
                        pi_tanh = distrax.Transformed(pi, bijector=tanh_bijector)
                        action, log_prob = pi_tanh.sample_and_log_prob(seed=rng)
                        breakpoint()

                        # Q-vals for actor loss
                        # q1_values = batched_q(
                        #     q1_params, 
                        #     (obs, dones, avail_actions), action
                        # )
                        # q2_values = batched_q(
                        #     q2_params, 
                        #     (obs, dones, avail_actions), action
                        # )
                        q1_values = q.apply(
                            q1_params, 
                            (obs, dones, avail_actions), action
                        )
                        q2_values = q.apply(
                            q2_params, 
                            (obs, dones, avail_actions), action
                        )
                        q_value = jnp.minimum(q1_values, q2_values)
                        
                        # actor loss with entropy
                        actor_loss = jnp.mean(alpha * log_prob - q_value)

                        breakpoint() # for checking actor loss
                        return actor_loss, log_prob
                    
                    def alpha_loss_fn(log_alpha, log_pi, target_entropy):
                        return jnp.mean(-jnp.exp(log_alpha) * (log_pi + target_entropy))
                    
                    
                    # Q networks loss and gradient 
                    # Janky reshape to revert from buffer back to what we expect to see
                    # breakpoint()
                    breakpoint()
                    # obs = batch.obs.swapaxes(1, 2)
                    # dones = batch.done.swapaxes(1, 2)
                    # action = batch.action.swapaxes(1, 2)
                    # next_obs = batch.next_obs.swapaxes(1, 2)
                    # reward = batch.reward.swapaxes(1, 2)
                    obs = batch.obs
                    dones = batch.done
                    action = batch.action
                    next_obs = batch.next_obs
                    reward = batch.reward

                    # avail_actions =  jnp.zeros( # avail_actions
                    #         (config["BATCH_SIZE"], env.num_agents, config["NUM_ENVS"], config["ACT_DIM"])
                        # ) # this is unused for assistax but useful in other implementations
                    avail_actions =  jnp.zeros( # avail_actions
                            (env.num_agents, config["NUM_ENVS"]*config["BATCH_SIZE"], config["ACT_DIM"])
                        ) # this is unused for assistax but useful in other implementations
                    
                    avail_actions = jax.lax.stop_gradient(avail_actions)

                    # next_act_mean, next_act_std = batched_actor(
                    #     train_state.actor.params, 
                    #     (next_obs, dones, avail_actions)) 
                    
                    breakpoint()
                    # next_act_mean, next_act_std = actor.apply(
                    #     train_state.actor.params, 
                    #     (next_obs, dones, avail_actions)) 
                    next_act_mean, next_act_std = train_state.actor.apply_fn(
                        actor_params, 
                        (next_obs, dones, avail_actions),
                    )

                    # TODO: verify I'm using the right distribution ...
                    next_pi = distrax.MultivariateNormalDiag(next_act_mean, next_act_std)
                    next_pi_tanh = distrax.Transformed(next_pi, bijector=tanh_bijector)
                    next_action, next_log_prob = next_pi_tanh.sample_and_log_prob(seed=rng) # these have shape (batch_size, num_agents, num_envs, act_dim) as expected for the qnetworks
                    
                    breakpoint() # for checking actions
                    
                    # compute q target
                    # next_q1 = q.apply(
                    #     train_state.q1_target, 
                    #     (next_obs, dones, avail_actions), next_action # double check is it next action or next_env_act? probs the latter
                    # )
                    # next_q2 = q.apply(
                    # train_state.q2_target, 
                    #     (next_obs, dones, avail_actions), next_action
                    # )
                    next_q1 = train_state.q1.apply_fn(
                        train_state.q1_target, 
                        (next_obs, dones, avail_actions), next_action
                    )
                    next_q2 = train_state.q2.apply_fn(
                        train_state.q2_target, 
                        (next_obs, dones, avail_actions), next_action
                    )
            
                    next_q = jnp.minimum(next_q1, next_q2)
                    breakpoint() # for checking the minimum vals
                    next_q = next_q - jnp.exp(train_state.log_alpha) * next_log_prob

                    target_q = reward + config["GAMMA"] * (1.0 - dones) * next_q
                    
                    q_grad_fun = jax.value_and_grad(q_loss_fn, has_aux=True)
                    (q_loss, (q1_loss, q2_loss)), q_grads = q_grad_fun(
                        train_state.q1.params, 
                        train_state.q2.params, 
                        obs, 
                        dones, 
                        action, 
                        target_q,
                        avail_actions,)

                    # actor loss and gradient 
                    actor_grad_fun = jax.value_and_grad(actor_loss_fn, has_aux=True)
                    (actor_loss, log_probs), actor_grads = actor_grad_fun(
                        train_state.actor.params,
                        train_state.q1.params,
                        train_state.q2.params,
                        obs,
                        dones,
                        jnp.exp(train_state.log_alpha),
                        actor_update_rng,
                        avail_actions,
                    )
                    breakpoint() # for checking log alpha

                    # alphaloss and gradient update
                    alpha_grad_fn = jax.value_and_grad(alpha_loss_fn)
                    temperature_loss, alpha_grad = alpha_grad_fn(train_state.log_alpha, log_probs, target_entropy)
                    alpha_updates, new_alpha_opt_state = alpha_opt.update(
                    alpha_grad, 
                    train_state.alpha_opt_state, 
                    train_state.log_alpha
                    )
                    new_log_alpha = optax.apply_updates(train_state.log_alpha, alpha_updates)
                    
                    # update networks
                    breakpoint()
                    new_actor_train_state = train_state.actor.apply_gradients(grads=actor_grads)
                    new_q1_train_state = train_state.q1.apply_gradients(grads=q_grads)
                    new_q2_train_state = train_state.q2.apply_gradients(grads=q_grads)

                    new_q1_target = optax.incremental_update(
                        new_q1_train_state.params,
                        train_state.q1_target,
                        config["TAU"],
                    )
                    new_q2_target = optax.incremental_update(
                        new_q2_train_state.params,
                        train_state.q2_target,
                        config["TAU"],
                    )
                    
                    new_train_state = SACTrainStates(
                        actor=new_actor_train_state,
                        q1=new_q1_train_state,
                        q2=new_q2_train_state,
                        q1_target=new_q1_target,
                        q2_target=new_q2_target,
                        log_alpha=new_log_alpha,
                        alpha_opt_state=new_alpha_opt_state,
                    )
                    
                    # new_train_state = train_state.apply_gradients(
                    #     actor_grads=actor_grads,
                    #     q1_grads=q_grads,
                    #     q2_grads=q_grads,
                    #     tau=config["TAU"],
                    #     new_log_alpha=new_log_alpha,
                    #     new_alpha_opt_state=new_alpha_opt_state,
                    # ) # this is messy as I don't only pass gradients maybe I should split off alpha update
                    # breakpoint()
                    metrics = {
                        'critic_loss': q_loss,
                        'q1_loss': q1_loss,
                        'q2_loss': q2_loss,
                        'actor_loss': actor_loss,
                        'alpha_loss': temperature_loss,
                        'alpha': jnp.exp(new_train_state.log_alpha),
                        'log_probs': log_probs.mean(),
                        "next_log_probs": next_log_prob.mean(),
                    }


                    return (new_train_state, buffer_state), metrics
                
                # TODO: figure out how to implement updates per step i.e. how to update the networks multiple times per step

                _, sample_rng = jax.random.split(runner_state.rng)

                # train_state, metrics = _update_networks(runner_state.train_states, batch.experience)
                # breakpoint()
                update_rngs = jax.random.split(sample_rng, config["NUM_SAC_UPDATES"])
                (train_state, buffer_state), metrics = jax.lax.scan(_update_networks, (runner_state.train_states, new_buffer_state), update_rngs)
                metrics = jax.tree.map(lambda x: x.mean(), metrics)
                # only store train state after epochs

                # breakpoint()
                
                runner_state = RunnerState(
                    train_states=train_state, # replace trainstate
                    env_state=runner_state.env_state,
                    last_obs=runner_state.last_obs,
                    last_done=runner_state.last_done,
                    t=runner_state.t,
                    buffer_state=buffer_state,
                    rng=runner_state.rng,
                )

                return runner_state, metrics
            
            runner_state, metrics = jax.lax.scan(
                _update_step, runner_state, None, config["SCAN_STEPS"]
            )
            metrics = jax.tree.map(lambda x: x.mean(), metrics)

            if save_train_state:
                metrics.update({"actor_train_state": runner_state.train_states.actor})
                metrics.update({"q1_train_state": runner_state.train_states.q1})
                metrics.update({"q2_train_state": runner_state.train_states.q2})

            return runner_state, metrics
        
        # Exploration before training
        # breakpoint()
        # explore_runner_state, explore_traj_batch = jax.lax.scan(
        #         _explore, runner_state, None, config["EXPLORE_SCAN_STEPS"]
        #     )

        # explore_buffer_state = rb.add(
        #     runner_state.buffer_state,
        #     jax.tree.map(lambda x: jnp.moveaxis(x,2,1), explore_traj_batch) # move batch axis to start
        # )

        # breakpoint()
        
        # # changed this to reflect the explore info gathered for training
        # explore_runner_state = RunnerState(
        #     train_states=explore_runner_state.train_states,
        #     env_state=explore_runner_state.env_state,
        #     last_obs=explore_runner_state.last_obs,
        #     last_done=explore_runner_state.last_done,
        #     t=runner_state.t,
        #     buffer_state= explore_buffer_state,
        #     rng=runner_state.rng
        # )

        final_runner_state, checkpoint_metrics = jax.lax.scan(
            _checkpoint_step, explore_runner_state, None, config["NUM_CHECKPOINTS"]
        ) # change 1 to config["NUM_CHECKPOINTS"] eventually 
        return {"runner_state": final_runner_state, "metrics": checkpoint_metrics}
    
    return train

def make_evaluation(config):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    config["OBS_DIM"] = get_space_dim(env.observation_space(env.agents[0]))
    config["ACT_DIM"] = get_space_dim(env.action_space(env.agents[0]))
    env = LogWrapper(env, replace_info=True)
    max_steps = env.episode_length
    tanh_bijector = distrax.Block(distrax.Tanh(), ndims=1)

    def run_evaluation(rng, train_state, log_env_state=False):
        rng_reset, rng_env = jax.random.split(rng)
        rngs_reset = jax.random.split(rng_reset, config["NUM_EVAL_EPISODES"])
        obsv, env_state = jax.vmap(env.reset)(rngs_reset)
        init_dones = jnp.zeros((env.num_agents, config["NUM_EVAL_EPISODES"],), dtype=bool)

        runner_state = EvalState(
            train_states=train_state,
            env_state=env_state,
            last_obs=obsv,
            last_done=init_dones,
            update_step=0,
            rng=rng_env,
        )
        def _env_step(runner_state, unused):
            
            rng = runner_state.rng
            obs_batch = batchify(runner_state.last_obs, env.agents)
            avail_actions = jax.vmap(env.get_avail_actions)(runner_state.env_state.env_state)
            avail_actions = jax.lax.stop_gradient(
                batchify(avail_actions, env.agents)
            )
            ac_in = (obs_batch, runner_state.last_done, avail_actions)

            # SELECT ACTION
            
            rng, action_rng = jax.random.split(rng)
            (actor_mean, actor_std) = runner_state.train_states.apply_fn(
                runner_state.train_states.params, 
                ac_in
                )
                
            # pi_normal = distrax.Normal(actor_mean, actor_std)
            # pi_tanh_normal = distrax.Transformed(pi_normal, bijector=distrax.Tanh())
            # pi_tanh = distrax.Independent(pi_tanh_normal, 1)
            # action, log_prob = pi_tanh.sample_and_log_prob(seed=action_rng)

            pi = distrax.MultivariateNormalDiag(actor_mean, actor_std)
            pi_tanh = distrax.Transformed(pi, bijector=tanh_bijector)

            action, log_prob = pi_tanh.sample_and_log_prob(seed=action_rng)

            env_act = unbatchify(action, env.agents)

            #STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["NUM_EVAL_EPISODES"])
            obsv, env_state, reward, done, info = jax.vmap(env.step)(
                rng_step, runner_state.env_state, env_act,
            )
            done_batch = batchify(done, env.agents)
            info = jax.tree_util.tree_map(lambda x: x.swapaxes(0,1), info)
            
            # q1_value = runner_state.train_states.q1.apply_fn(
            #     runner_state.train_states.q1.params,
            #     ac_in, action
            # )

            # q2_value = runner_state.train_states.q2.apply_fn(
            #     runner_state.train_states.q2.params,
            #     ac_in, action
            # )
            
            eval_info = EvalInfo(
                env_state=(env_state if log_env_state else None),
                done=done,
                action=action,
                reward=reward,
                log_prob=log_prob,
                obs=obs_batch,
                info=info,
                avail_actions=avail_actions,
            )
            runner_state = EvalState(
                train_states=runner_state.train_states,
                env_state=env_state,
                last_obs=obsv,
                last_done=done_batch,
                update_step=runner_state.update_step,
                rng=rng,
            )
            return runner_state, eval_info

        _, eval_info = jax.lax.scan(
            _env_step, runner_state, None, max_steps
        )

        return eval_info
    return env, run_evaluation

@hydra.main(version_base=None, config_path="config", config_name="isac_mabrax")
def main(config):
    config = OmegaConf.to_container(config, resolve=True)

    # IMPORT FUNCTIONS BASED ON ARCHITECTURE
    # match (config["network"]["recurrent"], config["network"]["agent_param_sharing"]):
    #     case (False, False):
    #         from ippo_ff_nps_mabrax import make_train as make_train
    #         from ippo_ff_nps_mabrax import make_evaluation as make_evaluation
    #     case (False, True):
    #         from ippo_ff_ps_mabrax import make_train as make_train
    #         from ippo_ff_ps_mabrax import make_evaluation as make_evaluation
    #     case (True, False):
    #         from ippo_rnn_nps_mabrax import make_train as make_train
    #         from ippo_rnn_nps_mabrax import make_evaluation as make_evaluation
    #     case (True, True):
    #         from ippo_rnn_ps_mabrax import make_train as make_train
    #         from ippo_rnn_ps_mabrax import make_evaluation as make_evaluation

    rng = jax.random.PRNGKey(config["SEED"])
    train_rng, eval_rng = jax.random.split(rng)
    train_rngs = jax.random.split(train_rng, config["NUM_SEEDS"])    
    with jax.disable_jit(config["DISABLE_JIT"]):
        train_jit = jax.jit(
            make_train(config, save_train_state=True),
            device=jax.devices()[config["DEVICE"]]
        )
        # first run (includes JIT)
        out = jax.vmap(train_jit, in_axes=(0, None, None, None))(
            train_rngs,
            config["LR"], config["ENT_COEF"], config["CLIP_EPS"]
        )

        # SAVE TRAIN METRICS
        EXCLUDED_METRICS = ["actor_train_state", "q1_train_state", "q2_train_state"]
        jnp.save("metrics.npy", {
            key: val
            for key, val in out["metrics"].items()
            if key not in EXCLUDED_METRICS
            },
            allow_pickle=True
        )

        # SAVE PARAMS
        env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

        all_train_states_actor = out["metrics"]["actor_train_state"]
        all_train_states_q1 = out["metrics"]["q1_train_state"]
        all_train_states_q2 = out["metrics"]["q2_train_state"]
        final_train_state_actor = out["runner_state"].train_states.actor
        final_train_state_q1 = out["runner_state"].train_states.q1
        final_train_state_q2 = out["runner_state"].train_states.q2

        safetensors.flax.save_file(
            flatten_dict(all_train_states_actor.params, sep='/'),
            "actor_all_params.safetensors"
        )

        safetensors.flax.save_file(
            flatten_dict(all_train_states_q1.params, sep='/'),
            "q1_all_params.safetensors"
        )

        safetensors.flax.save_file(
            flatten_dict(all_train_states_q2.params, sep='/'),
            "q2_all_params.safetensors"
        )

        if config["network"]["agent_param_sharing"]:
            safetensors.flax.save_file(
                flatten_dict(final_train_state_actor.params, sep='/'),
                "actor_final_params.safetensors"
            )

            safetensors.flax.save_file(
                flatten_dict(final_train_state_q1.params, sep='/'),
                "q1_final_params.safetensors"
            )
            
            safetensors.flax.save_file(
                flatten_dict(final_train_state_q2.params, sep='/'),
                "q2_final_params.safetensors"
            )
        else:
            # split by agent
            split_actor_params = _unstack_tree(
                jax.tree.map(lambda x: x.swapaxes(0,1), final_train_state_actor.params)
            )
            for agent, params in zip(env.agents, split_actor_params):
                safetensors.flax.save_file(
                    flatten_dict(params, sep='/'),
                    f"actor_{agent}.safetensors",
                )

            split_q1_params = _unstack_tree(
                jax.tree.map(lambda x: x.swapaxes(0,1), final_train_state_q1.params)
            )
            for agent, params in zip(env.agents, split_q1_params):
                safetensors.flax.save_file(
                    flatten_dict(params, sep='/'),
                    f"q1_{agent}.safetensors",
                )

            split_q2_params = _unstack_tree(
                jax.tree.map(lambda x: x.swapaxes(0,1), final_train_state_q2.params)
            )
            for agent, params in zip(env.agents, split_q2_params):
                safetensors.flax.save_file(
                    flatten_dict(params, sep='/'),
                    f"q2_{agent}.safetensors",
                )
            
            

        # TODO: implement evalution
       
        # Assume the first 2 dimensions are batch dims
        batch_dims = jax.tree.leaves(_tree_shape(all_train_states_actor.params))[:2]
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
        split_trainstate = jax.jit(_flatten_and_split_trainstate)(all_train_states_actor)
        eval_env, run_eval = make_evaluation(config)
        eval_jit = jax.jit(
            run_eval,
            static_argnames=["log_env_state"],
        )
        eval_vmap = jax.vmap(eval_jit, in_axes=(None, 0, None))
 
        evals = _concat_tree([
            eval_vmap(eval_rng, ts, False)
            for ts in tqdm(split_trainstate, desc="Evaluation batches")
        ])
        evals = jax.tree.map(
            lambda x: x.reshape((*batch_dims, *x.shape[1:])),
            evals
        )

        # COMPUTE RETURNS
        first_episode_returns = _compute_episode_returns(evals)
        first_episode_returns = first_episode_returns["__all__"]
        mean_episode_returns = first_episode_returns.mean(axis=-1)

        std_error = first_episode_returns.std(axis=-1) / jnp.sqrt(first_episode_returns.shape[-1])

        ci_lower = mean_episode_returns - 1.96 * std_error
        ci_upper = mean_episode_returns + 1.96 * std_error


        # SAVE RETURNS
        jnp.save("returns.npy", mean_episode_returns)
        jnp.save("returns_ci_lower.npy", ci_lower)
        jnp.save("returns_ci_upper.npy", ci_upper)

        # RENDER
        # Run episodes for render (saving env_state at each timestep)
        # eval_final = eval_jit(eval_rng, _tree_take(final_train_state, 0, axis=0), True)
        # first_episode_done = jnp.cumsum(eval_final.done["__all__"], axis=0, dtype=bool)
        # first_episode_rewards = eval_final.reward["__all__"] * (1-first_episode_done)
        # first_episode_returns = first_episode_rewards.sum(axis=0)
        # episode_argsort = jnp.argsort(first_episode_returns, axis=-1)
        # worst_idx = episode_argsort.take(0,axis=-1)
        # best_idx = episode_argsort.take(-1, axis=-1)
        # median_idx = episode_argsort.take(episode_argsort.shape[-1]//2, axis=-1)

        # from brax.io import html
        # worst_episode = _take_episode(
        #     eval_final.env_state.env_state.pipeline_state, first_episode_done,
        #     time_idx=-1, eval_idx=worst_idx,
        # )
        # median_episode = _take_episode(
        #     eval_final.env_state.env_state.pipeline_state, first_episode_done,
        #     time_idx=-1, eval_idx=median_idx,
        # )
        # best_episode = _take_episode(
        #     eval_final.env_state.env_state.pipeline_state, first_episode_done,
        #     time_idx=-1, eval_idx=best_idx,
        # )
        # html.save("final_worst.html", eval_env.sys, worst_episode)
        # html.save("final_median.html", eval_env.sys, median_episode)
        # html.save("final_best.html", eval_env.sys, best_episode)


if __name__ == "__main__":
    main()

       



