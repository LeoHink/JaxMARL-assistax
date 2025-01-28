import os
import os.path as osp
import uuid
import pandas as pd
import jax
import jax.numpy as jnp
import chex
import distrax
import flax.linen as nn
from flax import struct
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from flax.traverse_util import flatten_dict, unflatten_dict
import safetensors.flax
from jaxmarl.environments.multi_agent_env import State, MultiAgentEnv
from jaxmarl.wrappers.baselines import JaxMARLWrapper
from typing import Sequence, NamedTuple, Any, Dict, Optional, Callable, Tuple
import functools
from omegaconf import OmegaConf


def _tree_take(pytree, indices, axis=None):
    return jax.tree.map(lambda x: x.take(indices, axis=axis), pytree)


def _stack_tree(pytree_list, axis=0):
    return jax.tree.map(
        lambda *leaf: jnp.stack(leaf, axis=axis),
        *pytree_list
    )

@struct.dataclass
class ActorCriticOutput:
    pi: Optional[chex.Array | Tuple[chex.Array,chex.Array]] = None
    V: Optional[chex.Array] = None
    hstate: Optional[chex.Array] = None

# Define Network architectures.
# TODO would be nice to import these instead of copying them here.
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


class IPPOActorCritic(nn.Module):
    config: Dict

    @nn.compact
    def __call__(self, hstate, x):
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

        ActorCriticOutput(
            pi=pi,
            V=jnp.squeeze(critic, axis=-1),
            hstate=None,
        )


class IPPOActorCriticRNN(nn.Module):
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
        pi = (actor_mean, jnp.exp(actor_log_std))

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

        ActorCriticOutput(
            pi=pi,
            V=jnp.squeeze(critic, axis=-1),
            hstate=hstate,
        )


@struct.dataclass
class ZooState:
    agent_uuid: str
    apply_fn: Callable = struct.field(pytree_node=False)
    params: Dict
    hstate_reset_fn: Callable = struct.field(pytree_node=False, default=lambda x: None)


@struct.dataclass
class LoadNetworkState:
    apply_fn: Callable = struct.field(pytree_node=False)
    params: Dict
    hstate_reset_fn: Callable = struct.field(pytree_node=False, default=lambda x: None)
    pop_size: int = 1


@struct.dataclass
class LoadAgentState:
    _state: State
    ag_idx: Dict[str, chex.Array]
    load_agent_actions: Dict[str, chex.Array]
    hstate: Optional[Dict[str, chex.Array]] = None

    def __getattr__(self, name: str):
        return getattr(self._state, name)


class ZooManager:
    """Class for managing access to the agent zoo."""

    def __init__(self, zoo_path: str):
        self.zoo_path = zoo_path
        self.index_path = osp.join(zoo_path, "index.csv")
        self.index_cols = [
            "agent_uuid",
            "scenario",
            "scenario_agent_id",
            "algorithm",
            "is_rnn",
            "rnn_dim",
        ]

        self._init_zoo(zoo_path)
        self.index = pd.read_csv(self.index_path)

    def load_agent(self, agent_uuid: str) -> ZooState:
        """Load an agent from the zoo given an agent UUID."""
        apply_fn, hstate_reset_fn = self._load_architecture(agent_uuid)
        return ZooState(
            agent_uuid=agent_uuid,
            apply_fn=apply_fn,
            hstate_reset_fn=hstate_reset_fn,
            params=self._load_safetensors(agent_uuid),
        )

    def _load_safetensors(self, agent_uuid: str) -> Dict:
        return unflatten_dict(
            safetensors.flax.load_file(osp.join(self.zoo_path, "params", agent_uuid+".safetensors")),
            sep='/'
        )

    def _save_safetensors(self, agent_uuid, param_dict):
        safetensors.flax.save_file(
            flatten_dict(param_dict, sep='/'),
            osp.join(self.zoo_path, "params", agent_uuid+".safetensors")
        )

    def _load_config(self, agent_uuid: str) -> Dict:
        config_path = osp.join(self.zoo_path, "config", agent_uuid+".yaml")
        return OmegaConf.to_container(
            OmegaConf.load(config_path),
            resolve=True
        )

    def _save_config(self, agent_uuid: str, config):
        if hasattr(config["OBS_DIM"], "item"):
            config["OBS_DIM"] = config["OBS_DIM"].item()
        if hasattr(config["ACT_DIM"], "item"):
            config["ACT_DIM"] = config["ACT_DIM"].item()
        if ("GOBS_DIM" in config) and hasattr(config["GOBS_DIM"], "item"):
            config["GOBS_DIM"] = config["GOBS_DIM"].item()

        OmegaConf.save(
            config,
            osp.join(self.zoo_path, "config", agent_uuid+".yaml")
        )


    def _load_architecture(self, agent_uuid: str) -> Tuple[Callable, Callable]:
        agent_config = self._load_config(agent_uuid)
        is_recurrent = agent_config["network"]["recurrent"]
        recurrent_dim_size = agent_config["network"].get("gru_hidden_dim")

        def _no_rnn_hstate_reset_fn(key):
            return None

        def _zero_rnn_hstate_reset_fn(key):
            return jnp.zeros((recurrent_dim_size,))

        if is_recurrent:
            apply_fn = IPPOActorCriticRNN(config=agent_config).apply
            hstate_reset_fn = _zero_rnn_hstate_reset_fn
        else:
            apply_fn = IPPOActorCritic(config=agent_config).apply
            hstate_reset_fn = _no_rnn_hstate_reset_fn
        return apply_fn, hstate_reset_fn

    def _init_zoo(self, zoo_path):
        """Initialises the zoo directory."""
        os.makedirs(zoo_path, exist_ok=True)
        os.makedirs(osp.join(zoo_path, "config"), exist_ok=True)
        os.makedirs(osp.join(zoo_path, "params"), exist_ok=True)
        if not osp.exists(self.index_path):
            with open(self.index_path, "w", encoding="utf-8") as f:
                f.write(','.join(self.index_cols) + '\n')

    def _write_index(self, index_dict):
        """Writes the details of the current agent to the index file."""
        with open(self.index_path, "a", encoding="utf-8") as f:
            f.write(','.join(str(index_dict[col]) for col in self.index_cols) + '\n')

    def save_agent(self, config, param_dict, scenario_agent_id):
        """Saves the current agent to the zoo."""
        agent_uuid = str(uuid.uuid4())
        # save params
        self._save_safetensors(agent_uuid, param_dict)
        self._save_config(agent_uuid, config)
        self._write_index({
            "agent_uuid": agent_uuid,
            "scenario": config["ENV_NAME"],
            "scenario_agent_id": scenario_agent_id,
            "algorithm": config["ALGORITHM"],
            "is_rnn": config["network"]["recurrent"],
            "rnn_dim": config["network"].get("gru_hidden_dim", 0),
        })


class LoadAgentWrapper(JaxMARLWrapper):
    def __init__(self, env: MultiAgentEnv, load_agents: Dict[str, LoadNetworkState]):
        super().__init__(env)
        self.loaded_agents = list(load_agents.keys())
        self.loaded_params = load_agents
        self.agents = [
            agent
            for agent in self._env.agents
            if agent not in self.loaded_agents
        ]
        self.num_agents = len(self.agents)
        self.num_loaded_agents = len(self.loaded_agents)

    @classmethod
    def load_from_zoo(
        cls,
        env: MultiAgentEnv,
        zoo: ZooManager | str,
        load_agents_uuids: Dict[str, str|list[str]],
    ):
        """Loads agents from a zoo using ZooManager."""
        if isinstance(zoo, str):
            zoo = ZooManager(zoo_path=zoo)
        load_agents = {}
        for agent, agent_uuids in load_agents_uuids.items():
            if isinstance(agent_uuids, str):
                zoo_state = zoo.load_agent(agent_uuids)
                load_agents[agent] = LoadNetworkState(
                    apply_fn=jax.vmap(zoo_state.apply_fn, in_axes=(0,None,None)),
                    hstate_reset_fn=zoo_state.hstate_reset_fn,
                    params=jax.tree.map(lambda x: jnp.expand_dims(x, 0), zoo_state.params),
                    pop_size=1,
                )
            else:
                zoo_states = [zoo.load_agent(agent_uuid) for agent_uuid in agent_uuids]
                # TODO Assumes that all agents have the same architecture
                load_agents[agent] = LoadNetworkState(
                    apply_fn=jax.vmap(zoo_states[0].apply_fn, in_axes=(0,None,None)),
                    hstate_reset_fn=zoo_states[0].hstate_reset_fn,
                    params=_stack_tree([zoo_state.params for zoo_state in zoo_states]),
                    pop_size=len(zoo_states),
                )
        return cls(env, load_agents)

    def take_internal_action(
        self,
        key: chex.PRNGKey,
        obs: Dict[str, chex.Array],
        dones: Dict[str, bool],
        avail_actions: Dict[str, chex.Array],
        hstate: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], Dict[str, chex.Array]]:
        """Compute the action taken by each of the loaded agents and the new RNN hidden state.

        The dict returned contains the actions that all agents in a given population would take.
        So the tree shape has form {agent_id: (pop_size, act_size)}.
        """
        internal_action = {}
        n_hstate = {}
        for agent, train_state in self.loaded_params.items():
            key, _key = jax.random.split(key)
            network_out = train_state.apply_fn(
                train_state.params,
                hstate[agent],
                (obs[agent], dones[agent], avail_actions[agent])
            )
            pi = distrax.MultivariateNormalDiag(*network_out.pi)
            action = pi.sample(seed=_key)
            internal_action[agent] = action
        return internal_action, n_hstate

    def reset_internal_hstates(self, key: chex.PRNGKey) -> Dict[str, chex.Array]:
        """Reset the hstate for each of the loaded agents."""
        return {
            agent: train_state.hstate_reset_fn(_key)
            for _key, (agent, train_state) in zip(
                jax.random.split(key, self.num_loaded_agents),
                self.loaded_params.items()
            )
        }

    def reset_agent_index(self, key: chex.PRNGKey) -> Dict[str, chex.Array]:
        """Reset the agent population ID for each of the loaded agents."""
        return {
            agent: jax.random.randint(_key, shape=(), minval=0, maxval=train_state.pop_size)
            for _key, (agent, train_state) in zip(
                jax.random.split(key, self.num_loaded_agents),
                self.loaded_params.items()
            )
        }

    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], LoadAgentState]:
        """Resets the environment and initialises the loaded agent state."""
        key_env, key_hstate, key_action, key_ag_idx = jax.random.split(key, 4)
        obs, state = self._env.reset(key_env)
        dones = {agent: False for agent in self.loaded_agents}
        avail_actions = self._env.get_avail_actions(state)
        hstate = self.reset_internal_hstates(key_hstate)
        load_agent_actions, hstate = self.take_internal_action(
            key_action, obs, dones, avail_actions, hstate
        )
        ag_idx = self.reset_agent_index(key_ag_idx)
        load_agent_actions = jax.tree.map(lambda i, a: a[i], ag_idx, load_agent_actions)
        state = LoadAgentState(
            _state=state,
            load_agent_actions=load_agent_actions,
            hstate=hstate,
            ag_idx=ag_idx,
        )
        return obs, state

    def step(
        self,
        key: chex.PRNGKey,
        state: LoadAgentState,
        actions: Dict[str, chex.Array],
        reset_state: Optional[LoadAgentState]=None,
    ):
        """Performs step transitions in the environment."""

        key_step, key_reset, key_action, key_ag_idx = jax.random.split(key, 4)

        # read in the loaded agent actions from the state
        actions = {**state.load_agent_actions, **actions}

        obs_st, states_st, rewards, dones, infos = self._env.step_env(
            key_step, state._state, actions
        )
        if reset_state is None:
            obs_re, states_re = self._env.reset(key_reset)
            ag_idx_re = self.reset_agent_index(key_ag_idx)
        else:
            states_re = reset_state
            obs_re = self.get_obs(states_re)
            ag_idx_re = reset_state.ag_idx

        # Auto-reset environment based on termination
        states = jax.tree.map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), states_re, states_st,
        )
        obs = jax.tree.map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), obs_re, obs_st
        )
        ag_idx = jax.tree.map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), ag_idx_re, state.ag_idx
        )

        # Take the next action with the loaded agents
        avail_actions = self._env.get_avail_actions(state)
        load_agent_actions, load_agent_hstate = self.take_internal_action(
            key_action, obs, dones, avail_actions, state.hstate,
        )
        load_agent_actions = jax.tree.map(lambda i, a: a[i], ag_idx, load_agent_actions)
        states = LoadAgentState(
            _state=states,
            load_agent_actions=load_agent_actions,
            hstate=load_agent_hstate,
            ag_idx=ag_idx,
        )

        return obs, states, rewards, dones, infos
