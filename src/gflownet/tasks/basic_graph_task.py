import bz2
import os
import pickle
import pathlib
from omegaconf import OmegaConf
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch_geometric.data as gd
from networkx.algorithms.isomorphism import is_isomorphic
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch_scatter import scatter_logsumexp
from tqdm import tqdm

from gflownet.algo.config import TBVariant
from gflownet.algo.flow_matching import FlowMatching
from gflownet.algo.trajectory_balance import TrajectoryBalance
from gflownet.config import Config
from gflownet.envs.basic_graph_ctx import BasicGraphContext
from gflownet.envs.graph_building_env import (
    Graph,
    GraphAction,
    GraphActionCategorical,
    GraphActionType,
    GraphBuildingEnv,
)
from gflownet.models.graph_transformer import GraphTransformer, GraphTransformerGFN
from gflownet.trainer import FlatRewards, GFNAlgorithm, GFNTask, GFNTrainer, RewardScalar
from gflownet.utils.conditioning import LogZConditional

def n_clique_reward(g, n=4):
    cliques = list(nx.algorithms.clique.find_cliques(g))
    # The number of cliques each node belongs to
    num_cliques = np.bincount(sum(cliques, []))
    cliques_match = [len(i) == n for i in cliques]
    return np.mean(cliques_match) - np.mean(num_cliques)


def colored_n_clique_reward(g, n=4):
    cliques = list(nx.algorithms.clique.find_cliques(g))
    # The number of cliques each node belongs to
    num_cliques = np.bincount(sum(cliques, []))
    colors = {i: g.nodes[i]["v"] for i in g.nodes}

    def color_match(c):
        return np.bincount([colors[i] for i in c]).max() >= n - 1

    cliques_match = [float(len(i) == n) * (1 if color_match(i) else 0.5) for i in cliques]
    return np.maximum(np.sum(cliques_match) - np.sum(num_cliques) + len(g) - 1, -10)


def even_neighbors_reward(g):
    total_correct = 0
    for n in g:
        num_diff_colr = 0
        c = g.nodes[n]["v"]
        for i in g.neighbors(n):
            num_diff_colr += int(g.nodes[i]["v"] != c)
        total_correct += int(num_diff_colr % 2 == 0) - (1 if num_diff_colr == 0 else 0)
    return np.float32((total_correct - len(g.nodes) if len(g.nodes) > 3 else -5) * 10 / 7)


def count_reward(g):
    ncols = np.bincount([g.nodes[i]["v"] for i in g], minlength=2)
    return np.float32(-abs(ncols[0] + ncols[1] / 2 - 3) / 4 * 10)


def generate_two_col_data(data_root, max_nodes=7):
    atl = nx.generators.atlas.graph_atlas_g()
    # Filter out disconnected graphs
    conn = [i for i in atl if 1 <= len(i.nodes) <= max_nodes and nx.is_connected(i)]
    # Create all possible two-colored graphs
    two_col_graphs = [nx.Graph()]
    print(len(conn))
    pb = tqdm(range(117142), disable=None)
    hashes = {}
    rejected = 0

    def node_eq(a, b):
        return a == b

    for g in conn:
        for i in range(2 ** len(g.nodes)):
            g = g.copy()
            for j in range(len(g.nodes)):
                bit = i % 2
                i //= 2
                g.nodes[j]["v"] = bit
            h = nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(g, node_attr="v")
            if h not in hashes:
                hashes[h] = [g]
                two_col_graphs.append(g)
            else:
                if not any(nx.algorithms.isomorphism.is_isomorphic(g, gp, node_eq) for gp in hashes[h]):
                    hashes[h].append(g)
                    two_col_graphs.append(g)
                else:
                    pb.set_description(f"{rejected}", refresh=False)
                    rejected += 1
            pb.update(1)
    with bz2.open(data_root + f"/two_col_{max_nodes}_graphs.pkl.bz", "wb") as f:
        pickle.dump(two_col_graphs, f)
    return two_col_graphs


def load_two_col_data(data_root, max_nodes=7, generate_if_missing=True):
    p = data_root + f"/two_col_{max_nodes}_graphs.pkl.bz"
    print("Loading", p)
    if not os.path.exists(p) and generate_if_missing:
        return generate_two_col_data(data_root, max_nodes=max_nodes)
    with bz2.open(p, "rb") as f:
        data = pickle.load(f)
    return data


class GraphTransformerRegressor(GraphTransformer):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.g2o = torch.nn.Linear(kw["num_emb"] * 2, 1)

    def forward(self, g: gd.Batch, cond: torch.Tensor):
        per_node_pred, per_graph_pred = super().forward(g, cond)
        return self.g2o(per_graph_pred)[:, 0]


class LogZDataset(Dataset):
    def __init__(
        self,
        logZs,
        batch_size_per_logZ=64,
    ):

        data = []
        for logz in logZs:
            data.append(logz * torch.ones(batch_size_per_logZ))
        data = torch.cat(data).unsqueeze(-1)

        self.data = data
        self.idcs = np.arange(len(self.data))

    def __len__(self):
        return len(self.idcs)

    def __getitem__(self, idx):
        idx = self.idcs[idx]
        logZ = self.data[idx]
        return logZ
    

class TwoColorGraphDataset(Dataset):
    def __init__(
        self,
        data,
        ctx,
        train=True,
        output_graphs=False,
        split_seed=142857,
        ratio=0.9,
        max_nodes=7,
        reward_func="const",
        reward_reshape: bool = False,
        reward_corrupt: bool = False,
        reward_shuffle: bool = False,
        reward_temper: bool = False,
        reward_skewed_random: bool = False,
        reward_param: float = 0.0,
    ):
        self.data = data
        self.ctx = ctx
        self.output_graphs = output_graphs
        self.reward_func = reward_func
        self.reward_reshape = reward_reshape
        self.reward_corrupt = reward_corrupt
        self.reward_shuffle = reward_shuffle
        self.reward_temper = reward_temper
        self.reward_skewed_random = reward_skewed_random
        self.reward_param = reward_param
        self.idcs = [0]
        self.max_nodes = max_nodes
        if data is None:
            return

        idcs = np.arange(len(data))
        rng = np.random.default_rng(split_seed)
        rng.shuffle(idcs)
        if train:
            self.idcs = idcs[: int(np.floor(ratio * len(data)))]
        else:
            self.idcs = idcs[int(np.floor(ratio * len(data))) :]

        print(train, self.idcs.shape)
        self._gc = nx.complete_graph(7)
        self._enum_edges = list(self._gc.edges)
        self.compute_Fsa = False
        self.compute_normalized_Fsa = False
        self.regress_to_F = False

        # pre-compute log_rewards and apply selected reward trasnformation(s)
        log_rewards = self.pre_compute_rewards()
        self.adjusted_log_rewards, adjusted_log_rewards = None, log_rewards
        if self.reward_reshape:
            adjusted_log_rewards = self.monotonic_skew_reward_values(adjusted_log_rewards, lam=self.reward_param)
        if self.reward_corrupt:
            adjusted_log_rewards = self.corrupt_reward_values(adjusted_log_rewards, std=self.reward_param)
        if self.reward_shuffle:
            adjusted_log_rewards = self.shuffle_reward_values(adjusted_log_rewards)
        if self.reward_temper:
            adjusted_log_rewards = self.temper_reward_values(adjusted_log_rewards, beta=self.reward_param)
        if self.reward_skewed_random:
            adjusted_log_rewards = self.skewed_random_values(size=len(adjusted_log_rewards), sparse_reward=self.reward_param)

        if self.reward_reshape or self.reward_corrupt or self.reward_shuffle:
            self.adjusted_log_rewards = adjusted_log_rewards

    def __len__(self):
        return len(self.idcs)

    def reward(self, g):
        if self.adjusted_log_rewards is not None:
            g_idx = self.get_graph_idx(g, self.data)
            return self.adjusted_log_rewards[g_idx]
        else:
            return self.reward_type(g)
            
    def reward_type(self, g):
        if len(g.nodes) > self.max_nodes:
            return -100
        if self.reward_func == "cliques":
            return colored_n_clique_reward(g)
        elif self.reward_func == "even_neighbors":
            return even_neighbors_reward(g)
        elif self.reward_func == "count":
            return count_reward(g)
        elif self.reward_func == "const":
            return np.float32(0)
    
    def monotonic_skew_reward_values(self, log_rewards, lam=0.1):
        """
        Apply monotonic trasnformation on reward values
        """
        return self.adjust_reward_skew(log_rewards, lam)

    def corrupt_reward_values(self, log_rewards, std=1.0):
        """
        Corrupt reward values with noised. Used to 
        emulate "Rethinking Generalization" experiments, but for 
        GFlowNets
            TODO: 
                - Currently only for Guassian noise.
                  Could add implementation for Laplace and others.
                - Currently noise is just over one seed
        """
        if std <= 0.:
            return log_rewards
        rng = np.random.default_rng(12345)
        noise = rng.normal(loc=0.0, scale=std, size=np.array(log_rewards).shape)
        return list(log_rewards + noise)

    def shuffle_reward_values(self, log_rewards):
        """
        Shuffles reward value pairing for given graphs. Used to 
        emulate "Rethinking Generalization" experiments, but for 
        GFlowNets
        """
        rng = np.random.default_rng(12345)
        aranged_ids = np.arange(start=0, stop=len(log_rewards))
        rand_ids = rng.choice(aranged_ids, size=aranged_ids.shape, replace=False)
        shuffled_log_rewards = np.array(log_rewards)[rand_ids]
        return list(shuffled_log_rewards)

    def temper_reward_values(self, log_rewards, beta=1.0):
        """
        Temper rewards for pre-computed log_rewards.
        """
        return list(np.array(log_rewards) * (1.0 / beta))

    def skewed_random_values(self, size_log_rewards, sparse_reward=0.0):
        """
        Defines random log-rewards sampled from Rayleigh dsitribution.
        Emulates log-reward skew to high and low rewards. 'Sparser' rewards
        skew log-reward distribution to higher mass around lower rewards.
        """
        rng = np.random.default_rng(12345)
        if sparse_reward > 0.0:
            x = rng.rayleigh(2.6, size=size_log_rewards) - 10
            idcs = (x > 0)
            x[idcs] = 0
            idcs = (x < -10)
            x[idcs] = -10
        else:
            x = - rng.rayleigh(2.6, size=size_log_rewards) 
            idcs = (x > 0)
            x[idcs] = 0
            idcs = (x < -10)
            x[idcs] = -10
        return x

    def adjust_reward_skew(self, log_rewards, lam=0.1):
        """
        Skew the reward function towards favouring higher reward
        values. 
        """
        r_bins = list(set(log_rewards)) 
        mono_weights = np.exp(- lam * np.array(r_bins))
        log_rewards_skew = []
        
        for r in log_rewards:
            i = np.where(r_bins == r)[0][0]
            log_rewards_skew.append(mono_weights[i] * r)
            
        log_rewards_skew = np.array(log_rewards_skew) / np.min(log_rewards_skew) * np.min(r_bins)
        return list(log_rewards_skew)
    
    def get_graph_idx(self, g, states, default=None):
        def iso(u, v):
            return is_isomorphic(u, v, lambda a, b: a == b, lambda a, b: a == b)

        h = hashg(g)
        if h not in self._hash_to_graphs:
            if default is not None:
                return default
            else:
                print("Graph not found in cache", h)
                for i in g.nodes:
                    print(i, g.nodes[i])
                for i in g.edges:
                    print(i, g.edges[i])
        bucket = self._hash_to_graphs[h]
        if len(bucket) == 1:
            return bucket[0]
        for i in bucket:
            if iso(states[i], g):
                return i
        if default is not None:
            return default
        raise ValueError(g)
      
    def hashg_for_graphs(self):
        states = self.data
        _hash_to_graphs = {}
        states_hash = [hashg(i) for i in tqdm(states, disable=True)]
        for i, h in enumerate(states_hash):
            _hash_to_graphs[h] = _hash_to_graphs.get(h, list()) + [i]
        return _hash_to_graphs

    def pre_compute_rewards(self):
        self._hash_to_graphs = self.hashg_for_graphs()
        log_rewards = [
            self.reward_type(self.data[self.get_graph_idx(g, self.data)])
            for g in self.data
            ]
        return log_rewards

    def collate_fn(self, batch):
        graphs, rewards, idcs = zip(*batch)
        batch = self.ctx.collate(graphs)
        if self.regress_to_F:
            batch.y = torch.as_tensor([self.epc.mdp_graph.nodes[i]["F"] for i in idcs])
        else:
            batch.y = torch.as_tensor(rewards)
        if self.compute_Fsa:
            all_targets = []
            for data_idx in idcs:
                targets = [
                    torch.zeros_like(getattr(self.epc._Data[data_idx], i.mask_name)) - 100
                    for i in self.ctx.action_type_order
                ]
                for neighbor in list(self.epc.mdp_graph.neighbors(data_idx)):
                    for _, edge in self.epc.mdp_graph.get_edge_data(data_idx, neighbor).items():
                        a, F = edge["a"], edge["F"]
                        targets[a[0]][a[1], a[2]] = F
                if self.compute_normalized_Fsa:
                    logZ = torch.log(sum([i.exp().sum() for i in targets]))
                    targets = [i - logZ for i in targets]
                all_targets.append(targets)
            batch.y = torch.cat([torch.cat(i).flatten() for i in zip(*all_targets)])
        return batch

    def __getitem__(self, idx):
        idx = self.idcs[idx]
        g = self.data[idx]
        r = torch.tensor(self.reward(g).reshape((1,)))
        if self.output_graphs:
            return self.ctx.graph_to_Data(g), r, idx
        else:
            return g, r


class BasicGraphTask(GFNTask):
    def __init__(
        self,
        cfg: Config,
        dataset: TwoColorGraphDataset,
        rng: np.random.Generator = None,
    ):
        self.dataset = dataset
        self.cfg = cfg
        self.rng = rng
        self.logZ_conditional = LogZConditional(cfg, rng)

    def flat_reward_transform(self, y: Tensor) -> FlatRewards:
        return FlatRewards(y.float())

    def sample_conditional_information(self, n: int, train_it: int = 0):
        if self.cfg.cond.logZ.sample_dist is not None:
            return self.logZ_conditional.sample(n)
        else:
            return {"encoding": torch.zeros((n, 1))}

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
        return RewardScalar(flat_reward[:, 0].float())

    def compute_flat_rewards(self, mols: List[Graph]) -> Tuple[FlatRewards, Tensor]:
        if not len(mols):
            return FlatRewards(torch.zeros((0, 1))), torch.zeros((0,)).bool()
        is_valid = torch.ones(len(mols)).bool()
        flat_rewards = torch.tensor([self.dataset.reward(i) for i in mols]).float().reshape((-1, 1))
        return FlatRewards(flat_rewards), is_valid

    def encode_conditional_information(self, info):
        if self.cfg.cond.logZ.sample_dist is not None:
            encoding = self.logZ_conditional.encode(info)
            return {"beta": torch.ones(len(info)), "encoding": encoding.float(), "preferences": torch.tensor(info).float()}
        else:
            encoding = torch.zeros((len(info), 1))
            return {"beta": torch.ones(len(info)), "encoding": encoding.float(), "preferences": info.float()}


class UnpermutedGraphEnv(GraphBuildingEnv):
    """When using a tabular model, we want to always give the same graph node order, this environment
    just makes sure that happens"""

    def set_epc(self, epc):
        self.epc = epc

    def step(self, g: Graph, ga: GraphAction):
        g = super().step(g, ga)
        # get_graph_idx hashes the graph and so returns the same idx for the same graph up to isomorphism/node order
        return self.epc.states[self.epc.get_graph_idx(g)]


class BasicGraphTaskTrainer(GFNTrainer):
    cfg: Config
    training_data: TwoColorGraphDataset
    test_data: TwoColorGraphDataset

    def set_default_hps(self, cfg: Config):
        cfg.opt.learning_rate = 1e-4
        cfg.opt.weight_decay = 1e-8
        cfg.opt.momentum = 0.9
        cfg.opt.adam_eps = 1e-8
        cfg.opt.lr_decay = 20000
        cfg.opt.clip_grad_param = 10
        cfg.opt.clip_grad_type = "none"  # "norm"
        cfg.algo.max_nodes = 7
        cfg.algo.global_batch_size = 64
        cfg.model.num_emb = 96
        cfg.model.num_layers = 8
        cfg.algo.valid_offline_ratio = 0
        cfg.algo.tb.do_correct_idempotent = True  # Important to converge to the true p(x)
        cfg.algo.tb.variant = TBVariant.SubTB1
        cfg.algo.tb.do_parameterize_p_b = False
        cfg.algo.illegal_action_logreward = -30  # Although, all states are legal here, this shouldn't matter
        cfg.num_workers = 8
        cfg.algo.train_random_action_prob = 0.01
        cfg.log_sampled_data = False
        # Because we're using a RepeatedPreferencesDataset
        cfg.algo.valid_sample_cond_info = True # this should be true (was false o.g.)?
        cfg.algo.offline_ratio = 0

    def setup(self):
        mcfg = self.cfg.task.basic_graph
        max_nodes = self.cfg.algo.max_nodes
        print(self.cfg.log_dir)
        self.rng = np.random.default_rng(142857)
        if mcfg.do_tabular_model:
            self.env = UnpermutedGraphEnv()
        else:
            self.env = GraphBuildingEnv()
        self._data = load_two_col_data(self.cfg.task.basic_graph.data_root, max_nodes=max_nodes)
        if self.cfg.cond.logZ.sample_dist is not None:
            self.ctx = BasicGraphContext(max_nodes, num_cond_dim=self.cfg.cond.logZ.num_thermometer_dim + 1, graph_data=self._data, output_gid=True)
        else:
            self.ctx = BasicGraphContext(max_nodes, num_cond_dim=1, graph_data=self._data, output_gid=True)
        self.ctx.use_graph_cache = mcfg.do_tabular_model
        self._do_supervised = self.cfg.task.basic_graph.do_supervised

        self.training_data = TwoColorGraphDataset(
            self._data, 
            self.ctx, 
            train=True, 
            ratio=mcfg.train_ratio, 
            max_nodes=max_nodes, 
            reward_func=mcfg.reward_func, 
            reward_reshape=mcfg.reward_reshape, 
            reward_corrupt=mcfg.reward_corrupt,
            reward_shuffle=mcfg.reward_shuffle,
            reward_temper=mcfg.reward_temper,
            reward_param=mcfg.reward_param
        )
        self.test_data = TwoColorGraphDataset(
            self._data, self.ctx, 
            train=False, 
            ratio=mcfg.train_ratio, 
            max_nodes=max_nodes, 
            reward_func=mcfg.reward_func, 
            reward_reshape=mcfg.reward_reshape, 
            reward_corrupt=mcfg.reward_corrupt,
            reward_shuffle=mcfg.reward_shuffle,
            reward_temper=mcfg.reward_temper,
            reward_param=mcfg.reward_param
        )

        self.exact_prob_cb = ExactProbCompCallback(
            self,
            self.training_data.data,
            self.device,
            cache_root=self.cfg.task.basic_graph.data_root,
            cache_path=self.cfg.task.basic_graph.data_root + f"/two_col_epc_cache_{max_nodes}.pkl",
            log_rewards=self.training_data.adjusted_log_rewards if mcfg.reward_reshape or mcfg.reward_corrupt or mcfg.reward_shuffle else None,
            logits_shuffle=mcfg.logits_shuffle,
        )
        if mcfg.do_tabular_model:
            self.env.set_epc(self.exact_prob_cb)

        if self._do_supervised and not self.cfg.task.basic_graph.regress_to_Fsa:
            model = GraphTransformerRegressor(
                x_dim=self.ctx.num_node_dim,
                e_dim=self.ctx.num_edge_dim,
                g_dim=1,
                num_emb=self.cfg.model.num_emb,
                num_layers=self.cfg.model.num_layers,
                num_heads=self.cfg.model.graph_transformer.num_heads,
                ln_type=self.cfg.model.graph_transformer.ln_type,
            )
        elif mcfg.do_tabular_model:
            model = TabularHashingModel(self.exact_prob_cb)
            if 0:
                model.set_values(self.exact_prob_cb)
        else:
            model = GraphTransformerGFN(
                self.ctx,
                self.cfg,
                do_bck=self.cfg.algo.tb.do_parameterize_p_b,
            )
        if not self._do_supervised:
            self.test_data = RepeatedPreferenceDataset(np.zeros((32, 1)), 8)

        self.model = self.sampling_model = model
        params = [i for i in self.model.parameters()]
        if self.cfg.opt.opt == "adam":
            self.opt = torch.optim.Adam(
                params,
                self.cfg.opt.learning_rate,
                (self.cfg.opt.momentum, 0.999),
                weight_decay=self.cfg.opt.weight_decay,
                eps=self.cfg.opt.adam_eps,
            )
        elif self.cfg.opt.opt == "SGD":
            self.opt = torch.optim.SGD(
                params, self.cfg.opt.learning_rate, self.cfg.opt.momentum, weight_decay=self.cfg.opt.weight_decay
            )
        elif self.cfg.opt.opt == "RMSProp":
            self.opt = torch.optim.RMSprop(params, self.cfg.opt.learning_rate, weight_decay=self.cfg.opt.weight_decay)
        self.lr_sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda steps: 2 ** (-steps / self.cfg.opt.lr_decay))

        algo = self.cfg.algo.method
        if algo == "TB" or algo == "subTB":
            self.algo = TrajectoryBalance(self.env, self.ctx, self.rng, self.cfg)
        elif algo == "FM":
            self.algo = FlowMatching(self.env, self.ctx, self.rng, self.cfg)

        self.algo.model_is_autoregressive = False

        #if self.cfg.cond.logZ.sample_dist is None:
        #    assert "offline_ration == 0 but using conditional logZ for online model", self.cfg.algo.offline_ratio == 0
        self.task = BasicGraphTask(
            self.cfg,
            self.training_data,
            np.random.default_rng(self.cfg.seed),
        )

        if self.cfg.algo.flow_reg:
            dist_params = self.cfg.cond.logZ.dist_params
            num_logZ = self.cfg.cond.logZ.num_valid_logZ_samples
            if self.cfg.cond.logZ.sample_dist is not None:
                logZs = np.linspace(dist_params[0], dist_params[1], num_logZ).tolist()
                self.test_cond_logZs_data = LogZDataset(logZs, batch_size_per_logZ=self.cfg.algo.global_batch_size)
            if self.cfg.algo.supervised_reward_predictor is not None:
                self.algo.model_supervised_reward_predictor = GraphTransformerRegressor(
                    x_dim=self.ctx.num_node_dim,
                    e_dim=self.ctx.num_edge_dim,
                    g_dim=1,
                    num_emb=self.cfg.model.num_emb,
                    num_layers=self.cfg.model.num_layers,
                    num_heads=self.cfg.model.graph_transformer.num_heads,
                    ln_type=self.cfg.model.graph_transformer.ln_type,
                )
                print("Loading supervised trained model for unseen reward prediction ...")
                load_path = self.cfg.algo.supervised_reward_predictor + mcfg.reward_func + '/model_state.pt'
                model_pre_state = torch.load(load_path, map_location=self.cfg.device)
                self.algo.model_supervised_reward_predictor.load_state_dict(model_pre_state['models_state_dict'][0])
                self.algo.model_supervised_reward_predictor.to(self.cfg.device)
                print("Done")

        # initialize and load model for interpolated sampling
        if self.cfg.algo.dir_model_pretrain_for_sampling is not None:
            if self._do_supervised and not self.cfg.task.basic_graph.regress_to_Fsa:
                self.model_pretrain_for_sampling = GraphTransformerRegressor(
                    x_dim=self.ctx.num_node_dim,
                    e_dim=self.ctx.num_edge_dim,
                    g_dim=1,
                    num_emb=self.cfg.model.num_emb,
                    num_layers=self.cfg.model.num_layers,
                    num_heads=self.cfg.model.graph_transformer.num_heads,
                    ln_type=self.cfg.model.graph_transformer.ln_type,
                )
                print("Loading pre-trained model for sampling...")
                model_pre_state = torch.load(self.cfg.algo.dir_model_pretrain_for_sampling, map_location=self.cfg.device)
                self.model_pretrain_for_sampling.load_state_dict(model_pre_state['models_state_dict'][0])
                print("Done")
            else:
                self.model_pretrain_for_sampling = GraphTransformerGFN(
                    self.ctx,
                    self.cfg,
                    do_bck=self.cfg.algo.tb.do_parameterize_p_b,
                )
                print("Loading pre-trained model for sampling...")
                model_pre_state = torch.load(self.cfg.algo.dir_model_pretrain_for_sampling, map_location=self.cfg.device)
                self.model_pretrain_for_sampling.load_state_dict(model_pre_state['models_state_dict'][0])
                print("Done")
        else:  
            self.model_pretrain_for_sampling = None

        # For offline training -- set p(x) to be used for sampling x ~ p(x)
        if isinstance(model, GraphTransformerGFN):
            # select use of true log_Z
            if self.cfg.algo.use_true_log_Z:
                self.cfg.algo.true_log_Z = float(self.exact_prob_cb.logZ)
            # select x ~ p(x) sampling
            if self.cfg.algo.offline_sampling_g_distribution == "log_rewards": # x ~ R(x)/Z
                self.log_sampling_g_distribution = self.exact_prob_cb.true_log_probs
            elif self.cfg.algo.offline_sampling_g_distribution == "log_p": # x ~ p(x; \theta)
                self.log_sampling_g_distribution = self.exact_prob_cb.compute_prob(model.to(self.cfg.device))[0].cpu().numpy()[:-1]
            elif self.cfg.algo.offline_sampling_g_distribution == "l2_log_error_gfn" or self.cfg.algo.offline_sampling_g_distribution == "l1_error_gfn": # x ~ ||p(x; \theta) - p(x)||
                model_log_probs = self.exact_prob_cb.compute_prob(model.to(self.cfg.device))[0].cpu().numpy()[:-1]
                true_log_probs = self.exact_prob_cb.true_log_probs
                err = []
                for lq, lp in zip(model_log_probs, true_log_probs):
                    if self.cfg.algo.offline_sampling_g_distribution == "l2_log_error_gfn":
                        err.append((lq - lp)**2)
                    else:
                        err.append(np.abs(np.exp(lq) - np.exp(lp)))
                err = np.array(err)
                err = err / np.sum(err)
                self.log_sampling_g_distribution = np.log(err)
            elif self.cfg.algo.offline_sampling_g_distribution == "uniform": # x ~ Unif(x)
                self.log_sampling_g_distribution = -1 * np.ones_like(self.exact_prob_cb.true_log_probs) # uniform distribution
            elif self.cfg.algo.offline_sampling_g_distribution == "random":
                rng = np.random.default_rng(self.cfg.seed)
                self.log_sampling_g_distribution = rng.uniform(0, 10, len(self.exact_prob_cb.true_log_probs))
            else: 
                self.log_sampling_g_distribution = None
        self.sampling_tau = self.cfg.algo.sampling_tau
        self.mb_size = self.cfg.algo.global_batch_size
        self.clip_grad_param = self.cfg.opt.clip_grad_param
        self.clip_grad_callback = {
            "value": (lambda params: torch.nn.utils.clip_grad_value_(params, self.clip_grad_param)),
            "norm": (lambda params: torch.nn.utils.clip_grad_norm_(params, self.clip_grad_param)),
            "none": (lambda x: None),
        }[self.cfg.opt.clip_grad_type]

        self.algo.task = self.task
        if self.cfg.task.basic_graph.test_split_type == "random":
            pass
        elif self.cfg.task.basic_graph.test_split_type == "bck_traj":
            train_idcs, test_idcs = self.exact_prob_cb.get_bck_trajectory_test_split(
                self.cfg.task.basic_graph.train_ratio
            )
            self.training_data.idcs = train_idcs
            self.test_data.idcs = test_idcs
        elif self.cfg.task.basic_graph.test_split_type == "subtrees":
            train_idcs, test_idcs = self.exact_prob_cb.get_subtree_test_split(
                self.cfg.task.basic_graph.train_ratio, self.cfg.task.basic_graph.test_split_seed
            )
            self.training_data.idcs = train_idcs
            self.test_data.idcs = test_idcs
        if not self._do_supervised or self.cfg.task.basic_graph.regress_to_Fsa:
            self._callbacks = {"true_px_error": self.exact_prob_cb}
        else:
            self._callbacks = {}
            
        os.makedirs(self.cfg.log_dir, exist_ok=True)
        print("\n\nHyperparameters:\n")
        yaml = OmegaConf.to_yaml(self.cfg)
        print(yaml)
        with open(pathlib.Path(self.cfg.log_dir) / "hps.yaml", "w") as f:
            f.write(yaml)

    def build_callbacks(self):
        return self._callbacks

    def step(self, loss: Tensor):
        loss.backward()
        for i in self.model.parameters():
            self.clip_grad_callback(i)
        self.opt.step()
        self.opt.zero_grad()
        self.lr_sched.step()
        if self.sampling_tau > 0:
            for a, b in zip(self.model.parameters(), self.sampling_model.parameters()):
                b.data.mul_(self.sampling_tau).add_(a.data * (1 - self.sampling_tau))


class RepeatedPreferenceDataset(TwoColorGraphDataset):
    def __init__(self, preferences, repeat):
        self.prefs = preferences
        self.repeat = repeat

    def __len__(self):
        return len(self.prefs) * self.repeat

    def __getitem__(self, idx):
        assert 0 <= idx < len(self)
        return torch.tensor(self.prefs[int(idx // self.repeat)])


def hashg(g):
    return nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(g, node_attr="v")


class TabularHashingModel(torch.nn.Module):
    """A tabular model to ensure that the objectives converge to the correct solution."""

    def __init__(self, epc):
        super().__init__()
        self.epc = epc
        self.action_types = [GraphActionType.Stop, GraphActionType.AddNode, GraphActionType.AddEdge]
        # This makes a big array which is then sliced and reshaped into logits.
        # We're using the masks's shapes to determine the size of the table because they're the same shape
        # as the logits. The [1] is the F(s) prediction used for SubTB.
        num_total = 0
        self.slices = []
        self.shapes = []
        print("Making table...")
        for gid in tqdm(range(len(self.epc.states))):
            this_slice = [num_total]
            self.shapes.append(
                [
                    epc._Data[gid].stop_mask.shape,
                    epc._Data[gid].add_node_mask.shape,
                    epc._Data[gid].add_edge_mask.shape,
                    [1],
                ]
            )
            ns = [np.prod(i) for i in self.shapes[-1]]
            this_slice += list(np.cumsum(ns) + num_total)
            num_total += sum(ns)
            self.slices.append(this_slice)
        self.table = nn.Parameter(torch.zeros((num_total,)))
        # For TB we have to have a unique parameter for logZ
        self._logZ = nn.Parameter(torch.zeros((1,)))
        print("Made table of size", num_total)

    def __call__(self, g: gd.Batch, cond_info):
        """This ignores cond_info, which we don't use anyways for now, but beware"""
        ns = [self.slices[i] for i in g.gid.cpu()]
        shapes = [self.shapes[i] for i in g.gid.cpu()]
        items = [[self.table[a:b].reshape(s) for a, b, s in zip(n, n[1:], ss)] for n, ss in zip(ns, shapes)]
        logits = zip(*[i[0:3] for i in items])
        logF_s = torch.stack([i[-1] for i in items])
        masks = [GraphTransformerGFN._action_type_to_mask(None, t, g) for t in self.action_types]
        return (
            GraphActionCategorical(
                g,
                logits=[torch.cat(i, 0) * m - 1000 * (1 - m) for i, m in zip(logits, masks)],
                keys=[
                    GraphTransformerGFN._graph_part_to_key[GraphTransformerGFN._action_type_to_graph_part[t]]
                    for t in self.action_types
                ],
                masks=masks,
                types=self.action_types,
            ),
            logF_s,
        )

    def set_values(self, epc):
        """Set the values of the table to the true values of the MDP. This tabular model should have 0 error."""
        for i in tqdm(range(len(epc.states))):
            for neighbor in list(epc.mdp_graph.neighbors(i)):
                for _, edge in epc.mdp_graph.get_edge_data(i, neighbor).items():
                    a, F = edge["a"], edge["F"]
                    self.table.data[self.slices[i][a[0]] + a[1] * self.shapes[i][a[0]][1] + a[2]] = F
            self.table.data[self.slices[i][3]] = epc.mdp_graph.nodes[i]["F"]
        self._logZ.data = torch.tensor(epc.mdp_graph.nodes[0]["F"]).float()

    def logZ(self, cond_info: Tensor):
        return self._logZ.tile(cond_info.shape[0]).reshape((-1, 1))  # Why is the reshape necessary?


class ExactProbCompCallback:
    ctx: BasicGraphContext
    trial: BasicGraphTaskTrainer
    mdp_graph: nx.DiGraph

    def __init__(
        self,
        trial,
        states,
        dev,
        mbs=128,
        cache_root=None,
        cache_path=None,
        do_save_px=True,
        log_rewards=None,
        logits_shuffle=False,
        tqdm_disable=None,
        ctx=None,
        env=None,
    ):
        self.trial = trial
        self.ctx = trial.ctx if trial is not None else ctx
        self.env = trial.env if trial is not None else env
        self.mbs = mbs
        self.dev = dev
        self.states = states
        self.cache_root = cache_root
        self.cache_path = cache_path
        self.mdp_graph = None
        if self.cache_path is not None:
            self.load_cache(self.cache_path)
        if log_rewards is None:
            self.log_rewards = np.array(
                [self.trial.training_data.reward(i) for i in tqdm(self.states, disable=tqdm_disable)]
            )
        else:
            self.log_rewards = log_rewards
        self.logZ = np.log(np.sum(np.exp(self.log_rewards)))
        self.true_log_probs = self.log_rewards - self.logZ
        self.logits_shuffle = logits_shuffle
        # This is reward-dependent
        if self.mdp_graph is not None:
            self.recompute_flow()
        self.do_save_px = do_save_px
        if do_save_px:
            os.makedirs(self.trial.cfg.log_dir, exist_ok=True)
        self._save_increment = 0

    def load_cache(self, cache_path):
        print("Loading cache @", cache_path)
        cache = torch.load(open(cache_path, "rb"))
        self.mdp_graph = cache["mdp"]
        self._Data = cache["Data"]
        self._hash_to_graphs = cache["hashmap"]
        bs, ids = cache["batches"], cache["idces"]
        print("Done")
        self.precomputed_batches, self.precomputed_indices = (
            [i.to(self.dev) for i in bs],
            [[(j[0].to(self.dev), j[1].to(self.dev)) for j in i] for i in ids],
        )

    def compute_metrics(self, log_probs, state_flows, log_rewards_estimate, valid_batch_ids=None):
        log_probs = log_probs.cpu().numpy()[:-1].clip(-10_000, 0)
        state_flows = state_flows.cpu().numpy().flatten()
        log_rewards_estimate = log_rewards_estimate.cpu().numpy().flatten()
        log_rewards = np.asarray(self.log_rewards)
        lp, p = log_probs, np.exp(log_probs)
        lq, q = self.true_log_probs, np.exp(self.true_log_probs)
        self.trial.model_log_probs, self.trial.true_log_probs = log_probs, self.true_log_probs
        mae_log_probs = np.mean(abs(lp - lq))
        js_log_probs = (p * (np.log(p/2 + q/2 + 1e-38) - lp) + q * (np.log(p/2 + q/2 + 1e-38)  - lq)).sum() / 2
        jeff_log_probs = (p * (lp - lq) + q * (lq - lp)).sum() / 2
        mae_log_rewards = np.mean(abs(log_rewards_estimate - log_rewards)) 
        print("L1 logpx error", mae_log_probs, "JS divergence", js_log_probs)

        if self.do_save_px and self.trial.cfg.cond.logZ.sample_dist is None:
            torch.save(log_probs, open(self.trial.cfg.log_dir + f"/log_px_{self._save_increment}.pt", "wb"))
            self._save_increment += 1

        metrics_dict = {}
        if valid_batch_ids is not None:
            lp_valid, p_valid = log_probs[valid_batch_ids], np.exp(log_probs[valid_batch_ids])
            lq_valid, q_valid = self.true_log_probs[valid_batch_ids], np.exp(self.true_log_probs[valid_batch_ids])
            test_mae_log_probs = np.mean(abs(lp_valid - lq_valid))
            metrics_dict["test_graphs-L1_logpx_error"] = test_mae_log_probs
            if self.trial.cfg.algo.dir_model_pretrain_for_sampling is None:        
                if isinstance(log_rewards, list):
                    test_mae_log_rewards = np.mean(abs(log_rewards_estimate[valid_batch_ids] - np.array(log_rewards)[valid_batch_ids]))
                else:
                    test_mae_log_rewards = np.mean(abs(log_rewards_estimate[valid_batch_ids] - log_rewards[valid_batch_ids]))
                metrics_dict["test_graphs-L1_log_R_error"] = test_mae_log_rewards
         
        metrics_dict["L1_logpx_error"] = mae_log_probs
        metrics_dict["JS_divergence"] = js_log_probs
        metrics_dict["L1_log_R_error"] = mae_log_rewards

        return metrics_dict 

    def on_validation_end(self, metrics, valid_batch_ids=None):
        # Compute exact sampling probabilities of the model, last probability is p(illegal), remove it.
        if self.trial.cfg.cond.logZ.sample_dist is not None:
            logZ_true = self.logZ * torch.ones((1, 1)) #* torch.ones((1, self.trial.cfg.cond.logZ.num_thermometer_dim + 1)).to(self.dev)
            logZ_true_enc = self.trial.task.encode_conditional_information(logZ_true)
            cond_info = logZ_true_enc['encoding'].squeeze(0).to(self.dev)
            log_probs, state_flows, log_rewards_estimate = self.compute_prob(self.trial.model, cond_info=cond_info) # compute once using correct logZ
            metrics_true_logZ = self.compute_metrics(log_probs, state_flows, log_rewards_estimate, valid_batch_ids)
            
            if self.do_save_px:
                torch.save(log_probs, open(self.trial.cfg.log_dir + f"/log_px_val_iter_{self._save_increment}_logZ_{logZ_true.mean()}.pt", "wb"))
            
            dist_params = self.trial.cfg.cond.logZ.dist_params
            num_logZ = self.trial.cfg.cond.logZ.num_valid_logZ_samples
            metrics_range_logZ = {k: [v] for k, v in metrics_true_logZ.items()}

            for logz in np.linspace(dist_params[0], dist_params[1], num_logZ).tolist(): # select size of range for logZ's
                logZ_sampled = logz * torch.ones((1, 1)) #* torch.ones((1, self.trial.cfg.cond.logZ.num_thermometer_dim + 1)).to(self.dev)
                logZ_sampled_enc = self.trial.task.encode_conditional_information(logZ_sampled)
                cond_info = logZ_sampled_enc['encoding'].squeeze(0).to(self.dev)
                log_probs, state_flows, log_rewards_estimate = self.compute_prob(self.trial.model, cond_info=cond_info)
                metrics_tmp = self.compute_metrics(log_probs, state_flows, log_rewards_estimate, valid_batch_ids)

                if self.do_save_px:
                    torch.save(log_probs, open(self.trial.cfg.log_dir + f"/log_px_val_iter_{self._save_increment}_logZ_{logz}.pt", "wb"))

                for k in metrics_range_logZ.keys():
                    metrics_range_logZ[k].append(metrics_tmp[k])

            for k, v in metrics_range_logZ.items():
                metrics[k] = np.array(v)

            if self.do_save_px:
                self._save_increment += 1

        else:
            log_probs, state_flows, log_rewards_estimate = self.compute_prob(self.trial.model)
            metrics_pre = self.compute_metrics(log_probs, state_flows, log_rewards_estimate, valid_batch_ids)
            for k, v in metrics_pre.items():
                metrics[k] = np.array(v)

    def get_graph_idx(self, g, default=None):
        def iso(u, v):
            return is_isomorphic(u, v, lambda a, b: a == b, lambda a, b: a == b)

        h = hashg(g)
        if h not in self._hash_to_graphs:
            if default is not None:
                return default
            else:
                print("Graph not found in cache", h)
                for i in g.nodes:
                    print(i, g.nodes[i])
                for i in g.edges:
                    print(i, g.edges[i])
        bucket = self._hash_to_graphs[h]
        if len(bucket) == 1:
            return bucket[0]
        for i in bucket:
            if iso(self.states[i], g):
                return i
        if default is not None:
            return default
        raise ValueError(g)

    def compute_cache(self, tqdm_disable=None):
        states, mbs, dev = self.states, self.mbs, self.dev
        mdp_graph = nx.MultiDiGraph()
        self.precomputed_batches = []
        self.precomputed_indices = []
        self._hash_to_graphs = {}
        states_hash = [hashg(i) for i in tqdm(states, disable=tqdm_disable)]
        self._Data = states_Data = gd.Batch.from_data_list(
            [self.ctx.graph_to_Data(i) for i in tqdm(states, disable=tqdm_disable)]
        )
        for i, h in enumerate(states_hash):
            self._hash_to_graphs[h] = self._hash_to_graphs.get(h, list()) + [i]

        for bi in tqdm(range(0, len(states), mbs), disable=tqdm_disable):
            bs = states[bi : bi + mbs]
            bD = states_Data[bi : bi + mbs]
            indices = list(range(bi, bi + len(bs)))
            batch = self.ctx.collate(bD).to(dev)
            self.precomputed_batches.append(batch)
            actions = [[] for i in range(len(bs))]
            offset = 0
            for u, i in enumerate(ctx.action_type_order):
                # /!\ This assumes mask.shape == cat.logit[i].shape
                mask = getattr(batch, i.mask_name)
                batch_key = GraphTransformerGFN._graph_part_to_key[GraphTransformerGFN._action_type_to_graph_part[i]]
                batch_idx = (
                    getattr(batch, f"{batch_key}_batch" if batch_key != "x" else "batch")
                    if batch_key is not None
                    else torch.arange(batch.num_graphs, device=dev)
                )
                mslice = (
                    batch._slice_dict[batch_key]
                    if batch_key is not None
                    else torch.arange(batch.num_graphs + 1, device=dev)
                )
                for j in mask.nonzero().cpu().numpy():
                    # We're using nonzero above to enumerate all positions, but we still need to check
                    # if the mask is nonzero since we only want the legal actions.
                    # We *don't* wan't mask.nonzero() because then `k` would be wrong
                    k = j[0] * mask.shape[1] + j[1] + offset
                    jb = batch_idx[j[0]].item()
                    actions[jb].append((u, j[0] - mslice[jb].item(), j[1], k))
                offset += mask.numel()
            all_indices = []
            for jb, j_acts in enumerate(actions):
                end_indices = []
                being_indices = []
                for *a, srcidx in j_acts:
                    idx = indices[jb]
                    sp = self.env.step(bs[jb], self.ctx.aidx_to_GraphAction(bD[jb], a[:3])) if a[0] != 0 else bs[jb]
                    spidx = self.get_graph_idx(sp, len(states))
                    if a[0] == 0 or spidx >= len(states):
                        end_indices.append((idx, spidx, srcidx))
                        mdp_graph.add_edge(idx, spidx, srci=srcidx, term=True, a=a)
                    else:
                        being_indices.append((idx, spidx, srcidx))
                        mdp_graph.add_edge(idx, spidx, srci=srcidx, term=False, a=a)
                all_indices.append((torch.tensor(end_indices).T.to(dev), torch.tensor(being_indices).T.to(dev)))
            self.precomputed_indices.append(all_indices)
        self.mdp_graph = mdp_graph

    def save_cache(self, path):
        with open(path, "wb") as f:
            torch.save(
                {
                    "batches": [i.cpu() for i in self.precomputed_batches],
                    "idces": [[(j[0].cpu(), j[1].cpu()) for j in i] for i in self.precomputed_indices],
                    "Data": self._Data,
                    "mdp": self.mdp_graph,
                    "hashmap": self._hash_to_graphs,
                },
                f,
            )

    def compute_prob(self, model, cond_info=None, tqdm_disable=None):
        # +1 to count illegal actions prob (may not be applicable to well-masked envs)
        prob_of_being_t = torch.zeros(len(self.states) + 1).to(self.dev) - 100
        prob_of_being_t[0] = 0
        prob_of_ending_t = torch.zeros(len(self.states) + 1).to(self.dev) - 100
        state_log_flows = torch.zeros((len(self.states), 1)).to(self.dev) 
        log_rewards_estimate = torch.zeros((len(self.states), 1)).to(self.dev)
        if cond_info is None:
            cond_info = torch.zeros((self.mbs, self.ctx.num_cond_dim)).to(self.dev)
        if cond_info.ndim == 1:
            cond_info = cond_info[None, :] * torch.ones((self.mbs, 1)).to(self.dev)
        if cond_info.ndim == 2 and cond_info.shape[0] == 1:
            cond_info = cond_info * torch.ones((self.mbs, 1)).to(self.dev)
        # Note: visiting the states in order works because the ordering here is a natural topological sort.
        # Wrong results otherwise.
        for bi, batch, pre_indices in zip(
            tqdm(range(0, len(self.states), self.mbs), disable=tqdm_disable),
            self.precomputed_batches,
            self.precomputed_indices,
        ):
            bs = self.states[bi : bi + self.mbs]
            # This isn't even right:
            # indices = list(range(bi, bi + len(bs)))
            # non_terminals = [(i, j) for i, j in zip(bs, indices) if not self.is_terminal(i)]
            # if not len(non_terminals):
            #     continue
            # bs, indices = zip(*non_terminals)
            with torch.no_grad():
                cat, *_, mo = model(batch, cond_info[: len(bs)])
            logprobs = torch.cat([i.flatten() for i in cat.logsoftmax()])

            state_log_flows[bi : bi + len(bs)] = mo
            log_rewards_estimate[bi : bi + len(bs)] = mo + cat.logsoftmax()[0]

            for end_indices, being_indices in pre_indices:
                if being_indices.shape[0] > 0:
                    s_idces, sp_idces, a_idces = being_indices
                    src = prob_of_being_t[s_idces] + logprobs[a_idces]
                    inter = scatter_logsumexp(src, sp_idces, dim_size=prob_of_being_t.shape[-1])
                    prob_of_being_t = torch.logaddexp(inter, prob_of_being_t)
                    # prob_of_being_t = scatter_add(
                    #    (prob_of_being_t[s_idces] + logprobs[a_idces]).exp(), sp_idces, out=prob_of_being_t.exp()
                    # ).log()
                if end_indices.shape[0] > 0:
                    s_idces, sp_idces, a_idces = end_indices
                    src = prob_of_being_t[s_idces] + logprobs[a_idces]
                    inter = scatter_logsumexp(src, sp_idces, dim_size=prob_of_ending_t.shape[-1])
                    prob_of_ending_t = torch.logaddexp(inter, prob_of_ending_t)
                    # prob_of_ending_t = scatter_add(
                    #    (prob_of_being_t[s_idces] + logprobs[a_idces]).exp(), sp_idces, out=prob_of_ending_t.exp()
                    # ).log()
        return prob_of_ending_t, state_log_flows, log_rewards_estimate

    def recompute_flow(self, tqdm_disable=None):
        g = self.mdp_graph
        if self.logits_shuffle:
            rng = np.random.default_rng(seed=142857)
            for i in g:
                g.nodes[i]["F"] = -100
            for i in tqdm(list(range(len(g)))[::-1], disable=tqdm_disable):
                p = sorted(list(g.predecessors(i)), reverse=True)
                num_back = len([n for n in p if n != i])
                for j in p:
                    if j == i:
                        g.nodes[j]["F"] = rng.uniform(-10, 0)
                        g.edges[(i, i, 0)]["F"] = rng.uniform(-10, 0)
                    else:
                        #backflow = np.log(np.exp(g.nodes[i]["F"]) / num_back)
                        g.nodes[j]["F"] = rng.uniform(-10, 0)
                        # Here we're making a decision to split flow backwards equally for idempotent actions
                        # from the same state. I think it's ok?
                        ed = g.get_edge_data(j, i)
                        for k, vs in ed.items():
                            g.edges[(j, i, k)]["F"] = rng.uniform(-10, 0)
        else:
            for i in g:
                g.nodes[i]["F"] = -100
            for i in tqdm(list(range(len(g)))[::-1], disable=tqdm_disable):
                p = sorted(list(g.predecessors(i)), reverse=True)
                num_back = len([n for n in p if n != i])
                for j in p:
                    if j == i:
                        g.nodes[j]["F"] = np.logaddexp(g.nodes[j]["F"], self.log_rewards[j])
                        g.edges[(i, i, 0)]["F"] = self.log_rewards[j].item()
                    else:
                        backflow = np.log(np.exp(g.nodes[i]["F"]) / num_back)
                        g.nodes[j]["F"] = np.logaddexp(g.nodes[j]["F"], backflow)
                        # Here we're making a decision to split flow backwards equally for idempotent actions
                        # from the same state. I think it's ok?
                        ed = g.get_edge_data(j, i)
                        for k, vs in ed.items():
                            g.edges[(j, i, k)]["F"] = np.log(np.exp(backflow) / len(ed))

    def get_bck_trajectory_test_split(self, r, seed=142857):
        test_set = set()
        n = int((1 - r) * len(self.states))
        np.random.seed(seed)
        while len(test_set) < n:
            i0 = np.random.randint(len(self.states))
            s0 = self.states[i0]
            if len(s0.nodes) < 7:  # TODO: unhardcode this?
                continue
            s = s0
            idx = i0
            while len(s.nodes) > 5:  # TODO: unhardcode this?
                test_set.add(idx)
                actions = [
                    (u, a.item(), b.item())
                    for u, ra in enumerate(self.ctx.bck_action_type_order)
                    for a, b in getattr(self._Data[idx], ra.mask_name).nonzero()
                ]
                action = actions[np.random.randint(len(actions))]
                gaction = self.ctx.aidx_to_GraphAction(self._Data[idx], action, fwd=False)
                s = self.env.step(s, gaction)
                idx = self.get_graph_idx(s)  # This finds the graph index taking into account isomorphism
                s = self.states[idx]  # We still have to get the original graph so that the Data instance is correct
        train_set = list(set(range(len(self.states))).difference(test_set))
        test_set = list(test_set)
        np.random.shuffle(train_set)
        return train_set, test_set

    def get_subtree_test_split(self, r, seed=142857):
        cache_path = f"{self.cache_root}/subtree_split_{r}_{seed}.pkl"
        if self.cache_root is not None:
            if os.path.exists(cache_path):
                return pickle.load(open(cache_path, "rb"))
        test_set = set()
        n = int((1 - r) * len(self.states))
        np.random.seed(seed)
        start_states_idx, available_start_states, start_states = [], [], []
        edge_limit = 11
        while len(test_set) < n:
            num_ss = len([i for i in start_states_idx if i not in test_set])
            if num_ss == 0 or len(available_start_states) == 0:
                start_states, start_states_idx = zip(
                    *[(s0, i) for i, s0 in enumerate(self.states) if len(s0.nodes) == 6 and len(s0.edges) >= edge_limit]
                )
                available_start_states = list(range(len(start_states)))
                edge_limit -= 1
            assi = np.random.randint(len(available_start_states))
            ssi = available_start_states.pop(assi)
            s0 = start_states[ssi]
            i0 = self.get_graph_idx(s0)
            if i0 in test_set:
                continue
            stack = [(s0, i0)]
            while len(stack):
                s, i = stack.pop()
                if i in test_set:
                    continue
                test_set.add(i)
                actions = [
                    (u, a.item(), b.item())
                    for u, ra in enumerate(self.ctx.action_type_order)
                    if ra != GraphActionType.Stop
                    for a, b in getattr(self._Data[i], ra.mask_name).nonzero()
                ]
                for action in actions:
                    gaction = self.ctx.aidx_to_GraphAction(self._Data[i], action, fwd=True)
                    sp = self.env.step(s, gaction)
                    ip = self.get_graph_idx(sp)  # This finds the graph index taking into account isomorphism
                    if ip in test_set:
                        continue
                    sp = self.states[ip]  # We still have to get the original graph so that the Data instance is correct
                    stack.append((sp, ip))
        train_set = list(set(range(len(self.states))).difference(test_set))
        test_set = list(test_set)
        np.random.shuffle(train_set)
        if self.cache_root is not None:
            pickle.dump((np.array(train_set), np.array(test_set)), open(cache_path, "wb"))
        return train_set, test_set


class Regression(GFNAlgorithm):
    regress_to_Fsa: bool = False
    loss_type: str = "MSE"
    model_is_autoregressive = False

    def compute_batch_losses(self, model, batch, **kw):
        if self.regress_to_Fsa:
            fwd_cat, *other = model(batch, torch.zeros((batch.num_graphs, 1), device=batch.x.device))
            mask = torch.cat([i.flatten() for i in fwd_cat.masks])
            pred = torch.cat([i.flatten() for i in fwd_cat.logits]) * mask
            batch.y = batch.y * mask
        else:
            pred = model(batch, torch.zeros((batch.num_graphs, 1), device=batch.x.device))
        if self.loss_type == "MSE":
            loss = (pred - batch.y).pow(2).mean()
        elif self.loss_type == "MAE":
            loss = abs(pred - batch.y).mean()
        else:
            raise NotImplementedError
        return loss, {"loss": loss}


class BGSupervisedTrainer(BasicGraphTaskTrainer):
    def setup(self):
        super().setup()
        self.algo = Regression()
        self.algo.loss_type = self.cfg.task.basic_graph.supervised_loss
        self.algo.regress_to_Fsa = self.cfg.task.basic_graph.regress_to_Fsa
        self.log_sampling_g_distribution = self.cfg.algo.offline_sampling_g_distribution
        self.training_data.output_graphs = True
        self.test_data.output_graphs = True
        if self.cfg.task.basic_graph.regress_to_P_F:
            # P_F is just the normalized Fsa, so this flag must be on
            assert self.cfg.task.basic_graph.regress_to_Fsa

        for i in [self.training_data, self.test_data]:
            i.compute_Fsa = self.cfg.task.basic_graph.regress_to_Fsa
            i.regress_to_F = self.cfg.task.toy_seq.regress_to_F
            i.compute_normalized_Fsa = self.cfg.task.basic_graph.regress_to_P_F
            #i.compute_Fsa = self.cfg.task.basic_graph.regress_to_Fsa
            #i.regress_to_F = self.cfg.task.basic_graph.regress_to_Fsa
            #i.compute_normalized_Fsa = self.cfg.task.basic_graph.regress_to_Fsa
            i.epc = self.exact_prob_cb

    def build_training_data_loader(self) -> DataLoader:
        return torch.utils.data.DataLoader(
            self.training_data,
            batch_size=self.mb_size,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.num_workers > 0,
            shuffle=True,
            collate_fn=self.training_data.collate_fn,
        )

    def build_validation_data_loader(self) -> DataLoader:
        return torch.utils.data.DataLoader(
            self.test_data,
            batch_size=self.mb_size,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.num_workers > 0,
            collate_fn=self.test_data.collate_fn,
        )


def main():
    # Launch a test job

    hps = {
        "num_training_steps": 20000,
        "validate_every": 100,
        "num_workers": 0,
        "log_dir": "./logs/basic_graphs/run_6n_pb2",
        "log_tags": None,
        "model": {"num_layers": 2, "num_emb": 256},
        "opt": {"adam_eps": 1e-8, "learning_rate": 3e-4},
        "algo": {
            "global_batch_size": 64,
            "tb": {"variant": "SubTB1", "do_parameterize_p_b": False},
            "max_nodes": 7,
            "offline_ratio": 0 / 4,
        },
        "task": {"basic_graph": {"do_supervised": False, "do_tabular_model": False, "train_ratio": 1}},  #
    }
    if hps["task"]["basic_graph"]["do_supervised"]:
        trial = BGSupervisedTrainer(hps)
    else:
        trial = BasicGraphTaskTrainer(hps)
        torch.set_num_threads(1)
    trial.verbose = True
    trial.print_every = 1
    trial.run()


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        # Example call:
        # python basic_graph_task.py --recompute-all ./data/basic_graphs 7
        if sys.argv[1] == "--recompute-all":
            max_nodes = 7 if len(sys.argv) == 3 else int(sys.argv[3])
            states = load_two_col_data(sys.argv[2], max_nodes, generate_if_missing=True)
            env = GraphBuildingEnv()
            ctx = BasicGraphContext(max_nodes, num_cond_dim=1, graph_data=states, output_gid=True)
            epc = ExactProbCompCallback(
                None, states, torch.device("cpu"), ctx=ctx, env=env, do_save_px=False, log_rewards=1
            )
            epc.compute_cache()
            epc.save_cache(sys.argv[2] + f"/two_col_epc_cache_{max_nodes}.pkl")
        else:
            raise ValueError(sys.argv)
    else:
        main()
