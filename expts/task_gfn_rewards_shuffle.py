import sys
import itertools

root = "/logs/gfn_TB_rewards_shuffle"
counter = itertools.count()

base_hps = {
    "num_training_steps": 100000,
    "validate_every": 1000,
    "num_workers": 8,
    "pickle_mp_messages": True, # when using 1 or mor worker always have this True (otherwise slow)
    "model": {
        "num_layers": 8, 
        "num_emb": 128,
        "graph_transformer": {
            "num_heads": 4,
            "num_mlp_layers": 2, 
            },
        },
    "opt": {"learning_rate": 1e-4},
    "device": 'cuda',
}


base_algo_hps = {
    "global_batch_size": 1024, #256
    "max_nodes": 7,
    "offline_ratio": 0 / 4,
}

hps = [
    {
        **base_hps,
        "log_dir": f"{root}/run_{next(counter)}/",
        "log_tags": ["gfn_rewards_shuffle"],
        
        "task": {
        "basic_graph": {
            "test_split_seed": seed, 
            "do_supervised": False, 
            "do_tabular_model": False, 
            "regress_to_P_F": False,
            "regress_to_Fsa": False,
            "train_ratio": 0.9,
            "reward_func": reward, 
            "reward_shuffle": shuffle, #"reward_shuffle": shuffle,
            },
        },  
        
        "algo": {
            **base_algo_hps,
            **algo,
        },
        
    }
    for reward in ['count', 'even_neighbors', 'cliques']
    for shuffle in [False, True]
    for seed in [1]
    for algo in [
        {
            "method": "TB", # either TB or FM
            "tb": {"variant": "SubTB1", "do_parameterize_p_b": False},
        },
    ]
]

from gflownet.tasks.basic_graph_task import BasicGraphTaskTrainer

trial = BasicGraphTaskTrainer(hps[int(sys.argv[1])])
#trial.print_every = 1
trial.run()