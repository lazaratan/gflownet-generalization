import sys
import itertools

root = "/mnt/ps/home/CORP/lazar.atanackovic/project/gflownet-runs/logs/gfn_train_dist_logZ_l1reg_Nov_2"
counter = itertools.count()

base_hps = {
    "num_training_steps": 100000,
    "validate_every": 1000, # use 1000 might be faster
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
    "opt": {"learning_rate": 1e-4}, # o.g. 1e-4
    "device": 'cuda',
}


base_algo_hps = {
    "global_batch_size": 256,
    "max_nodes": 7,
    "offline_ratio": 1,
}

hps = [
    {
        **base_hps,
        "log_dir": f"{root}/run_{next(counter)}/",
        "log_tags": ["gfn_l1_reg_logZ_train_dist_v3"],
        
        "task": {
        "basic_graph": {
            "test_split_seed": seed, 
            "do_supervised": False, 
            "do_tabular_model": False, 
            "regress_to_P_F": False,
            "regress_to_Fsa": False,
            "train_ratio": 0.9, # set this to 1 maybe??
            "reward_func": 'count', 
            },
        },  
        
        "algo": {
            **base_algo_hps,
            "offline_sampling_g_distribution": 'uniform',
            "l1_reg_log_Z_lambda": lam,
            **algo,
        },
        
    }
    for lam in [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 0.0]
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
trial.print_every = 1
trial.run()