import sys
import itertools

root = "/logs/gfn_offline_full_train_dist_sample"
counter = itertools.count()

base_hps = {
    "num_training_steps": 100000,
    "validate_every": 200, # use 1000 might be faster
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
    #"opt": {"learning_rate": 1e-4}, # o.g. 1e-4
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
        "log_tags": ["gfn_offline_train_dist_sample"],
        "seed": seed,
        "opt": {"learning_rate": 1e-4}, # o.g. 1e-4,
        
        "task": {
        "basic_graph": {
            "test_split_seed": seed, #142857, 
            "do_supervised": False, 
            "do_tabular_model": False, 
            "regress_to_P_F": False,
            "regress_to_Fsa": False,
            "train_ratio": 1.0,
            "reward_func": reward, 
            },
        },  
        
        "algo": {
            **base_algo_hps,
            "offline_sampling_g_distribution": data_distribution,
            "use_true_log_Z": False,
            **algo,
        },
        
    }
    for reward in ['count', 'even_neighbors', 'cliques']
    for data_distribution in ["uniform", "log_rewards", "log_p", "l2_log_error_gfn", "l1_error_gfn"] 
    for seed in [1, 2, 3]
    #for lr in [5e-4, 1e-4, 5e-5, 1e-5]
    for algo in [
        {
            "method": "TB", # either TB or FM
            "tb": {
                "variant": "SubTB1", #"TB", 
                "do_parameterize_p_b": False,
                "cum_subtb": False,
                },
        },
    ]
]

from gflownet.tasks.basic_graph_task import BasicGraphTaskTrainer

trial = BasicGraphTaskTrainer(hps[int(sys.argv[1])])
trial.print_every = 1
trial.run()