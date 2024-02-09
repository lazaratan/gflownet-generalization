import sys
import itertools

root = "/logs/distilled_rewards_shuffle" # use for reward shuffle (with env structure)
#root = "/logs/distilled_logits_shuffle" # use for random logits (no env structure)
counter = itertools.count()

base_hps = {
    "num_training_steps": 100000,
    "validate_every": 1000,
    "num_workers": 4,
    "pickle_mp_messages": True, # when using 1 or mor worker always have this True (otherwise slow)
    "model": {
        "num_layers": 8, #8 
        "num_emb": 128, #128
        "graph_transformer": {
            "num_heads": 4,
            "num_mlp_layers": 2, 
            },
        },
    "opt": {"learning_rate": 1e-4},
    "device": 'cuda',
}

base_algo_hps = {
    "global_batch_size": 2048, #256
    "max_nodes": 7,
    "offline_ratio": 0 / 4,
}

hps = [
    {
        **base_hps,
        "log_dir": f"{root}/run_{next(counter)}/",
        "log_tags": ["distilled_logits_shuffle"],
        
        "task": {
        "basic_graph": {
            "test_split_seed": seed, 
            "do_supervised": True, 
            "do_tabular_model": False, 
            "regress_to_P_F": True,
            "regress_to_Fsa": True,
            "train_ratio": 0.9,
            "reward_func": reward, 
            "reward_reshape": False,
            "reward_corrupt": False,
            "reward_shuffle": shuffle, # if logit shuffle use "logits_shuffle": shuffle,
            "reward_param": 0.0,
            },
        },  
        
        "algo": {
            **base_algo_hps,
            #**algo,
        },
        
    }
    for reward in ['count', 'even_neighbors', 'cliques']
    for shuffle in [True, False]
    for seed in [1, 2, 3]
    #for algo in [
        #{
        #    "method": "TB", # either TB or FM
        #    "tb": {"variant": "SubTB1", "do_parameterize_p_b": False},
        #},
        #{
        #    "method": "FM", # either TB or FM
        #    "fm": {"correct_idempotent": False, "balanced_loss": False, "leaf_coef": 10, "epsilon": 1e-38},
        #,
    #]
]

from gflownet.tasks.basic_graph_task import BGSupervisedTrainer

trial = BGSupervisedTrainer(hps[int(sys.argv[1])])
trial.print_every = 1
trial.run()