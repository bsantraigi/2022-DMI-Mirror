"""
Returns the bash commands that user need to run to test for
all checkpoints available in wandb.

TODO: Filtering of wandb runs based on metrics available/post-hoc analysis
"""
import pandas as pd
import wandb
from collections import defaultdict
import numpy as np
from tqdm.auto import tqdm
import functools
api = wandb.Api()
entity, project = "c2ai", "infonce-dialog"  # set to your entity and project
# runs = api.runs(entity + "/" + project)

# List of selected runs
runs = {
    "SMILE": "c2ai/estimators-dev/19fvh822",
    "MINE": "c2ai/estimators-dev/wk1fl77f",
    "JSD": "c2ai/estimators-dev/1ub3ncd5",
    # "InfoNCE-S": "c2ai/estimators-dev/12jx0875",
    "InfoNCE-S": "c2ai/infonce-dialog/225eou9o",
    "InfoNCE": "c2ai/infonce-dialog/1n32ixh7"
}

# runs = {
#     "DMI-rMax-small": "36ktqr4w",
#     "TwinDMI-rMax-small": "2cf0hpwh",
#     "RoB-DMI-rMax-12L": "1tcufpav",
#     "DMI-DD-tiny": "1n32ixh7",
#     "DMI-r1M/cc-small (Symm)": "24ubltnf",
#     "DMI-r1M/cc-small": "29j7e5n5"}
# runs = [api.run(f"{entity}/{project}/{r}") for k, r in runs.items()]
# print(list(runs))

summary_list, config_list, name_list = [], [], []
accepted_runs = []
all_history = {}
for i, k in enumerate(runs):
    r = runs[k]
    # run = api.run(f"{entity}/{project}/{r}")
    run = api.run(f"{r}")

    print(f"\n======= [{i + 1}/{len(runs)}] Pulling {run.name} ========")
    df_run_valid = run.history(keys=["auc", "valid_loss", "mutual_info"])
    df_run_train = run.history(keys=["train_loss"])
    df_run_train['train_mi'] = np.log(run.config['batch_size']) - df_run_train['train_loss']
    for file in filter(lambda x: x.name.endswith(".pth"), run.files()):
        print(file.name)
        accepted_runs.append(
            f"bash finetune_pipeline.sh {run.id} ./data {file.name} 2>&1 | tee logs/{run.id}_{file.name}.log")

    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files
    summary_dict = {
        'best_valid_loss': df_run_valid['valid_loss'].min(),
        'best_valid_mi': df_run_valid['mutual_info'].max(),
        'best_auc': df_run_valid['auc'].max(),
    }
    df_run = pd.merge(df_run_train, df_run_valid, how="outer", on="_step")
    df_run.sort_values(by="_step", inplace=True)
    all_history[k] = df_run
    df_run.to_csv(f"checkpoints/pretraining_logs/{run.id}.csv", index=False)

    print(summary_dict)
    # summary_list.append(run.summary._json_dict)
    summary_list.append(summary_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append(
        {k: v for k, v in run.config.items()
         if not k.startswith('_')})

    # .name is the human-readable name of the run.
    name_list.append(run.name)

# print(f"{run.config.keys()}")
# summary_keys = '\n'.join(sorted(filter(lambda x: 'gradient' not in x, list(run.summary._json_dict.keys()))))
# print(f"{summary_keys}")

bucket = defaultdict(lambda:[None]*len(name_list))
bucket['names'] = name_list
for i, summary in enumerate(summary_list):
    for k, v in summary.items():
        bucket[k][i] = v

for i, config in enumerate(config_list):
    for k, v in config.items():
        bucket[k][i] = v

runs_df = pd.DataFrame(bucket)

runs_df.to_csv("checkpoints/pretraining_logs/1.csv", index=False)

for metric in ["auc", "valid_loss", "mutual_info", "train_loss", "train_mi"]:
    print(metric)
    df_by_metric = []
    df_final = None
    for name, df_log in all_history.items():
        df_log = df_log[["_step", metric]]
        df_log = df_log.rename(columns={metric: name})
        df_by_metric.append(df_log)
        if df_final is None:
            df_final = df_log
        else:
            df_final = pd.merge(df_final, df_log, how="outer", on="_step")
    # df_final = functools.reduce(lambda left, right: pd.merge(left, right, how="outer", on="_step"), df_by_metric)
    df_final.sort_values(by="_step", inplace=True)
    df_final.to_csv(f"checkpoints/pretraining_logs/{metric}.csv", index=False)