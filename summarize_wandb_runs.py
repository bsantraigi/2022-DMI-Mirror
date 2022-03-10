"""
Returns the bash commands that user need to run to test for
all checkpoints available in wandb.

TODO: Filtering of wandb runs based on metrics available/post-hoc analysis
"""
import pandas as pd
import wandb
from tqdm.auto import tqdm

api = wandb.Api()
entity, project = "c2ai", "infonce-dialog"  # set to your entity and project
# entity, project = "bsantraigi", "infonce-dialog-setup"
runs = api.runs(entity + "/" + project)

summary_list, config_list, name_list = [], [], []
accepted_runs = []
for i, run in enumerate(runs):
    print(f"\n======= [{i+1}/{len(runs)}] Pulling {run.name} ========")
    df_run_valid = run.history(keys=["auc", "valid_loss", "mutual_info"])
    df_run_train = run.history(keys=["train_loss"])
    if df_run_valid.shape[0] == 0:
        print("Empty run...")
        continue
    for file in filter(lambda x: x.name.endswith(".pth"), run.files()):
        print(file.name)
        accepted_runs.append(f"bash finetune_pipeline.sh {run.id} ./data {file.name} 2>&1 | tee logs/{run.id}_{file.name}.log")
            
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files
    param_dict = run.config
    results_dict = {
        'codename': run.name,
        'id': run.id,
        'sweep': run.sweepName,
        'best_loss_step': df_run_valid[df_run_valid.valid_loss == df_run_valid.valid_loss.min()].iloc[0]['_step'],
        'best_valid_loss': df_run_valid['valid_loss'].min(),
        'best_valid_mi': df_run_valid['mutual_info'].max(),
        'best_auc_step': df_run_valid[df_run_valid.auc == df_run_valid.auc.max()].iloc[0]['_step'],
        'best_auc': df_run_valid['auc'].max(),
        'state': run.state,
    }
    summary_dict = {**results_dict, **param_dict}
    # print(summary_dict)
    # summary_list.append(run.summary._json_dict)
    summary_list.append(summary_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append(
        {k: v for k,v in run.config.items()
         if not k.startswith('_')})

    # .name is the human-readable name of the run.
    name_list.append(run.name)

# print(f"{run.config.keys()}")
# summary_keys = '\n'.join(sorted(filter(lambda x: 'gradient' not in x, list(run.summary._json_dict.keys()))))
# print(f"{summary_keys}")

runs_df = pd.DataFrame(summary_list)

runs_df.to_csv(f"checkpoints/{project}[{entity}].csv", index=False)

print("\n============= Final filtered commands ==============\n")
for cmd in accepted_runs:
    print(cmd)