"""
Returns the bash commands that user need to run to test for
all checkpoints available in wandb.

TODO: Filtering of wandb runs based on metrics available/post-hoc analysis
"""
import pandas as pd
import wandb
from tqdm.auto import tqdm
from pprint import pprint
import re

api = wandb.Api()
entity, project = "bsantraigi", "dmi-baselines"  # set to your entity and project
# entity, project = "bsantraigi", "infonce-dialog-setup"
runs = api.runs(entity + "/" + project)

tasks = ['b77', 'swda', 'e/intent', 'mutual', 'mutual_plus', 'dd++', 'dd++/adv', 'dd++/cross', 'dd++/full', 'eintent']

results_list = []

try:
    for i, run in enumerate(runs):
        print(f"\n======= [{i+1}/{len(runs)}] Pulling {run.name} ========")
        config = run.config
        if config['task'] not in tasks:
            print(f"[Wrong Task]->{config['task']} skipping")
            continue
        summary = run.summary
        print(run.config)
        print(run.summary)

        task = config['task']
        if task == "eintent":
            # two names used for eintent, so merging them!
            task = "e/intent"

        result = {
            'task': task,
            'full_finetune': str(config['full_finetune']),
            'encode_together': str(config.get('encode_together', "False")),
        }
        
        try:
            # result['model_name'] = re.match(r"checkpoints/([^/]+)/.*", config['checkpoint_path']).group(1)
            result['model_name'] = config['model']
        except TypeError as e:
            print(e)
            print(f"Cannot parse model name: {config['model']}")
            continue

        metrics = [key for key in summary.keys() if key in ['MRR', 'R@1', 'R@2', 'test_accuracy']]
        print(metrics)

        if len(metrics) > 1:
            for metric in metrics:
                result[f"{task}_{metric}"] = round(summary[metric]*100, 3)
        elif len(metrics) == 1:
            result[f"{task}"] = round(summary[metrics[0]], 3)
        
        pprint(result)
        results_list.append(result)

        # if len(results_list) > 100:
        #     break

    results_df = pd.DataFrame(results_list)
    print(results_df.head())

    # task to column mapping
    def task_to_col_mapping(frame):
        # The metric to use for finding best run for each (model, task, config)
        mapping = {
            'b77': 'b77',
            'swda': 'swda',
            'e/intent': 'e/intent',
            'mutual': 'mutual_R@1',
            'mutual_plus': 'mutual_plus_R@1',
            'dd++': 'dd++',
            'dd++/adv': 'dd++/adv',
            'dd++/cross':'dd++/cross',
            'dd++/full':'dd++/full'
        }
        return mapping[frame.iloc[0]['task']]

    # Group by individual (model, task, config) and find the best run
    groups = results_df.groupby(['model_name', 'task', 'full_finetune', 'encode_together'])
    results_df = groups.apply(lambda frame: frame.loc[frame[task_to_col_mapping(frame)] == frame[task_to_col_mapping(frame)].max()])
    # Fix the multilevel column header
    results_df.reset_index(inplace=True, drop=True)

    # Remove task column now and merge the rows for a (model, config) -> the final result row for each model
    results_df.drop("task", axis=1, inplace=True)
    # because of previous groupby only one row will have value for each task
    # so, doing a group.max() will remove the na's and merge the results from all task into a single row
    results_df = results_df.groupby(['model_name', 'full_finetune', 'encode_together']).max()
    results_df.reset_index(inplace=True)

    # reorder the columns as per the google-sheet
    results_df = results_df[['model_name', 'full_finetune', 'encode_together', 'b77', 'swda', \
        'e/intent', 'mutual_R@1', 'mutual_R@2', 'mutual_MRR', 'mutual_plus_R@1', 'mutual_plus_R@2', \
        'mutual_plus_MRR', 'dd++', 'dd++/adv', 'dd++/cross', 'dd++/full']]
    results_df.to_excel("logs/wandb_baselines_table.ods", index=False)

except KeyboardInterrupt as e:
    print("\n\n[Ctrl-C] Stopping process...")