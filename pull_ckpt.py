import wandb
import argparse

def cmdline_args():
    # Make parser object
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--entity", type=str, required=True, help="entity (team or user id)")
    p.add_argument("-p", "--project", type=str, required=True, help="project name")
    p.add_argument("-r", "--run_id", type=str, required=True, help="run_id")

    return (p.parse_args())

if __name__ == '__main__':
    args = cmdline_args()
    print(args)

    api = wandb.Api()

    entity = args.entity
    project = args.project
    run_id = args.run_id
    run = api.run(f"{entity}/{project}/{run_id}")

    print(f"RUN: {run}")
    for file in run.files():
        print(f"Downloading {file.name}")
        file.download(root=f"checkpoints/{run_id}", replace=True)
