trap "exit" INT

# SETUP
# 1. Finetune can be run on 1 GPU VMs
# 2. Each _SET_ along with it's variables can be passed to individual GPU-vms
# 3. Run get_data.sh to download the new required data to ./data directory and SMI-legacy checkpoints.
# 4. Run the pull_ckpt.py commands below to download the models to checkpoints/ directory.
# 5. Logs will now be sent to wandb (along with errors, if any)

# 28/7/2021
## SET 1
python pull_ckpt.py -e c2ai -p infonce-dialog -r azmsz48p # no need to run if folder and pth files exists in checkpoints/azmsz48p

RUN="azmsz48p"
FILE="model_best_auc"
CKPT="checkpoints/$RUN/$FILE.pth"
VOCAB="blender"
DATA_PATH="./data"
LOG="./logs/$RUN.$FILE"
python -u run_finetune.py -task banking77 -voc $VOCAB -lr 1e-3 -scdl -bs 32 -ep 50 --data_path "$DATA_PATH" -ckpt "$CKPT" --tracking 1 --no_tqdm
python -u run_finetune.py -task swda -voc $VOCAB -lr 1e-4 -scdl -bs 256 -ep 40 --data_path "$DATA_PATH" -ckpt "$CKPT" --tracking 1 --no_tqdm
python -u run_finetune.py -task mutual_plus -voc $VOCAB -lr 2e-3 -wtl -scdl -bs 32 -ep 100 --data_path "$DATA_PATH" -ckpt "$CKPT" --tracking 1 --no_tqdm
python -u run_finetune.py -task e/intent -voc $VOCAB -lr 2e-4 -wtl -scdl -bs 32 -ep 50 --data_path "$DATA_PATH" -ckpt "$CKPT" --tracking 1 --no_tqdm
python -u run_finetune.py -task dd++ -voc $VOCAB -lr 1e-5 -scdl -bs 32 -ep 50 --data_path "$DATA_PATH" -ckpt "$CKPT" --tracking 1 --no_tqdm
python -u run_finetune.py -task dd++/adv -voc $VOCAB -lr 1e-5 -scdl -bs 32 -ep 50 --data_path "$DATA_PATH" -ckpt "$CKPT" --tracking 1 --no_tqdm
python -u run_finetune.py -task dd++/cross -voc $VOCAB -lr 1e-5 -scdl -bs 32 -ep 50 --data_path "$DATA_PATH" -ckpt "$CKPT" --tracking 1 --no_tqdm
python -u run_finetune.py -task dd++/full -voc $VOCAB -lr 1e-5 -scdl -bs 32 -ep 50 --data_path "$DATA_PATH" -ckpt "$CKPT" --tracking 1 --no_tqdm

## SET 2
python pull_ckpt.py -e c2ai -p infonce-dialog -r 32uy2cr8

RUN="32uy2cr8"
FILE="model_best_auc_ddp"
CKPT="checkpoints/$RUN/$FILE.pth"
VOCAB="bert"
DATA_PATH="./data"
LOG="./logs/$RUN.$FILE"
python -u run_finetune.py -task banking77 -voc $VOCAB -lr 1e-3 -scdl -bs 32 -ep 50 --data_path "$DATA_PATH" -ckpt "$CKPT" --tracking 1 --no_tqdm
python -u run_finetune.py -task swda -voc $VOCAB -lr 1e-4 -scdl -bs 256 -ep 40 --data_path "$DATA_PATH" -ckpt "$CKPT" --tracking 1 --no_tqdm
python -u run_finetune.py -task mutual -voc $VOCAB -lr 2e-3 -wtl -scdl -bs 32 -ep 100 --data_path "$DATA_PATH" -ckpt "$CKPT" --tracking 1 --no_tqdm
python -u run_finetune.py -task mutual_plus -voc $VOCAB -lr 2e-3 -wtl -scdl -bs 32 -ep 100 --data_path "$DATA_PATH" -ckpt "$CKPT" --tracking 1 --no_tqdm
python -u run_finetune.py -task e/intent -voc $VOCAB -lr 2e-4 -wtl -scdl -bs 32 -ep 50 --data_path "$DATA_PATH" -ckpt "$CKPT" --tracking 1 --no_tqdm
python -u run_finetune.py -task dd++ -voc $VOCAB -lr 1e-5 -scdl -bs 32 -ep 50 --data_path "$DATA_PATH" -ckpt "$CKPT" --tracking 1 --no_tqdm
python -u run_finetune.py -task dd++/adv -voc $VOCAB -lr 1e-5 -scdl -bs 32 -ep 50 --data_path "$DATA_PATH" -ckpt "$CKPT" --tracking 1 --no_tqdm
python -u run_finetune.py -task dd++/cross -voc $VOCAB -lr 1e-5 -scdl -bs 32 -ep 50 --data_path "$DATA_PATH" -ckpt "$CKPT" --tracking 1 --no_tqdm
python -u run_finetune.py -task dd++/full -voc $VOCAB -lr 1e-5 -scdl -bs 32 -ep 50 --data_path "$DATA_PATH" -ckpt "$CKPT" --tracking 1 --no_tqdm