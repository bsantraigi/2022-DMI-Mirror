trap "exit" INT

source auto_eval/preamble.sh

# B77 + SWDA
# RUN TESTS -- non roberta-init -- except dd++

# B77
python run_finetune.py -task banking77 \
  -ff \
  -voc roberta -lr 1e-3 -scdl \
  -bs 64 -ep 25 \
  -ckpt "checkpoints/${MODEL_NAME_PATH}" \
  -t 1 --no_tqdm

# SWDA
python run_finetune.py -task swda \
  -ff \
  -voc roberta -lr 1e-3 -scdl \
  -bs 64 -ep 25 \
  -ckpt "checkpoints/${MODEL_NAME_PATH}" \
  --no_tqdm -t 1



