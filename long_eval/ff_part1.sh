trap "exit" INT

source auto_eval/preamble.sh

# MUTUAL + EIntent

# RUN TESTS -- non roberta-init -- except dd++
# 4 mutual

python run_finetune.py -task mutual_plus \
  -et -ff \
  -voc roberta -lr 1e-3 -wtl -scdl \
  -bs 32 -ep 25 \
  -ckpt "checkpoints/${MODEL_NAME_PATH}" \
  -t 1 --no_tqdm

python run_finetune.py -task mutual_plus \
  -ff \
  -voc roberta -lr 1e-3 -wtl -scdl \
  -bs 32 -ep 25 \
  -ckpt "checkpoints/${MODEL_NAME_PATH}" \
  -t 1 --no_tqdm

python run_finetune.py -task mutual \
  -et -ff \
  -voc roberta -lr 1e-3 -wtl -scdl \
  -bs 32 -ep 25 \
  -ckpt "checkpoints/${MODEL_NAME_PATH}" \
  -t 1 --no_tqdm

python run_finetune.py -task mutual \
  -ff \
  -voc roberta -lr 1e-3 -wtl -scdl \
  -bs 32 -ep 25 \
  -ckpt "checkpoints/${MODEL_NAME_PATH}" \
  -t 1 --no_tqdm


# EIntent
python run_finetune.py -task e/intent \
  -ff \
  -voc roberta -lr 1e-3 -scdl \
  -bs 64 -ep 25 \
  -ckpt "checkpoints/${MODEL_NAME_PATH}" \
  -t 1 --no_tqdm

