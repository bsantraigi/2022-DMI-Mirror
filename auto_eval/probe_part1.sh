trap "exit" INT

source auto_eval/preamble.sh

# MUTUAL + EIntent

# RUN TESTS -- non roberta-init -- except dd++
# 4 mutual

python run_finetune.py -task mutual_plus \
  -et \
  -voc roberta -lr 1e-3 -wtl -scdl \
  -bs 32 -ep 10 \
  -ckpt "checkpoints/${MODEL_NAME_PATH}" \
  -t 1 --no_tqdm

python run_finetune.py -task mutual_plus \
  -voc roberta -lr 1e-3 -wtl -scdl \
  -bs 32 -ep 10 \
  -ckpt "checkpoints/${MODEL_NAME_PATH}" \
  -t 1 --no_tqdm

python run_finetune.py -task mutual \
  -et \
  -voc roberta -lr 1e-3 -wtl -scdl \
  -bs 32 -ep 10 \
  -ckpt "checkpoints/${MODEL_NAME_PATH}" \
  -t 1 --no_tqdm

python run_finetune.py -task mutual \
  -voc roberta -lr 1e-3 -wtl -scdl \
  -bs 32 -ep 10 \
  -ckpt "checkpoints/${MODEL_NAME_PATH}" \
  -t 1 --no_tqdm

# EIntent
python run_finetune.py -task e/intent \
  -voc roberta -lr 1e-3 -scdl \
  -bs 64 -ep 10 \
  -ckpt "checkpoints/${MODEL_NAME_PATH}" \
  -t 1 --no_tqdm


