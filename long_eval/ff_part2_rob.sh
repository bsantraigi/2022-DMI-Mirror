trap "exit" INT

source auto_eval/preamble.sh

# RUN TESTS -- all non roberta-init

# DD++ no et
python run_finetune.py -task dd++ \
  -rob -ff \
  -voc roberta -lr 5e-4 -scdl \
  -bs 28 -ep 5 \
  -ckpt "checkpoints/${MODEL_NAME_PATH}" \
  -t 1 -ntq

python run_finetune.py -task dd++/adv \
  -rob -ff \
  -voc roberta -lr 5e-4 -scdl \
  -bs 28 -ep 5 \
  -ckpt "checkpoints/${MODEL_NAME_PATH}" \
  -t 1 -ntq

python run_finetune.py -task dd++/cross \
  -rob -ff \
  -voc roberta -lr 1e-4 -scdl \
  -bs 28 -ep 5 \
  -ckpt "checkpoints/${MODEL_NAME_PATH}" \
  -t 1 -ntq

python run_finetune.py -task dd++/full \
  -rob -ff \
  -voc roberta -lr 5e-4 -scdl \
  -bs 28 -ep 5 \
  -ckpt "checkpoints/${MODEL_NAME_PATH}" \
  -t 1 -ntq

