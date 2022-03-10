trap "exit" INT

source auto_eval/preamble.sh

# RUN TESTS -- all non bert-init

# DD++ no et
python run_finetune.py -task dd++ \
  -ff \
  -voc bert -lr 5e-4 -scdl \
  -bs 28 -ep 5 \
  -ckpt "checkpoints/${MODEL_NAME_PATH}" \
  -t 1 -ntq

python run_finetune.py -task dd++/adv \
  -ff \
  -voc bert -lr 5e-4 -scdl \
  -bs 28 -ep 5 \
  -ckpt "checkpoints/${MODEL_NAME_PATH}" \
  -t 1 -ntq

python run_finetune.py -task dd++/cross \
  -ff \
  -voc bert -lr 1e-4 -scdl \
  -bs 28 -ep 5 \
  -ckpt "checkpoints/${MODEL_NAME_PATH}" \
  -t 1 -ntq

python run_finetune.py -task dd++/full \
  -ff \
  -voc bert -lr 5e-4 -scdl \
  -bs 28 -ep 5 \
  -ckpt "checkpoints/${MODEL_NAME_PATH}" \
  -t 1 -ntq

