trap "exit" INT

source auto_eval/preamble.sh

# DD++ with et
python run_finetune.py -task dd++ \
  -ff -et \
  -voc roberta -lr 5e-4 -scdl \
  -bs 32 -ep 5 \
  -ckpt "checkpoints/${MODEL_NAME_PATH}" \
  -t 1 -ntq

python run_finetune.py -task dd++/adv \
  -ff -et \
  -voc roberta -lr 5e-4 -scdl \
  -bs 32 -ep 5 \
  -ckpt "checkpoints/${MODEL_NAME_PATH}" \
  -t 1 -ntq

python run_finetune.py -task dd++/cross \
  -ff -et \
  -voc roberta -lr 1e-4 -scdl \
  -bs 32 -ep 5 \
  -ckpt "checkpoints/${MODEL_NAME_PATH}" \
  -t 1 -ntq

python run_finetune.py -task dd++/full \
  -ff -et \
  -voc roberta -lr 5e-4 -scdl \
  -bs 32 -ep 5 \
  -ckpt "checkpoints/${MODEL_NAME_PATH}" \
  -t 1 -ntq