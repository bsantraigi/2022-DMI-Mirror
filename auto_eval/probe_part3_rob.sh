trap "exit" INT

source auto_eval/preamble.sh

# DD++ with et
python run_finetune.py -task dd++ -rob \
  -et \
  -voc roberta -lr 1e-3 -scdl \
  -bs 32 -ep 2 \
  -ckpt "checkpoints/${MODEL_NAME_PATH}" \
  -t 1 -ntq

python run_finetune.py -task dd++/adv -rob \
  -et \
  -voc roberta -lr 1e-3 -scdl \
  -bs 32 -ep 2 \
  -ckpt "checkpoints/${MODEL_NAME_PATH}" \
  -t 1 -ntq

python run_finetune.py -task dd++/cross -rob \
  -et \
  -voc roberta -lr 1e-4 -scdl \
  -bs 32 -ep 5 \
  -ckpt "checkpoints/${MODEL_NAME_PATH}" \
  -t 1 -ntq

python run_finetune.py -task dd++/full -rob \
  -et \
  -voc roberta -lr 1e-3 -scdl \
  -bs 32 -ep 2 \
  -ckpt "checkpoints/${MODEL_NAME_PATH}" \
  -t 1 -ntq