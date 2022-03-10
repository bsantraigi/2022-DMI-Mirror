# SETUP!
LINK="https://iitkgpacin-my.sharepoint.com/:u:/g/personal/bishal_santra_iitkgp_ac_in/ES6i1zR1RSlMm53RXYqeIb8BK_H0N82CpBoiJ8n3_mzCGw?e=8GfYph"
wget "$LINK&download=1" -O model_best_auc.pth
mkdir -p checkpoints/RobSMI-2Sep/
mv model_best_auc.pth checkpoints/RobSMI-2Sep/

CKPT=checkpoints/RobSMI-2Sep/model_best_auc.pth

# 1
(python -u run_finetune.py -task banking77 -voc roberta -rob -lr 8e-4 -scdl -bs 64 -ep 35 -ckpt $CKPT -t 1 --no_tqdm;
python -u run_finetune.py -task mutual -voc roberta -rob -lr 8e-6 -wtl -scdl -bs 64 -ep 70 -ckpt $CKPT -t 1 --no_tqdm) &


(python -u run_finetune.py -task mutual_plus -voc roberta -rob -lr 5e-6 -wtl -bs 64 -ep 70 -ckpt $CKPT -t 1 --no_tqdm;
python -u run_finetune.py -task e/intent -voc roberta -rob -lr 8e-4 -scdl -bs 256 -ep 32 -ckpt $CKPT -t 1 --no_tqdm)



# 2
python -u run_finetune.py -task swda -voc roberta -rob -lr 1e-3 -scdl -ep 50 -bs 64 -ckpt $CKPT --no_tqdm -t 1

# 3
python -u run_finetune.py -task dd++/full -voc roberta -rob -lr 1.2e-5 -scdl -bs 64 -ep 40 -ckpt $CKPT -t 1 --no_tqdm

# 4
python -u run_finetune.py -task dd++/cross -voc roberta -rob -lr 1.2e-5 -scdl -bs 64 -ep 36 -ckpt $CKPT -t 1 --no_tqdm

# 5
python -u run_finetune.py -task dd++/adv -voc roberta -rob -lr 5e-5 -scdl -bs 64 -ep 36 -ckpt $CKPT -t 1 --no_tqdm

# 6
python -u run_finetune.py -task dd++ -voc roberta -rob -lr 5e-5 -scdl -bs 64 -ep 36 -ckpt $CKPT -t 1 --no_tqdm
