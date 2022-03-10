# 2. DMI-Large-RoB - 0AD37F85
LINK="https://iitkgpacin-my.sharepoint.com/:u:/g/personal/bishal_santra_iitkgp_ac_in/EYiGx8QxqNBOhIuZhPXCGX0BCKYcT3tWXJ7F5cR63dbVYg?e=ehhE7X"
wget "$LINK&download=1" -O model_best_auc.pth
mkdir -p checkpoints/DMI-L-RoB-20Oct/
mv model_best_auc.pth checkpoints/DMI-L-RoB-20Oct/

# 1
(python -u run_finetune.py -task banking77 -voc roberta -rob -robname roberta-large -lr 8e-4 -scdl -bs 30 -ep 35 -ckpt checkpoints/DMI-L-RoB-20Oct/model_best_auc.pth -t 1 --no_tqdm;
python -u run_finetune.py -task mutual -voc roberta -rob -robname roberta-large -lr 8e-6 -wtl -scdl -bs 30 -ep 70 -ckpt checkpoints/DMI-L-RoB-20Oct/model_best_auc.pth -t 1 --no_tqdm) &

(python -u run_finetune.py -task mutual_plus -voc roberta -rob -robname roberta-large -lr 5e-6 -wtl -bs 30 -ep 70 -ckpt checkpoints/DMI-L-RoB-20Oct/model_best_auc.pth -t 1 --no_tqdm;
python -u run_finetune.py -task e/intent -voc roberta -rob -robname roberta-large -lr 8e-4 -scdl -bs 100 -ep 32 -ckpt checkpoints/DMI-L-RoB-20Oct/model_best_auc.pth -t 1 --no_tqdm)


# 2
python -u run_finetune.py -task swda -voc roberta -rob -robname roberta-large -lr 1e-3 -scdl -ep 50 -bs 30 -ckpt checkpoints/DMI-L-RoB-20Oct/model_best_auc.pth --no_tqdm -t 1

# 3
python -u run_finetune.py -task dd++/full -voc roberta -rob -robname roberta-large -lr 1.2e-5 -scdl -bs 30 -ep 40 -ckpt checkpoints/DMI-L-RoB-20Oct/model_best_auc.pth -t 1 --no_tqdm

# 4
python -u run_finetune.py -task dd++/cross -voc roberta -rob -robname roberta-large -lr 1.2e-5 -scdl -bs 30 -ep 36 -ckpt checkpoints/DMI-L-RoB-20Oct/model_best_auc.pth -t 1 --no_tqdm

# 5
python -u run_finetune.py -task dd++/adv -voc roberta -rob -robname roberta-large -lr 5e-5 -scdl -bs 30 -ep 36 -ckpt checkpoints/DMI-L-RoB-20Oct/model_best_auc.pth -t 1 --no_tqdm

# 6
python -u run_finetune.py -task dd++ -voc roberta -rob -robname roberta-large -lr 5e-5 -scdl -bs 30 -ep 36 -ckpt checkpoints/DMI-L-RoB-20Oct/model_best_auc.pth -t 1 --no_tqdm