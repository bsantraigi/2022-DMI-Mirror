# SETUP!
LINK="https://iitkgpacin-my.sharepoint.com/:u:/g/personal/bishal_santra_iitkgp_ac_in/ES_dG9Xaje5LoKO3zWJGAjABR4NoD6H9MKwPAM5EHLHEqQ?e=bQFxpm"
wget "$LINK&download=1" -O model_best_auc.pth
mkdir -p checkpoints/TwinSMI-2Sep/
mv model_best_auc.pth checkpoints/TwinSMI-2Sep/

LINK="https://iitkgpacin-my.sharepoint.com/:u:/g/personal/bishal_santra_iitkgp_ac_in/EYfYiQpVPmlBphcXv-1sSrQB-ITdlo0KbzQzBOMKrlZO8Q?e=aLJPnj"
wget "$LINK&download=1" -O model_best_auc.pth
mkdir -p checkpoints/SMILE-2Sep/
mv model_best_auc.pth checkpoints/SMILE-2Sep/

LINK="https://iitkgpacin-my.sharepoint.com/:u:/g/personal/bishal_santra_iitkgp_ac_in/EdoVTdNvOXdDq1sP9wbLaXsBBFHLkXp-qfymcQq1mgfIuw?e=oB8lzX"
wget "$LINK&download=1" -O model_best_auc.pth
mkdir -p checkpoints/InfoNCE-S-25Aug/
mv model_best_auc.pth checkpoints/InfoNCE-S-25Aug/


# 1. 
CKPT=checkpoints/TwinSMI-2Sep/model_best_auc.pth

(python -u run_finetune.py -task banking77 -voc bert -lr 8e-4 -scdl -bs 64 -ep 35 -ckpt $CKPT -t 1 --no_tqdm;
python -u run_finetune.py -task mutual -voc bert -lr 8e-6 -wtl -scdl -bs 64 -ep 70 -ckpt $CKPT -t 1 --no_tqdm) &

(python -u run_finetune.py -task mutual_plus -voc bert -lr 5e-6 -wtl -bs 64 -ep 70 -ckpt $CKPT -t 1 --no_tqdm;
python -u run_finetune.py -task e/intent -voc bert -lr 8e-4 -scdl -bs 256 -ep 32 -ckpt $CKPT -t 1 --no_tqdm)

python -u run_finetune.py -task swda -voc bert -lr 1e-3 -scdl -ep 50 -bs 64 -ckpt $CKPT --no_tqdm -t 1

# 2. 
CKPT=checkpoints/TwinSMI-2Sep/model_best_auc.pth

(python -u run_finetune.py -task dd++/adv -voc bert -lr 5e-5 -scdl -bs 64 -ep 36 -ckpt $CKPT -t 1 --no_tqdm;
python -u run_finetune.py -task dd++ -voc bert -lr 5e-5 -scdl -bs 64 -ep 36 -ckpt $CKPT -t 1 --no_tqdm) &

(python -u run_finetune.py -task dd++/full -voc bert -lr 1.2e-5 -scdl -bs 64 -ep 40 -ckpt $CKPT -t 1 --no_tqdm;
python -u run_finetune.py -task dd++/cross -voc bert -lr 1.2e-5 -scdl -bs 64 -ep 36 -ckpt $CKPT -t 1 --no_tqdm)

# 3. 
CKPT=checkpoints/SMILE-2Sep/model_best_auc.pth

(python -u run_finetune.py -task banking77 -voc bert -lr 8e-4 -scdl -bs 64 -ep 35 -ckpt $CKPT -t 1 --no_tqdm;
python -u run_finetune.py -task mutual -voc bert -lr 8e-6 -wtl -scdl -bs 64 -ep 70 -ckpt $CKPT -t 1 --no_tqdm) &

(python -u run_finetune.py -task mutual_plus -voc bert -lr 5e-6 -wtl -bs 64 -ep 70 -ckpt $CKPT -t 1 --no_tqdm;
python -u run_finetune.py -task e/intent -voc bert -lr 8e-4 -scdl -bs 256 -ep 32 -ckpt $CKPT -t 1 --no_tqdm)

python -u run_finetune.py -task swda -voc bert -lr 1e-3 -scdl -ep 50 -bs 64 -ckpt $CKPT --no_tqdm -t 1

# 4. 
CKPT=checkpoints/SMILE-2Sep/model_best_auc.pth

(python -u run_finetune.py -task dd++/adv -voc bert -lr 5e-5 -scdl -bs 64 -ep 36 -ckpt $CKPT -t 1 --no_tqdm;
python -u run_finetune.py -task dd++ -voc bert -lr 5e-5 -scdl -bs 64 -ep 36 -ckpt $CKPT -t 1 --no_tqdm) &

(python -u run_finetune.py -task dd++/full -voc bert -lr 1.2e-5 -scdl -bs 64 -ep 40 -ckpt $CKPT -t 1 --no_tqdm;
python -u run_finetune.py -task dd++/cross -voc bert -lr 1.2e-5 -scdl -bs 64 -ep 36 -ckpt $CKPT -t 1 --no_tqdm)

# 5.
CKPT=checkpoints/InfoNCE-S-25Aug/model_best_auc.pth

(python -u run_finetune.py -task banking77 -voc bert -lr 8e-4 -scdl -bs 64 -ep 35 -ckpt $CKPT -t 1 --no_tqdm;
python -u run_finetune.py -task mutual -voc bert -lr 8e-6 -wtl -scdl -bs 64 -ep 70 -ckpt $CKPT -t 1 --no_tqdm) &

(python -u run_finetune.py -task mutual_plus -voc bert -lr 5e-6 -wtl -bs 64 -ep 70 -ckpt $CKPT -t 1 --no_tqdm;
python -u run_finetune.py -task e/intent -voc bert -lr 8e-4 -scdl -bs 256 -ep 32 -ckpt $CKPT -t 1 --no_tqdm)

python -u run_finetune.py -task swda -voc bert -lr 1e-3 -scdl -ep 50 -bs 64 -ckpt $CKPT --no_tqdm -t 1

# 6.
CKPT=checkpoints/InfoNCE-S-25Aug/model_best_auc.pth

(python -u run_finetune.py -task dd++/adv -voc bert -lr 5e-5 -scdl -bs 64 -ep 36 -ckpt $CKPT -t 1 --no_tqdm;
python -u run_finetune.py -task dd++ -voc bert -lr 5e-5 -scdl -bs 64 -ep 36 -ckpt $CKPT -t 1 --no_tqdm) &

(python -u run_finetune.py -task dd++/full -voc bert -lr 1.2e-5 -scdl -bs 64 -ep 40 -ckpt $CKPT -t 1 --no_tqdm;
python -u run_finetune.py -task dd++/cross -voc bert -lr 1.2e-5 -scdl -bs 64 -ep 36 -ckpt $CKPT -t 1 --no_tqdm)


