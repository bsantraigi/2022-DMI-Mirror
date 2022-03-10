# Setup
bash get_data.sh

LINK="https://iitkgpacin-my.sharepoint.com/:u:/g/personal/bishal_santra_iitkgp_ac_in/ESB9hjM-5JFJm8bJlKPIpBYBtcKKeChEE3W_1YOXggRHvA?e=6Z5JDV"
wget "$LINK&download=1" -O model_best_auc.pth
mkdir -p checkpoints/RobSMI-6Sep/
mv model_best_auc.pth checkpoints/RobSMI-6Sep/

LINK="https://iitkgpacin-my.sharepoint.com/:u:/g/personal/bishal_santra_iitkgp_ac_in/EUQsF4novnhApTlOvfTeQS0BSdyP91aJFhH4t4T3Wzy8sw?e=pLj1hP"
wget "$LINK&download=1" -O model_best_auc.pth
mkdir -p checkpoints/TwinSMI-6Sep/
mv model_best_auc.pth checkpoints/TwinSMI-6Sep/


# 1
(python -u run_finetune.py -task banking77 -voc bert -lr 8e-4 -scdl -ff -bs 64 -ep 35 -ckpt checkpoints/TwinSMI-6Sep/model_best_auc.pth -t 1 --no_tqdm;
python -u run_finetune.py -task mutual -voc bert -lr 8e-6 -wtl -scdl -ff -bs 64 -ep 70 -ckpt checkpoints/TwinSMI-6Sep/model_best_auc.pth -t 1 --no_tqdm) &


(python -u run_finetune.py -task mutual_plus -voc bert -lr 5e-6 -wtl -ff -bs 64 -ep 70 -ckpt checkpoints/TwinSMI-6Sep/model_best_auc.pth -t 1 --no_tqdm;
python -u run_finetune.py -task e/intent -voc bert -lr 8e-4 -scdl -ff -bs 256 -ep 32 -ckpt checkpoints/TwinSMI-6Sep/model_best_auc.pth -t 1 --no_tqdm)



# 2
python -u run_finetune.py -task swda -voc bert -lr 1e-3 -scdl -ep 50 -ff -bs 64 -ckpt checkpoints/TwinSMI-6Sep/model_best_auc.pth --no_tqdm -t 1

# 3
python -u run_finetune.py -task dd++/full -voc bert -lr 1.2e-5 -scdl -ff -bs 64 -ep 40 -ckpt checkpoints/TwinSMI-6Sep/model_best_auc.pth -t 1 --no_tqdm

# 4
python -u run_finetune.py -task dd++/cross -voc bert -lr 1.2e-5 -scdl -ff -bs 64 -ep 36 -ckpt checkpoints/TwinSMI-6Sep/model_best_auc.pth -t 1 --no_tqdm

# 5
python -u run_finetune.py -task dd++/adv -voc bert -lr 5e-5 -scdl -ff -bs 64 -ep 36 -ckpt checkpoints/TwinSMI-6Sep/model_best_auc.pth -t 1 --no_tqdm

# 6
python -u run_finetune.py -task dd++ -voc bert -lr 5e-5 -scdl -ff -bs 64 -ep 36 -ckpt checkpoints/TwinSMI-6Sep/model_best_auc.pth -t 1 --no_tqdm

# ============================

# 1
(python -u run_finetune.py -task banking77 -voc roberta -rob -lr 8e-4 -scdl -ff -bs 64 -ep 35 -ckpt checkpoints/RobSMI-6Sep/model_best_auc.pth -t 1 --no_tqdm;
python -u run_finetune.py -task mutual -voc roberta -rob -lr 8e-6 -wtl -scdl -ff -bs 64 -ep 70 -ckpt checkpoints/RobSMI-6Sep/model_best_auc.pth -t 1 --no_tqdm) &


(python -u run_finetune.py -task mutual_plus -voc roberta -rob -lr 5e-6 -wtl -ff -bs 64 -ep 70 -ckpt checkpoints/RobSMI-6Sep/model_best_auc.pth -t 1 --no_tqdm;
python -u run_finetune.py -task e/intent -voc roberta -rob -lr 8e-4 -scdl -ff -bs 256 -ep 32 -ckpt checkpoints/RobSMI-6Sep/model_best_auc.pth -t 1 --no_tqdm)



# 2
python -u run_finetune.py -task swda -voc roberta -rob -lr 1e-3 -scdl -ep 50 -ff -bs 64 -ckpt checkpoints/RobSMI-6Sep/model_best_auc.pth --no_tqdm -t 1

# 3
python -u run_finetune.py -task dd++/full -voc roberta -rob -lr 1.2e-5 -scdl -ff -bs 64 -ep 40 -ckpt checkpoints/RobSMI-6Sep/model_best_auc.pth -t 1 --no_tqdm

# 4
python -u run_finetune.py -task dd++/cross -voc roberta -rob -lr 1.2e-5 -scdl -ff -bs 64 -ep 36 -ckpt checkpoints/RobSMI-6Sep/model_best_auc.pth -t 1 --no_tqdm

# 5
python -u run_finetune.py -task dd++/adv -voc roberta -rob -lr 5e-5 -scdl -ff -bs 64 -ep 36 -ckpt checkpoints/RobSMI-6Sep/model_best_auc.pth -t 1 --no_tqdm

# 6
python -u run_finetune.py -task dd++ -voc roberta -rob -lr 5e-5 -scdl -ff -bs 64 -ep 36 -ckpt checkpoints/RobSMI-6Sep/model_best_auc.pth -t 1 --no_tqdm


