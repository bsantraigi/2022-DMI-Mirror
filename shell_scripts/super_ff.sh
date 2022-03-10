"""
1. InfoNCE-S (8L) - E051F136
2. SMILE (8L) - A1B51263
3. Rob-SMI (12L) - F0AD3838
4. Twin-SMI (8L) - F4DFD937
5. InfoNCE-S Tiny (4L) - 4EE1AE25

# Aug 29
1. https://iitkgpacin-my.sharepoint.com/:u:/g/personal/bishal_santra_iitkgp_ac_in/EcqMzxpYaJpNu_dqzAi0sj8B9aFDQCmZRruarOETkAmcgg?e=teZCAE
2. https://iitkgpacin-my.sharepoint.com/:u:/g/personal/bishal_santra_iitkgp_ac_in/EcNFUzLSjpNIgar4DIO9sZcB071U_In6NlaXETa32XSw3w?e=LhMblu
3. https://iitkgpacin-my.sharepoint.com/:u:/g/personal/bishal_santra_iitkgp_ac_in/EdsF_W5SFG9LoRwWteezzvQBexC7RDY7tZtgNdSYxnRusg?e=Yb4Vig
4. https://iitkgpacin-my.sharepoint.com/:u:/g/personal/bishal_santra_iitkgp_ac_in/ESkYTSNFdUtGquHh_UuuKp0B-mRFus8fR_o1lAIEQfWP3w?e=hOVgmu

# 
"""

LINK="https://iitkgpacin-my.sharepoint.com/:u:/g/personal/bishal_santra_iitkgp_ac_in/EcQsFQr9FUZLjdsIT6O8TbkBHc-UaxLWXo1Bf3OsbkwZIg?e=cSTDY9"
wget "$LINK&download=1" -O model_best_auc.pth
mkdir -p checkpoints/DD-Tiny-Sym/
mv model_best_auc.pth checkpoints/DD-Tiny-Sym/

LINK="https://iitkgpacin-my.sharepoint.com/:u:/g/personal/bishal_santra_iitkgp_ac_in/ESB9hjM-5JFJm8bJlKPIpBYBtcKKeChEE3W_1YOXggRHvA?e=6Z5JDV"
wget "$LINK&download=1" -O model_best_auc.pth
mkdir -p checkpoints/RobSMI-6Sep/
mv model_best_auc.pth checkpoints/RobSMI-6Sep/

LINK="https://iitkgpacin-my.sharepoint.com/:u:/g/personal/bishal_santra_iitkgp_ac_in/EUQsF4novnhApTlOvfTeQS0BSdyP91aJFhH4t4T3Wzy8sw?e=pLj1hP"
wget "$LINK&download=1" -O model_best_auc.pth
mkdir -p checkpoints/TwinSMI-6Sep/
mv model_best_auc.pth checkpoints/TwinSMI-6Sep/


"""
LINK=""
wget "$LINK&download=1" -O model_best_auc.pth
mkdir -p checkpoints//
mv model_best_auc.pth checkpoints//

CKPT=checkpoints//model_best_auc.pth

(python -u run_finetune.py -task banking77 -voc bert -lr 8e-4 -scdl -bs 64 -ep 35 -ckpt $CKPT -t 1 --no_tqdm;
python -u run_finetune.py -task mutual -voc bert -lr 8e-6 -wtl -scdl -bs 64 -ep 70 -ckpt $CKPT -t 1 --no_tqdm;
python -u run_finetune.py -task dd++/adv -voc bert -lr 5e-5 -scdl -bs 64 -ep 36 -ckpt $CKPT -t 1 --no_tqdm;
python -u run_finetune.py -task dd++ -voc bert -lr 5e-5 -scdl -bs 64 -ep 36 -ckpt $CKPT -t 1 --no_tqdm) &

(python -u run_finetune.py -task mutual_plus -voc bert -lr 5e-6 -wtl -bs 64 -ep 70 -ckpt $CKPT -t 1 --no_tqdm;
python -u run_finetune.py -task e/intent -voc bert -lr 8e-4 -scdl -bs 256 -ep 32 -ckpt $CKPT -t 1 --no_tqdm;
python -u run_finetune.py -task dd++/full -voc bert -lr 1.2e-5 -scdl -bs 64 -ep 40 -ckpt $CKPT -t 1 --no_tqdm;
python -u run_finetune.py -task dd++/cross -voc bert -lr 1.2e-5 -scdl -bs 64 -ep 36 -ckpt $CKPT -t 1 --no_tqdm)

python -u run_finetune.py -task swda -voc bert -lr 1e-3 -scdl -ep 50 -bs 64 -ckpt $CKPT --no_tqdm -t 1
"""