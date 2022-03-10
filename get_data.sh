# Downloading from Onedrive (public read-only share links)
# Reddit datasets were generated using this repo -> https://github.com/bsantraigi/dstc8-reddit-corpus

# SMI-Legacy Checkpoints
LINK="https://iitkgpacin-my.sharepoint.com/:u:/g/personal/bishal_santra_iitkgp_ac_in/EWlmCLQgeuhBjZoecRH92AkBmi8KD_VwN0M5qruIZqpQFg?e=Um8SHU"
wget -nv "$LINK&download=1" -O tmp.zip
unzip -o tmp.zip -d checkpoints/
rm tmp.zip

# Wizard-of-Wikipedia (WoW)
wget http://parl.ai/downloads/wizard_of_wikipedia/wizard_of_wikipedia.tgz
mkdir -p data/WoW
tar -xvf wizard_of_wikipedia.tgz -C data/WoW/ && rm wizard_of_wikipedia.tgz

# Reddit
# 100k

LINK="https://iitkgpacin-my.sharepoint.com/:u:/g/personal/bishal_santra_iitkgp_ac_in/EcXssTSG_9NFk8sedjw97D4B9Z8oPlPPtY5y6XjVAmPsew?e=6BQwk8"
wget "$LINK&download=1" -O tmp.zip
unzip -o tmp.zip -d data/
rm tmp.zip

# 5k
LINK="https://iitkgpacin-my.sharepoint.com/:u:/g/personal/bishal_santra_iitkgp_ac_in/EZ-cmwqNWbVAr2tMFUlfsloB7Z80KxyJDR5AqIHujnTENQ?e=usaFOB"
wget "$LINK&download=1" -O tmp.zip
unzip -o tmp.zip -d data/
rm tmp.zip

# 1M

LINK="https://iitkgpacin-my.sharepoint.com/:u:/g/personal/bishal_santra_iitkgp_ac_in/ESU9MojrQFJDgXxNLHHFKpUBKpBAemJDNuwzRSQkwxA-qA?e=RVTg9V"
wget "$LINK&download=1" -O tmp.zip
unzip -o tmp.zip -d data/
rm tmp.zip

# 1M/cc
LINK="https://iitkgpacin-my.sharepoint.com/:u:/g/personal/bishal_santra_iitkgp_ac_in/EcRUdz8PpdlPhL4bIG0JMDUB-ae93BtEipqmccUSZoRXOw?e=DCwZTm"
wget "$LINK&download=1" -O tmp.zip
unzip -o tmp.zip -d data/
rm tmp.zip

# TODO: Mutual
git clone https://github.com/Nealcly/MuTual.git data/mutual

# TODO: dd++
git clone https://github.com/iitmnlp/DailyDialog-plusplus.git data/dailydialog_pp

# TODO: e/intent
git clone https://github.com/anuradha1992/EmpatheticIntents.git data/e_intents

# TODO: dstc7
#wget -nc http://parl.ai/downloads/dstc7/dstc7_v2.tgz
#mkdir -p data/ubuntu
#tar -xvf dstc7_v2.tgz -C ./data/ubuntu

# TODO: dnli
gdown "https://drive.google.com/u/0/uc?id=1WtbXCv3vPB5ql6w0FVDmAEMmWadbrCuG" -O dnli.zip
unzip dnli.zip dnli/* -d data/
rm dnli.zip

# New Download Format!

wget https://github.com/Nealcly/MuTual/archive/refs/heads/master.zip -O data/mutual.zip
wget https://github.com/iitmnlp/DailyDialog-plusplus/archive/refs/heads/master.zip -O data/dd++.zip
wget https://github.com/anuradha1992/EmpatheticIntents/archive/refs/heads/master.zip -O data/eintent.zip
gdown "https://drive.google.com/u/0/uc?id=1WtbXCv3vPB5ql6w0FVDmAEMmWadbrCuG" -O data/dnli.zip