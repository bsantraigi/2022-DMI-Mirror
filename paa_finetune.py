import os.path
import pickle
import time
import math
import argparse
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BlenderbotTokenizer, BertTokenizer, BertTokenizerFast, RobertaTokenizerFast
from datasets import load_dataset
import wandb

from models import SMIForClassification, Legacy, SMIForRegression
# import datautils
from utils import task_to_keys, pprint_args

import run_finetune
import json
import datautils

def load_paa_data(task_name, split, data_path, eou_token):
    # Sample -> {"context": "", "response": "", "label": ""}
    # is_adversarial = False
    # is_full = False
    # if "/" in split:
    #     split, mode = split.split("/")

    data = []
    if task_name == "paa/ctr":
        df = pd.read_csv(f"{data_path}/PAA_CTR/{split}.tsv", delimiter="\t")
        # print(df.head())
        for _, line in df.iterrows():
            sample = line.to_dict()
            con = sample['Query']
            res = sample['Suggestion']
            ctr = sample['ActualCTR']
            data.append({"context": con, "response": res, "value": ctr})
    elif task_name == "paa/labels":
        df = pd.read_csv(f"{data_path}/PAA_Labels/{split}.tsv", delimiter="\t", names=["lang", "Query", "Suggestion", "Label"])
        # print(df.head())
        for _, line in df.iterrows():
            sample = line.to_dict()
            con = sample['Query']
            res = sample['Suggestion']
            label = sample['Label']
            data.append({"context": con, "response": res, "label": label})
    return data


class PaaDataset(Dataset):
    def __init__(self, task_name, tokenizer, hf_dataset, keys, num_inputs, data_path, split='train', max_len=200,
                 encode_together=False, is_classification=True):
        if isinstance(tokenizer, BlenderbotTokenizer):
            self.CLS = tokenizer.bos_token_id
            self.EOU = "__eou__"
        elif isinstance(tokenizer, BertTokenizer) or isinstance(tokenizer, BertTokenizerFast):
            self.CLS = tokenizer.cls_token_id
            self.EOU = "__eou__"
        elif isinstance(tokenizer, RobertaTokenizerFast):
            self.CLS = tokenizer.cls_token_id
            self.EOU = tokenizer.sep_token
        else:
            raise Exception(f"Reached Hell: Tokenizer not supported {tokenizer}")

        data = load_paa_data(task_name, split, data_path, self.EOU)

        self.task_name = task_name
        self.data = data
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.keys = keys
        self.num_inputs = num_inputs
        self.data_path = data_path
        self.encode_together = encode_together

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        entry = self.data[index]
        text = entry[self.keys["input_1"]]

        if self.encode_together:
            text = text + f" {self.EOU} " + entry[self.keys["input_2"]]
        attn_mask, text = datautils.data_swda.tok_n_pad(self.tokenizer, text, max_len=self.max_len, cls_token_id=self.CLS, left_truncate=True)

        if is_classification:
            label = int(entry[self.keys["label"]])
        else:
            label = torch.tensor(float(entry[self.keys["value"]]), dtype=torch.float32)

        if self.num_inputs == 2 and not self.encode_together:
            resp = entry[self.keys["input_2"]]
            attn_mask, resp = datautils.data_swda.tok_n_pad(self.tokenizer, resp, max_len=self.max_len, cls_token_id=self.CLS)

        # input_ids = [1] + input_ids  # append <s> token - CLS for blenderbot
        # # TODO: Verify this next line
        # attn_mask = [0] + attn_mask  # unmask the [CLS]

        # label = torch.tensor(label, dtype=torch.int64).view(1)

        if self.num_inputs == 2 and not self.encode_together:
            return text, resp, label
        else:
            return text, label


def cmdline_args():
    # Make parser object
    p = argparse.ArgumentParser()
    p.add_argument("-task", "--task", type=str, required=True, choices=['paa/ctr', 'paa/labels'],
                   help="Select a PAA task.")
    p.add_argument("-dp", "--data_path", type=str, default='./data/',
                   help="path to the root data folder.")
    p.add_argument("-voc", "--vocab", type=str, choices=["bert", "blender", "roberta"], required=True,
                   help="mention which tokenizer was used for pretraining? bert or blender")
    p.add_argument("-et", "--encode_together", action="store_true", help="in case of 2 inputs, "
                                                                         "should we encode them as [C _eou_ R] "
                                                                         "or separately.")

    p.add_argument("-rob", "--roberta_init", action="store_true",
                   help="Initialize transformer-encoder with roberta weights?")
    p.add_argument("-robname", "--roberta_name", type=str, default="roberta-base",
                   help="name of checkpoint from huggingface")
    p.add_argument("-bs", "--batch_size", type=int, default=128, help="batch size during pretraining")
    p.add_argument("-ep", "--epochs", type=int, default=10, help="epochs for pretraining")
    # p.add_argument("-vi", "--val_interval", type=int, default=1000, help="validation interval during training")
    # p.add_argument("-li", "--log_interval", type=int, default=100, help="logging interval during training")
    p.add_argument("-lr", "--learning_rate", type=float, default=1e-4, help="set learning rate")
    p.add_argument("-sf", "--slowness_factor", type=float, default=100, help="core_model_lr=lr/slowness")
    p.add_argument("-ff", "--full_finetune", action="store_true", help="the script, by default, probes "
                                                                       "the pretrained model. set this flag to "
                                                                       "finetune the full model.")
    p.add_argument("-lg", "--legacy", action="store_true", help="use legacy CPC model checkpoints.")
    # p.add_argument("-t", "--tracking", default=0, type=int, choices=[0, 1],
    #                help="whether to track training+validation loss wandb")
    p.add_argument("-scdl", "--use_scheduler", action="store_true",
                   help="whether to use a warmup+decay schedule for LR")
    p.add_argument("-wtl", "--use_weighted_loss", action="store_true",
                   help="whether to use class weights in Cross-Entropy loss")
    p.add_argument("-ckpt", "--checkpoint_path", type=str, default=None, help="Path to the .pth model checkpoint file.")
    p.add_argument("-ntq", "--no_tqdm", action="store_true", help="disable tqdm to create concise log files!")
    p.add_argument("-t", "--tracking", default=0, type=int, choices=[0, 1],
                   help="whether to track training+validation loss wandb")

    return (p.parse_args())


if __name__ == '__main__':
    _args = cmdline_args()

    # WANDB
    if _args.tracking:
        raise Exception("Tracking not enabled for PAA tasks.")
        # 1. Start a new run
        # wandb.init(project='smi-finetune', entity='c2ai', config=_args)

        # 2. Save model inputs and hyperparameters
        # Access all hyperparameter values through wandb.config
        # args = wandb.config
    else:
        args = _args



    pprint_args(_args)
    # print(args)

    # if args.roberta_init:
    #     print(f"[WARNING] Initializing from Roberta-base. This will OVERRIDE all arg config parameters...")
    #     print("..........................................................................................\n")
    #     if args.roberta_name == "roberta-base":
    #         args.d_model = 768
    #         args.projection = 768
    #         args.encoder_layers = 12
    #         args.encoder_heads = 12
    #         args.dim_feedforward = 3072
    #     elif args.roberta_name == "roberta-large":
    #         args.d_model = 1024
    #         args.projection = 1024
    #         args.encoder_layers = 24
    #         args.encoder_heads = 16
    #         args.dim_feedforward = 4096
    #     args.vocab = "roberta"


    # LEGACY
    if not args.legacy:
        assert args.checkpoint_path is not None, "Checkpoint path is required."
        assert os.path.isfile(args.checkpoint_path), "Checkpoint path is invalid: No such file"

    # VOCAB
    if args.vocab == "roberta":
        tokenizer = RobertaTokenizerFast.from_pretrained(args.roberta_name)
    else:
        if args.vocab == "blender":
            mname = 'facebook/blenderbot-3B'
            tokenizer = BlenderbotTokenizer.from_pretrained(mname)
        else:
            mname = 'bert-base-uncased'
            tokenizer = BertTokenizerFast.from_pretrained(mname)
        tokenizer.add_special_tokens({'sep_token': '__eou__'})

    print(f"\nVocab Size: {len(tokenizer)}")

    # DATASET
    MAX_SEQ_LEN = 200

    print(f"{args.task} Dataset")
    if args.task not in task_to_keys:
        raise Exception(f"Not there yet! {args.task}")

    num_inputs = 2 if "input_2" in task_to_keys[args.task] else 1
    task_keys = task_to_keys[args.task]
    is_classification = 'label' in task_keys

    if args.task in ["paa/ctr", "paa/labels"]:
        # TODO: classification task from local data
        s1, s2, s3 = task_keys['splits']
        train_data = PaaDataset(args.task, tokenizer, None, task_keys, num_inputs,
                                data_path=args.data_path, split=s1, max_len=MAX_SEQ_LEN,
                                encode_together=args.encode_together)
        valid_data = PaaDataset(args.task, tokenizer, None, task_keys, num_inputs,
                                data_path=args.data_path, split=s2, max_len=MAX_SEQ_LEN,
                                encode_together=args.encode_together)
        test_data = PaaDataset(args.task, tokenizer, None, task_keys, num_inputs,
                               data_path=args.data_path, split=s3, max_len=MAX_SEQ_LEN,
                               encode_together=args.encode_together)

        num_classes = task_keys['num_classes']
    else:
        raise Exception("This script only handles PAA tasks.")

    if args.encode_together:
        num_inputs = 1
        print("Dataloading complete && encode_together=True >> RESET num_inputs=1")

    if is_classification:
        # DECIDE class weights
        if args.use_weighted_loss:
            labels = [x[-1] for x in train_data]
            cnt = Counter(labels)
            weights = torch.tensor([0.0]*num_classes)
            total = sum(cnt.values())
            for i in cnt:
                weights[i] = total/cnt[i]

            print(f"Weighted Loss -> class weights are {weights}")
            criterion = nn.CrossEntropyLoss(torch.tensor(weights))
        else:
            criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    print(f"SPLIT: train ({len(train_data)}), valid ({len(valid_data)}), test ({len(test_data)})")

    BS = args.batch_size
    dataload = DataLoader(train_data, batch_size=BS, num_workers=0, shuffle=True)
    dataload_valid = DataLoader(valid_data, batch_size=BS, num_workers=0)
    dataload_test = DataLoader(test_data, batch_size=BS, num_workers=0)

    # Check if on cuda
    use_cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA:", use_cuda)

    # TRAIN
    if is_classification:
        if not args.legacy:
            clf = SMIForClassification(num_inputs=num_inputs,
                                       num_classes=num_classes,
                                       tokenizer=tokenizer,
                                       freeze=(not args.full_finetune),
                                       checkpoint_path=args.checkpoint_path,
                                       roberta_init=args.roberta_init,
                                       roberta_name=args.roberta_name
                                       )
        else:
            print("#**   USING LEGACY CPC MODELS **#")
            # The following import is a temp fix to make torch.load work!
            # torch.load needs class def in same namespace to work
            from models.legacy import PositionalEncoding, embedding, transformer, Projection
            clf = Legacy.SMIForClassification(num_inputs=num_inputs,
                                       num_classes=num_classes,
                                       tokenizer=tokenizer,
                                       freeze=(not args.full_finetune))
    else:
        # REGRESSION
        clf = SMIForRegression(num_inputs=num_inputs,
                               tokenizer=tokenizer,
                               freeze=(not args.full_finetune),
                               checkpoint_path=args.checkpoint_path,
                               roberta_init=args.roberta_init,
                               roberta_name=args.roberta_name
                               )

    clf = clf.train()
    if use_cuda:
        clf.to(device)
        criterion.to(device)

    # print(len(train_data.word2idx))
    print("Training starts")

    # pass GLOBAL VARIABLES to run_finetune
    run_finetune.args = args
    # run_finetune.device = device

    run_finetune.trainIters(clf, args.epochs, dataload, dataload_test, dataload_valid, loss_fn=criterion,
               learning_rate=args.learning_rate, freeze=(not args.full_finetune), is_classification=is_classification)

    # EVAL
    # clf = torch.load("clf.pth")
    # clf = clf.eval()