import os
import argparse
import logging

import torch

logger = logging.getLogger(__name__)

import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from transformers import BlenderbotTokenizer, BertTokenizer

from datautils.data_dialog import DialogData

from utils import pprint_args
from pretrain import evaluate
from models import SMIForClassification, Legacy

# The following import is a temp fix to make torch.load work!
# torch.load needs class def in same namespace to work
from models.legacy import PositionalEncoding, embedding, transformer, Projection

# =============================== DDP ====================================
# DDP Guides:
# https://spell.ml/blog/pytorch-distributed-data-parallel-XvEaABIAAB8Ars0e
# https://pytorch.org/tutorials/intermediate/dist_tuto.html
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
# ========================================================================


def combined_validation(rank, test_loader, model, model_opt, args=None):
    print_msg = lambda x: print(f"[RANK {rank}]: {x}")
    with torch.no_grad():
        # auc-roc calculation
        valid_losses = []
        valid_mi = []
        y_pred = []
        y_test = []
        c_vectors = []
        r_vectors = []
        for entry in tqdm(test_loader, disable=args.no_tqdm, desc="Validation"):
            # ============================ DDP =======================================
            eff_batch_size = len(entry[0])//args.world_size
            if eff_batch_size >= 1:
                batch_context = entry[0][eff_batch_size*rank:eff_batch_size*(rank+1)].to(rank)
                batch_response = entry[1][eff_batch_size*rank:eff_batch_size*(rank+1)].to(rank)
            else:
                continue
            # =======================================================================
            # vloss, score, mi = evaluate(rank, batch_context, batch_response, model, model_opt, args=args)
            mask_ctx = (batch_context == 0)
            mask_rsp = (batch_response == 0)
            c_t, z_t = model(batch_context, batch_response, mask_ctx, mask_rsp)
            c_vectors.extend(c_t.cpu().numpy())
            r_vectors.extend(z_t.cpu().numpy())

        # print_msg(f"Samples: {len(y_pred)}")
        c_vectors = np.row_stack(c_vectors)
        r_vectors = np.row_stack(r_vectors)
        score = c_vectors @ r_vectors.T
        y_pred.extend(score.ravel())
        y_test.extend(np.eye(score.shape[0]).ravel())
        auc = roc_auc_score(y_test, y_pred)
        print_msg(f"Score matrix shape {score.shape}")
        print_msg(f"\n*** Eval AUC: {auc} | Eval AUC / Num positives: {np.sum(y_test)} | Eval Dataset: {len(test_loader.dataset)}\n")
    return auc, np.mean(valid_losses[:-1]), np.mean(valid_mi)


def model_launcher(rank, container):

    # dataload = container['train']
    dataload_valid = container['valid']
    dataload_test = container['test']
    args = container['args']
    tokenizer = container['tokenizer']

    if not args.legacy:
        clf = SMIForClassification(num_inputs=1,
                                   num_classes=2,
                                   tokenizer=tokenizer,
                                   freeze=True,
                                   checkpoint_path=args.checkpoint_path)
    else:
        print("#**   USING LEGACY CPC MODELS **#")
        clf = Legacy.SMIForClassification(num_inputs=1,
                                          num_classes=2,
                                          tokenizer=tokenizer,
                                          freeze=True)

    # clf = SMIForClassification(num_inputs=1,
    #                            num_classes=2,
    #                            tokenizer=tokenizer,
    #                            freeze=True,
    #                            checkpoint_path=args.checkpoint_path)
    clf.to(rank)

    model = DDP(clf.cpc, device_ids=[rank])

    # Set random seed within each process to make sure (ddp-)model initializations are same
    # set_random_seeds(random_seed=1234)

    # MODEL
    # pt_model = SMI(vocab_size=len(tokenizer), d_model=args.d_model, projection_size=args.projection,
    #                encoder_layers=args.encoder_layers, encoder_heads=args.encoder_heads).to(rank)
    # pt_model.train()
    # ddp_model = DDP(pt_model, device_ids=[rank])
    #
    # num_params = sum(p.numel() for p in pt_model.parameters() if p.requires_grad)
    # print_msg(f'Total number of trainable parameters:  {str(num_params / float(1000000))}M')

    # Check if on cuda
    #     print_msg("CUDA:", use_cuda)
    #     if use_cuda:
    #         model.cuda()

    # TRAIN
    # print_msg(len(train_data.word2idx))
    print("Validate")
    auc, valid_loss_mean, valid_mi_mean = combined_validation(0, dataload_valid, model, model_opt=None, args=args)


def init_training_process(rank, size, container, proc_entry_fn=model_launcher, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    proc_entry_fn(rank, container)


def cmdline_args():
    # Make parser object
    p = argparse.ArgumentParser()
    p.add_argument("-dp", "--data_path", type=str, default='./data/',
                   help="path to the root data folder.")
    p.add_argument("-voc", "--vocab", type=str, choices=["bert", "blender"], required=True,
                   help="mention which tokenizer was used for pretraining? bert or blender")

    p.add_argument("-bs", "--batch_size", type=int, default=128, help="batch size during pretraining")
    p.add_argument("-lg", "--legacy", action="store_true", help="use legacy CPC model checkpoints.")

    p.add_argument("-ckpt", "--checkpoint_path", type=str, default=None, help="Path to the .pth model checkpoint file.")
    p.add_argument("-ntq", "--no_tqdm", action="store_true", help="disable tqdm to create concise log files!")

    p.add_argument("-ws", "--world_size", type=int, default=1, help="world size when using DDP with pytorch.")
    p.add_argument("-es", "--estimator", type=str,
                   choices=["infonce", "jsd", "nwj", "tuba", "dv", "smile", "infonce/td"], default="infonce",
                   help="which MI estimator is used as the loss function.")

    return (p.parse_args())

if __name__ == '__main__':
    args = cmdline_args()
    pprint_args(args)

    # Tokenizer
    if args.vocab == "blender":
        mname = 'facebook/blenderbot-3B'
        tokenizer = BlenderbotTokenizer.from_pretrained(mname)
    else:
        mname = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(mname)
    tokenizer.add_special_tokens({'sep_token': '__eou__'})

    print(f"\nVocab Size: {len(tokenizer)}")

    # DATA
    # if args.dataset == "dd":
    #     train_data_path = os.path.join(args.data_path, "dailydialog/dialogues_train.txt")
    # elif args.dataset == "r5k":
    #     train_data_path = os.path.join(args.data_path, "reddit_5k/train_dialogues.txt")
    # elif args.dataset == 'r100k':
    #     train_data_path = os.path.join(args.data_path, "reddit_100k/train_dialogues.txt")
    # elif args.dataset == 'r1M':
    #     train_data_path = os.path.join(args.data_path, "reddit_1M/train_dialogues.txt")
    # else:
    #     raise Exception(f"Not ready yet: {args.dataset}")

    valid_data_path = os.path.join(args.data_path, "dailydialog/dialogues_valid.txt")
    test_data_path = os.path.join(args.data_path, "dailydialog/dialogues_test.txt")

    # READ DATA
    # train_data = DialogData(data_path=train_data_path, tokenizer=tokenizer)
    valid_data = DialogData(data_path=valid_data_path, tokenizer=tokenizer)
    test_data = DialogData(data_path=test_data_path, tokenizer=tokenizer)


    # SHUFFLE with DDP will need special care
    # dataload = DataLoader(train_data, batch_size=BS, num_workers=0, pin_memory=False, shuffle=True)
    dataload_valid = DataLoader(valid_data, batch_size=args.batch_size, num_workers=0, pin_memory=False)
    dataload_test = DataLoader(test_data, batch_size=args.batch_size, num_workers=0, pin_memory=False)

    print('Data loaded')

    logger.warning("DDP disabled >> Launcing single process training.")
    args.distdp = False
    args.world_size = 1
    container = {
        # 'train': dataload,
        'valid': dataload_valid,
        'test': dataload_test,
        'tokenizer': tokenizer,
        'args': args
    }


    init_training_process(0, 1, container, model_launcher)
