import os.path
import pickle
import time
import math
import argparse
from collections import Counter

import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BlenderbotTokenizer, BertTokenizer, BertTokenizerFast, RobertaTokenizerFast
from datasets import load_dataset
import wandb

from models import SMIForClassification, Legacy
import datautils
from utils import task_to_keys, pprint_args


# Check if on cuda
use_cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(context, label, clf, clf_opt, scdl, loss_fn, response=None):
    clf_opt.zero_grad()

    if use_cuda:
        context = context.cuda()
        label = label.cuda()
        if response is not None:
            response = response.cuda()

    label = label.squeeze()

    if response is None:
        logits = clf(context)  # (batch, n_class)
    else:
        logits = clf(context, response=response)  # (batch, n_class)
    loss = loss_fn(logits, label)

    if torch.isnan(loss):
        pass
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(clf.parameters(), 1.0)
        clf_opt.step()
        if args.use_scheduler:
            scdl.step()
    return float(loss.item())


def evaluate(context, label, clf, loss_fn, response=None):
    with torch.no_grad():
        context = context.to(device)
        label = label.to(device)
        if response is None:
            logits = clf(context)  # (batch, n_class)
        else:
            response = response.to(device)
            logits = clf(context, response=response)  # (batch, n_class)

        label = label.squeeze()
        loss = loss_fn(logits, label)

        preds = torch.argmax(logits, dim=1)

        return float(loss.item()), label.cpu().numpy(), preds.cpu().numpy()


def recall_at_k(model, seq_retrieval_datloader, epoch, K = (1, 2)):
    """
    !Recall At K for retrieval downstream tasks!

    :param model: Classifier Model
    :param seq_retrieval_datloader: batch size must be = neg_pool + 1
         first item of each batch must be the positive item!
    :param K: values of k
    :return: recall_at_k for all k, mrr
    """
    recalls = {}
    for k_val in K:
        recalls[k_val] = 0.
    mrr = 0
    model.eval()
    all_probs = []
    all_labels = []
    for entry in tqdm(seq_retrieval_datloader, position=0, leave=True, disable=args.no_tqdm):
        if args.encode_together:
            assert len(entry) == 2
            # context-response or dual input classification
            context, label = entry
            response = None
        else:
            assert len(entry) == 3
            # context-response or dual input classification
            context, response, label = entry
        assert len(entry[0]) == seq_retrieval_datloader.batch_size

        # v_loss, y_true, y_pred = evaluate(entry[0], entry[2], model, response=entry[1])
        with torch.no_grad():
            context = context.to(device)
            label = label.to(device)
            if not args.encode_together:
                response = response.to(device)

            logits = torch.softmax(clf(context, response=response), dim=1)  # (batch, n_class)
            probs = logits[:, 1]
            all_probs.append(probs)

            label = label.squeeze().tolist()
            all_labels.append(label)

    # for k_val in K:
    #     r = 0
    #     for probs in all_probs:
    #         values, indices = torch.topk(probs, k=k_val, dim=0)
    #         if 0 in indices.tolist():  # and probs[0,0]<probs[0,1]:
    #             r += 1
    #     recalls[k_val] = r/len(all_probs)
    # for probs in all_probs:
    #     all_ranks = torch.argsort(-probs)  # descending
    #     index = all_ranks.tolist().index(0)
    #     mrr += 1/(1+index)
    # mrr = mrr/len(all_probs)

    # New code for MRR and recall
    # '0' is not always positive anymore

    # TODO: Dump test_dataloader + all_probs and all_labels
    dump_prefix = args.checkpoint_path.replace('.pth', f"/{args.task}_dump")
    os.makedirs(dump_prefix, exist_ok=True)
    print(f"Pickle Dump Location: {dump_prefix}")
    with open(f"{dump_prefix}/all_prob_E{epoch}.pickle", "wb") as fp:
        pickle.dump(all_probs, fp)
    with open(f"{dump_prefix}/all_label_E{epoch}.pickle", "wb") as fp:
        pickle.dump(all_labels, fp)
    with open(f"{dump_prefix}/data.pickle", "wb") as fp:
        pickle.dump(seq_retrieval_datloader.dataset.data, fp)

    for probs, label in zip(all_probs, all_labels):
        assert len(label) == 4  # Only for mutual
        pos = label.index(1)
        all_ranks = torch.argsort(-probs)  # descending
        pos_rank = all_ranks.tolist().index(pos)
        mrr += 1/(1+pos_rank)
        for k_val in K:
            if (1+pos_rank) <= k_val:
                recalls[k_val] += 1

    mrr = mrr/len(all_probs)
    for k_val in K:
        recalls[k_val] /= len(all_probs)

    return recalls, mrr


def trainIters(clf, epochs, train_loader, test_loader, valid_loader, loss_fn, learning_rate=5e-5, freeze=True,
               is_classification=True):
    start = time.time()
    if is_classification:
        if freeze:
            clf_opt = AdamW(clf.classifier.parameters(), lr=learning_rate, eps=1e-8)
        else:
            """
            DMI Classifiers have
            - cpc (the pretrained core)
            - classifier
            """
            clf_opt = AdamW([
                # Slow weights
                {"params": clf.cpc.parameters(), 'lr': learning_rate/args.slowness_factor},
                # Fast weights
                {"params": clf.classifier.parameters(), 'lr': learning_rate}
            ], eps=1e-8)
            # clf_opt = AdamW(clf.parameters(), lr=learning_rate, eps=1e-8)
    else:
        if freeze:
            clf_opt = AdamW(clf.affine.parameters(), lr=learning_rate, eps=1e-8)
        else:
            """
            DMI Regression have
            - cpc (the pretrained core)
            - affine
            """
            clf_opt = AdamW([
                # Slow weights
                {"params": clf.cpc.parameters(), 'lr': learning_rate/args.slowness_factor},
                # Fast weights
                {"params": clf.affine.parameters(), 'lr': learning_rate}
            ], eps=1e-8)
            # clf_opt = AdamW(clf.parameters(), lr=learning_rate, eps=1e-8)
    print("\n[INFO] Freeze core DMI?: ", freeze)
    print(clf_opt)
    if args.use_scheduler:
        scdl = get_linear_schedule_with_warmup(clf_opt, num_warmup_steps=0, num_training_steps=len(train_loader) * epochs)
    else:
        scdl = None


    print("Initialised optimisers")

    valid_score = []
    best_valid_score = 100000.0  # any large number
    best_auc = 0.  # any large number

    track_record = []

    for epoch in range(epochs):
        print(f'\n====================\n\tEpochs: {epoch}\n====================\n')
        clf.train()
        train_losses = []
        pbar = tqdm(train_loader, position=0, leave=True, disable=args.no_tqdm)
        for entry in pbar:
            if len(entry) == 2:
                # single input classification
                loss = train(entry[0], entry[1], clf, clf_opt, scdl, loss_fn)
            else:
                # context-response or dual input classification
                loss = train(entry[0], entry[2], clf, clf_opt, scdl, loss_fn, response=entry[1])
            if not math.isnan(loss):
                train_losses.append(loss)
            pbar.set_description(f"Train Loss: {np.mean(train_losses[-20:]):.4f}")

        print('Train loss: ', np.mean(train_losses))

        # validation log
        val_loss = []
        val_true = []
        val_pred = []
        clf.eval()
        for entry in tqdm(valid_loader, position=0, leave=True, disable=args.no_tqdm):
            if len(entry) == 2:
                # single input classification
                v_loss, y_true, y_pred = evaluate(entry[0], entry[1], clf, loss_fn)
            else:
                # context-response or dual input classification
                v_loss, y_true, y_pred = evaluate(entry[0], entry[2], clf, loss_fn, response=entry[1])

            if not math.isnan(v_loss):
                val_loss.append(v_loss)
                val_true.extend(y_true)
                val_pred.extend(y_pred)

        if is_classification:
            val_loss = np.mean(val_loss)  # .item()
            val_acc = accuracy_score(val_true, val_pred)

            print("Validation Loss: ", val_loss)
            print("Validation Acc: ", val_acc)
            if args.tracking:
                wandb.log({
                    "val_loss": val_loss,
                    "val_acc": val_acc
                }, step=epoch)
            if clf.num_classes == 2:
                val_f1 = f1_score(val_true, val_pred)
                print("Validation F1: ", val_f1)
                if args.tracking:
                    wandb.log({
                        "val_f1": val_f1
                    }, step=epoch)
        else:            
            val_loss = np.mean(val_loss)  # .item()
            print("Validation Loss: ", val_loss)
            if args.tracking:
                wandb.log({
                    "val_loss": val_loss
                }, step=epoch)

        # Save trained classification/regression model
        if val_loss < best_valid_score:
            try:
                # Wandb config works with this
                writable_args = args.as_dict()
            except AttributeError:
                # If it fails then we have argparse output
                writable_args = vars(args)

            ckpt_prefix = args.checkpoint_path.replace('.pth', "")
            os.makedirs(ckpt_prefix,exist_ok=True)
            ckpt_name = os.path.join(ckpt_prefix, f"{args.task.replace('/', '_')}_ckpt_loss.pth")            
            torch.save({
                'epoch': epoch,
                'model_state_dict': clf.state_dict(),
                'optim_state_dict': clf_opt.state_dict(),
                'loss': val_loss,
                'args': writable_args
            }, ckpt_name)
            print(f"[Loss] Model saved for current epoch, V.Loss = {val_loss}")
            print("Model saved for current epoch")
            print(f"[CKPT] Location: {ckpt_name}")
            best_valid_score = min(best_valid_score, val_loss)
        # if val_loss < best_valid_score:
        #     torch.save(clf, "clf.pth")
        #     print("Model saved for current epoch")
        #     best_valid_score = min(best_valid_score, val_loss)

        valid_score.append(val_loss)

        if args.task in ["mutual", "mutual_plus", "dstc7", "paa"]:
            retrieval_test_loader = DataLoader(test_data, batch_size=test_data.num_neg_samples + 1, num_workers=0)
            rk, mrr = recall_at_k(clf, retrieval_test_loader, epoch=epoch)
            perf = {
                "val_f1": val_f1,
                "Test MRR": mrr
            }
            print(f"Test MRR: {mrr}")
            for k in rk:
                print(f"Recall@{k} = {rk[k]}")
                perf[f"Test Recall@{k}"] = rk[k]
            track_record.append(perf)
        elif is_classification:
            # Testing log
            test_loss = []
            test_true = []
            test_pred = []
            for entry in tqdm(test_loader, position=0, leave=True, disable=args.no_tqdm):
                if len(entry) == 2:
                    # single input classification
                    v_loss, y_true, y_pred = evaluate(entry[0], entry[1], clf, loss_fn=loss_fn)
                else:
                    # context-response or dual input classification
                    v_loss, y_true, y_pred = evaluate(entry[0], entry[2], clf, loss_fn=loss_fn, response=entry[1])

                if not math.isnan(v_loss):
                    test_loss.append(v_loss)
                    test_true.extend(y_true)
                    test_pred.extend(y_pred)

            test_loss = np.mean(test_loss)  # .item()
            test_acc = accuracy_score(test_true, test_pred)  # .item()

            # TODO: Dump test_dataloader + all_probs and all_labels
            dump_prefix = args.checkpoint_path.replace('.pth', f"/{args.task}_dump")
            os.makedirs(dump_prefix,exist_ok=True)
            print(f"Pickle Dump Location: {dump_prefix}")
            with open(f"{dump_prefix}/pred_E{epoch}.pickle", "wb") as fp:
                pickle.dump(test_pred, fp)
            with open(f"{dump_prefix}/true_E{epoch}.pickle", "wb") as fp:
                pickle.dump(test_true, fp)

            print("Test Loss: ", test_loss)
            print("Test Acc: ", test_acc)

            track_record.append({
                "val_acc": val_acc,
                "test_acc": test_acc
            })
        else:
            # Regression
            # Testing log
            test_loss = []
            test_true = []
            test_pred = []
            for entry in tqdm(test_loader, position=0, leave=True, disable=args.no_tqdm):
                if len(entry) == 2:
                    # single input classification
                    v_loss, y_true, y_pred = evaluate(entry[0], entry[1], clf, loss_fn=loss_fn)
                else:
                    # context-response or dual input classification
                    v_loss, y_true, y_pred = evaluate(entry[0], entry[2], clf, loss_fn=loss_fn, response=entry[1])

                if not math.isnan(v_loss):
                    test_loss.append(v_loss)
                    test_true.extend(y_true)
                    test_pred.extend(y_pred)

            test_loss = np.mean(test_loss)  # .item()

            # TODO: Dump test_dataloader + all_probs and all_labels
            dump_prefix = args.checkpoint_path.replace('.pth', f"/{args.task}_dump")
            os.makedirs(dump_prefix,exist_ok=True)
            print(f"Pickle Dump Location: {dump_prefix}")
            with open(f"{dump_prefix}/pred_E{epoch}.pickle", "wb") as fp:
                pickle.dump(test_pred, fp)
            with open(f"{dump_prefix}/true_E{epoch}.pickle", "wb") as fp:
                pickle.dump(test_true, fp)

            print("Test Loss: ", test_loss)

            track_record.append({
                "val_loss": val_loss,
                "test_loss": test_loss
            })

    if args.task in ["mutual", "mutual_plus", "dstc7", "paa"]:
        for ei, tr in enumerate(track_record):
            print(f"E: {ei}")
            print(tr)
        best = max(track_record, key=lambda x: x["val_f1"])
        # print(f"\n* BEST of all (by Val_F1)\n\tVal F1: {best['val_f1']}\n\tTest Recall@1: {best['test_rk1']}")
        final_perf = {
            "best_val_f1": best['val_f1']
        }
        print(f"\n* BEST of all (by Val_F1)")
        for k in best:
            print(f"\t\t{k}: {best[k]}")
            final_perf[f"best_{k.replace(' ', '_').lower()}"] = best[k]

        if args.tracking:
            for k in final_perf:
                wandb.run.summary[k] = final_perf[k]
    elif is_classification:
        for ei, tr in enumerate(track_record):
            print(f"E: {ei}")
            print(tr)
        best = max(track_record, key=lambda x: x["val_acc"])
        print(f"\n* BEST of all (by Val_Acc)\n\tVal Accuracy: {best['val_acc']}\n\tTest Accuracy: {best['test_acc']}")
        if args.tracking:
            wandb.run.summary["best_val_accuracy"] = best['val_acc']
            wandb.run.summary["best_test_accuracy"] = best['test_acc']
    else:
        for ei, tr in enumerate(track_record):
            print(f"E: {ei}")
            print(tr)
        best = min(track_record, key=lambda x: x["val_loss"])
        print(f"\n* BEST of all (by Val_Loss)\n\tVal Loss: {best['val_loss']}\n\tTest Loss: {best['test_loss']}")
        if args.tracking:
            wandb.run.summary["best_val_loss"] = best['val_loss']
            wandb.run.summary["best_test_loss"] = best['test_loss']


def cmdline_args():
    # Make parser object
    p = argparse.ArgumentParser()
    p.add_argument("-task", "--task", type=str, default='swda',
                   help="task from huggingface. Format: glue/taskX or swda etc.")
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
        # 1. Start a new run
        wandb.init(project='smi-finetune', entity='c2ai', config=_args)

        # 2. Save model inputs and hyperparameters
        # Access all hyperparameter values through wandb.config
        args = wandb.config

        # 3. Log gradients and model parameters
        # wandb.watch(model)
        # for batch_idx, (data, target) in enumerate(train_loader):
        #     ...
    else:
        args = _args

    pprint_args(_args)
    # print(args)

    devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    if args.task in ["mutual", "mutual_plus", "dstc7", "paa"]:
        # Retrieval task from local data
        # TODO: DSTC7
        if args.task in ["mutual", "mutual_plus"]:
            train_data = datautils.RetrievalDatasetCLF(args.task, 3, "train", args.data_path, tokenizer,
                                                       encode_together=args.encode_together)
            train_data, valid_data = train_data.split_train_valid(0.9)
            test_data = datautils.RetrievalDatasetCLF(args.task, 3, "validation", args.data_path, tokenizer,
                                                      encode_together=args.encode_together)
            # Test data doesn't have labels!
            # test_data = datautils.RetrievalDatasetCLF(args.task, 3, "test", args.data_path, tokenizer)

            num_classes = train_data.num_classes
        elif args.task in ["paa"]:
            train_data = datautils.RetrievalDatasetCLF(args.task, 3, "train", args.data_path, tokenizer,
                                                       encode_together=args.encode_together)
            valid_data = datautils.RetrievalDatasetCLF(args.task, 3, "validation", args.data_path, tokenizer,
                                                       encode_together=args.encode_together)
            test_data = datautils.RetrievalDatasetCLF(args.task, 3, "test", args.data_path, tokenizer,
                                                      encode_together=args.encode_together)

            num_classes = train_data.num_classes
        else:
            raise NotImplementedError(f"Coming soon... {args.task}")
    elif args.task in ["dd++", "dd++/adv", "dd++/cross", "dd++/full", "e/intent", "dnli"]:
        # TODO: classification task from local data
        s1, s2, s3 = task_keys['splits']
        train_data = datautils.TaskDataset(args.task, tokenizer, None, task_keys, num_inputs,
                                           data_path=args.data_path, split=s1, max_len=MAX_SEQ_LEN,
                                           encode_together=args.encode_together)
        valid_data = datautils.TaskDataset(args.task, tokenizer, None, task_keys, num_inputs,
                                           data_path=args.data_path, split=s2, max_len=MAX_SEQ_LEN,
                                           encode_together=args.encode_together)
        test_data = datautils.TaskDataset(args.task, tokenizer, None, task_keys, num_inputs,
                                          data_path=args.data_path, split=s3, max_len=MAX_SEQ_LEN,
                                          encode_together=args.encode_together)

        num_classes = task_keys['num_classes']
    else:
        if "glue" in args.task:
            hf_dataset = load_dataset("glue", args.task.split("/")[-1])
        else:
            hf_dataset = load_dataset(args.task)

        if 'splits' in task_keys:
            s1, s2, s3 = task_keys['splits']
        else:
            s1, s2, s3 = "train", "validation", "test"
        train_data = datautils.TaskDataset(args.task, tokenizer, hf_dataset, task_keys, num_inputs,
                                           data_path=None, split=s1, max_len=MAX_SEQ_LEN,
                                           encode_together=args.encode_together)
        if len(hf_dataset) == 3:
            valid_data = datautils.TaskDataset(args.task, tokenizer, hf_dataset, task_keys, num_inputs,
                                               data_path=None, split=s2, max_len=MAX_SEQ_LEN,
                                               encode_together=args.encode_together)
        else:
            # no validation?
            train_data, valid_data = datautils.split_dataset_train_valid(train_data, 0.8)
        test_data = datautils.TaskDataset(args.task, tokenizer, hf_dataset, task_keys, num_inputs,
                                          data_path=None, split=s3, max_len=MAX_SEQ_LEN,
                                          encode_together=args.encode_together)

        num_classes = hf_dataset['train'].features[task_keys['label']].num_classes

    if args.encode_together:
        num_inputs = 1
        print("Dataloading complete && encode_together=True >> RESET num_inputs=1")

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

    print(f"SPLIT: train ({len(train_data)}), valid ({len(valid_data)}), test ({len(test_data)})")

    BS = args.batch_size
    dataload = DataLoader(train_data, batch_size=BS, num_workers=0, shuffle=True)
    dataload_valid = DataLoader(valid_data, batch_size=BS, num_workers=0)
    dataload_test = DataLoader(test_data, batch_size=BS, num_workers=0)

    # Check if on cuda
    print("CUDA:", use_cuda)

    # TRAIN
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
    clf = clf.train()
    if use_cuda:
        clf.to(device)
        criterion.to(device)

    # print(len(train_data.word2idx))
    print("Training starts")

    trainIters(clf, args.epochs, dataload, dataload_test, dataload_valid, loss_fn=criterion,
               learning_rate=args.learning_rate, freeze=(not args.full_finetune))

    # EVAL
    # clf = torch.load("clf.pth")
    # clf = clf.eval()
