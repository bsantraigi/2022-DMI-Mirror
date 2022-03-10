import zipfile
import json
import warnings
warnings.filterwarnings("ignore")

import re
import copy
import random
import glob
from functools import partial

import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BlenderbotTokenizer, RobertaTokenizerFast, BertTokenizerFast


def text_words(t, filter_disfluency=True):
        """
        Source: https://github.com/NathanDuran/Switchboard-Corpus
        Tokenized version of the utterance; filter_disfluency=True
        will remove the special utterance notation to make the results
        look more like printed text. The tokenization itself is just
        spitting on whitespace, with no other simplification. The
        return value is a list of str instances.
        """
        # t = self.text
        if filter_disfluency:
            t = re.sub(r"([+/\}\[\]]|\{\w)", "", t)
        return ' '.join(re.split(r"\s+", t.strip()))
#e.g. text_words(swda_data['train'][0]['text'])

class SWDA_NathanDuran(Dataset):
    """
    TODO: Implement this class. Data in data/nathanduran_swda/
    """
    pass


def split_dataset_train_valid(dataset, train_valid_ratio=0.9):
    split = dataset.data.train_test_split(test_size=1-train_valid_ratio)
    train = TaskDataset(dataset.task_name, dataset.tokenizer, split, dataset.keys, dataset.num_inputs,
                        dataset.data_path, split="train")
    valid = TaskDataset(dataset.task_name, dataset.tokenizer, split, dataset.keys, dataset.num_inputs,
                        dataset.data_path, split="test")
    print(f"Splitting data: {len(dataset)} -> train {len(train)}, valid {len(valid)}")
    return train, valid


def tok_n_pad(tokenizer, text, max_len, cls_token_id, left_truncate=False):
    input_ids = tokenizer.encode(text, add_special_tokens=False)  # remove  </s> from tokens
    txt_len = len(input_ids)
    attn_mask = [0] * txt_len
    if txt_len < max_len:
        input_ids = input_ids + [tokenizer.pad_token_id] * (max_len - txt_len - 1)  # remove tail / pad tail
        attn_mask = attn_mask + [1] * (max_len - txt_len - 1)
    else:
        if left_truncate:
            input_ids = input_ids[txt_len - max_len + 1:]
            attn_mask = attn_mask[txt_len - max_len + 1:]
        else:
            input_ids = input_ids[:max_len - 1]
            attn_mask = attn_mask[:max_len - 1]

    input_ids = [cls_token_id] + input_ids
    attn_mask = [0] + attn_mask

    text = torch.tensor(input_ids, dtype=torch.long).view(max_len)
    attn_mask = torch.tensor(attn_mask, dtype=torch.bool).view(max_len)
    return attn_mask, text


def load_dailydial_pp(split, data_path, eou_token):
    # Sample -> {"context": "", "response": "", "label": ""}
    is_adversarial = False
    is_full = False
    if "/" in split:
        split, mode = split.split("/")
        if mode == "adv":
            is_adversarial = True
        elif mode == "full":
            is_full = True

    data = []
    with open(f"{data_path}/dailydialog_pp/dataset/{split}.json") as f:
        for line in f:
            sample = json.loads(line)
            con = f" {eou_token} ".join(sample["context"])
            # Positive
            for pos in sample["positive_responses"]:
                data.append({"context": con, "response": pos, "label": 1})
            # Negative
            if is_adversarial:
                for adv in sample["adversarial_negative_responses"]:
                    data.append({"context": con, "response": adv, "label": 0})
            elif is_full:
                for adv in sample["adversarial_negative_responses"]:
                    data.append({"context": con, "response": adv, "label": 0})
                for neg in sample["random_negative_responses"]:
                    data.append({"context": con, "response": neg, "label": 0})
            else:
                for neg in sample["random_negative_responses"]:
                    data.append({"context": con, "response": neg, "label": 0})
    return data


def load_e_intent(split, data_path):
    data = []
    with open(f"{data_path}/e_intents/datasets/train_data/{split}.txt") as f:
        for line in f:
            label, _, text = line.split(" ", 2)
            data.append({"context": text.strip(), "label": int(label.strip())})
    return data


def load_dnli(split, data_path):
    data = []
    label_map ={
        "negative": 0,
        "positive": 1,
        "neutral": 2
    }
    with open(f"{data_path}/dnli/dialogue_nli/dialogue_nli_{split}.jsonl") as f:
        f = json.loads(f.read())
        for line in f:
            s1 = line['sentence1']
            s2 = line['sentence2']
            label = label_map[line["label"]]
            data.append({"sentence1": s1.strip(), "sentence2": s2.strip(), "label": label})
    return data

class TaskDataset(Dataset):
    def __init__(self, task_name, tokenizer, hf_dataset, keys, num_inputs, data_path, split='train', max_len=200, encode_together=False):
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

        if task_name.split("/")[0] == "dd++":
            data = load_dailydial_pp(split, data_path, self.EOU)
        elif task_name == "e/intent":
            data = load_e_intent(split, data_path)
        elif task_name == "dnli":
            data = load_dnli(split, data_path)
        else:
            data = hf_dataset[split]
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
        if self.task_name == "swda":
            text = text_words(entry[self.keys["input_1"]])
        else:
            text = entry[self.keys["input_1"]]

        if self.encode_together:
            text = text + f" {self.EOU} " + entry[self.keys["input_2"]]
        attn_mask, text = tok_n_pad(self.tokenizer, text, max_len=self.max_len, cls_token_id=self.CLS, left_truncate=True)

        label = int(entry[self.keys["label"]])

        if self.num_inputs == 2 and not self.encode_together:
            resp = entry[self.keys["input_2"]]
            attn_mask, resp = tok_n_pad(self.tokenizer, resp, max_len=self.max_len, cls_token_id=self.CLS)

        # input_ids = [1] + input_ids  # append <s> token - CLS for blenderbot
        # # TODO: Verify this next line
        # attn_mask = [0] + attn_mask  # unmask the [CLS]

        # label = torch.tensor(label, dtype=torch.int64).view(1)

        if self.num_inputs == 2 and not self.encode_together:
            return text, resp, label
        else:
            return text, label


class RetrievalDatasetCLF(Dataset):
    def __init__(self, task_name, num_neg_samples, split, data_root, tokenizer, ctx_max_len=200, rsp_max_len=50, encode_together=False):
        if task_name in ["mutual", "mutual_plus"]:
            self.num_neg_samples = num_neg_samples if split != "test" else -1
        else:
            self.num_neg_samples = num_neg_samples
        self.task_name = task_name
        self.ctx_max_len = ctx_max_len
        self.rsp_max_len = rsp_max_len
        self.num_classes = 2
        self.tokenizer = tokenizer
        self.encode_together = encode_together

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

        # Actual data loading
        self.split = split
        self.data = []
        if self.task_name in ["mutual", "mutual_plus"]:
            if split == "validation":
                split = "dev"
            # self.con = []
            # self.pos = []
            # self.neg = []
            with zipfile.ZipFile(f"{data_root}/mutual.zip") as data_zip:
                pattern = rf"MuTual-master/data/{task_name}/{split}/.*.txt"
                rep = re.compile(pattern)
                print(rep)
                all_files = data_zip.namelist()
                fList = [x for x in all_files if rep.match(x)]

                for fi in tqdm.tqdm(fList, desc=f"LOADING({task_name}:{split})"):
                    with data_zip.open(fi) as fh:
                        obj = json.load(fh)
                        # Preprocessing
                        # Remove the m: / f: speaker tags
                        speaker_regex = re.compile(r" *[mf] *: *", flags=re.IGNORECASE)
                        obj["article"] = f" {self.EOU} ".join(speaker_regex.split(obj['article'])[1:])
                        temp = []
                        for opt in obj['options']:
                            temp.append(speaker_regex.sub("", opt))
                        obj["options"] = temp

                    assert len(obj['options']) == 4
                    if split != "test":
                        ans = ord(obj['answers']) - ord('A')
                        for x_index, x in enumerate(obj["options"]):
                            if x_index == ans:
                                self.data.append((obj["article"], x, 1))
                            else:
                                self.data.append((obj["article"], x, 0))
                    else:
                        for x in obj["options"]:
                            self.data.append((obj["article"], x, -1))
        elif self.task_name in ["paa"]:
            if split == "validation":
                split = "dev"
            # self.con = []
            # self.pos = []
            # self.neg = []
            path = f"{data_root}/PAA_downstream/{split}.jsonl"
            with open(path) as fh:
                for fi in fh:
                    obj = json.loads(fi)
                    # Preprocessing
                    # Remove the m: / f: speaker tags
                    obj["article"] = f" {self.EOU} ".join(obj['context'])

                    assert len(obj['options']) == 4
                    ans = ord(obj['answers']) - ord('A')
                    for x_index, x in enumerate(obj["options"]):
                        if x_index == ans:
                            self.data.append((obj["article"], x, 1))
                        else:
                            self.data.append((obj["article"], x, 0))
        elif self.task_name == "dstc7":
            pass
        else:
            raise NotImplementedError(f"Umm... What now? [{task_name} not found]")

    def split_train_valid(self, train_valid_ratio):
        assert self.split != "test", "Do not split test set in retrieval"
        train = copy.deepcopy(self)
        valid = copy.deepcopy(self)

        # split
        divisor = (self.num_neg_samples + 1)
        K = int(train_valid_ratio*len(train) // divisor)
        K = int(K * divisor)
        train.data = train.data[:K]
        valid.data = valid.data[K:]

        print(f"Splitting data: {len(self)} -> train {len(train)}, valid {len(valid)}")
        return train, valid

    def __getitem__(self, index):
        context, candidate, label = self.data[index]
        if self.encode_together:
            context = context + f" {self.EOU} " + candidate
            _, context = tok_n_pad(self.tokenizer, context, max_len=self.ctx_max_len, cls_token_id=self.CLS,
                                   left_truncate=True)
            # _, candidate = tok_n_pad(self.tokenizer, candidate, max_len=self.rsp_max_len, cls_token_id=self.CLS)

            return context, label
        else:
            _, context = tok_n_pad(self.tokenizer, context, max_len=self.ctx_max_len, cls_token_id=self.CLS, left_truncate=True)
            _, candidate = tok_n_pad(self.tokenizer, candidate, max_len=self.rsp_max_len, cls_token_id=self.CLS)

            return context, candidate, label

    def __len__(self):
        return len(self.data)


if __name__=="__main__":


    mname = 'facebook/blenderbot-3B'
    tokenizer = BlenderbotTokenizer.from_pretrained(mname)

    # mname = 'bert-base-uncased'
    # tokenizer = BertTokenizer.from_pretrained(mname)
    tokenizer.add_special_tokens({'sep_token': '__eou__'})

    """SWDA
        """
    # datas = SWDA()
    # dataload = DataLoader(datas, batch_size=8)

    """Mutual
    """
    data = RetrievalDatasetCLF("mutual_plus", 3, "train", "./data", tokenizer)
    print(len(data))
    # dl = torch.utils.data.DataLoader(data, batch_size=10, shuffle=True)
    # for x in dl:
    #     print(x)
    #     break
    #
    # data = RetrievalDatasetCLF("mutual", 3, "validation", "./data", tokenizer)
    # print(len(data))
    #
    # data = RetrievalDatasetCLF("mutual", 3, "test", "./data", tokenizer)
    # print(len(data))
    # dl = torch.utils.data.DataLoader(data, batch_size=10)
    # for x in dl:
    #     print(x)
    #     break
