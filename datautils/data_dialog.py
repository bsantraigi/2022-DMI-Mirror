import torch
from torch.utils.data import Dataset, DataLoader
import tqdm
from transformers import BlenderbotTokenizer, BertTokenizer, BertTokenizerFast, RobertaTokenizerFast, GPT2TokenizerFast
import random
import pandas as pd

# Ignore warnings
import warnings
import json
warnings.filterwarnings("ignore")

MAX_CTX_LEN = 300
MAX_RESP_LEN = 60

from torch.nn.utils.rnn import pad_sequence
from .r1m_preprocess import filter_dialogs

class DialogData(Dataset):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, data_path, tokenizer, max_items=50000, max_ctx_len=MAX_CTX_LEN, max_resp_len=MAX_RESP_LEN, 
                 reddit_filter_enabled=False, unsupervised_adrl=False):
        super(DialogData, self).__init__()
        _file = data_path

        print("File:", _file)

        self.unsupervised_adrl = unsupervised_adrl
        self.supervised_adrl = False
        self.prep_tokenizer_info(tokenizer, max_ctx_len, max_resp_len)

        self.dial_data = []
        
        with open(_file) as f:
            for line in tqdm.auto.tqdm(f, desc="Loading data"):
                # if len(self.data) > max_items:
                #     break  # Enough for now
                Full_D = line.strip().strip("__eou__").split(" __eou__ ")
                self.dial_data.append(Full_D)
        
        if reddit_filter_enabled:
            _, self.dial_data = filter_dialogs(self.dial_data)
            
        self.extract_cr_pairs()

    def extract_cr_pairs(self):
        # Empty context fix!
        self.BASE_CTX_LENGTH = 3 if self.unsupervised_adrl or self.supervised_adrl else 2
        
        self.data = []
        for Full_D in tqdm.auto.tqdm(self.dial_data, desc="Unrolling dialogs"):
            if len(Full_D) >= self.BASE_CTX_LENGTH:
                for j in range(self.BASE_CTX_LENGTH, len(Full_D) + 1):
                    D = Full_D[:j]
                    D = [f"{u} {self.EOU}" for u in D]
                    # C = f" {self.EOU} ".join(D[:-1]).strip()
                    # R = D[-1].strip()
                    # mid = len(D)//2
                    # C = " __eou__ ".join(D[:mid])
                    # R = " __eou__ ".join(D[mid:])

                    # self.data.append([C, R])
                    self.data.append(D)

        print(f"Loaded {len(self.data)} CR-samples.")
        print(f"Samples:", self.data[random.randint(0, len(self.data))])

    def prep_tokenizer_info(self, tokenizer, max_ctx_len, max_resp_len):
        self.tokenizer = tokenizer
        self.max_ctx_len = max_ctx_len
        self.max_resp_len = max_resp_len
        if isinstance(tokenizer, BlenderbotTokenizer):
            self.CLS = tokenizer.bos_token_id
            self.EOU = "__eou__"
        elif isinstance(tokenizer, GPT2TokenizerFast):
            # Token new token added because we may init the model with actual DialoGPT weights
            self.CLS = tokenizer.cls_token_id # yet, we added this new token!
            self.EOU = tokenizer.bos_token
            # self.EOU = "__eou__"
        elif isinstance(tokenizer, BertTokenizer) or isinstance(tokenizer, BertTokenizerFast):
            self.CLS = tokenizer.cls_token_id
            self.EOU = "__eou__"
        elif isinstance(tokenizer, RobertaTokenizerFast):
            # Token new token added because we may init the model with actual Roberta weights
            self.CLS = tokenizer.cls_token_id
            self.EOU = tokenizer.sep_token
        else:
            raise Exception(f"Reached Hell: Tokenizer not supported {tokenizer}")
        self.pad_token_id = tokenizer.pad_token_id

    def collate_fn(self, batch):
        if self.unsupervised_adrl or self.supervised_adrl:
            morphed_batch = pd.DataFrame(batch).to_dict(orient="list")
            final_batch = {}
            for relation, sub_batch in morphed_batch.items():
                ctx_tokens, rsp_tokens = zip(*sub_batch)
                ctx = pad_sequence(ctx_tokens, batch_first=True, padding_value=self.pad_token_id)
                rsp = pad_sequence(rsp_tokens, batch_first=True, padding_value=self.pad_token_id)
                final_batch[relation] = (ctx, rsp)
            return final_batch
        else:
            ctx_tokens, rsp_tokens = zip(*batch)
            ctx = pad_sequence(ctx_tokens, batch_first=True, padding_value=self.pad_token_id)
            rsp = pad_sequence(rsp_tokens, batch_first=True, padding_value=self.pad_token_id)
            return ctx, rsp

    def _preprocess(self, D, discourse_rel_type=0):
        # should be on cpu to support multiple workers in dataloader
        # for blender
        # c = self.tokenizer.encode("<s> " + C)
        # r = self.tokenizer.encode("<s> " + R)

        # for bert
        # c = self.tokenizer.encode(C)
        # r = self.tokenizer.encode(R)

        # R should be the single utterance element for each of the discourse relations below. 
        # If not single utterance is not possible, atleast use the smaller of the two elements.
        if discourse_rel_type == 0:
            # Type-0 [c,q];r
            C = " ".join(D[:-1])
            R = D[-1]
        elif discourse_rel_type == 1:
            # Type-1 q;r
            C = D[-2]
            R = D[-1]
        elif discourse_rel_type == 2:
            # Type-2 c;r
            C = " ".join(D[:-2])
            R = D[-1]
        elif discourse_rel_type == 3:
            # Type-3 [c,r];q
            C = " ".join(D[:-2] + [D[-1]])
            R = D[-2]
        elif discourse_rel_type == 4:
            # Type-4 c;[q,r]
            C = " ".join(D[:-2])
            R = " ".join(D[-2:])
        

        c = self.tokenizer.encode(C, add_special_tokens=False)
        r = self.tokenizer.encode(R, add_special_tokens=False)

        l1 = len(c)
        l2 = len(r)

        if l1 >= self.max_ctx_len:
            c = c[l1 - self.max_ctx_len + 1:]
        if l2 >= self.max_resp_len:
            r = r[:self.max_resp_len - 1]

        c = [self.CLS] + c
        r = [self.CLS] + r

        c = torch.tensor(c)
        r = torch.tensor(r)
        return c, r

    def __getitem__(self, index):
        D = self.data[index]
        if not self.unsupervised_adrl:
            c, r = self._preprocess(D, discourse_rel_type=0)
            return [c, r]
        else:
            # Type-0 [c,q];r
            # Type-1 q;r
            # Type-2 c;r
            # Type-3 [c,r];q
            # Type-4 c;[q,r]
            return {
                'cq~r': self._preprocess(D, discourse_rel_type=0),
                'q~r': self._preprocess(D, discourse_rel_type=1),
                'c~r': self._preprocess(D, discourse_rel_type=2),
                'cr~q': self._preprocess(D, discourse_rel_type=3),
                'c~qr': self._preprocess(D, discourse_rel_type=4)
            }
            

    def __len__(self):
        return len(self.data)


class RMaxData(DialogData):
    def __init__(self, data_path, tokenizer, max_items=50000, max_ctx_len=MAX_CTX_LEN, max_resp_len=MAX_RESP_LEN, 
                 reddit_filter_enabled=False, unsupervised_adrl=False):
        # Calling init of GRAND-PARENT class
        super(DialogData, self).__init__()
        assert ".json" in data_path, "Must be a r727 json file."
        _file = data_path

        print("File:", _file)

        self.unsupervised_adrl = unsupervised_adrl
        self.supervised_adrl = False
        self.prep_tokenizer_info(tokenizer, max_ctx_len, max_resp_len)

        # with open(_file) as f:
        # Load Dialogs from the json file
        sink = self.json_to_d(_file)
        self.dial_data = []
        for line in tqdm.auto.tqdm(sink, desc="Loading data"):
            # if len(self.data) > max_items:
            #     break  # Enough for now
            Full_D = line["dialog.txt"].strip().strip("__eou__").split(" __eou__ ")
            self.dial_data.append(Full_D)

        if reddit_filter_enabled:
            _, self.dial_data = filter_dialogs(self.dial_data)

        self.extract_cr_pairs()

    def json_to_d(self, json_path):
        sink = []
        SEP = " __eou__ "
        for index, input in tqdm.tqdm(enumerate(open(json_path)), leave=False, desc=f"Converting {json_path}"):
            data = json.loads(input)
            used_keys = ["context", "response"]
            dialog = data["context"] + SEP + data["response"] + SEP
            ut = 0
            while True:
                key = f"context/{ut}"
                if key in data:
                    dialog = data[key] + SEP + dialog
                    ut += 1
                    used_keys.append(key)
                else:
                    break
            meta = {k: v for k, v in data.items() if k not in used_keys}
            sink.append({
                "__key__": f"{json_path}:{index:05d}",
                "dialog.txt": dialog,
                "meta.json": meta
            })
        return sink


class WoWData(DialogData):
    def __init__(self, data_path, tokenizer, max_items=50000, max_ctx_len=MAX_CTX_LEN, max_resp_len=MAX_RESP_LEN, 
                 reddit_filter_enabled=False, unsupervised_adrl=False, supervised_adrl=False):
        # Calling init of GRAND-PARENT class
        super(DialogData, self).__init__()
        assert ".json" in data_path, "[WoWData] Must be a .json file."
        print("[WoWData] FOR TRAINING WITH S-ADRL ON WoW, IT IS RECOMMENDED THAT YOU SHUFFLE THE DATASET BEFORE USE.")
        _file = data_path

        print("File:", _file)

        self.unsupervised_adrl = unsupervised_adrl
        self.supervised_adrl = supervised_adrl
        self.prep_tokenizer_info(tokenizer, max_ctx_len, max_resp_len)

        # with open(_file) as f:
        # Load Dialogs from the json file
        self.dial_data, self.topic_mapper, self.knowledge_mapper = self.process_wow_json(_file)

        if reddit_filter_enabled:
            # _, self.dial_data = filter_dialogs(self.dial_data)
            raise Exception("WoW data cannot be filtered.")

        self.extract_cr_pairs()
        print(self.knowledge_mapper[0])

        # Diffuse knowledge to **all utterances**
        self.diffuse_knowledge()
        assert len(self.data) == len(self.diffused_knowledge), "Length mismatch in unrolled CR and K pairs."

    def diffuse_knowledge(self):
        self.diffused_knowledge = []
        
        for d_index in range(len(self.knowledge_mapper)):
            # knowledge for dialog at index
            # Remove a wrapper list from each turn
            kd = [x[0] if len(x) > 0 else "" for x in self.knowledge_mapper[d_index]]
            if len(kd) >= self.BASE_CTX_LENGTH:
                # Diffusion process
                # Forward direction
                for j in range(1, len(kd)):
                    if len(kd[j].strip()) == 0:
                        kd[j] = kd[j-1]

                # Reverse Direction
                for j in range(len(kd)-2, -1, -1):
                    if len(kd[j].strip()) == 0:
                        kd[j] = kd[j+1]

                # Skip the first utterance to align with CR-pairs
                # Initial utterances skipped based on self.BASE_CTX_LENGTH
                self.diffused_knowledge.extend(kd[(self.BASE_CTX_LENGTH-1):])

    def process_wow_json(self, json_path):
        with open(json_path) as f:
            data = json.load(f)

        print(f"[WoW] Length of data.json: {len(data)}")
        dialogs = [[turn['text'] for turn in x['dialog']] for x in data]
        topic_map = [[list(
            turn.get('checked_passage', {}).values()
            ) for turn in x['dialog']] for x in data]
        knowledge_map = [[list(
            map(
                lambda x: x.replace("no_passages_used", ""),
                turn.get('checked_sentence', {'default': ""}).values()
                )
            ) for turn in x['dialog']] for x in data]

        return dialogs, topic_map, knowledge_map

    def _preprocess(self, D, K, discourse_rel_type=0):
        # should be on cpu to support multiple workers in dataloader
        # for blender
        # c = self.tokenizer.encode("<s> " + C)
        # r = self.tokenizer.encode("<s> " + R)

        # for bert
        # c = self.tokenizer.encode(C)
        # r = self.tokenizer.encode(R)

        # R should be the single utterance element for each of the discourse relations below. 
        # If not single utterance is not possible, atleast use the smaller of the two elements.

        if discourse_rel_type < 5: # Falls under u-adrl
            return super()._preprocess(D, discourse_rel_type)
        else: # s-adrl
            if discourse_rel_type == 5:
                # Type-0 [c,q];k
                C = " ".join(D[:-1])
                R = K
            elif discourse_rel_type == 6:
                # Type-1 q;k
                C = D[-2]
                R = K
            elif discourse_rel_type == 7:
                # Type-2 r;k
                C = D[-1]
                R = K
            elif discourse_rel_type == 8:
                # ?? I(R;K|C)
                raise Exception("Cannot compute this!")
            elif discourse_rel_type == 9:
                # ?? I(Q;K|C)
                raise Exception("Cannot compute this!")
            

            c = self.tokenizer.encode(C, add_special_tokens=False)
            r = self.tokenizer.encode(R, add_special_tokens=False)

            l1 = len(c)
            l2 = len(r)

            if l1 >= self.max_ctx_len:
                c = c[l1 - self.max_ctx_len + 1:]
            if l2 >= self.max_resp_len:
                r = r[:self.max_resp_len - 1]

            c = [self.CLS] + c
            r = [self.CLS] + r

            c = torch.tensor(c)
            r = torch.tensor(r)
            return c, r

    def __getitem__(self, index):
        D = self.data[index]
        if (not self.unsupervised_adrl) and (not self.supervised_adrl):
            c, r = super()._preprocess(D, discourse_rel_type=0)
            return [c, r]
        
        if self.unsupervised_adrl:
            # Type-0 [c,q];r
            # Type-1 q;r
            # Type-2 c;r
            # Type-3 [c,r];q
            # Type-4 c;[q,r]
            tensor_dict = {
                'cq~r': super()._preprocess(D, discourse_rel_type=0),
                'q~r': super()._preprocess(D, discourse_rel_type=1),
                'c~r': super()._preprocess(D, discourse_rel_type=2),
                'cr~q': super()._preprocess(D, discourse_rel_type=3),
                'c~qr': super()._preprocess(D, discourse_rel_type=4)
            }
        else:
            # The default base
            # s-adrl will be added to this
            tensor_dict = {
                'cq~r': super()._preprocess(D, discourse_rel_type=0),
            }

        K = self.diffused_knowledge[index]        
        if self.supervised_adrl:
            # Type-0 [c,q];r
            # Type-1 q;r
            # Type-2 c;r
            # Type-3 [c,r];q
            # Type-4 c;[q,r]
            tensor_dict.update({
                'cq~k': self._preprocess(D, K, discourse_rel_type=5),
                'q~k': self._preprocess(D, K, discourse_rel_type=6),
                'r~k': self._preprocess(D, K, discourse_rel_type=7)
            })
        return tensor_dict


if __name__=="__main__":
    # tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    tokenizer = GPT2TokenizerFast.from_pretrained("microsoft/DialoGPT-medium")
    train_data = RMaxData(data_path="data/reddit-727m-small/test-00010-of-01000.json", tokenizer=tokenizer, reddit_filter_enabled=True)
    train_data = DialogData(data_path="data/dailydialog/dialogues_train.txt", tokenizer=tokenizer, reddit_filter_enabled=True)
    # train_data = DialogData(data_path="../dd/dialogues_train.txt", tokenizer=tokenizer)
    # print(f"*\tNum Samples: {len(train_data)}\n")
    # train_data = DialogData(data_path="../data/reddit_5k/train_dialogues.txt", tokenizer=tokenizer)
    # print(f"*\tNum Samples: {len(train_data)}\n")
    # train_data = DialogData(data_path="../data/reddit_100k/train_dialogues.txt", tokenizer=tokenizer)
    # print(f"*\tNum Samples: {len(train_data)}\n")
    # train_data = DialogData(data_path="./data/reddit_1M/train_dialogues.txt", tokenizer=tokenizer)
    # print(f"*\tNum Samples: {len(train_data)}\n")

