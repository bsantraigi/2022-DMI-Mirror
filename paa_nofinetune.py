import random
import argparse
import torch
from torch.nn.utils.rnn import pad_sequence
import json

from torch.utils import data
from tqdm.std import tqdm
from models import SMIForClassification

from transformers import BlenderbotTokenizer, BertTokenizer, BertTokenizerFast, RobertaTokenizerFast, GPT2TokenizerFast
from torch.utils.data import Dataset, DataLoader

MAX_CTX_LEN = 300
MAX_RESP_LEN = 60

class TaskDataset(Dataset):
    def __init__(self, input_file, tokenizer):

        self.prep_tokenizer_info(tokenizer, MAX_CTX_LEN, MAX_RESP_LEN)

        data_raw = []
        with open(input_file) as f:
            for line in f:
                line_obj = json.loads(line)
                data_raw.append(line_obj)
                
        # print(data_raw)
    
        data = []
        for x in data_raw:
            context = f" {self.EOU} ".join(x['context'])
            for j, y in enumerate(x['options']):
                data.append([context, y])
            
        self.data = data
        self.tokenizer = tokenizer
    
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

    def __len__(self):
        return len(self.data)
    
    def _preprocess(self, C, R):
        # print(C)
        # print(R)
        # should be on cpu to support multiple workers in dataloader
        # for blender
        # c = self.tokenizer.encode("<s> " + C)
        # r = self.tokenizer.encode("<s> " + R)

        # for bert
        # c = self.tokenizer.encode(C)
        # r = self.tokenizer.encode(R)

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
        # return self.data[index][0][0],self.data[index][1][0],self.data[index][2][0]
        C, R = self.data[index]
        c, r = self._preprocess(C, R)
        return [c, r]
   
    def collate_fn(self, batch):
        ctx_tokens, rsp_tokens = zip(*batch)
        ctx = pad_sequence(ctx_tokens, batch_first=True, padding_value=self.pad_token_id)
        rsp = pad_sequence(rsp_tokens, batch_first=True, padding_value=self.pad_token_id)
        return ctx, rsp
    
def cmdline_args():
    # Make parser object
    p = argparse.ArgumentParser()
    # p.add_argument("-task", "--task", type=str, default='swda',
    #                help="task from huggingface. Format: glue/taskX or swda etc.")
    # p.add_argument("-dp", "--data_path", type=str, default='./data/',
    #                help="path to the root data folder.")
    p.add_argument("-voc", "--vocab", type=str, choices=["bert", "blender", "roberta"], required=True,
                   help="mention which tokenizer was used for pretraining? bert or blender")
    # p.add_argument("-et", "--encode_together", action="store_true", help="in case of 2 inputs, "
    #                                                                      "should we encode them as [C _eou_ R] "
    #                                                                      "or separately.")

    p.add_argument("-rob", "--roberta_init", action="store_true",
                   help="Initialize transformer-encoder with roberta weights?")
    p.add_argument("-robname", "--roberta_name", type=str, default="roberta-base",
                   help="name of checkpoint from huggingface")
    # p.add_argument("-bs", "--batch_size", type=int, default=128, help="batch size during pretraining")
    # p.add_argument("-ep", "--epochs", type=int, default=10, help="epochs for pretraining")
    # p.add_argument("-vi", "--val_interval", type=int, default=1000, help="validation interval during training")
    # p.add_argument("-li", "--log_interval", type=int, default=100, help="logging interval during training")
    # p.add_argument("-lr", "--learning_rate", type=float, default=1e-4, help="set learning rate")
    # p.add_argument("-sf", "--slowness_factor", type=float, default=100, help="core_model_lr=lr/slowness")
    # p.add_argument("-ff", "--full_finetune", action="store_true", help="the script, by default, probes "
    #                                                                    "the pretrained model. set this flag to "
    #                                                                    "finetune the full model.")
    # p.add_argument("-lg", "--legacy", action="store_true", help="use legacy CPC model checkpoints.")
    # p.add_argument("-t", "--tracking", default=0, type=int, choices=[0, 1],
    #                help="whether to track training+validation loss wandb")
    # p.add_argument("-scdl", "--use_scheduler", action="store_true",
    #                help="whether to use a warmup+decay schedule for LR")
    # p.add_argument("-wtl", "--use_weighted_loss", action="store_true",
    #                help="whether to use class weights in Cross-Entropy loss")
    p.add_argument("-ckpt", "--checkpoint_path", type=str, default=None, help="Path to the .pth model checkpoint file.")
    # p.add_argument("-ntq", "--no_tqdm", action="store_true", help="disable tqdm to create concise log files!")
    # p.add_argument("-t", "--tracking", default=0, type=int, choices=[0, 1],
    #                help="whether to track training+validation loss wandb")
    
    p.add_argument("-i", "--input_file", type=str, required=True, help="Path to the input .jsonl file.")
    p.add_argument("-o", "--output_file", type=str, required=True, help="Path to the output .jsonl file.")

    return (p.parse_args())



if __name__=="__main__":
    args = cmdline_args()
    
    # Tokenizer
    if args.vocab == "roberta":
        tokenizer = RobertaTokenizerFast.from_pretrained(args.roberta_name)
    elif args.vocab == "dgpt-m":
        mname = "microsoft/DialoGPT-medium"
        tokenizer = GPT2TokenizerFast.from_pretrained(mname)
        tokenizer.add_special_tokens({'pad_token': '<pad/>'})
        tokenizer.add_special_tokens({'cls_token': '<cls/>'})
    else:
        if args.vocab == "blender":
            mname = 'facebook/blenderbot-3B'
            tokenizer = BlenderbotTokenizer.from_pretrained(mname)
        else:
            mname = 'bert-base-uncased'
            tokenizer = BertTokenizerFast.from_pretrained(mname)
        tokenizer.add_special_tokens({'sep_token': '__eou__'})

    print(f"\nVocab Size: {len(tokenizer)}")
    
    # The inputs parameters for classifier-MLP here doesn't matter. 
    # We will extract just take out the smi object later on.
    clf = SMIForClassification(num_inputs=2,
                                num_classes=10,
                                tokenizer=tokenizer,
                                freeze=False,
                                checkpoint_path=args.checkpoint_path,
                                roberta_init=args.roberta_init,
                                roberta_name=args.roberta_name
                                )
    
    # Get the actual pretrained model.
    # This removes the MLP layers, as we're not finetuning
    model = clf.cpc
    dataset = TaskDataset(args.input_file, tokenizer)
    # print(dataset[random.randint(0, 100)])
    dataloader = DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=16, shuffle=False)
    
    scores = []    
    for batch in tqdm(dataloader):
        ctx_batch, rsp_batch = batch[0], batch[1]
        # print(batch[0].shape, batch[1].shape)
    
        mask_ctx = (ctx_batch == dataset.pad_token_id)
        mask_rsp = (rsp_batch == dataset.pad_token_id)

        c_t, z_t = model(ctx_batch, rsp_batch, mask_ctx, mask_rsp)
        batch_scores = torch.sum(c_t*z_t, dim=1)
        # print(c_t.shape, z_t.shape, batch_scores.shape)
        scores.extend(batch_scores.cpu().tolist())
        
    
    data_raw = []
    with open(args.input_file) as f:
        for line in f:
            line_obj = json.loads(line)
            data_raw.append(line_obj)

    for i in range(len(data_raw)):
        k = len(data_raw[i]['options'])
        slice, scores = scores[:k], scores[k:]
        data_raw[i]['scores'] = slice

    assert len(scores) == 0
    
    with open(args.output_file, "w") as wf:
        for line in data_raw:
            wf.write(json.dumps(line)+"\n")
        
                