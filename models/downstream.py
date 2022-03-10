import torch
from torch import nn as nn

from .core import SMI, WrappedSMI


def is_ddp_module(state_dict):
    return 'module' in list(state_dict.keys())[0]

class SMIForClassification(nn.Module):
    def __init__(self, num_inputs, num_classes, tokenizer, freeze=True, hidden_size=256, dropout=0.1, checkpoint_path=None, roberta_init=False, roberta_name=""):
        super(SMIForClassification, self).__init__()
        assert checkpoint_path is not None
        self.num_classes = num_classes
        self.pad_token_id = tokenizer.pad_token_id

        device = torch.device("cpu") # Load to cpu first
        checkpoint = torch.load(checkpoint_path, map_location=device)
        args = checkpoint['args']
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        auc = checkpoint['auc']
        state_dict = checkpoint['model_state_dict']

        self.cpc = SMI(
            vocab_size=len(tokenizer),
            d_model=args['d_model'],
            projection_size=args['projection'],
            encoder_layers=args['encoder_layers'],
            encoder_heads=args['encoder_heads'],
            dim_feedforward=args.get('dim_feedforward', 2048), # 2048 is the default
            roberta_init=roberta_init,
            roberta_name=roberta_name
        )
        if is_ddp_module(state_dict):
            print("*** WARNING: Model was saved as ddp module. Extracting self.module...")
            self.wsmi = WrappedSMI(self.cpc)
            load_status = self.wsmi.load_state_dict(state_dict, strict=False)
            self.cpc = self.wsmi.module
        else:
            load_status = self.cpc.load_state_dict(state_dict, strict=False)
        missing, unexpected = load_status
        if len(unexpected) > 0:
            print(
                f"\n[WARNING] ***Some weights of the model checkpoint at were not used when "
                f"initializing: {unexpected}\n"
            )
        else:
            print(f"\n[INFO] ***All model checkpoint weights were used when initializing current model.\n")
        if len(missing) > 0:
            print(
                f"\n[WARNING] ***Some weights of current model were not initialized from the model checkpoint file "
                f"and are newly initialized: {missing}\n"
            )
        assert (len(missing) <= 2 and len(unexpected) <= 2), f"Too many missing/unexpected keys in checkpoint!!!"

        print(f"Loaded pretrained model\n\tEpoch: {epoch}\n\tLoss: {loss}\n\tAUC: {auc}")

        self.classifier = nn.Sequential(
            nn.Linear(num_inputs*self.cpc.d_model, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        self.freeze = freeze
        if freeze:
            print("** PROBING model.")
            self.cpc.eval()
        else:
            print("** FULL FINETUNE model.")

    def forward(self, context, response=None):
        if self.freeze:
            with torch.no_grad():
                c_t = self.cpc.forward_context_only(context, context == self.pad_token_id)
                if response is not None:
                    z_t = self.cpc.forward_context_only(response, response == self.pad_token_id)
                    c_t = torch.cat([c_t, z_t], dim=-1)
        else:
            c_t = self.cpc.forward_context_only(context, context == self.pad_token_id)
            if response is not None:
                z_t = self.cpc.forward_context_only(response, response == self.pad_token_id)
                c_t = torch.cat([c_t, z_t], dim=-1)

        logits = self.classifier(c_t)
        return logits


class SMIForRegression(nn.Module):
    def __init__(self, num_inputs, tokenizer, freeze=True, hidden_size=256, dropout=0.1, checkpoint_path=None, roberta_init=False, roberta_name=""):
        super(SMIForRegression, self).__init__()
        assert checkpoint_path is not None
        self.pad_token_id = tokenizer.pad_token_id

        device = torch.device("cpu") # Load to cpu first
        checkpoint = torch.load(checkpoint_path, map_location=device)
        args = checkpoint['args']
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        auc = checkpoint['auc']
        state_dict = checkpoint['model_state_dict']

        self.cpc = SMI(
            vocab_size=len(tokenizer),
            d_model=args['d_model'],
            projection_size=args['projection'],
            encoder_layers=args['encoder_layers'],
            encoder_heads=args['encoder_heads'],
            dim_feedforward=args.get('dim_feedforward', 2048), # 2048 is the default
            roberta_init=roberta_init,
            roberta_name=roberta_name
        )
        if is_ddp_module(state_dict):
            print("*** WARNING: Model was saved as ddp module. Extracting self.module...")
            self.wsmi = WrappedSMI(self.cpc)
            load_status = self.wsmi.load_state_dict(state_dict, strict=False)
            self.cpc = self.wsmi.module
        else:
            load_status = self.cpc.load_state_dict(state_dict, strict=False)
        missing, unexpected = load_status
        if len(unexpected) > 0:
            print(
                f"\n[WARNING] ***Some weights of the model checkpoint at were not used when "
                f"initializing: {unexpected}\n"
            )
        else:
            print(f"\n[INFO] ***All model checkpoint weights were used when initializing current model.\n")
        if len(missing) > 0:
            print(
                f"\n[WARNING] ***Some weights of current model were not initialized from the model checkpoint file "
                f"and are newly initialized: {missing}\n"
            )
        assert (len(missing) <= 2 and len(unexpected) <= 2), f"Too many missing/unexpected keys in checkpoint!!!"

        print(f"Loaded pretrained model\n\tEpoch: {epoch}\n\tLoss: {loss}\n\tAUC: {auc}")

        self.affine = nn.Sequential(
            nn.Linear(num_inputs*self.cpc.d_model, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        self.freeze = freeze
        if freeze:
            print("** PROBING model.")
            self.cpc.eval()
        else:
            print("** FULL FINETUNE model.")

    def forward(self, context, response=None):
        if self.freeze:
            with torch.no_grad():
                c_t = self.cpc.forward_context_only(context, context == self.pad_token_id)
                if response is not None:
                    z_t = self.cpc.forward_context_only(response, response == self.pad_token_id)
                    c_t = torch.cat([c_t, z_t], dim=-1)
        else:
            c_t = self.cpc.forward_context_only(context, context == self.pad_token_id)
            if response is not None:
                z_t = self.cpc.forward_context_only(response, response == self.pad_token_id)
                c_t = torch.cat([c_t, z_t], dim=-1)

        output = self.affine(c_t)
        return output