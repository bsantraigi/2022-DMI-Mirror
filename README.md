# InfoNCE-Dialog

PyTorch implementation for Contrastive Predictive Coding (https://arxiv.org/pdf/1807.03748.pdf) for Dialogue Data (text modality).

## TODO:

- [x] Verification
    - [x] Dataset + Dataloader
    - [x] Model: Transformer Encoder + CLS embedding
    - [x] Make sure train and validation loss are comparable. (proper normalization)
    - [x] Plot train vs valid loss 
    
## Requirements

- gdown
- wandb
- transformers
- datasets
- pytorch

## Files

```.
├── corr.py
├── create_data.py
├── create_test.py
├── data_deb.py
├── data_json.py
├── data_persona.py
├── data_processed.py
├── datautils
│   ├── data_dialog.py
│   ├── data_swda.py
│   └── __init__.py
├── dialog_eval_refactor.py
├── dialog_train_deb_refactor.py
├── environ.yml
├── filter_wandb_runs.py
├── finetune_pipeline.sh
├── get_data.sh
├── Makefile
├── models
│   ├── core.py
│   ├── downstream.py
│   ├── __init__.py
│   └── legacy.py
├── pretrain.py
├── pull_ckpt.sh
├── README.md
├── run_finetune.py
├── search.yaml
├── summarize_wandb_runs.py
├── tests.sh
└── utils
    ├── func_utils.py
    ├── generate_run_id.py
    ├── __init__.py
    └── task_to_keys.py
```

## Downstream Tasks [PROBING]

### Types of tasks:
- Single utterance classification `(MLP(concat(x)))`
- Context-Response similarity
  - Distance/Similarity between our representations `(cosine(c, r)/L2...)`
  - `MLP(concat(c, r))`
- Dual-utterance classification?
  - `MLP(concat(u, v))`


### GLUE

Adding GLUE tasks just because we can! :P

### DD++

|  	| (R->R, Sim) 	| (R->A, Sim) 	| (A->A, Sim) 	| (R+A->A, Sim) 	| (R->R, MLP) 	| (R->A, MLP) 	| (A->A, MLP) 	| (R+A->A, MLP) 	|
|-	|-	|-	|-	|-	|-	|-	|-	|-	|
| RoBERTa 	|  	|  	|  	|  	|  	|  	|  	|  	|
| BERT 	|  	|  	|  	|  	|  	|  	|  	|  	|
| T5 	|  	|  	|  	|  	|  	|  	|  	|  	|
| GPT-2 	|  	|  	|  	|  	|  	|  	|  	|  	|
| DialoGPT 	|  	|  	|  	|  	|  	|  	|  	|  	|
| Blender 	|  	|  	|  	|  	|  	|  	|  	|  	|
| DEB 	|  	|  	|  	|  	|  	|  	|  	|  	|
| SMI 	|  	|  	|  	|  	|  	|  	|  	|  	|

### All

|  Models | DD++ 	| DD++(Adversarial) 	| SWDA 	| Banking77 	| E-Intent 	| Mutual 	| Mutual++ 	| Ubuntu-DSTC7 	|
|-   |:-:	|:-:	|:-:	|:-:	|:-:	|:-:	|:-:	|:-:	|
|   |  Sim, MLP	| Sim, MLP 	|  	|  	|  	| Sim, MLP 	| Sim, MLP 	| Sim, MLP 	|
| RoBERTa 	|  	|  	|  	|  	|  	|  	|  	|  	|
| BERT 	|  	|  	|  	|  	|  	|  	|  	|  	|
| T5 	|  	|  	|  	|  	|  	|  	|  	|  	|
| GPT-2 	|  	|  	|  	|  	|  	|  	|  	|  	|
| DialoGPT 	|  	|  	|  	|  	|  	|  	|  	|  	|
| Blender 	|  	|  	|  	|  	|  	|  	|  	|  	|
| DEB 	|  	|  	|  	|  	|  	|  	|  	|  	|
| SMI 	|  	|  	|  	|  	|  	|  	|  	|  	|

## SWDA

- Total number of utterances: 199740
- Max utterance length: 132
- Mean utterance length: 9.62
- Total Number of dialogues: 1155
- Max dialogue length: 457
- Mean dialogue length: 172.94
- Vocabulary size: 22301
- Number of labels: 41
- Number of speakers: 2

Train set

- Number of dialogues: 1115
- Max dialogue length: 457
- Mean dialogue length: 172.55
- Number of utterances: 192390

Test set
- Number of dialogues: 19
- Max dialogue length: 330
- Mean dialogue length: 214.63
- Number of utterances: 4078

Val set
- Number of dialogues: 21
- Max dialogue length: 299
- Mean dialogue length: 155.81
- Number of utterances: 3272
