task_to_keys = {
    "glue/cola": {
        "input_1": "sentence",
        "label": "label"
    },

    "glue/mnli": {
        "input_1": "premise",
        "input_2": "hypothesis",
        "label": "label",
        "splits": ['train', 'validation_matched', 'test_matched']
    },

    "glue/mrpc": {
        "input_1": "sentence1",
        "input_2": "sentence2",
        "label": "label"
    },

    "glue/qnli": {
        "input_1": "question",
        "input_2": "sentence",
        "label": "label"
    },

    "glue/qqp": {
        "input_1": "question1",
        "input_2": "question2",
        "label": "label"
    },

    "glue/rte": {
        "input_1": "sentence1",
        "input_2": "sentence2",
        "label": "label"
    },

    "glue/sst2": {
        "input_1": "sentence",
        "label": "label"
    },

    "glue/stsb": {
        "input_1": "sentence1",
        "input_2": "sentence2",
        "label": "label"
    },

    "glue/wnli": {
        "input_1": "sentence1",
        "input_2": "sentence2",
        "label": "label"
    },

    "swda": {
        "input_1": "text",
        "label": "damsl_act_tag"
    },

    "banking77": {
        'input_1': "text",
        "label": "label"
    },

    "mutual": {
        "input_1": "context",
        "input_2": "response",
        "label": "label"
    },

    "mutual_plus": {
        "input_1": "context",
        "input_2": "response",
        "label": "label"
    },

    "dd++": {
        "input_1": "context",
        "input_2": "response",
        "label": "label",
        "num_classes": 2,
        "splits": ("train", "dev", "test")
    },

    "dd++/adv": {
        "input_1": "context",
        "input_2": "response",
        "label": "label",
        "num_classes": 2,
        "splits": ("train/adv", "dev/adv", "test/adv")
    },

    "dd++/cross": {
        "input_1": "context",
        "input_2": "response",
        "label": "label",
        "num_classes": 2,
        "splits": ("train", "dev/adv", "test/adv")
    },

    "dd++/full": {
        "input_1": "context",
        "input_2": "response",
        "label": "label",
        "num_classes": 2,
        "splits": ("train/full", "dev/adv", "test/adv")
    },

    "e/intent": {
        "input_1": "context",
        "label": "label",
        "num_classes": 41,
        "splits": ("train", "valid", "test")
    },

    "dnli": {
        "input_1": "sentence1",
        "input_2": "sentence2",
        "label": "label",
        "num_classes": 3,
        "splits": ("train", "dev", "test")
    },
    "paa": {
        "input_1": "context",
        "input_2": "response",
        "label": "label"
    },
    "paa/ctr": {
        "input_1": "context",
        "input_2": "response",
        "value": "value", # value is for regression
        "num_classes": 1, # Just a dummy placeholder
        "splits": ("train", "dev", "test")
    },
    "paa/labels": {
        "input_1": "context",
        "input_2": "response",
        "label": "label",
        "num_classes": 2,
        "splits": ("train", "dev", "test")
    },


}