"""Create Dummy data for PAA-downstream task(s)

DO NOT USE THIS IN PRODUCTION!!!

Examples:
    python paa_downstream_adapter.py -i data/PAA/PAAData.tsv -o data/PAA_downstream/train.jsonl  
"""

#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from collections import defaultdict
import re
import os
import argparse
from tqdm import tqdm
from copy import deepcopy
import json

def cmdline_args(s=None):
    # Make parser object

    p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input-path", required=True, help="path of the input tsv file to be processed")
    p.add_argument("-o", "--output-path", required=True, help="path of the dialog structured output txt file")

    if s is None:
        return (p.parse_args())
        
    return (p.parse_args(s))


# ## MAIN()

args = cmdline_args()
print(args)

# PAA_dir = "../data/PAA/"

df = pd.read_csv(args.input_path, delimiter="\t")

# print(df.head())
# print(df.columns)

# ### Preprocess PAA sessions

sessions = defaultdict(list)

for i, row in tqdm(df.iterrows(), desc="Processing tsv file", total=df.shape[0]):
    try:
        sess_key = row['Session Id']
        session_entries = row.to_dict()
        session_entries['PAA Questions'] = session_entries['PAA Questions'].split("__SUGGSEP__")

        # Interactions
        interactions = []
        next_interaction = session_entries['Dynamic Questions']
    #     num_interactions = re.findall(r"__NEXT__", curr_interaction)
        #try:
        for j in range(int(session_entries['Click Counts'])):
            try:
                curr_interaction, next_interaction = next_interaction.split("__NEXT__", 1)
            except ValueError as e:
                curr_interaction = next_interaction
            clicked_query, follow_ups = curr_interaction.split("__QUERYSEP__", 1)
            interactions.append({
                'clicked_q': clicked_query.replace("ClickedQ:", ""),
                'follow_ups': follow_ups.split("__SUGGSEP__")
            })
        #except:
        #    print(row, session_entries['Click Counts'])
        #    range(session_entries['Click Counts'])
        session_entries['interactions'] = interactions
        sessions[sess_key].append(session_entries)
    except:
        print(row)
        

# ### Create Dialogs

superset = []
pool = []
jsonlist = []
for k, sess_s in tqdm(sessions.items(), desc="Generating retrievals"):
    # Construct all [context, pool, answer]
    for sess in sess_s:
        conv = []
        pool = []
        conv.append(sess['Query'])
        pool.extend(sess['PAA Questions'])

        interactions = sess['interactions']
        for inter_i, act in enumerate(interactions):
            if act['clicked_q'] in conv:
                # Already selected -> skip it
                continue
            context = deepcopy(conv)
            options = deepcopy(pool)
            answer = act['clicked_q']
            
            # Pool has 4 relevant options only
            # With only one that user interacted
            while len(options) > 4:
                for j, a in enumerate(options):
                    if a != answer:
                        break
                options.remove(a)
            
            
            if ((answer not in context) and (answer in options) and (len(options) == 4)):
                jsonlist.append({
                    'id': f"{sess['Session Id']}_{inter_i}",
                    'context': context,
                    'answers': chr(options.index(answer)+ord('A')),
                    'answers_text': answer,
                    'options': options,
                })
            
            conv.append(act['clicked_q'])
            pool.extend(act['follow_ups'])
            pool = list(set(pool))
            
            # Remove all prior interactions from pool!
            pool.remove(answer)

        superset.append(conv)

os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

with open(args.output_path, "w") as fw:
    w = "\n".join([json.dumps(l) for l in jsonlist])
    fw.write(w)