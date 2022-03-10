"""PAA Data Adapter for DMI Pretraining
Examples:
    - python paa_adapter.py -i data/PAA/PAAData1.tsv -o data/PAA/train_dialogues.txt
    - python paa_adapter.py -i data/PAA/PAAData2.tsv -o data/PAA/valid_dialogues.txt
    - python paa_adapter.py -i data/PAA/PAAData3.tsv -o data/PAA/test_dialogues.txt
"""
#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from collections import defaultdict
import re
import os
import argparse
from tqdm import tqdm


def cmdline_args():
    # Make parser object

    p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input-path", required=True, help="path of the input tsv file to be processed")
    p.add_argument("-o", "--output-path", required=True, help="path of the dialog structured output txt file")


    return (p.parse_args())


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
#     while "__NEXT__" in next_interaction:


# sessions['a44aa80c88bf7c7b28c95bd3f484273b']


# ### Create Dialogs

superset = []
pool = []
for k, sess_s in tqdm(sessions.items(), desc="Generating conversations"):
    for sess in sess_s:
        conv = []
        conv.append(sess['Query'])
#         conv.append(" __comma__ ".join(sess['PAA Questions']))
        interactions = sess['interactions']
        for act in interactions:
            conv.append(act['clicked_q'])
#             conv.append(" __comma__ ".join(act['follow_ups']))
        superset.append(conv)
    
with open(args.output_path, "w") as f:
    for line in tqdm(superset, desc="Creating dialog file"):
        line = " __eou__ ".join(line) + " __eou__"
        # print(line)
        f.write(line + "\n")

# ### Incorporating all system options

# for k, sess_s in sessions.items():
#     for sess in sess_s:
#         conv = []
#         conv.append(sess['Query'])
#         conv.append(" __comma__ ".join(sess['PAA Questions']))
#         interactions = sess['interactions']
#         for act in interactions:
#             conv.append(act['clicked_q'])
#             conv.append(" __comma__ ".join(act['follow_ups']))
#         print(" __eou__ ".join(conv) + "__eou__")