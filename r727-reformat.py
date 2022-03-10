import webdataset as wds
import sys
import json
import os
import glob
import re
import tqdm
import tarfile
import io


if __name__=="__main__":
    SEP = " __eou__ "
    base_path = "/home/bishal/HULK/reddit-727m/20210727/train"
    raw_list = glob.glob(f"{base_path}*")
    start = 0
    end = 1
#    for read_file in tqdm.tqdm(sorted(raw_list)):
    for read_x in tqdm.tqdm(range(start, end+1)):
        read_file = f"{base_path}-{read_x:05d}-of-01000.json"
        i = re.search(r"(\d{5})-of-01000", read_file).group(1)
#         with tarfile.open(f"data/r727/train_dialogs_{i}.tar", "w") as tar:
#             for lx, line in enumerate(open(read_file)):
#                 b = line.strip().encode("utf-8")
#                 f = io.BytesIO(b)
#                 info = tarfile.TarInfo(f"{lx}.txt")
#                 info.size = len(b)
#                 tar.addfile(info, f)
        # Method 2
        
        sink = wds.TarWriter(f"data/r727/train_{i}.tar")
        for index, input in tqdm.tqdm(enumerate(open(read_file)), leave=False, desc=f"creating train_{i}.tar"):
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
            meta = {k:v for k,v in data.items() if k not in used_keys}
            sink.write({
                "__key__": f"sample-{i}/{index:05d}",
                "dialog.txt": dialog,
                "meta.json": meta
            })
#             if index == 100:
#                 break
        sink.close()            