import json
import pandas as pd
from config import *
from collections import Counter

def generate_rel():
    with open(SCHEMA_PATH) as f:
        rel_list = []
        for line in f.readlines():
            info = json.loads(line)
            rel_list.append(info['predicate'])
        rel_dict = {v: k for k, v in enumerate(rel_list)}
        df = pd.DataFrame(rel_dict.items())
        df.to_csv(REL_PATH, header=None, index=None)

def train_rel_count():
    with open(TRAIN_JSON_PATH) as f:
        rel_list = []
        for line in f.readlines():
            info = json.loads(line)
            rel_list += [spo['predicate'] for spo in info['spo_list']]
    print(Counter(rel_list).most_common())

if __name__ == '__main__':
    generate_rel()
    train_rel_count()