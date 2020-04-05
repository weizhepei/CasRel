#! -*- coding:utf-8 -*-


import json
from tqdm import tqdm
import codecs
import numpy as np

RANDOM_SEED = 2019

rel_set = set()


train_data = []


with open('train.json') as f:
    for l in tqdm(f):
        a = json.loads(l)
        if not a['relations']:
            continue
        line = {
                'text': a['sentext'].lstrip('\"').strip('\r\n').rstrip('\"'),
                'triple_list': [(i['em1'], i['rtext'], i['em2']) for i in a['relations'] if i['rtext'] != 'None']
               }
        if not line['triple_list']:
            continue
        train_data.append(line)
        for rm in a['relations']:
            if rm['rtext'] != 'None':
                rel_set.add(rm['rtext'])


id2rel = {i:j for i,j in enumerate(sorted(rel_set))}
rel2id = {j:i for i,j in id2rel.items()}

with codecs.open('rel2id.json', 'w', encoding='utf-8') as f:
    json.dump([id2rel, rel2id], f, indent=4, ensure_ascii=False)

train_len = len(train_data)
random_order = list(range(train_len))
np.random.seed(RANDOM_SEED)
np.random.shuffle(random_order)

dev_data = [train_data[i] for i in random_order[:int(0.005 * train_len)]]
train_data = [train_data[i] for i in random_order[int(0.005 * train_len):]]

with codecs.open('dev_triples.json', 'w', encoding='utf-8') as f:
    json.dump(dev_data, f, indent=4, ensure_ascii=False)

with codecs.open('train_triples.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, indent=4, ensure_ascii=False)


test_data = []


with open('test.json') as f:
    for l in tqdm(f):
        a = json.loads(l)
        if not a['relations']:
            continue
        line = {
                'text': a['sentext'].lstrip('\"').strip('\r\n').rstrip('\"'),
                'triple_list': [(i['em1'], i['rtext'], i['em2']) for i in a['relations'] if i['rtext'] != 'None']
               }
        if not line['triple_list']:
            continue
        test_data.append(line)


with codecs.open('test_triples.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, indent=4, ensure_ascii=False)

