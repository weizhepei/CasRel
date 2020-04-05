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
        if not a['relationMentions']:
            continue
        line = {
                'text': a['sentText'].lstrip('\"').strip('\r\n').rstrip('\"'),
                'triple_list': [(i['em1Text'], i['label'], i['em2Text']) for i in a['relationMentions'] if i['label'] != 'None']
               }
        if not line['triple_list']:
            continue
        train_data.append(line)
        for rm in a['relationMentions']:
            if rm['label'] != 'None':
                rel_set.add(rm['label'])


id2rel = {i:j for i,j in enumerate(sorted(rel_set))}
rel2id = {j:i for i,j in id2rel.items()}

with codecs.open('rel2id.json', 'w', encoding='utf-8') as f:
    json.dump([id2rel, rel2id], f, indent=4, ensure_ascii=False)

with codecs.open('train_triples.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, indent=4, ensure_ascii=False)


test_data = []


with open('test.json') as f:
    for l in tqdm(f):
        a = json.loads(l)
        if not a['relationMentions']:
            continue
        line = {
                'text': a['sentText'].lstrip('\"').strip('\r\n').rstrip('\"').replace(',', ' ,').replace("'s", " 's").lstrip().rstrip().rstrip('.'),
                'triple_list': [(i['em1Text'].replace("'s", " 's"), i['label'], i['em2Text'].replace("'s", " 's")) for i in a['relationMentions'] if i['label'] != 'None']
               }
        if not line['triple_list']:
            continue
        test_data.append(line)


test_len = len(test_data)
random_order = list(range(test_len))
np.random.seed(RANDOM_SEED)
np.random.shuffle(random_order)

dev_data = [test_data[i] for i in random_order[:int(0.1 * test_len)]]
test_data = [test_data[i] for i in random_order[int(0.1 * test_len):]]


with codecs.open('test_triples.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, indent=4, ensure_ascii=False)

with codecs.open('dev_triples.json', 'w', encoding='utf-8') as f:
    json.dump(dev_data, f, indent=4, ensure_ascii=False)
