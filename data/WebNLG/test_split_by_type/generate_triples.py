#! -*- coding:utf-8 -*-


import json
from tqdm import tqdm
import codecs


test_normal = []
test_epo = []
test_seo = []

with open('test_normal.json') as f:
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
        spo_num = len(line['triple_list'])
        test_normal.append(line)

with open('test_epo.json') as f:
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
        spo_num = len(line['triple_list'])
        test_epo.append(line)

with open('test_seo.json') as f:
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
        spo_num = len(line['triple_list'])
        test_seo.append(line)

with codecs.open('test_triples_normal.json', 'w', encoding='utf-8') as f:
    json.dump(test_normal, f, indent=4, ensure_ascii=False)

with codecs.open('test_triples_epo.json', 'w', encoding='utf-8') as f:
    json.dump(test_epo, f, indent=4, ensure_ascii=False)

with codecs.open('test_triples_seo.json', 'w', encoding='utf-8') as f:
    json.dump(test_seo, f, indent=4, ensure_ascii=False)

