#! -*- coding:utf-8 -*-


import json
from tqdm import tqdm
import codecs


test_1 = []
test_2 = []
test_3 = []
test_4 = []
test_other = []

with open('test.json') as f:
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
        if spo_num == 1:
            test_1.append(line)
        elif spo_num == 2:
            test_2.append(line)
        elif spo_num == 3:
            test_3.append(line)
        elif spo_num == 4:
            test_4.append(line)
        else:
            test_other.append(line)


with codecs.open('test_triples_1.json', 'w', encoding='utf-8') as f:
    json.dump(test_1, f, indent=4, ensure_ascii=False)

with codecs.open('test_triples_2.json', 'w', encoding='utf-8') as f:
    json.dump(test_2, f, indent=4, ensure_ascii=False)

with codecs.open('test_triples_3.json', 'w', encoding='utf-8') as f:
    json.dump(test_3, f, indent=4, ensure_ascii=False)

with codecs.open('test_triples_4.json', 'w', encoding='utf-8') as f:
    json.dump(test_4, f, indent=4, ensure_ascii=False)

with codecs.open('test_triples_5.json', 'w', encoding='utf-8') as f:
    json.dump(test_other, f, indent=4, ensure_ascii=False)
