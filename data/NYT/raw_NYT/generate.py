import json
import numpy as np

def is_normal_triple(triples, is_relation_first=False):
    entities = set()
    for i, e in enumerate(triples):
        key = 0 if is_relation_first else 2
        if i % 3 != key:
            entities.add(e)
    return len(entities) == 2 * int(len(triples) / 3)

def is_multi_label(triples, is_relation_first=False):
    if is_normal_triple(triples, is_relation_first):
        return False
    if is_relation_first:
        entity_pair = [tuple(triples[3 * i + 1: 3 * i + 3]) for i in range(int(len(triples) / 3))]
    else:
        entity_pair = [tuple(triples[3 * i: 3 * i + 2]) for i in range(int(len(triples) / 3))]
    # if is multi label, then, at least one entity pair appeared more than once
    return len(entity_pair) != len(set(entity_pair))

def is_over_lapping(triples, is_relation_first=False):
    if is_normal_triple(triples, is_relation_first):
        return False
    if is_relation_first:
        entity_pair = [tuple(triples[3 * i + 1: 3 * i + 3]) for i in range(int(len(triples) / 3))]
    else:
        entity_pair = [tuple(triples[3 * i: 3 * i + 2]) for i in range(int(len(triples) / 3))]
    # remove the same entity_pair, then, if one entity appear more than once, it's overlapping
    entity_pair = set(entity_pair)
    entities = []
    for pair in entity_pair:
        entities.extend(pair)
    entities = set(entities)
    return len(entities) != 2 * len(entity_pair)

def load_data(in_file, word_dict, rel_dict, out_file, normal_file, epo_file, seo_file):
    with open(in_file, 'r') as f1, open(out_file, 'w') as f2, open(normal_file, 'w') as f3, open(epo_file, 'w') as f4, open(seo_file, 'w') as f5:
        cnt_normal = 0
        cnt_epo = 0
        cnt_seo = 0
        lines = f1.readlines()
        for line in lines:
            line = json.loads(line)
            print(len(line))
            lengths, sents, spos = line[0], line[1], line[2]
            print(len(spos))
            print(len(sents))
            for i in range(len(sents)):
                new_line = dict()
                #print(sents[i])
                #print(spos[i])
                tokens = [word_dict[i] for i in sents[i]]
                sent = ' '.join(tokens)
                new_line['sentText'] = sent
                triples = np.reshape(spos[i], (-1,3))
                relationMentions = []
                for triple in triples:
                    rel = dict()
                    rel['em1Text'] = tokens[triple[0]]
                    rel['em2Text'] = tokens[triple[1]]
                    rel['label'] = rel_dict[triple[2]]
                    relationMentions.append(rel)
                new_line['relationMentions'] = relationMentions
                f2.write(json.dumps(new_line) + '\n')
                if is_normal_triple(spos[i]):
                    f3.write(json.dumps(new_line) + '\n')
                if is_multi_label(spos[i]):
                    f4.write(json.dumps(new_line) + '\n')
                if is_over_lapping(spos[i]):
                    f5.write(json.dumps(new_line) + '\n')

if __name__ == '__main__':
    file_name = 'valid.json'
    output = 'new_valid.json'
    output_normal = 'new_valid_normal.json'
    output_epo = 'new_valid_epo.json'
    output_seo = 'new_valid_seo.json'
    with open('relations2id.json', 'r') as f1, open('words2id.json', 'r') as f2:
        rel2id = json.load(f1)
        words2id = json.load(f2)
    rel_dict = {j:i for i,j in rel2id.items()}
    word_dict = {j:i for i,j in words2id.items()}
    load_data(file_name, word_dict, rel_dict, output, output_normal, output_epo, output_seo)
