import os
import json
import argparse
import multiprocessing as mp
import re
from multiprocessing import Pool
from typing import List
import pickle
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser(description='preprocess')
parser.add_argument('--task',type=str, metavar='N',
                    help='dataset name')
parser.add_argument('--workers', default=2, type=int, metavar='N',
                    help='number of workers')
parser.add_argument('--train-path', default='', type=str, metavar='N',
                    help='path to training data')
parser.add_argument('--valid-path', default='', type=str, metavar='N',
                    help='path to valid data')
parser.add_argument('--test-path', default='', type=str, metavar='N',
                    help='path to test data')

args = parser.parse_args()
mp.set_start_method('fork')


def _check_sanity(relation_id_to_str: dict):
    # We directly use normalized relation string as a key for training and evaluation,
    # make sure no two relations are normalized to the same surface form
    relation_str_to_id = {}
    for rel_id, rel_str in relation_id_to_str.items():
        if rel_str is None:
            continue
        if rel_str not in relation_str_to_id:
            relation_str_to_id[rel_str] = rel_id
        elif relation_str_to_id[rel_str] != rel_id:
            assert False, 'ERROR: {} and {} are both normalized to {}' \
                .format(relation_str_to_id[rel_str], rel_id, rel_str)
    return


def _normalize_relations(examples: List[dict], normalize_fn, is_train: bool):
    relation_id_to_str = {}
    for ex in examples:
        rel_str = normalize_fn(ex['relation'])
        relation_id_to_str[ex['relation']] = rel_str
        ex['relation'] = rel_str

    _check_sanity(relation_id_to_str)

    if is_train:
        out_path = '{}/relations.json'.format(os.path.dirname(args.train_path))
        with open(out_path, 'w', encoding='utf-8') as writer:
            json.dump(relation_id_to_str, writer, ensure_ascii=False, indent=4)
            print('Save {} relations to {}'.format(len(relation_id_to_str), out_path))


wn18rr_id2ent = {}
wn18rr_id2imgpath = {}


def _load_wn18rr_texts(path: str):
    global wn18rr_id2ent, wn18rr_id2imgpath
    lines = open(path, 'r', encoding='utf-8').readlines()
    for line in lines:
        fs = line.strip().split('\t')
        assert len(fs) == 3, 'Invalid line: {}'.format(line.strip())
        entity_id, word, desc = fs[0], fs[1].replace('__', ''), fs[2]
        wn18rr_id2ent[entity_id] = (entity_id, word, desc, wn18rr_id2imgpath.get(entity_id, ''))
    print('Load {} entities from {}'.format(len(wn18rr_id2ent), path))


def _process_line_wn18rr(line: str) -> dict:
    fs = line.strip().split('\t')
    assert len(fs) == 3, 'Expect 3 fields for {}'.format(line)
    head_id, relation, tail_id = fs[0], fs[1], fs[2]
    if args.task == 'WN9_ind' or 'WN9':
        head_id = head_id[1:]
        tail_id = tail_id[1:]
    _, head, _, _ = wn18rr_id2ent[head_id]
    _, tail, _, _ = wn18rr_id2ent[tail_id]
    example = {'head_id': head_id,
               'head': head,
               'relation': relation,
               'tail_id': tail_id,
               'tail': tail}
    return example


def preprocess_wn18rr(path):
    if not wn18rr_id2imgpath:
        _load_wn18rr_imgpath()
    if not wn18rr_id2ent:
        _load_wn18rr_texts('{}/wordnet-mlj12-definitions.txt'.format(os.path.dirname(path)))
    lines = open(path, 'r', encoding='utf-8').readlines()
    pool = Pool(processes=args.workers)
    examples = pool.map(_process_line_wn18rr, lines)
    pool.close()
    pool.join()

    _normalize_relations(examples, normalize_fn=lambda rel: rel.replace('_', ' ').strip(),
                         is_train=(path == args.train_path))

    out_path = path + '.json'
    json.dump(examples, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    print('Save {} examples to {}'.format(len(examples), out_path))
    return examples


fb15k_id2ent = {}
fb15k_id2desc = {}
fb15k_id2imgpath = {}


def _load_fb15k237_wikidata(path: str):
    global fb15k_id2ent, fb15k_id2desc, fb15k_id2imgpath
    lines = open(path, 'r', encoding='utf-8').readlines()
    for line in lines:
        fs = line.strip().split('\t')
        assert len(fs) == 2, 'Invalid line: {}'.format(line.strip())
        entity_id, name = fs[0], fs[1]
        if args.task in ['YAGO15K', 'DB15K']:
            name = fs[0]
        name = name.replace('_', ' ').strip()
        if entity_id not in fb15k_id2desc:
            print('No desc found for {}'.format(entity_id))
        if entity_id not in fb15k_id2imgpath:
            print('No img found for {}'.format(entity_id))
        fb15k_id2ent[entity_id] = (entity_id, name, fb15k_id2desc.get(entity_id, ''),
                                   fb15k_id2imgpath.get(entity_id, ''))
    print('Load {} entity names from {}'.format(len(fb15k_id2ent), path))


def _load_fb15k237_desc(path: str):
    global fb15k_id2desc
    lines = open(path, 'r', encoding='utf-8').readlines()
    for line in lines:
        fs = line.strip().split('\t')
        assert len(fs) == 2, 'Invalid line: {}'.format(line.strip())
        entity_id, desc = fs[0], fs[1]
        fb15k_id2desc[entity_id] = _truncate(desc, 50)
    print('Load {} entity descriptions from {}'.format(len(fb15k_id2desc), path))


def _load_fb15k237_imgpath(path: str):
    global fb15k_id2imgpath
    img_paths = pickle.load(open(path, 'rb'))
    for k, v in img_paths.items():
        name = '/m/' + k[2:]
        if os.path.exists(v) and Image.open(v):
            fb15k_id2imgpath[name] = v
        else:
            print('image path of ' + k + ': ' + v + 'not valid.')
    print('Load {} entity image paths from {}'.format(len(fb15k_id2imgpath), path))


def _load_wn18rr_imgpath():
    global wn18rr_id2imgpath
    wndir = './img_data/wnimgs'
    entities = os.listdir(wndir)
    for ent in entities:
        name = ent[1:]
        if os.path.exists(wndir + '/' + ent):
            imgs = os.listdir(wndir + '/' + ent)
            if Image.open(wndir + '/' + ent + '/' + imgs[0]):
                wn18rr_id2imgpath[name] = wndir + '/' + ent + '/' + imgs[0]
        else:
            print("no image for entity " + ent)
    print('Load {} entity image paths from {}'.format(len(wn18rr_id2imgpath), wndir))


def _normalize_fb15k237_relation(relation: str) -> str:
    tokens = relation.replace('./', '/').replace('_', ' ').strip().split('/')
    dedup_tokens = []
    for token in tokens:
        if token not in dedup_tokens[-3:]:
            dedup_tokens.append(token)
    # leaf words are more important (maybe)
    relation_tokens = dedup_tokens[::-1]
    relation = ' '.join([t for idx, t in enumerate(relation_tokens)
                         if idx == 0 or relation_tokens[idx] != relation_tokens[idx - 1]])
    return relation


def _normalize_fb15k_relation(relation: str) -> str:
    tokens = relation.replace('./', '/').replace('_', ' ').strip().split('/')
    dedup_tokens = []
    for token in tokens:
        # if token not in dedup_tokens[-2:]:
        dedup_tokens.append(token)
    # leaf words are more important (maybe)
    relation_tokens = dedup_tokens[::-1]
    relation = ' '.join([t for idx, t in enumerate(relation_tokens)])
    return relation


def _process_line_fb15k237(line: str) -> dict:
    fs = line.strip().split('\t')
    assert len(fs) == 3, 'Expect 3 fields for {}'.format(line)
    head_id, relation, tail_id = fs[0], fs[1], fs[2]

    _, head, _, _ = fb15k_id2ent[head_id]
    _, tail, _, _ = fb15k_id2ent[tail_id]
    example = {'head_id': head_id,
               'head': head,
               'relation': relation,
               'tail_id': tail_id,
               'tail': tail}
    return example


def preprocess_fb15k237(path):
    if not fb15k_id2desc:
        _load_fb15k237_desc('{}/FB15k_mid2description.txt'.format(os.path.dirname(path)))
    # if not fb15k_id2imgpath:
    # _load_fb15k237_imgpath('./data/FB15k237/fb15k_best_img.pickle')
    if not fb15k_id2ent:
        _load_fb15k237_wikidata('{}/FB15k_mid2name.txt'.format(os.path.dirname(path)))

    lines = open(path, 'r', encoding='utf-8').readlines()
    pool = Pool(processes=args.workers)
    examples = pool.map(_process_line_fb15k237, lines)
    pool.close()
    pool.join()

    if args.task.lower() == 'fb15k':
        _normalize_relations(examples, normalize_fn=_normalize_fb15k_relation, is_train=(path == args.train_path))
    else:
        _normalize_relations(examples, normalize_fn=_normalize_fb15k237_relation, is_train=(path == args.train_path))

    out_path = path + '.json'
    json.dump(examples, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    print('Save {} examples to {}'.format(len(examples), out_path))
    return examples


wiki5m_id2rel = {}
wiki5m_id2ent = {}
wiki5m_id2text = {}


def _truncate(text: str, max_len: int):
    return ' '.join(text.split()[:max_len])


def _has_none_value(ex: dict) -> bool:
    return any(v is None for v in ex.values())


def dump_all_entities(examples, out_path, id2text: dict, id2imgpath: dict):
    id2entity = {}
    relations = set()

    if not id2imgpath:
        for ex in examples:
            head_id = ex['head_id']
            relations.add(ex['relation'])
            if head_id not in id2entity:
                id2entity[head_id] = {'entity_id': head_id,
                                      'entity': ex['head'],
                                      'entity_desc': id2text[head_id],
                                      }
            tail_id = ex['tail_id']
            if tail_id not in id2entity:
                id2entity[tail_id] = {'entity_id': tail_id,
                                      'entity': ex['tail'],
                                      'entity_desc': id2text[tail_id],
                                      }
    else:
        for ex in examples:
            head_id = ex['head_id']
            relations.add(ex['relation'])
            if head_id not in id2entity:
                id2entity[head_id] = {'entity_id': head_id,
                                      'entity': ex['head'],
                                      'entity_desc': id2text[head_id],
                                      'entity_imgpath': id2imgpath[head_id]}
            tail_id = ex['tail_id']
            if tail_id not in id2entity:
                id2entity[tail_id] = {'entity_id': tail_id,
                                      'entity': ex['tail'],
                                      'entity_desc': id2text[tail_id],
                                      'entity_imgpath': id2imgpath[tail_id]}
    print('Get {} entities, {} relations in total'.format(len(id2entity), len(relations)))

    json.dump(list(id2entity.values()), open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)


def main():
    all_examples = []
    for path in [args.train_path, args.valid_path, args.test_path]:
        assert os.path.exists(path)
        print('Process {}...'.format(path))
        if args.task.lower() in ['wn18rr', 'wn18rr_ind', 'wn9', 'wn9_ind', 'wn18']:
            all_examples += preprocess_wn18rr(path)
        elif args.task.lower() in ['fb15k237', 'fb15k237_ind', 'fb15k']:
            all_examples += preprocess_fb15k237(path)
        else:
            assert False, 'Unknown task: {}'.format(args.task)

    if args.task.lower() in ['wn18rr', 'wn18rr_ind', 'wn9', 'wn9_ind', 'wn18']:
        id2text = {k: v[2] for k, v in wn18rr_id2ent.items()}
        id2imgpath = {k: v[3] for k, v in wn18rr_id2ent.items()}
    elif args.task.lower() in ['fb15k237', 'fb15k237_ind', 'fb15k']:
        id2text = {k: v[2] for k, v in fb15k_id2ent.items()}
        id2imgpath = {k: v[3] for k, v in fb15k_id2ent.items()}
    else:
        assert False, 'Unknown task: {}'.format(args.task)

    dump_all_entities(all_examples,
                      out_path='{}/entities.json'.format(os.path.dirname(args.train_path)),
                      id2text=id2text,
                      id2imgpath=id2imgpath)
    print('Done')


if __name__ == '__main__':
    tasks = ['FB15k237_ind', 'WN18RR_ind', 'WN9_ind']
    if args.task != '':
        main()
    else:
        for task in tasks:
            args.task = task
            if not args.train_path:
                args.train_path = "./data/" + args.task + "/train.txt"
            if not args.valid_path:
                args.valid_path = "./data/" + args.task + "/valid.txt"
            if not args.test_path:
                args.test_path = "./data/" + args.task + "/test.txt"
            main()
