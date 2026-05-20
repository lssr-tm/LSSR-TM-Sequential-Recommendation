import os
import random
from collections import defaultdict
from typing import List
import numpy as np
from tqdm import tqdm
TRAIN_FILENAME = 'train_corpus_total_dual.txt'
TEST_FILENAME = 'test_corpus_total_dual.txt'
SPLIT_SEED = 2026
PROTOCOL_MARKER = 'xlong_public_v2_user_latest_no_timestamp'

def _resolve_xlong_dir(file_path: str) -> str:
    if os.path.isdir(file_path):
        return file_path
    base = os.path.basename(file_path)
    if base in {TRAIN_FILENAME, TEST_FILENAME, 'graph_emb.txt'}:
        return os.path.dirname(file_path)
    return os.path.dirname(file_path)

def _split_csv_ints(text: str) -> List[int]:
    text = text.strip()
    if not text:
        return []
    text = text.replace(' ', '')
    return [int(x) for x in text.split(',') if x != '']

def _split_neg_items(text: str) -> List[int]:
    text = text.strip()
    if not text:
        return []
    if ',' in text:
        return [int(x) for x in text.split(',') if x != '']
    return [int(text)]

def _safe_int(value, default=-1):
    try:
        return int(value)
    except Exception:
        return default

def _read_public_rows(path: str):
    rows = []
    with open(path, 'r', encoding='utf8') as f:
        for line in tqdm(f.readlines()):
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 5:
                continue
            index_raw = parts[0]
            user_raw = parts[1]
            item_seq_raw = _split_csv_ints(parts[2])
            if len(item_seq_raw) == 0:
                continue
            pos_item_raw = int(parts[3])
            neg_items_raw = _split_neg_items(parts[4])
            rows.append({'index_raw': index_raw, 'index_int': _safe_int(index_raw, default=-1), 'user_raw': user_raw, 'item_seq_raw': item_seq_raw, 'pos_item_raw': pos_item_raw, 'neg_items_raw': neg_items_raw})
    return rows

def _build_id_maps(train_rows, test_rows):
    user_map = {}
    item_map = {}
    next_user = 1
    next_item = 1

    def get_user(u):
        nonlocal next_user
        if u not in user_map:
            user_map[u] = next_user
            next_user += 1
        return user_map[u]

    def get_item(i):
        nonlocal next_item
        if i not in item_map:
            item_map[i] = next_item
            next_item += 1
        return item_map[i]
    for row in train_rows + test_rows:
        get_user(row['user_raw'])
        for item in row['item_seq_raw']:
            get_item(item)
        get_item(row['pos_item_raw'])
        for item in row['neg_items_raw']:
            get_item(item)
    return (user_map, item_map)

def _convert_rows(rows, user_map, item_map):
    converted = []
    for row in rows:
        converted.append({'user': user_map[row['user_raw']], 'click_seq': [item_map[x] for x in row['item_seq_raw']], 'pos_item': item_map[row['pos_item_raw']], 'neg_items': [item_map[x] for x in row['neg_items_raw'] if x in item_map]})
    return converted

def _row_rank_key(row):
    return (row.get('index_int', -1), len(row.get('item_seq_raw', [])), row.get('pos_item_raw', -1))

def _per_user_latest_train_val_split(train_rows_raw):
    if len(train_rows_raw) <= 1:
        return (train_rows_raw, train_rows_raw)
    grouped = defaultdict(list)
    for row in train_rows_raw:
        grouped[row['user_raw']].append(row)
    train_rows = []
    val_rows = []
    for _, rows in grouped.items():
        rows_sorted = sorted(rows, key=_row_rank_key)
        if len(rows_sorted) == 1:
            train_rows.extend(rows_sorted)
        else:
            val_rows.append(rows_sorted[-1])
            train_rows.extend(rows_sorted[:-1])
    if len(val_rows) == 0 and len(train_rows) > 1:
        rows_sorted = sorted(train_rows, key=_row_rank_key)
        val_rows = [rows_sorted[-1]]
        train_rows = rows_sorted[:-1]
    return (train_rows, val_rows)

def _serialize_rows(rows, path):
    with open(path, 'w', encoding='utf8') as f:
        for row in rows:
            seq_str = ' '.join(map(str, row['click_seq']))
            neg_str = ' '.join(map(str, row['neg_items'])) if row['neg_items'] else ''
            f.write(f"{row['user']}\t{seq_str}\t{row['pos_item']}\t{neg_str}\n")

def split_seq_data(file_path):
    dst_path = _resolve_xlong_dir(file_path)
    train_src = os.path.join(dst_path, TRAIN_FILENAME)
    test_src = os.path.join(dst_path, TEST_FILENAME)
    if not os.path.exists(train_src) or not os.path.exists(test_src):
        raise FileNotFoundError('XLong public files not found. Please place train_corpus_total_dual.txt and test_corpus_total_dual.txt under the same xlong directory.')
    train_path = os.path.join(dst_path, 'xlong_seq_train.txt')
    val_path = os.path.join(dst_path, 'xlong_seq_val.txt')
    test_path = os.path.join(dst_path, 'xlong_seq_test.txt')
    meta_path = os.path.join(dst_path, 'xlong_seq_meta.txt')
    train_rows_raw = _read_public_rows(train_src)
    test_rows_raw = _read_public_rows(test_src)
    user_map, item_map = _build_id_maps(train_rows_raw, test_rows_raw)
    train_rows_raw, val_rows_raw = _per_user_latest_train_val_split(train_rows_raw)
    train_rows = _convert_rows(train_rows_raw, user_map, item_map)
    val_rows = _convert_rows(val_rows_raw, user_map, item_map)
    test_rows = _convert_rows(test_rows_raw, user_map, item_map)
    _serialize_rows(train_rows, train_path)
    _serialize_rows(val_rows, val_path)
    _serialize_rows(test_rows, test_path)
    with open(meta_path, 'w', encoding='utf8') as f:
        f.write(f'{len(user_map)}\t{len(item_map)}\n')
        f.write(f'protocol\t{PROTOCOL_MARKER}\n')
    return (train_path, val_path, test_path, meta_path)

def _sample_extra_negatives(existing_negs, neg_num, click_seq, pos_item, max_item_num):
    negs = []
    seen = set(click_seq)
    seen.add(pos_item)
    for item in existing_negs:
        if item <= 0 or item > max_item_num:
            continue
        if item in seen:
            continue
        if item not in negs:
            negs.append(item)
        if len(negs) >= neg_num:
            return negs[:neg_num]
    while len(negs) < neg_num:
        sampled = random.randint(1, max_item_num)
        if sampled in seen or sampled in negs:
            continue
        negs.append(sampled)
    return negs

def load_seq_data(file_path, mode, seq_len, neg_num, max_item_num, contain_user=False, contain_time=False):
    users, click_seqs, pos_items, neg_items, dis_seqs = ([], [], [], [], [])
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 3:
                continue
            user = int(parts[0])
            click_seq = [int(x) for x in parts[1].split(' ') if x != '']
            pos_item = int(parts[2])
            existing_negs = [int(x) for x in parts[3].split(' ') if x != ''] if len(parts) >= 4 else []
            if len(click_seq) >= seq_len:
                padded_seq = click_seq[-seq_len:]
            else:
                padded_seq = [0] * (seq_len - len(click_seq)) + click_seq
            dis_seq = [0] * seq_len
            neg_item = _sample_extra_negatives(existing_negs, neg_num, click_seq, pos_item, max_item_num)
            users.append(user)
            click_seqs.append(padded_seq)
            dis_seqs.append(dis_seq)
            pos_items.append(pos_item)
            neg_items.append(neg_item)
    if len(users) == 0:
        raise ValueError(f'No valid samples loaded from {file_path}')
    data = list(zip(users, click_seqs, dis_seqs, pos_items, neg_items))
    if mode == 'train':
        random.Random(SPLIT_SEED).shuffle(data)
    users, click_seqs, dis_seqs, pos_items, neg_items = zip(*data)
    output = {'click_seq': np.array(click_seqs, dtype=np.int32), 'pos_item': np.array(pos_items, dtype=np.int32), 'neg_item': np.array(neg_items, dtype=np.int32)}
    if contain_user:
        output['user'] = np.array(users, dtype=np.int32)
    if contain_time:
        output['dis_seq'] = np.array(dis_seqs, dtype=np.int32)
    return output
