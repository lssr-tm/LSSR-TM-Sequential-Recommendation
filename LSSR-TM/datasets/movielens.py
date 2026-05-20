import os
import random
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from collections import defaultdict
from tensorflow.keras.preprocessing.sequence import pad_sequences
MAX_ITEM_NUM = 3953
MAX_USER_NUM = 6041

def split_data(file_path):
    dst_path = os.path.dirname(file_path)
    train_path = os.path.join(dst_path, 'ml_train.txt')
    val_path = os.path.join(dst_path, 'ml_val.txt')
    test_path = os.path.join(dst_path, 'ml_test.txt')
    meta_path = os.path.join(dst_path, 'ml_meta.txt')
    users, items = (set(), set())
    item_count = dict()
    history = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            user, item, score, timestamp = line.strip().split('::')
            users.add(int(user))
            items.add(int(item))
            item_count.setdefault(item, 0)
            item_count[item] += 1
            history.setdefault(int(user), [])
            history[int(user)].append([item, timestamp])
    random.shuffle(list(users))
    num = 0
    cold_list = []
    for key, value in item_count.items():
        if value < 20:
            num += 1
            cold_list.append(key)
    pd.DataFrame(cold_list).to_csv(os.path.join(dst_path, 'cold_list.csv'), index=False, header=None)
    print(f'[MovieLens] cold items (count < 20): {num}, total unique items: {len(item_count)}')
    with open(train_path, 'w') as f1, open(val_path, 'w') as f2, open(test_path, 'w') as f3:
        for user in users:
            hist = history[int(user)]
            hist.sort(key=lambda x: x[1])
            for idx, value in enumerate(hist):
                if idx == len(hist) - 1:
                    f3.write(str(user) + '\t' + value[0] + '\n')
                elif idx == len(hist) - 2:
                    f2.write(str(user) + '\t' + value[0] + '\n')
                else:
                    f1.write(str(user) + '\t' + value[0] + '\n')
    with open(meta_path, 'w') as f:
        f.write(str(max(users)) + '\t' + str(max(items)))
    return (train_path, val_path, test_path, meta_path)

def split_seq_data(file_path):
    dst_path = os.path.dirname(file_path)
    train_path = os.path.join(dst_path, 'ml_seq_train.txt')
    val_path = os.path.join(dst_path, 'ml_seq_val.txt')
    test_path = os.path.join(dst_path, 'ml_seq_test.txt')
    meta_path = os.path.join(dst_path, 'ml_seq_meta.txt')
    users, items = (set(), set())
    history = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            user, item, score, timestamp = line.strip().split('::')
            users.add(int(user))
            items.add(int(item))
            history.setdefault(int(user), [])
            history[int(user)].append([item, timestamp])
        random.shuffle(list(users))
    with open(train_path, 'w') as f1, open(val_path, 'w') as f2, open(test_path, 'w') as f3:
        for user, hist in history.items():
            hist_u = history[int(user)]
            hist_u.sort(key=lambda x: x[1])
            hist = [x[0] for x in hist_u]
            time = [x[1] for x in hist_u]
            f1.write(str(user) + '\t' + ' '.join(hist[:-2]) + '\t' + ' '.join(time[:-2]) + '\n')
            f2.write(str(user) + '\t' + ' '.join(hist[:-2]) + '\t' + ' '.join(time[:-2]) + '\t' + hist[-2] + '\n')
            f3.write(str(user) + '\t' + ' '.join(hist[:-1]) + '\t' + ' '.join(time[:-1]) + '\t' + hist[-1] + '\n')
    with open(meta_path, 'w') as f:
        f.write(str(max(users)) + '\t' + str(max(items)))
    return (train_path, val_path, test_path, meta_path)

def load_data(file_path, neg_num, max_item_num):
    data = np.array(pd.read_csv(file_path, delimiter='\t'))
    np.random.shuffle(data)
    neg_items = []
    for i in tqdm(range(len(data))):
        neg_item = [random.randint(1, max_item_num) for _ in range(neg_num)]
        neg_items.append(neg_item)
    return {'user': data[:, 0].astype(int), 'pos_item': data[:, 1].astype(int), 'neg_item': np.array(neg_items)}

def load_seq_data(file_path, mode, seq_len, neg_num, max_item_num, contain_user=False, contain_time=False):
    users, click_seqs, time_seqs, pos_items, neg_items = ([], [], [], [], [])
    day_seqs, hour_seqs, dis_seqs = ([], [], [])
    with open(file_path) as f:
        lines = f.readlines()
        for line in tqdm(lines):
            if mode == 'train':
                user, click_seq, time_seq = line.split('\t')
                click_seq = click_seq.split(' ')
                click_seq = [int(x) for x in click_seq]
                time_seq = time_seq.split(' ')
                abs_time = np.array(list(map(lambda x: [time.localtime(int(x)).tm_yday, time.localtime(int(x)).tm_hour + 1] if x != 0 else [0, 0], time_seq)))
                day_seq = abs_time[:, 0].tolist()
                hour_seq = abs_time[:, 1].tolist()
                hour_abs = np.array([int(x) // 3600 if int(x) != 0 else 0 for x in time_seq], dtype='int64')
                dis_seq = np.concatenate(([0], np.diff(hour_abs).astype('int64')))
                dis_seq = np.where(dis_seq >= 0, dis_seq + 1, 0)
                dis_seq = np.clip(dis_seq, 0, 500).astype('int32').tolist()
                for i in range(len(click_seq) - 1):
                    if i + 1 >= seq_len:
                        tmp = click_seq[i + 1 - seq_len:i + 1]
                        tmp2 = time_seq[i + 1 - seq_len:i + 1]
                        tmp3 = day_seq[i + 1 - seq_len:i + 1]
                        tmp4 = hour_seq[i + 1 - seq_len:i + 1]
                        tmp5 = dis_seq[i + 1 - seq_len:i + 1]
                    else:
                        tmp = [0] * (seq_len - i - 1) + click_seq[:i + 1]
                        tmp2 = [0] * (seq_len - i - 1) + time_seq[:i + 1]
                        tmp3 = [0] * (seq_len - i - 1) + day_seq[:i + 1]
                        tmp4 = [0] * (seq_len - i - 1) + hour_seq[:i + 1]
                        tmp5 = [0] * (seq_len - i - 1) + dis_seq[:i + 1]
                    neg_item = [random.randint(1, max_item_num) for _ in range(neg_num)]
                    users.append(int(user))
                    click_seqs.append(tmp)
                    time_seqs.append(tmp2)
                    day_seqs.append(tmp3)
                    hour_seqs.append(tmp4)
                    dis_seqs.append(tmp5)
                    pos_items.append(click_seq[i + 1])
                    neg_items.append(neg_item)
            else:
                user, click_seq, time_seq, pos_item = line.split('\t')
                click_seq = click_seq.split(' ')
                click_seq = [int(x) for x in click_seq]
                time_seq = time_seq.split(' ')
                abs_time = np.array(list(map(lambda x: [time.localtime(int(x)).tm_yday, time.localtime(int(x)).tm_hour + 1] if x != 0 else [0, 0], time_seq)))
                day_seq = abs_time[:, 0].tolist()
                hour_seq = abs_time[:, 1].tolist()
                hour_abs = np.array([int(x) // 3600 if int(x) != 0 else 0 for x in time_seq], dtype='int64')
                dis_seq = np.concatenate(([0], np.diff(hour_abs).astype('int64')))
                dis_seq = np.where(dis_seq >= 0, dis_seq + 1, 0)
                dis_seq = np.clip(dis_seq, 0, 500).astype('int32').tolist()
                if len(click_seq) >= seq_len:
                    tmp = click_seq[len(click_seq) - seq_len:]
                    tmp2 = time_seq[len(time_seq) - seq_len:]
                    tmp3 = day_seq[len(day_seq) - seq_len:]
                    tmp4 = hour_seq[len(hour_seq) - seq_len:]
                    tmp5 = dis_seq[len(dis_seq) - seq_len:]
                else:
                    tmp = [0] * (seq_len - len(click_seq)) + click_seq[:]
                    tmp2 = [0] * (seq_len - len(time_seq)) + time_seq[:]
                    tmp3 = [0] * (seq_len - len(day_seq)) + day_seq[:]
                    tmp4 = [0] * (seq_len - len(hour_seq)) + hour_seq[:]
                    tmp5 = [0] * (seq_len - len(dis_seq)) + dis_seq[:]
                neg_item = [random.randint(1, max_item_num) for _ in range(neg_num)]
                users.append(int(user))
                click_seqs.append(tmp)
                time_seqs.append(tmp2)
                day_seqs.append(tmp3)
                hour_seqs.append(tmp4)
                dis_seqs.append(tmp5)
                pos_items.append(int(pos_item))
                neg_items.append(neg_item)
    data = list(zip(users, click_seqs, time_seqs, day_seqs, hour_seqs, dis_seqs, pos_items, neg_items))
    random.shuffle(data)
    users, click_seqs, time_seqs, day_seqs, hour_seqs, dis_seqs, pos_items, neg_items = zip(*data)
    data = {'click_seq': np.array(click_seqs), 'pos_item': np.array(pos_items), 'neg_item': np.array(neg_items)}
    if contain_user:
        data['user'] = np.array(users)
    if contain_time:
        data['day_seq'] = np.array(day_seqs)
        data['hour_seq'] = np.array(hour_seqs)
        data['dis_seq'] = np.array(dis_seqs)
    return data

def _gen_negative_samples(neg_num, item_list, max_num):
    for i in range(neg_num):
        neg = random.randint(1, max_num)
        yield neg
"\ndef generate_movielens(file_path, neg_num):\n    with open(file_path, 'r') as f:\n        for line in f:\n            user, pos_item = line.split('\t')\n            neg_item = [random.randint(1, MAX_ITEM_NUM) for _ in range(neg_num)]\n            yield int(user), int(pos_item), neg_item\n"
'\ndef generate_ml(file_path, neg_num):\n    return tf.data.Dataset.from_generator(\n        generator=generate_movielens,\n        args=[file_path, neg_num],\n        output_signature=(\n            tf.TensorSpec(shape=(), dtype=tf.int32),\n            tf.TensorSpec(shape=(), dtype=tf.int32),\n            tf.TensorSpec(shape=(neg_num,), dtype=tf.int32),\n        )\n    )\n'

def generate_seq_data(file_path, seq_len, neg_num):
    with open(file_path, 'r') as f:
        user, hist = f.readline().split('\t')
        hist = hist.split(',')
        hist = [int(item) for item in hist]
        hist_len = len(hist)
        if hist_len < seq_len:
            hist = [0] * (seq_len - hist_len) + hist
            gen_neg = _gen_negative_samples(neg_num, hist, MAX_ITEM_NUM)
            neg_hist = [neg_item for neg_item in gen_neg]
            mask_hist = [0] * (seq_len - hist_len) + [1] * hist_len
        else:
            hist = hist[hist_len - seq_len:]
            gen_neg = _gen_negative_samples(neg_num, hist, MAX_ITEM_NUM)
            neg_hist = [neg_item for neg_item in gen_neg]
            mask_hist = [1] * seq_len
        yield {'user': int(user), 'hist': hist, 'neg_list': neg_hist, 'mask_hist': mask_hist}

def create_ml_1m_dataset(file, trans_score=2, test_neg_num=100):
    print('==========Data Preprocess Start=============')
    data_df = pd.read_csv(file, sep='::', engine='python', names=['user_id', 'item_id', 'label', 'Timestamp'])
    data_df['item_count'] = data_df.groupby('item_id')['item_id'].transform('count')
    data_df = data_df[data_df.item_count >= 5]
    data_df = data_df[data_df.label >= trans_score]
    data_df = data_df.sort_values(by=['user_id', 'Timestamp'])
    print('============Negative Sampling===============')
    train_data, val_data, test_data = (defaultdict(list), defaultdict(list), defaultdict(list))
    item_id_max = data_df['item_id'].max()
    for user_id, df in tqdm(data_df[['user_id', 'item_id']].groupby('user_id')):
        pos_list = df['item_id'].tolist()

        def gen_neg():
            neg = pos_list[0]
            while neg in set(pos_list):
                neg = random.randint(1, item_id_max)
            return neg
        neg_list = [gen_neg() for i in range(len(pos_list) + test_neg_num - 1)]
        for i in range(1, len(pos_list)):
            if i == len(pos_list) - 1:
                test_data['user_id'].append(user_id)
                test_data['pos_id'].append(pos_list[i])
                test_data['neg_id'].append(neg_list[i:])
            elif i == len(pos_list) - 2:
                val_data['user_id'].append(user_id)
                val_data['pos_id'].append(pos_list[i])
                val_data['neg_id'].append(neg_list[i])
            else:
                train_data['user_id'].append(user_id)
                train_data['pos_id'].append(pos_list[i])
                train_data['neg_id'].append(neg_list[i])
    random.shuffle(train_data)
    random.shuffle(val_data)
    train = {'user': np.array(train_data['user_id']), 'pos_item': np.array(train_data['pos_id']), 'neg_item': np.array(train_data['neg_id']).reshape((-1, 1))}
    val = {'user': np.array(val_data['user_id']), 'pos_item': np.array(val_data['pos_id']), 'neg_item': np.array(val_data['neg_id']).reshape((-1, 1))}
    test = {'user': np.array(test_data['user_id']), 'pos_item': np.array(test_data['pos_id']), 'neg_item': np.array(test_data['neg_id'])}
    print('============Data Preprocess End=============')
    return (train, val, test)

def create_seq_ml_1m_dataset(file, trans_score=1, seq_len=40, test_neg_num=100):
    print('==========Data Preprocess Start=============')
    data_df = pd.read_csv(file, sep='::', engine='python', names=['user_id', 'item_id', 'label', 'Timestamp'])
    data_df['item_count'] = data_df.groupby('item_id')['item_id'].transform('count')
    data_df = data_df[data_df.item_count >= 5]
    data_df = data_df[data_df.label >= trans_score]
    data_df = data_df.sort_values(by=['user_id', 'Timestamp'])
    print('============Negative Sampling===============')
    train_data, val_data, test_data = (defaultdict(list), defaultdict(list), defaultdict(list))
    item_id_max = data_df['item_id'].max()
    for user_id, df in tqdm(data_df[['user_id', 'item_id']].groupby('user_id')):
        pos_list = df['item_id'].tolist()

        def gen_neg():
            neg = pos_list[0]
            while neg in set(pos_list):
                neg = random.randint(1, item_id_max)
            return neg
        neg_list = [gen_neg() for i in range(len(pos_list) + test_neg_num)]
        for i in range(1, len(pos_list)):
            hist_i = pos_list[:i]
            if i == len(pos_list) - 1:
                test_data['hist'].append(hist_i)
                test_data['pos_id'].append(pos_list[i])
                test_data['neg_id'].append(neg_list[i:])
            elif i == len(pos_list) - 2:
                val_data['hist'].append(hist_i)
                val_data['pos_id'].append(pos_list[i])
                val_data['neg_id'].append([neg_list[i]])
            else:
                train_data['hist'].append(hist_i)
                train_data['pos_id'].append(pos_list[i])
                train_data['neg_id'].append([neg_list[i]])
    user_num, item_num = (data_df['user_id'].max() + 1, data_df['item_id'].max() + 1)
    random.shuffle(train_data)
    random.shuffle(val_data)
    print('==================Padding===================')
    train = {'click_seq': pad_sequences(train_data['hist'], maxlen=seq_len), 'pos_item': np.array(train_data['pos_id']), 'neg_item': np.array(train_data['neg_id'])}
    val = {'click_seq': pad_sequences(val_data['hist'], maxlen=seq_len), 'pos_item': np.array(val_data['pos_id']), 'neg_item': np.array(val_data['neg_id'])}
    test = {'click_seq': pad_sequences(test_data['hist'], maxlen=seq_len), 'pos_item': np.array(test_data['pos_id']), 'neg_item': np.array(test_data['neg_id'])}
    print('============Data Preprocess End=============')
    return (user_num, item_num, train, val, test)
