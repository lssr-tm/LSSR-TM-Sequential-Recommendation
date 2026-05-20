import os
import random
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

def split_data(file_path):
    dst_path = os.path.dirname(file_path)
    train_path = os.path.join(dst_path, 'steam_train.txt')
    val_path = os.path.join(dst_path, 'steam_val.txt')
    test_path = os.path.join(dst_path, 'steam_test.txt')
    meta_path = os.path.join(dst_path, 'steam_meta.txt')
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        users, items = (set(), dict())
        user_idx, item_idx = (1, 1)
        history = {}
        item_count = {}
        for line in tqdm(lines):
            user = re.findall('u\\\'username\\\': u(\\"[^\\"]+\\"|\\\'[^\\\']+\\\')', line)[0]
            item = re.findall("u\\'product_id\\': u\\'([^\\']+?)\\'", line)[0]
            timestamp = re.findall("u\\'date\\': u\\'([^\\']+?)\\'", line)[0]
            users.add(user)
            if items.get(item) is None:
                items[item] = str(item_idx)
                item_count.setdefault(str(item_idx), 1)
                item_idx += 1
            else:
                item_count[items[item]] += 1
            history.setdefault(user, [])
            history[user].append([items[item], timestamp])
    num = 0
    cold_list = [[0, 0]]
    for key, value in item_count.items():
        if value < 20:
            num += 1
            cold_list.append([key, 1])
        else:
            cold_list.append([key, 0])
    print(f'[Steam] cold items (count < 20): {num}, total unique items: {len(item_count)}')
    pd.DataFrame(cold_list, columns=['item_id', 'is_cold']).to_csv(os.path.join(dst_path, 'cold_list.csv'), index=False)
    return
    with open(train_path, 'w') as f1, open(val_path, 'w') as f2, open(test_path, 'w') as f3:
        for user in users:
            hist = history[user]
            if len(hist) < 5:
                continue
            hist.sort(key=lambda x: x[1])
            for idx, value in enumerate(hist):
                if idx == len(hist) - 1:
                    f3.write(str(user_idx) + '\t' + value[0] + '\n')
                elif idx == len(hist) - 2:
                    f2.write(str(user_idx) + '\t' + value[0] + '\n')
                else:
                    f1.write(str(user_idx) + '\t' + value[0] + '\n')
            user_idx += 1
    with open(meta_path, 'w') as f:
        f.write(str(user_idx - 1) + '\t' + str(item_idx - 1))
    return (train_path, val_path, test_path, meta_path)

def split_seq_data(file_path):
    dst_path = os.path.dirname(file_path)
    train_path = os.path.join(dst_path, 'steam_seq_train.txt')
    val_path = os.path.join(dst_path, 'steam_seq_val.txt')
    test_path = os.path.join(dst_path, 'steam_seq_test.txt')
    meta_path = os.path.join(dst_path, 'steam_seq_meta.txt')
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        users, items = (set(), dict())
        user_idx, item_idx = (1, 1)
        history = {}
        for line in tqdm(lines):
            user = re.findall('u\\\'username\\\': u(\\"[^\\"]+\\"|\\\'[^\\\']+\\\')', line)[0]
            item = re.findall("u\\'product_id\\': u\\'([^\\']+?)\\'", line)[0]
            timestamp = re.findall("u\\'date\\': u\\'([^\\']+?)\\'", line)[0]
            users.add(user)
            if items.get(item) is None:
                items[item] = str(item_idx)
                item_idx += 1
            history.setdefault(user, [])
            history[user].append([items[item], timestamp])
    with open(train_path, 'w') as f1, open(val_path, 'w') as f2, open(test_path, 'w') as f3:
        for user in users:
            hist_u = history[user]
            if len(hist_u) < 5:
                continue
            hist_u.sort(key=lambda x: x[1])
            hist = [x[0] for x in hist_u]
            time = [x[1] for x in hist_u]
            f1.write(str(user_idx) + '\t' + ' '.join(hist[:-2]) + '\t' + ' '.join(time[:-2]) + '\n')
            f2.write(str(user_idx) + '\t' + ' '.join(hist[:-2]) + '\t' + ' '.join(time[:-2]) + '\t' + hist[-2] + '\n')
            f3.write(str(user_idx) + '\t' + ' '.join(hist[:-1]) + '\t' + ' '.join(time[:-1]) + '\t' + hist[-1] + '\n')
            user_idx += 1
    with open(meta_path, 'w') as f:
        f.write(str(user_idx - 1) + '\t' + str(item_idx - 1))
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
                time_seq = time_seq.strip('\n').split(' ')
                parsed_dt = [datetime.strptime(x, '%Y-%m-%d') for x in time_seq]
                day_seq = [dt.timetuple().tm_yday for dt in parsed_dt]
                hour_seq = [0 for _ in parsed_dt]
                dis_seq = np.array([0] + [(parsed_dt[i] - parsed_dt[i - 1]).days + 1 for i in range(1, len(parsed_dt))], dtype='int32')
                dis_seq = np.clip(dis_seq, 0, 500).tolist()
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
                parsed_dt = [datetime.strptime(x, '%Y-%m-%d') for x in time_seq]
                day_seq = [dt.timetuple().tm_yday for dt in parsed_dt]
                hour_seq = [0 for _ in parsed_dt]
                dis_seq = np.array([0] + [(parsed_dt[i] - parsed_dt[i - 1]).days + 1 for i in range(1, len(parsed_dt))], dtype='int32')
                dis_seq = np.clip(dis_seq, 0, 500).tolist()
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
