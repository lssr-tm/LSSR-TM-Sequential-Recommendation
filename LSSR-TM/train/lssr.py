import os
import sys
import gc


def _get_arg_value(argv, name, default=None):
    if name in argv:
        idx = argv.index(name)
        if idx + 1 < len(argv):
            return argv[idx + 1]
        return default
    prefix = name + '='
    for arg in argv:
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return default


_cli_gpu = _get_arg_value(sys.argv, '--gpu', None)
if _cli_gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = _cli_gpu

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')
os.environ.pop('TF_GPU_ALLOCATOR', None)


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import argparse
import csv
import random
from time import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

from reclearn.data.feature_column import sparseFeature
from reclearn.evaluator import eval_pos_neg

try:
    from model import LSSR
except ImportError:
    from LSSR.model import LSSR
from datasets import beauty, games, movielens as ml, steam, xlong


NEG_NUM = 4
EMBED_DIM = 128
LEARNING_RATE = 0.0005
EPOCHS = 40
BATCH_SIZE = 512
EVAL_BATCH_SIZE = 128
TOPK = 10
TEST_NEG_NUM = 100
X_LONG_PROTOCOL_MARKER = 'xlong_public_v2_user_latest_no_timestamp'

BASE_MODEL_PARAMS = {
    'blocks': 2,
    'num_heads': 1,
    'ffn_hidden_unit': 64,
    'num_expert': 3,
    'expert_units': [128, 128],
    'dnn_dropout': 0.2,
    'layer_norm_eps': 1e-6,
    'use_l2norm': False,
    'loss_name': 'binary_cross_entropy',
    'embed_reg': 1e-6,
}


DATASET_CONFIG = {
    'ml-1m': {
        'module': ml,
        'default_raw_path': './data/ratings.dat',
        'seq_len': 100,
        'seq_files': ('ml_seq_train.txt', 'ml_seq_val.txt', 'ml_seq_test.txt', 'ml_seq_meta.txt')
    },
    'beauty': {
        'module': beauty,
        'default_raw_path': './data/ratings_Beauty.csv',
        'seq_len': 50,
        'seq_files': ('beauty_seq_train.txt', 'beauty_seq_val.txt', 'beauty_seq_test.txt', 'beauty_seq_meta.txt')
    },
    'games': {
        'module': games,
        'default_raw_path': './data/ratings_Video_Games.csv',
        'seq_len': 50,
        'seq_files': ('games_seq_train.txt', 'games_seq_val.txt', 'games_seq_test.txt', 'games_seq_meta.txt')
    },
    'steam': {
        'module': steam,
        'default_raw_path': './data/steam_reviews.json',
        'seq_len': 50,
        'seq_files': ('steam_seq_train.txt', 'steam_seq_val.txt', 'steam_seq_test.txt', 'steam_seq_meta.txt')
    },
    'xlong': {
        'module': xlong,
        'default_raw_path': './data/xlong/train_corpus_total_dual.txt',
        'seq_len': 500,
        'seq_files': ('xlong_seq_train.txt', 'xlong_seq_val.txt', 'xlong_seq_test.txt', 'xlong_seq_meta.txt')
    }
}

ABLATIONS = {
    'full': {
        'use_timestamp': True,
        'use_moe': True,
        'use_gate_unit': True,
        'use_abs_time': True,
        'use_st2lt_transfer': True
    },
    'wo_timestamp': {
        'use_timestamp': False,
        'use_moe': True,
        'use_gate_unit': True,
        'use_abs_time': False,
        'use_st2lt_transfer': True
    },
    'wo_moe': {
        'use_timestamp': True,
        'use_moe': False,
        'use_gate_unit': True,
        'use_abs_time': True,
        'use_st2lt_transfer': True
    },
    'wo_gate': {
        'use_timestamp': True,
        'use_moe': True,
        'use_gate_unit': False,
        'use_abs_time': True,
        'use_st2lt_transfer': True
    },
    'wo_st2lt': {
        'use_timestamp': True,
        'use_moe': True,
        'use_gate_unit': True,
        'use_abs_time': True,
        'use_st2lt_transfer': False
    }
}

EXP_PLAN_A = [
    ('ml-1m', 'wo_gate'),
    ('beauty', 'wo_gate'),
    ('steam', 'wo_timestamp'),
    ('steam', 'wo_moe'),
    ('steam', 'wo_gate'),
    ('games', 'wo_timestamp'),
    ('games', 'wo_moe'),
    ('games', 'wo_gate'),
    ('xlong', 'full'),
    ('xlong', 'wo_moe'),
    ('xlong', 'wo_gate'),
]


EXP_PLAN_B = [
    ('beauty', 'wo_st2lt'),
    ('steam', 'wo_st2lt'),
    ('xlong', 'wo_st2lt'),
]


def configure_gpu_memory():
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print('[GPU] No visible GPU detected. Running on CPU.')
        return

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print(f'[GPU] set_memory_growth failed on {gpu}: {e}')

    logical_gpus = tf.config.list_logical_devices('GPU')
    print(f'[GPU] physical={len(gpus)}, logical={len(logical_gpus)}, visible={os.environ.get("CUDA_VISIBLE_DEVICES", "(inherit)")}')


def cleanup_memory(*objs):
    for obj in objs:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    try:
        K.clear_session()
    except Exception:
        pass
    gc.collect()


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_seq_paths(dataset_name, raw_path):
    cfg = DATASET_CONFIG[dataset_name]
    data_dir = os.path.dirname(raw_path)
    filenames = cfg['seq_files']
    return tuple(os.path.join(data_dir, name) for name in filenames)


def _should_resplit_xlong(train_path, val_path, test_path, meta_path):
    required = [train_path, val_path, test_path, meta_path]
    if not all(os.path.exists(path) for path in required):
        return True
    try:
        with open(meta_path, 'r', encoding='utf8') as f:
            meta_text = f.read()
        return X_LONG_PROTOCOL_MARKER not in meta_text
    except Exception:
        return True


def prepare_seq_files(dataset_name, raw_path=None, force_resplit=False):
    cfg = DATASET_CONFIG[dataset_name]
    module = cfg['module']
    raw_path = raw_path or cfg['default_raw_path']
    train_path, val_path, test_path, meta_path = get_seq_paths(dataset_name, raw_path)

    if dataset_name == 'xlong':
        need_resplit = force_resplit or _should_resplit_xlong(train_path, val_path, test_path, meta_path)
    else:
        need_resplit = force_resplit or (not all(os.path.exists(path) for path in [train_path, val_path, test_path, meta_path]))

    if not need_resplit:
        return train_path, val_path, test_path, meta_path

    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f'Raw file not found: {raw_path}. '
            f'You can either place the raw file there, or prepare the four sequence files first: '
            f'{train_path}, {val_path}, {test_path}, {meta_path}.'
        )

    return module.split_seq_data(file_path=raw_path)


def load_data_for_dataset(dataset_name, train_path, val_path, test_path, seq_len, max_item_num):
    module = DATASET_CONFIG[dataset_name]['module']

    train_data = module.load_seq_data(
        train_path, 'train', seq_len, NEG_NUM, max_item_num,
        contain_user=True, contain_time=True
    )


    val_fit_data = module.load_seq_data(
        val_path, 'val', seq_len, NEG_NUM, max_item_num,
        contain_user=True, contain_time=True
    )


    val_eval_data = module.load_seq_data(
        val_path, 'val', seq_len, TEST_NEG_NUM, max_item_num,
        contain_user=True, contain_time=True
    )


    test_data = module.load_seq_data(
        test_path, 'test', seq_len, TEST_NEG_NUM, max_item_num,
        contain_user=True, contain_time=True
    )

    return train_data, val_fit_data, val_eval_data, test_data


def append_row(csv_path, fieldnames, row):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return float('nan')


def _shrink_array(arr):
    arr = np.asarray(arr)
    if arr.dtype == np.int64:
        return arr.astype(np.int32, copy=False)
    if arr.dtype == np.float64:
        return arr.astype(np.float32, copy=False)
    return arr


def _sanitize_data(data):
    if isinstance(data, dict):
        return {k: _sanitize_data(v) for k, v in data.items()}
    if isinstance(data, (tuple, list)):
        return type(data)(_sanitize_data(v) for v in data)
    return _shrink_array(data)


def _infer_num_samples(data):
    if isinstance(data, dict):
        first = next(iter(data.values()))
        return len(first)
    if isinstance(data, (tuple, list)):
        return _infer_num_samples(data[0])
    return len(data)


def build_tf_dataset(data, batch_size, shuffle=False, shuffle_buffer=10000, prefetch_batches=1):
    data = _sanitize_data(data)
    options = tf.data.Options()
    options.experimental_deterministic = True

    if isinstance(data, tuple) and len(data) == 2:
        dataset = tf.data.Dataset.from_tensor_slices((data[0], data[1]))
        num_samples = _infer_num_samples(data[0])
    else:
        dataset = tf.data.Dataset.from_tensor_slices(data)
        num_samples = _infer_num_samples(data)

    dataset = dataset.with_options(options)

    if shuffle:
        buffer_size = min(max(1, num_samples), shuffle_buffer)
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=2026, reshuffle_each_iteration=True)

    dataset = dataset.batch(batch_size, drop_remainder=False)
    if prefetch_batches and prefetch_batches > 0:
        dataset = dataset.prefetch(prefetch_batches)
    return dataset


def maybe_print_memory(tag=''):
    try:
        info = tf.config.experimental.get_memory_info('GPU:0')
        current_gb = info.get('current', 0) / (1024 ** 3)
        peak_gb = info.get('peak', 0) / (1024 ** 3)
        print(f'[GPU-MEM] {tag} current={current_gb:.2f} GB, peak={peak_gb:.2f} GB')
    except Exception:
        pass


def validate_experiment_request(dataset_name, ablation_name):
    if dataset_name == 'xlong' and ablation_name == 'wo_timestamp':
        raise ValueError(
            'Public XLong release does not provide real timestamps. '
            'Please run xlong with --ablation full / wo_moe / wo_gate / wo_st2lt only. '
            'In the XLong setting, all valid variants automatically disable timestamp inputs.'
        )


def _print_ablation_hint(ablation_name):
    if ablation_name == 'wo_gate':
        print('[w/o ADP gate] remove the candidate-aware adaptive fusion gate and replace it with fixed equal-weight fusion.')
    elif ablation_name == 'wo_st2lt':
        print('[w/o ST->LT transfer] long-term branch trained with p_long = e_u alone (dim d).')
        print('[w/o ST->LT transfer] uses a completely independent MoE module with no structural dependency on short-term output.')


def run_one_experiment(dataset_name, ablation_name, args):
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    if ablation_name not in ABLATIONS:
        raise ValueError(f'Unknown ablation: {ablation_name}')
    validate_experiment_request(dataset_name, ablation_name)

    cleanup_memory()
    seed_everything(args.seed)

    cfg = DATASET_CONFIG[dataset_name]
    seq_len = args.seq_len if args.seq_len is not None else cfg['seq_len']
    raw_path = args.raw_path if args.raw_path else cfg['default_raw_path']

    print('=' * 80)
    print(f'Dataset         : {dataset_name}')
    print(f'Ablation        : {ablation_name}')
    print(f'Raw path        : {raw_path}')
    print(f'Seq len         : {seq_len}')
    print(f'Seed            : {args.seed}')
    print(f'Train batch     : {args.batch_size}')
    print(f'Eval batch      : {args.eval_batch_size}')
    print(f'Learning rate   : {args.learning_rate}')
    print(f'Clipnorm        : {args.clipnorm}')
    print(f'Loss            : {args.loss_name}')
    print(f'Use l2norm      : {args.use_l2norm}')
    print(f'Embed reg       : {args.embed_reg}')
    print(f'Use tf.data     : {args.use_tfdata}')
    if dataset_name == 'xlong':
        print('[XLong] public release has no real timestamps; timestamp branch will be disabled automatically.')
        print('[XLong] validation is derived from public train by holding out the latest available training instance per user.')
    _print_ablation_hint(ablation_name)
    print('=' * 80)

    model = None
    train_data = val_fit_data = val_eval_data = test_data = None
    train_dataset = val_dataset = None

    try:
        train_path, val_path, test_path, meta_path = prepare_seq_files(
            dataset_name=dataset_name,
            raw_path=raw_path,
            force_resplit=args.force_resplit
        )

        with open(meta_path, 'r', encoding='utf8') as f:
            max_user_num, max_item_num = [int(x) for x in f.readline().strip('\n').split('\t')]

        fea_cols = {
            'item': sparseFeature('item', max_item_num + 1, EMBED_DIM),
            'user': sparseFeature('user', max_user_num + 1, EMBED_DIM)
        }

        train_data, val_fit_data, val_eval_data, test_data = load_data_for_dataset(
            dataset_name, train_path, val_path, test_path, seq_len, max_item_num
        )

        if args.use_tfdata:
            train_dataset = build_tf_dataset(
                train_data,
                batch_size=args.batch_size,
                shuffle=True,
                shuffle_buffer=args.shuffle_buffer,
                prefetch_batches=args.prefetch_batches
            )
            val_dataset = build_tf_dataset(
                val_fit_data,
                batch_size=args.batch_size,
                shuffle=False,
                prefetch_batches=args.prefetch_batches
            )


            val_eval_data = _sanitize_data(val_eval_data)
            test_data = _sanitize_data(test_data)
        else:
            train_data = _sanitize_data(train_data)
            val_fit_data = _sanitize_data(val_fit_data)
            val_eval_data = _sanitize_data(val_eval_data)
            test_data = _sanitize_data(test_data)

        model_params = dict(BASE_MODEL_PARAMS)
        model_params['seq_len'] = seq_len
        model_params['use_l2norm'] = bool(args.use_l2norm)
        model_params['loss_name'] = args.loss_name
        model_params['embed_reg'] = args.embed_reg
        model_params['shared_gate_init'] = args.shared_gate_init
        model_params.update(ABLATIONS[ablation_name])

        if dataset_name == 'xlong':
            model_params['use_timestamp'] = False
            model_params['use_abs_time'] = False

        print('Model switches   : ' + ', '.join(f'{k}={v}' for k, v in model_params.items() if k.startswith('use_')))

        model = LSSR(fea_cols, seed=args.seed, **model_params)
        optimizer = Adam(learning_rate=args.learning_rate, clipnorm=args.clipnorm) if args.clipnorm > 0 else Adam(learning_rate=args.learning_rate)
        model.compile(optimizer=optimizer)

        os.makedirs(args.results_dir, exist_ok=True)
        history_csv = os.path.join(args.results_dir, f'{dataset_name}_{ablation_name}_seed{args.seed}_history_valselect.csv')
        summary_csv = os.path.join(args.results_dir, 'ablation_summary_valselect.csv')
        best_ckpt = os.path.join(args.results_dir, f'best_{dataset_name}_{ablation_name}_seed{args.seed}.weights.h5')

        best_epoch = 0
        best_val_metrics = {'hr': -1.0, 'mrr': -1.0, 'ndcg': -1.0}
        best_test_metrics = {'hr': -1.0, 'mrr': -1.0, 'ndcg': -1.0}

        no_improve = 0

        for epoch in range(1, args.epochs + 1):
            t1 = time()

            if args.use_tfdata:
                history = model.fit(
                    train_dataset,
                    epochs=1,
                    validation_data=val_dataset,
                    verbose=args.fit_verbose
                )
            else:
                history = model.fit(
                    x=train_data,
                    epochs=1,
                    validation_data=val_fit_data,
                    batch_size=args.batch_size,
                    verbose=args.fit_verbose
                )

            t2 = time()

            train_loss = _safe_float(history.history.get('loss', [float('nan')])[0])
            val_loss = _safe_float(history.history.get('val_loss', [float('nan')])[0])

            if not np.isfinite(train_loss) or not np.isfinite(val_loss):
                print(f'[STOP] Non-finite loss detected at epoch {epoch}: loss={train_loss}, val_loss={val_loss}')
                break


            eval_dict = eval_pos_neg(
                model,
                val_eval_data,
                ['hr', 'mrr', 'ndcg'],
                TOPK,
                args.eval_batch_size
            )

            fit_time = t2 - t1
            eval_time = time() - t2

            hr = _safe_float(eval_dict.get('hr'))
            mrr = _safe_float(eval_dict.get('mrr'))
            ndcg = _safe_float(eval_dict.get('ndcg'))

            if not np.all(np.isfinite([hr, mrr, ndcg])):
                print(f'[STOP] Non-finite validation metrics detected at epoch {epoch}: {eval_dict}')
                break

            print(
                'Iteration %d Fit [%.1f s], ValEval [%.1f s]: loss = %.6f, val_loss = %.6f, ValHR = %.4f, ValMRR = %.4f, ValNDCG = %.4f'
                % (epoch, fit_time, eval_time, train_loss, val_loss, hr, mrr, ndcg)
            )
            maybe_print_memory(f'after epoch {epoch}')

            append_row(
                history_csv,
                [
                    'dataset', 'ablation', 'seed', 'epoch',
                    'loss', 'val_loss',
                    'val_hr', 'val_mrr', 'val_ndcg',
                    'fit_time', 'eval_time',
                    'use_timestamp', 'use_moe', 'use_gate_unit', 'use_st2lt_transfer'
                ],
                {
                    'dataset': dataset_name,
                    'ablation': ablation_name,
                    'seed': args.seed,
                    'epoch': epoch,
                    'loss': train_loss,
                    'val_loss': val_loss,
                    'val_hr': hr,
                    'val_mrr': mrr,
                    'val_ndcg': ndcg,
                    'fit_time': fit_time,
                    'eval_time': eval_time,
                    'use_timestamp': model_params.get('use_timestamp'),
                    'use_moe': model_params.get('use_moe'),
                    'use_gate_unit': model_params.get('use_gate_unit'),
                    'use_st2lt_transfer': model_params.get('use_st2lt_transfer'),
                }
            )


            if ndcg > best_val_metrics['ndcg']:
                best_epoch = epoch
                best_val_metrics = {
                    'hr': hr,
                    'mrr': mrr,
                    'ndcg': ndcg
                }
                model.save_weights(best_ckpt)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f'[Early Stop] No improvement on validation NDCG for {args.patience} epochs. Stop at epoch {epoch}.')
                    break


        if best_epoch > 0 and os.path.exists(best_ckpt):
            model.load_weights(best_ckpt)
            test_eval = eval_pos_neg(
                model,
                test_data,
                ['hr', 'mrr', 'ndcg'],
                TOPK,
                args.eval_batch_size
            )

            best_test_metrics = {
                'hr': _safe_float(test_eval.get('hr')),
                'mrr': _safe_float(test_eval.get('mrr')),
                'ndcg': _safe_float(test_eval.get('ndcg'))
            }
        else:
            print('[WARN] No best checkpoint found. Test metrics are invalid.')

        append_row(
            summary_csv,
            [
                'dataset', 'ablation', 'seed', 'best_epoch',
                'val_hr', 'val_mrr', 'val_ndcg',
                'test_hr', 'test_mrr', 'test_ndcg',
                'seq_len', 'raw_path', 'loss_name', 'use_l2norm', 'embed_reg',
                'train_batch', 'eval_batch',
                'use_timestamp', 'use_moe', 'use_gate_unit', 'use_st2lt_transfer'
            ],
            {
                'dataset': dataset_name,
                'ablation': ablation_name,
                'seed': args.seed,
                'best_epoch': best_epoch,
                'val_hr': best_val_metrics['hr'],
                'val_mrr': best_val_metrics['mrr'],
                'val_ndcg': best_val_metrics['ndcg'],
                'test_hr': best_test_metrics['hr'],
                'test_mrr': best_test_metrics['mrr'],
                'test_ndcg': best_test_metrics['ndcg'],
                'seq_len': seq_len,
                'raw_path': raw_path,
                'loss_name': args.loss_name,
                'use_l2norm': bool(args.use_l2norm),
                'embed_reg': args.embed_reg,
                'train_batch': args.batch_size,
                'eval_batch': args.eval_batch_size,
                'use_timestamp': model_params.get('use_timestamp'),
                'use_moe': model_params.get('use_moe'),
                'use_gate_unit': model_params.get('use_gate_unit'),
                'use_st2lt_transfer': model_params.get('use_st2lt_transfer'),
            }
        )

        print(
            'Best Validation -> epoch=%d, ValHR=%.4f, ValMRR=%.4f, ValNDCG=%.4f'
            % (best_epoch, best_val_metrics['hr'], best_val_metrics['mrr'], best_val_metrics['ndcg'])
        )
        print(
            'Final Test      -> HR=%.4f, MRR=%.4f, NDCG=%.4f'
            % (best_test_metrics['hr'], best_test_metrics['mrr'], best_test_metrics['ndcg'])
        )

    finally:
        cleanup_memory(model, train_data, val_fit_data, val_eval_data, test_data, train_dataset, val_dataset)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run LSSR/LSSR-TM ablation experiments.'
    )
    parser.add_argument('--dataset', type=str, default=None, choices=list(DATASET_CONFIG.keys()))
    parser.add_argument('--ablation', type=str, default=None, choices=list(ABLATIONS.keys()))
    parser.add_argument('--run_plan', type=str, default=None, choices=['plan_a', 'plan_b'])
    parser.add_argument('--raw_path', type=str, default=None,
                        help='Optional raw data path. When omitted, the dataset default path is used.')
    parser.add_argument('--seq_len', type=int, default=None,
                        help='Override default sequence length of the selected dataset.')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--eval_batch_size', type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE)
    parser.add_argument('--clipnorm', type=float, default=1.0,
                        help='Gradient clipping norm. Set 0 to disable.')
    parser.add_argument('--loss_name', type=str, default='binary_cross_entropy', choices=['bpr_loss', 'hinge_loss', 'binary_cross_entropy'])
    parser.add_argument('--use_l2norm', type=int, default=0, choices=[0, 1])
    parser.add_argument('--embed_reg', type=float, default=0.0)
    parser.add_argument('--shared_gate_init', type=float, default=0.5,
                        help='Compatibility-only argument; ignored when wo_gate removes the whole gate unit.')
    parser.add_argument('--use_tfdata', type=int, default=1, choices=[0, 1])
    parser.add_argument('--shuffle_buffer', type=int, default=10000)
    parser.add_argument('--prefetch_batches', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--gpu', type=str, default=None,
                        help='Example: "0" or "0,1". If omitted, current environment is kept.')
    parser.add_argument('--results_dir', type=str, default='./results/lssr_ablation')
    parser.add_argument('--force_resplit', action='store_true')
    parser.add_argument('--fit_verbose', type=int, default=1)
    parser.add_argument('--patience', type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    configure_gpu_memory()

    if args.run_plan == 'plan_a':
        experiment_list = EXP_PLAN_A
    elif args.run_plan == 'plan_b':
        experiment_list = EXP_PLAN_B
    else:
        if args.dataset is None or args.ablation is None:
            raise ValueError('Please provide both --dataset and --ablation, or use --run_plan plan_a / --run_plan plan_b.')
        experiment_list = [(args.dataset, args.ablation)]

    for dataset_name, ablation_name in experiment_list:
        run_one_experiment(dataset_name, ablation_name, args)


if __name__ == '__main__':
    main()
