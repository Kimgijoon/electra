import os
import time
import json
import logging
import warnings
import datetime

import numpy as np
from absl import app, flags, logging as logger
from sklearn.model_selection import train_test_split

import utils.tf_util as tfu
from utils.data_util import DataHelper
from src.classifier import Classifier


FLAGS = flags.FLAGS

flags.DEFINE_string('op', None, '[REQUIRED] Operation code to do.')
flags.mark_flag_as_required('op')

flags.DEFINE_string('data_dir', 'data', 'Path to input directory')
flags.DEFINE_string('config_dir', 'configs', 'Directory of config file')
flags.DEFINE_string('ckpt_dir', 'checkpoints', 'Directory of model checkpoints')
flags.DEFINE_string('logs_dir', 'logs', 'Directory with tensorboard log')

flags.DEFINE_integer('seed_number', None, 'Controls the shuffling applied to the data before applying the split')
flags.DEFINE_integer('chunk_size', 10000, 'The number of tfrecord dataset i.e. raw data divice to tfrecord of size N')
flags.DEFINE_string('split_ratio', '0.2, 0.1', 'Valid, test ratio in dataset')
flags.DEFINE_bool('verbose', True, 'If True, prints a message to stdout for each update')

# Data on which to evaluate the loss
flags.DEFINE_integer('batch_size', 16, 'train batch size')
flags.DEFINE_integer('epochs', 100, 'epochs')
flags.DEFINE_float('lr', 2e-5, 'learning rate')
flags.DEFINE_bool('is_validation', True, 'Flag whether data exists to evaluate the loss and the metric')
flags.DEFINE_integer('gpus', None, 'number of on GPUs on which to create model replicas')
flags.DEFINE_bool('use_xla', None, 'Flag that determine whether to use XLA(Accelerated Linear Algebra)')


fmt = '[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s'
formatter = logging.Formatter(fmt)
logger.get_absl_handler().setFormatter(formatter)

warnings.filterwarnings('ignore')


def print_func(msg, verbose):
    
    if verbose:
        logger.info(msg)


def main(_):
    
    start_time = time.time()
    print_func(f'[{FLAGS.op}] operation start', FLAGS.verbose)
    
    if FLAGS.op == 'preprocess':
        with open(f'{FLAGS.config_dir}/model_config.json', 'r') as f:
            model_config = json.load(f)
        
        os.makedirs(f'{FLAGS.data_dir}/preprocess', exist_ok=True)
        
        # define data ratio (train, valid, test)
        valid_ratio, test_ratio = [float(x) for x in FLAGS.split_ratio.split(', ')]        
        
        # load dataset
        dh = DataHelper('base')
        df = dh.read_parquet(f'{FLAGS.data_dir}/raw')
        
        # split train/test set and save test set
        x, y = df['text'].tolist(), df['label'].tolist()
        x_train_val, x_test, y_train_val, y_test = train_test_split(x,
                                                                    y,
                                                                    test_size=test_ratio,
                                                                    shuffle=True,
                                                                    stratify=y,
                                                                    random_state=FLAGS.seed_number)
        x_train, x_val, y_train, y_val = train_test_split(x_train_val,
                                                          y_train_val,
                                                          test_size=valid_ratio,
                                                          shuffle=True,
                                                          stratify=y_train_val,
                                                          random_state=FLAGS.seed_number)
        
        with open(f'{FLAGS.data_dir}/preprocess/test.txt', 'w') as f:
            for i, j in zip(x_test, y_test):
                f.write(i + '\t' + str(j) + '\n')
        print_func('[preprocess] save test set finish', FLAGS.verbose)
        
        train_prep_list, train_label = [], []
        for i in range(len(x_train)):
            problem = dh.test(x_train[i])
            train_prep_list.append(problem)
            train_label.append(y_train[i])
        
        val_prep_list, val_label = [], []
        for i in range(len(x_val)):
            problem = dh.test(x_val[i])
            val_prep_list.append(problem)
            val_label.append(y_val[i])
        
        print_func('[preprocess] replacement finish', FLAGS.verbose)
        
        # tokenize preprocessed problem data for model training
        train_tokens_list, val_tokens_list = [], []
        for problem, label in zip(train_prep_list, train_label):
            dic = dh.tokenize(problem, model_config['seq_len'])
            train_tokens_list.append([dic['input_ids'], dic['token_type_ids'], dic['attention_mask'], label])            
        for problem, label in zip(val_prep_list, val_label):
            dic = dh.tokenize(problem, model_config['seq_len'])
            val_tokens_list.append([dic['input_ids'], dic['token_type_ids'], dic['attention_mask'], label])
        
        print_func('[preprocess] tokenization finish', FLAGS.verbose)
        
        # save train/valid set to tfrecord format
        train_set = np.array_split(train_tokens_list, round(len(train_tokens_list) / FLAGS.chunk_size))
        val_set = np.array_split(val_tokens_list, round(len(val_tokens_list) / FLAGS.chunk_size))
        tfu.save_tfrecords(train_set, f'{FLAGS.data_dir}/preprocess', 'train')
        tfu.save_tfrecords(val_set, f'{FLAGS.data_dir}/preprocess', 'validation')
        
        model_config['train'] = sum([x.shape[0] for x in train_set])
        model_config['valid'] = sum([x.shape[0] for x in val_set])
        
        with open(f'{FLAGS.config_dir}/model_config.json', 'w') as f:
            f.write(json.dumps(model_config, ensure_ascii=False))
        print_func('[preprocess] save tfrecord finish', FLAGS.verbose)
        
    elif FLAGS.op == 'train':
        with open(f'{FLAGS.config_dir}/model_config.json', 'r') as f:
            configs = json.load(f)
        with open(f'{FLAGS.data_dir}/raw/class_info.json', 'r') as f:
            labels = json.load(f)
        class_num = len(labels)
        configs.update({'class_num': class_num})
        
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        ckpt_dir = f'{FLAGS.ckpt_dir}/{current_time}'
        log_dir = f'{FLAGS.logs_dir}/{current_time}'
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        batch_size = FLAGS.batch_size * FLAGS.gpus
        
        trainset = tfu.get_features_from_tfrecord(f'{FLAGS.data_dir}/preprocess',
                                                  batch_size=batch_size,
                                                  max_seq_len=configs['seq_len'],
                                                  shuffle_size=configs['train'],
                                                  class_number=class_num,
                                                  data_type='train')

        # if validation set exist,
        if FLAGS.is_validation:
            validset = tfu.get_features_from_tfrecord(f'{FLAGS.data_dir}/preprocess',
                                                    batch_size=batch_size,
                                                    max_seq_len=configs['seq_len'],
                                                    shuffle_size=configs['valid'],
                                                    class_number=class_num,
                                                    data_type='validation')
            model = Classifier(configs,
                               batch_size=FLAGS.batch_size,
                               epochs=FLAGS.epochs,
                               lr=FLAGS.lr,
                               num_gpus=FLAGS.gpus,
                               ckpt_path=ckpt_dir,
                               is_validation=FLAGS.is_validation,
                               is_training=True)
            model.train(trainset, validset, log_dir, FLAGS.use_xla)
        else:
            model = Classifier(configs,
                               batch_size=FLAGS.batch_size,
                               epochs=FLAGS.epochs,
                               lr=FLAGS.lr,
                               num_gpus=FLAGS.gpus,
                               ckpt_path=ckpt_dir,
                               is_training=True)
            model.train(trainset, summeries_path=log_dir, use_xla=FLAGS.use_xla)
    elif FLAGS.op == 'predict':
        dh = DataHelper('base')
        
        with open(f'{FLAGS.config_dir}/model_config.json', 'r') as f:
            configs = json.load(f)
        with open(f'{FLAGS.data_dir}/raw/class_info.json', 'r') as f:
            labels = json.load(f)
        class_num = len(labels)
        configs.update({'class_num': class_num})        
        idx_to_labels = {labels[key]: key for key in labels}
        
        with open('data/preprocess/test.txt', 'r') as f:
            data = f.read().splitlines()
        
        problem = dh.tokenize(dh.test(data[0]), configs['seq_len'])
        ckpt_dir = f'{FLAGS.ckpt_dir}'
        model = Classifier(configs,
                           ckpt_path=ckpt_dir,
                           is_training=False)
        prob, result = model.predict(problem)
        
    else:
        logger.info('''Usage: python main.py --op=[OPERATION] --some argument=[...]''')
    
    print_func(f'[{FLAGS.op}] operation end', FLAGS.verbose)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print_func(f'[{FLAGS.op}] Elapsed time: {datetime.timedelta(seconds=elapsed_time)}', FLAGS.verbose)
    

if __name__ == '__main__':

    app.run(main)
