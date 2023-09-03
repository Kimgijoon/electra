from typing import List, Optional, Union
from collections import OrderedDict

import ray
import numpy as np
import tensorflow as tf


def _create_int_feature(values: List[int]) -> tf.train.Feature:
    """Convert data to int64 list
    
    Args:
        values (List[int]): Data
        
    Returns:
        tf.train.Feature: Int64 features
    """    
    if type(values) is list:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    else:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[values]))


def _to_example(row: np.ndarray) -> tf.train.Example:
    """Creates a tf.Example message ready to be written to a file
    
    Args:
        data_arr (np.ndarray): Features
        
    Returns:
        tf.train.Example: Features message
    """
    input_ids, token_type_ids, attention_mask, label = row
    
    tfr_features = OrderedDict()
    tfr_features['input_ids'] = _create_int_feature(input_ids)
    tfr_features['token_type_ids'] = _create_int_feature(token_type_ids)
    tfr_features['attention_mask'] = _create_int_feature(attention_mask)
    tfr_features['label'] = _create_int_feature(label)

    example = tf.train.Example(features=tf.train.Features(feature=tfr_features))
    return example


@ ray.remote
def save_tfrecord(chunk: np.ndarray, path: str):
    """Create the history data of student to a TFRecord file
    
    Args:
        chunk (np.ndarray): Input dataset per chunk
        path (str): Filename of the dataset to be saved
    """    
    with tf.io.TFRecordWriter(path) as writer:
        for row in chunk:
            example = _to_example(row)
            writer.write(example.SerializeToString())


def save_tfrecords(chunks: List[np.ndarray],
                   output_path: str,
                   data_type: str):
    """Create the tfrecord files for a dataset using parallel

    Args:
        chunks (List[np.ndarray]): full dataset
        output_path (str): directory path
        data_type (str): flag that distinguishes whether the data is for train or valid
    """
    futures = [save_tfrecord.remote(chunk, f'{output_path}/{data_type}_{i}.tfrecords')
                for i, chunk in enumerate(chunks)]
    ray.get(futures)


def get_features_from_tfrecord(data_dir: str,
                               batch_size: int,
                               max_seq_len: int,
                               shuffle_size: int,
                               class_number: int,
                               seed_number: Optional[int]=None,
                               data_type: str='train') -> tf.data.Dataset:
    """Load tfrecord, parse features in the tfrecord and return TFRecordDataset class
    i.e. this function is input pipelines

    Args:
        data_dir (str): directory related to tfrecord files
        batch_size (int): number of samples per gradient update
        max_seq_len (int): maximum length of all sequences
        data_type (str): flag that distinguishes whether the data is for train or valid

    Returns:
        tf.data.Dataset: parsed TFRecordDataset
    """
    name_to_features = {
        'input_ids': tf.io.FixedLenFeature([max_seq_len], tf.int64),
        'token_type_ids': tf.io.FixedLenFeature([max_seq_len], tf.int64),
        'attention_mask': tf.io.FixedLenFeature([max_seq_len], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }

    def _parser(tfrecord):
        example = tf.io.parse_single_example(tfrecord, name_to_features)
        label = tf.one_hot(example.pop('label'), class_number)
        return example, label

    dataset = tf.data.Dataset.list_files(f'{data_dir}/{data_type}*')
    dataset = dataset.interleave(lambda x: tf.data.TFRecordDataset(x),
                                 cycle_length=4,
                                 num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(_parser, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=shuffle_size, seed=seed_number)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

