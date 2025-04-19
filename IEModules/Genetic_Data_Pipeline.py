import time
import sys
import os
import glob
import math
import threading
import concurrent.futures as cf
import random
import re
import gc
import shutil

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Input, Model, layers, metrics, losses, callbacks, optimizers, models, utils
from keras import backend as K
import keras_tuner as kt
from pyfaidx import Fasta

from .config import DATA_DIR, LOG_DIR, MODEL_DIR, MODULE_DIR, NOTEBOOK_DIR

# Global paths (adjust as needed)
DATA_DIR = "../Datasets/"
MODEL_DIR = "../Models/"

# Clear Keras session and collect garbage
K.clear_session()
gc.collect()

def test_function(variable):
    print(f"You entered {variable} into test_function")
    return variable
###########################################
# FASTA File Processing
###########################################

def trim_chr_genome(input_fasta, output_fasta):
    """
    Writes out a new FASTA that uses only the full sequences of chromosomes 1-22, X and Y.
    """
    valid_chroms = set([f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"])
    keep = False
    with open(input_fasta, "r") as fin, open(output_fasta, "w") as fout:
        for line in fin:
            if line.startswith(">"):
                chrom_name = line.strip()[1:].split()[0]
                keep = (chrom_name in valid_chroms)
                if keep:
                    fout.write(line)
                    print(chrom_name)
            else:
                if keep:
                    fout.write(line)


###########################################
# GTF Annotation Functions
###########################################

def load_gtf_annotations(gtf_file):
    """
    Loads a GTF file into a pandas DataFrame and converts the start coordinate (cstart) to 0-based indexing.
    """
    gtf_data = pd.read_csv(
        gtf_file, sep='\t', comment='#', header=None,
        names=['seqname', 'source', 'feature', 'cstart', 'cend',
               'score', 'strand', 'frame', 'attribute']
    )
    gtf_data['cstart'] = gtf_data['cstart'] - 1
    return gtf_data


def search_gtf_by_range(gtf_df, seqname, pos_min, pos_max, require_both=False):
    """
    Search a GTF DataFrame for rows with the given sequence name and with cstart and/or cend values within a range.
    """
    df = gtf_df[gtf_df['seqname'] == seqname]
    if require_both:
        condition = (
            (df['cstart'] >= pos_min) & (df['cstart'] <= pos_max) &
            (df['cend']   >= pos_min) & (df['cend']   <= pos_max)
        )
    else:
        condition = (
            ((df['cstart'] >= pos_min) & (df['cstart'] <= pos_max)) |
            ((df['cend']   >= pos_min) & (df['cend']   <= pos_max))
        )
    return df[condition]


def calculate_introns(gtf_df):
    """
    For each gene (grouped by gene_id in the attribute field), merge overlapping exons and then compute each intron as
    the gap between consecutive merged exons.
    """
    def get_gene_id(attr):
        m = re.search(r'gene_id\s+"([^"]+)"', attr)
        return m.group(1) if m else None

    if 'gene_id' not in gtf_df.columns:
        gtf_df = gtf_df.copy()
        gtf_df['gene_id'] = gtf_df['attribute'].apply(get_gene_id)
    
    intron_records = []
    for gene_id, group in gtf_df.groupby('gene_id'):
        if gene_id is None:
            continue

        # Use gene record if available; otherwise, default to first record in group.
        gene_rows = group[group['feature'] == 'gene']
        if not gene_rows.empty:
            seqname = gene_rows.iloc[0]['seqname']
            strand  = gene_rows.iloc[0]['strand']
        else:
            seqname = group.iloc[0]['seqname']
            strand  = group.iloc[0]['strand']

        exon_rows = group[group['feature'] == 'exon']
        if exon_rows.empty:
            continue

        exon_intervals = sorted(list(zip(exon_rows['cstart'], exon_rows['cend'])), key=lambda x: x[0])

        merged_exons = []
        for interval in exon_intervals:
            if not merged_exons:
                merged_exons.append(list(interval))
            else:
                last = merged_exons[-1]
                if interval[0] <= last[1]:
                    last[1] = max(last[1], interval[1])
                else:
                    merged_exons.append(list(interval))
        if len(merged_exons) < 2:
            continue

        for i in range(len(merged_exons) - 1):
            intron_start = merged_exons[i][1]
            intron_end   = merged_exons[i+1][0]
            if intron_end > intron_start:
                intron_records.append({
                    'seqname': seqname,
                    'feature': 'intron',
                    'cstart': intron_start,
                    'cend': intron_end,
                    'strand': strand
                })

    return pd.DataFrame(intron_records)


def swap_columns_if_needed(df, col_a, col_b):
    """
    If values in col_a are greater than col_b, swap them.
    """
    mask = df[col_a] > df[col_b]
    df.loc[mask, [col_a, col_b]] = df.loc[mask, [col_b, col_a]].values
    return df


###########################################
# TFRecord & FASTA Data Processing
###########################################

def compute_chunk_indices(fasta_file, chunk_size, shifts=[0]):
    """
    Create a list of (record_id, cstart, cend) tuples for each chunk in the FASTA, with optional shifts for data augmentation.
    """
    print('Running compute_chunk_indices with data augmentation')
    fa = Fasta(fasta_file)
    chunk_indices = []
    for record_id in fa.keys():
        seq_len = len(fa[record_id])
        for shift in shifts:
            for cstart in range(shift, seq_len, chunk_size):
                cend = min(cstart + chunk_size, seq_len)
                chunk_indices.append((record_id, cstart, cend))
    return chunk_indices


def one_hot_encode_reference(sequence):
    """
    One-hot encodes a DNA sequence.
    """
    n_base_encoder = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
        'N': [0, 0, 0, 0],
    }
    return [n_base_encoder.get(nuc, [0, 0, 0, 0]) for nuc in sequence]


def label_sequence_local(sequence_length, annotations, window=50):
    """
    Generates a label matrix for a sequence, assigning partial credit to bases near an annotation.
    """
    exon_cstart_binary   = np.zeros(sequence_length)
    exon_cend_binary     = np.zeros(sequence_length)
    intron_cstart_binary = np.zeros(sequence_length)
    intron_cend_binary   = np.zeros(sequence_length)

    for _, row in annotations.iterrows():
        cs = int(row['cstart'])
        ce = int(row['cend'])
        feat = row['feature'].strip().lower()
        if feat == 'exon':
            if 0 <= cs < sequence_length:
                exon_cstart_binary[cs] = 1
            if 0 <= ce - 1 < sequence_length:
                exon_cend_binary[ce - 1] = 1
        elif feat == 'intron':
            if 0 <= cs < sequence_length:
                intron_cstart_binary[cs] = 1
            if 0 <= ce - 1 < sequence_length:
                intron_cend_binary[ce - 1] = 1

    def smooth_binary(binary_arr, window):
        L = len(binary_arr)
        smooth_arr = np.zeros(L)
        annotation_indices = np.where(binary_arr == 1)[0]
        for idx in annotation_indices:
            smooth_arr[idx] = 1.0
            for d in range(1, window + 1):
                credit = 0.5 - (0.5 / window) * (d - 1)
                credit = max(credit, 0)
                left = idx - d
                right = idx + d
                if left >= 0:
                    smooth_arr[left] = max(smooth_arr[left], credit)
                if right < L:
                    smooth_arr[right] = max(smooth_arr[right], credit)
        return smooth_arr

    exon_cstart_smooth   = smooth_binary(exon_cstart_binary, window)
    exon_cend_smooth     = smooth_binary(exon_cend_binary, window)
    intron_cstart_smooth = smooth_binary(intron_cstart_binary, window)
    intron_cend_smooth   = smooth_binary(intron_cend_binary, window)

    labels = np.zeros((sequence_length, 5))
    labels[:, 1] = intron_cstart_smooth
    labels[:, 2] = intron_cend_smooth
    labels[:, 3] = exon_cstart_smooth
    labels[:, 4] = exon_cend_smooth

    max_annotation = np.max(labels[:, 1:], axis=1)
    labels[:, 0] = np.where(max_annotation == 1, 0, 1)

    return labels


def pad_labels(labels, target_length):
    """
    Pads the label matrix to the target length.
    """
    current_length = len(labels)
    if current_length < target_length:
        pad_length = target_length - current_length
        pad_array = np.tile(np.array([[1, 0, 0, 0, 0]]), (pad_length, 1))
        labels = np.concatenate([labels, pad_array], axis=0)
        labels = labels.tolist()
    return labels


def pad_encoded_seq(encoded_seq, target_length):
    """
    Pads an encoded sequence up to the target length.
    """
    seq_len = len(encoded_seq)
    pad_size = target_length - seq_len
    if pad_size > 0:
        encoded_seq += [[0, 0, 0, 0, 0]] * pad_size
    return encoded_seq


def build_chunk_data_for_indices(fasta_file, gtf_df, subset_indices, skip_empty=True, chunk_size=5000):
    """
    For each chunk defined by subset_indices, extracts the reference sequence,
    encodes it, labels it using GTF annotations, and yields the data.
    """
    print('Running build_chunk_data_for_indices')
    fa = Fasta(fasta_file)
    # Pre-group GTF data by (seqname, strand)
    grouped_gtf = {}
    for (seqname, strand), sub_df in gtf_df.groupby(['seqname', 'strand']):
        grouped_gtf[(seqname, strand)] = sub_df

    for (record_id, cstart, cend) in subset_indices:
        seq = str(fa[record_id][cstart:cend])
        base_encoded_4 = one_hot_encode_reference(seq)
        chunk_len = len(base_encoded_4)
        for strand_symbol in ['+', '-']:
            strand_flag = 1 if strand_symbol == '+' else 0
            encoded_seq_5 = [row + [strand_flag] for row in base_encoded_4]

            if (record_id, strand_symbol) not in grouped_gtf:
                labels = [[1, 0, 0, 0, 0]] * chunk_len
            else:
                sub_df = grouped_gtf[(record_id, strand_symbol)]
                overlap = sub_df[(sub_df['cstart'] < cend) & (sub_df['cend'] > cstart)].copy()
                if len(overlap) == 0:
                    labels = [[1, 0, 0, 0, 0]] * chunk_len
                else:
                    overlap['cstart'] = overlap['cstart'] - cstart
                    overlap['cend']   = overlap['cend']   - cstart
                    labels = label_sequence_local(chunk_len, overlap, window=100)
                    labels = labels.tolist()

            if skip_empty and all(lbl == [1, 0, 0, 0, 0] for lbl in labels):
                continue

            encoded_seq_5 = pad_encoded_seq(encoded_seq_5, chunk_size)
            labels = pad_labels(labels, chunk_size)

            X = np.array(encoded_seq_5, dtype=np.float32)
            y = np.array(labels, dtype=np.float32)
            yield (X, y, record_id, cstart, cend, strand_symbol, chunk_size)


###########################################
# TFRecord Serialization
###########################################

def float_feature_list(value_list):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value_list))


def int_feature_list(value_list):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value_list))


def bytes_feature(value):
    if isinstance(value, str):
        value = value.encode('utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_chunk_example(X, y, record_id, cstart, cend, strand_symbol, chunk_size=5000):
    """
    Serializes a single data chunk into a tf.train.Example.
    """
    X_flat = X.flatten().tolist()
    y_list = y.flatten().tolist()
    feature_dict = {
        'X':           float_feature_list(X_flat),
        'y':           float_feature_list(y_list),
        'record_id':   bytes_feature(record_id),
        'cstart':      int_feature_list([cstart]),
        'cend':        int_feature_list([cend]),
        'strand':      bytes_feature(strand_symbol),
        'chunk_size':  int_feature_list([chunk_size]),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example.SerializeToString()


###########################################
# TFRecord Writing with Threads/Processes
###########################################

def write_to_shard_with_threads(
    shard_id,
    shard_path,
    num_shards,
    all_indices,
    fasta_file,
    gtf_df,
    compression_type="GZIP",
    skip_empty=True,
    max_threads_per_process=4,
    chunk_size_input=5000
):
    """
    Writes data for a specific shard using threading to allow concurrent writes.
    """
    print('Running write_to_shard_with_threads')
    subset_indices = [idx for i, idx in enumerate(all_indices) if i % num_shards == shard_id]
    print(f'Shard subset indices gathered for shard {shard_id}')
    chunk_size_in = chunk_size_input
    options = tf.io.TFRecordOptions(compression_type=compression_type)
    writer = tf.io.TFRecordWriter(shard_path, options=options)
    lock = threading.Lock()

    def thread_worker(subset_indices_split):
        for X, y, record_id, cstart, cend, strand_symbol, chunk_size in build_chunk_data_for_indices(
            fasta_file, gtf_df, subset_indices_split, skip_empty=skip_empty, chunk_size=chunk_size_in
        ):
            try:
                example_str = serialize_chunk_example(X, y, record_id, cstart, cend, strand_symbol, chunk_size)
                with lock:
                    writer.write(example_str)
            except Exception as e:
                print(f"Error writing chunk: {e}")

    subset_splits = [subset_indices[i::max_threads_per_process] for i in range(max_threads_per_process)]
    with cf.ThreadPoolExecutor(max_threads_per_process) as thread_executor:
        thread_futures = [thread_executor.submit(thread_worker, subset_split) for subset_split in subset_splits]
        for future in thread_futures:
            future.result()
            worker_number = thread_futures.index(future)
            print(f'Thread executor {worker_number} completed for process executor {shard_id}.')
    writer.close()


def write_tfrecord_in_shards_hybrid(
    shard_prefix,
    fasta_file,
    gtf_df,
    num_shards=4,
    compression_type="GZIP",
    max_processes=4,
    max_threads_per_process=4,
    chunk_size=5000,
    skip_empty=True,
    shifts=[0]
):
    """
    Writes TFRecord data in multiple shards using multiprocessing for shards
    and threading within each shard.
    """
    print('Running write_tfrecord_in_shards_hybrid')
    all_indices = compute_chunk_indices(fasta_file, chunk_size, shifts=shifts)
    print('all_indices calculated')
    shard_paths = []
    for shard_id in range(num_shards):
        shard_path = f"{shard_prefix}-{shard_id:04d}.tfrecord"
        if compression_type == "GZIP":
            shard_path += ".gz"
        shard_paths.append(shard_path)
    print(shard_paths)

    with cf.ProcessPoolExecutor(max_workers=max_processes) as process_executor:
        futures = []
        for shard_id in range(num_shards):
            proc = process_executor.submit(
                write_to_shard_with_threads,
                shard_id,
                shard_paths[shard_id],
                num_shards,
                all_indices,
                fasta_file,
                gtf_df,
                compression_type,
                skip_empty,
                max_threads_per_process,
                chunk_size
            )
            futures.append(proc)
            print(f'Process executor {shard_id} has started')
        for future in futures:
            future.result()
            shard_number = futures.index(future)
            print(f"Process executor {shard_number} completed.")


###########################################
# TFRecord Parsing and Dataset Construction
###########################################

def parse_chunk_example(serialized_example):
    """
    Parses a serialized tf.train.Example back into tensors.
    """
    feature_spec = {
        'X':          tf.io.VarLenFeature(tf.float32),
        'y':          tf.io.VarLenFeature(tf.float32),
        'record_id':  tf.io.FixedLenFeature([], tf.string),
        'cstart':     tf.io.FixedLenFeature([1], tf.int64),
        'cend':       tf.io.FixedLenFeature([1], tf.int64),
        'strand':     tf.io.FixedLenFeature([], tf.string),
        'chunk_size': tf.io.FixedLenFeature([1], tf.int64),
    }
    parsed = tf.io.parse_single_example(serialized_example, feature_spec)
    chunk_size = parsed['chunk_size'][0]
    X_flat = tf.sparse.to_dense(parsed['X'])
    y_flat = tf.sparse.to_dense(parsed['y'])
    X_reshaped = tf.reshape(X_flat, [chunk_size, 5])
    y_reshaped = tf.reshape(y_flat, [chunk_size, 5])
    record_id = parsed['record_id']
    cstart = parsed['cstart'][0]
    cend = parsed['cend'][0]
    strand = parsed['strand']
    return X_reshaped, y_reshaped, record_id, cstart, cend, strand


def build_dataset_from_tfrecords(
    tfrecord_pattern,
    batch_size=28,
    compression_type='GZIP',
    shuffle_buffer=66000,
):
    files = tf.data.Dataset.list_files(tfrecord_pattern, shuffle=True)
    dataset = files.interleave(
        lambda fname: tf.data.TFRecordDataset(fname, compression_type=compression_type),
        cycle_length=4,
        block_length=1,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(8 * batch_size, reshuffle_each_iteration=True)
    dataset = dataset.unbatch()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def split_tfrecords(input_directory, output_directory, num_splits=24):
    """
    Splits each gzipped TFRecord in input_directory into a specified number of smaller shards.
    """
    current_directory = os.getcwd()
    input_path = os.path.join(current_directory, input_directory)
    output_path = os.path.join(current_directory, output_directory)
    os.makedirs(output_path, exist_ok=True)

    input_file_names = os.listdir(input_path)
    input_file_paths = [os.path.join(input_path, file) for file in input_file_names]

    for file in input_file_paths:
        basename = os.path.basename(file)
        set_index, remainder = basename.split("_inex_shard-")
        sub_index = remainder[:4]
        final_digit = sub_index[-1]

        total_records = 0
        for _ in tf.data.TFRecordDataset(file, compression_type="GZIP"):
            total_records += 1

        chunk_size = total_records // num_splits
        remainder_count = total_records % num_splits

        boundaries = []
        start = 0
        for i in range(num_splits):
            extra = 1 if i < remainder_count else 0
            end = start + chunk_size + extra
            boundaries.append(end)
            start = end

        writers = []
        for i in range(num_splits):
            sub_sub_index = f"{i:02d}"
            new_filename = f"{set_index}_{final_digit}_{sub_sub_index}_tiny_inex_shard.tfrecord.gz"
            new_filepath = os.path.join(output_path, new_filename)
            options = tf.io.TFRecordOptions(compression_type="GZIP")
            writer = tf.io.TFRecordWriter(new_filepath, options=options)
            writers.append(writer)

        current_index = 0
        current_shard = 0
        for record in tf.data.TFRecordDataset(file, compression_type="GZIP"):
            if current_index >= boundaries[current_shard]:
                current_shard += 1
            writers[current_shard].write(record.numpy())
            current_index += 1

        for writer in writers:
            writer.close()

        print(f"Processed {basename}: {total_records} records split into {num_splits} tiny shards.")


def stream_shuffled_records(input_dir, allowed_indices):
    """
    Iterates over TFRecord files in input_dir (whose filenames start with allowed indices),
    shuffling them and yielding one record per file.
    """
    file_paths = [os.path.join(input_dir, fname)
                  for fname in os.listdir(input_dir)
                  if any(fname.startswith(str(idx)) for idx in allowed_indices)]
    if not file_paths:
        raise ValueError("No TFRecord files found matching allowed indices.")
    file_iterators = [(fp, iter(tf.data.TFRecordDataset(fp, compression_type="GZIP")))
                      for fp in file_paths]
    while file_iterators:
        random.shuffle(file_iterators)
        next_file_iterators = []
        for fp, iterator in file_iterators:
            try:
                record = next(iterator)
                yield record
                next_file_iterators.append((fp, iterator))
            except StopIteration:
                print(f"File {fp} is exhausted and will be skipped.")
        file_iterators = next_file_iterators


def write_shuffled_records_to_single_tfrecord(input_dir, allowed_indices, output_filepath):
    """
    Writes records streamed from tiny shards into one big gzipped TFRecord file.
    """
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    writer = tf.io.TFRecordWriter(output_filepath, options=options)
    record_count = 0
    for record in stream_shuffled_records(input_dir, allowed_indices):
        writer.write(record.numpy())
        record_count += 1
        if record_count % 1000 == 0:
            print(f"{record_count} records written...")
    writer.close()
    print(f"Finished writing {record_count} records to {output_filepath}")


def test_dataset_from_tfrecords(
    tfrecord_pattern,
    batch_size=32,
    compression_type='GZIP',
    shuffle_buffer=75000
):
    """
    Loads and parses TFRecord files, returning a dataset for inspection.
    """
    files = tf.data.Dataset.list_files(tfrecord_pattern, shuffle=True)
    dataset = files.interleave(
        lambda fname: tf.data.TFRecordDataset(fname, compression_type=compression_type),
        cycle_length=4,
        block_length=1,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(8 * batch_size, reshuffle_each_iteration=True)
    dataset = dataset.unbatch()
    dataset = dataset.map(parse_chunk_example, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def tvt_split_multi_tfrecords(original_pattern, train_path, val_path, test_path, train_frac=0.8, val_frac=0.10):
    """
    Splits TFRecord files into train, validation, and test sets without parsing. 
    Input is a directory with one or more shards matching a pattern.
    """
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    train_writer = tf.io.TFRecordWriter(train_path, options=options)
    val_writer = tf.io.TFRecordWriter(val_path, options=options)
    test_writer = tf.io.TFRecordWriter(test_path, options=options)

    dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(original_pattern), compression_type='GZIP')
    num_records = sum(1 for _ in dataset)
    print(f"Total records found: {num_records}")

    train_size = int(train_frac * num_records)
    val_size   = int(val_frac * num_records)
    test_size  = num_records - train_size - val_size
    print(f"Splitting into -> Train: {train_size}, Val: {val_size}, Test: {test_size}")

    train_count, val_count, test_count = 0, 0, 0
    dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(original_pattern), compression_type='GZIP')
    dataset = dataset.shuffle(25000, reshuffle_each_iteration=True)

    for i, raw_record in enumerate(dataset):
        if i < train_size:
            train_writer.write(raw_record.numpy())
            train_count += 1
        elif i < train_size + val_size:
            val_writer.write(raw_record.numpy())
            val_count += 1
        else:
            test_writer.write(raw_record.numpy())
            test_count += 1

    train_writer.close()
    val_writer.close()
    test_writer.close()

    print(f"Final Split Counts -> Train: {train_count}, Val: {val_count}, Test: {test_count}")

    
def tvt_split_tfrecords(file_path, train_path, val_path, test_path, train_frac=0.8, val_frac=0.10):
    """
    Splits a single TFRecord file into train, validation, and test sets without parsing.
    """
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    train_writer = tf.io.TFRecordWriter(train_path, options=options)
    val_writer = tf.io.TFRecordWriter(val_path, options=options)
    test_writer = tf.io.TFRecordWriter(test_path, options=options)

    dataset = tf.data.TFRecordDataset(file_path, compression_type='GZIP')
    num_records = sum(1 for _ in dataset)
    print(f"Total records found: {num_records}")

    train_size = int(train_frac * num_records)
    val_size   = int(val_frac * num_records)
    test_size  = num_records - train_size - val_size
    print(f"Splitting into -> Train: {train_size}, Val: {val_size}, Test: {test_size}")

    train_count, val_count, test_count = 0, 0, 0
    dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(file_path), compression_type='GZIP')
    dataset = dataset.shuffle(25000, reshuffle_each_iteration=True)

    for i, raw_record in enumerate(dataset):
        if i < train_size:
            train_writer.write(raw_record.numpy())
            train_count += 1
        elif i < train_size + val_size:
            val_writer.write(raw_record.numpy())
            val_count += 1
        else:
            test_writer.write(raw_record.numpy())
            test_count += 1

    train_writer.close()
    val_writer.close()
    test_writer.close()

    print(f"Final Split Counts -> Train: {train_count}, Val: {val_count}, Test: {test_count}")


###########################################
# Additional TFRecord Conversion Functions
###########################################

def convert_labels_to_binary(x, y):
    """
    Converts labels so that any value not exactly 1 becomes 0.
    """
    y_binary = tf.cast(tf.equal(y, 1.0), y.dtype)
    return x, y_binary


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    if isinstance(value, str):
        value = value.encode('utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example_with_metadata_no_convert(x, y, record_id, cstart, cend, strand):
    """
    Serializes an example into a tf.train.Example. Assumes that labels are already binary.
    """
    x_flat = tf.reshape(x, [-1]).numpy().tolist()
    y_flat = tf.reshape(y, [-1]).numpy().tolist()
    chunk_size = int(x.shape[0])
    record_id_val = record_id.numpy() if isinstance(record_id, tf.Tensor) else record_id
    strand_val = strand.numpy() if isinstance(strand, tf.Tensor) else strand
    cstart_val = cstart.numpy() if isinstance(cstart, tf.Tensor) else cstart
    cend_val   = cend.numpy() if isinstance(cend, tf.Tensor) else cend
    cstart_int = int(cstart_val[0]) if isinstance(cstart_val, (list, tuple, np.ndarray)) else int(cstart_val)
    cend_int   = int(cend_val[0]) if isinstance(cend_val, (list, tuple, np.ndarray)) else int(cend_val)
    feature = {
        'X': _float_feature(x_flat),
        'y': _float_feature(y_flat),
        'record_id': _bytes_feature(record_id_val),
        'cstart': _int64_feature([cstart_int]),
        'cend': _int64_feature([cend_int]),
        'strand': _bytes_feature(strand_val),
        'chunk_size': _int64_feature([chunk_size])
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def convert_and_write_tfrecord(input_tfrecord, output_tfrecord, compression_type="GZIP"):
    """
    Reads an existing TFRecord, converts the labels to binary, and writes out a new TFRecord.
    """
    dataset = tf.data.TFRecordDataset(
        input_tfrecord,
        compression_type=compression_type,
        num_parallel_reads=tf.data.AUTOTUNE
    )
    dataset = dataset.map(parse_chunk_example, num_parallel_calls=tf.data.AUTOTUNE)
    def convert_sample(x, y, record_id, cstart, cend, strand):
        x, y_binary = convert_labels_to_binary(x, y)
        return x, y_binary, record_id, cstart, cend, strand
    dataset = dataset.map(convert_sample, num_parallel_calls=tf.data.AUTOTUNE)
    options = tf.io.TFRecordOptions(compression_type=compression_type)
    with tf.io.TFRecordWriter(output_tfrecord, options=options) as writer:
        for sample in dataset:
            X, y_binary, record_id, cstart, cend, strand = sample
            serialized_example = serialize_example_with_metadata_no_convert(
                X, y_binary, record_id, cstart, cend, strand)
            writer.write(serialized_example)


def build_initial_shards(shifts: list=[0]):
    """
    Function that converts GTF and FASTA into 
    TFRecord shards for later construction into a dataset.
    
    Steps:
      1. Trim the genome FASTA file to only include the desired chromosomes.
      2. Load GTF annotations, filter out chrM, and calculate introns.
      3. Write TFRecord shards using a hybrid (multiprocessing + threading) approach.
    """
    
    # Step 1. Trim the genome FASTA file to only include the desired chromosomes.
    input_fasta = os.path.join(DATA_DIR, "chr_genome.fa")
    output_fasta = os.path.join(DATA_DIR, "trim_chr_genome.fa")
    trim_chr_genome(input_fasta, output_fasta)
    
    # Step 2. Load GTF annotations, filter out chrM, and calculate introns.
    annotation_file = os.path.join(DATA_DIR, "basic_annotations.gtf")
    annotation_data = load_gtf_annotations(annotation_file)
    annotation_data = annotation_data[annotation_data["seqname"] != "chrM"]
    introns = calculate_introns(annotation_data)
    
    trimmed_annotation_data = annotation_data[["seqname", "feature", "cstart", "cend", "strand"]]
    IntronExonDF = pd.concat([trimmed_annotation_data, introns])
    
    # Swap columns for minus strand if needed.
    FixedIntronExonDF = swap_columns_if_needed(IntronExonDF, "cstart", "cend")
    Trimmed_Intron_Exon_DF = FixedIntronExonDF[
        ((FixedIntronExonDF["feature"] == "exon") |
         (FixedIntronExonDF["feature"] == "intron"))
    ]
    Trimmed_Intron_Exon_DF = Trimmed_Intron_Exon_DF[["seqname", "feature", "cstart", "cend", "strand"]]
    print(Trimmed_Intron_Exon_DF.sample(10))
    
    output_csv = os.path.join(DATA_DIR, "FinalIntronExonDF.csv")
    Trimmed_Intron_Exon_DF.to_csv(output_csv, index=False)
    
    # Step 3. Write TFRecord shards using a hybrid (multiprocessing + threading) approach.
    my_fasta = os.path.join(DATA_DIR, "trim_chr_genome.fa")
    my_gtf_df = pd.read_csv(output_csv)
    output_directory = os.path.join(DATA_DIR, "Final_Optimized_TFRecord_Shards")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Example shift for data augmentation (adjust as needed)
    shifts_serial = "".join("0000" if num == 0 else str(num) for num in shifts)
    my_prefix = os.path.join(output_directory, f"{shifts_serial}_inex_shard")
    
    write_tfrecord_in_shards_hybrid(
        shard_prefix=my_prefix,
        fasta_file=my_fasta,
        gtf_df=my_gtf_df,
        num_shards=4,
        compression_type="GZIP",
        max_processes=4,
        max_threads_per_process=2,
        chunk_size=5000,
        skip_empty=True,
        shifts=shifts
    )
    
    
def simple_split_tfrecords(input_directory, output_directory, num_splits):
    """
    Splits every .tfrecord.gz file in input_directory into num_splits shards.
    Each shard is written to output_directory with a simple naming scheme.
    """
    # Get all .tfrecord.gz files (ignoring directories like the temporary folder).
    files = [f for f in os.listdir(input_directory)
             if os.path.isfile(os.path.join(input_directory, f)) and f.endswith('.tfrecord.gz')]
    
    for file in files:
        filepath = os.path.join(input_directory, file)
        # Count records in the file.
        total_records = 0
        for _ in tf.data.TFRecordDataset(filepath, compression_type="GZIP"):
            total_records += 1

        # Determine the boundaries for splitting.
        chunk_size = total_records // num_splits
        remainder = total_records % num_splits
        boundaries = []
        start = 0
        for i in range(num_splits):
            extra = 1 if i < remainder else 0
            end = start + chunk_size + extra
            boundaries.append(end)
            start = end

        # Create writers for the shards.
        writers = []
        base_no_ext = os.path.splitext(file)[0]
        for i in range(num_splits):
            new_filename = f"{base_no_ext}_shard_{i:02d}.tfrecord.gz"
            new_filepath = os.path.join(output_directory, new_filename)
            options = tf.io.TFRecordOptions(compression_type="GZIP")
            writer = tf.io.TFRecordWriter(new_filepath, options=options)
            writers.append(writer)

        # Write records to each shard.
        current_index = 0
        shard_index = 0
        for record in tf.data.TFRecordDataset(filepath, compression_type="GZIP"):
            if current_index >= boundaries[shard_index]:
                shard_index += 1
            writers[shard_index].write(record.numpy())
            current_index += 1

        for w in writers:
            w.close()

        print(f"Split {file} into {num_splits} shards.")


def simple_stream_shuffled_records(directory):
    """
    Streams records randomly from all .tfrecord.gz files in the given directory.
    """
    file_paths = [os.path.join(directory, f) for f in os.listdir(directory)
                  if os.path.isfile(os.path.join(directory, f)) and f.endswith('.tfrecord.gz')]
    if not file_paths:
        raise ValueError("No TFRecord shards found in directory: " + directory)
    
    # Build iterators for each shard.
    file_iterators = [(fp, iter(tf.data.TFRecordDataset(fp, compression_type="GZIP")))
                      for fp in file_paths]
    
    while file_iterators:
        random.shuffle(file_iterators)
        next_file_iterators = []
        for fp, iterator in file_iterators:
            try:
                record = next(iterator)
                yield record
                next_file_iterators.append((fp, iterator))
            except StopIteration:
                print(f"Shard {fp} is exhausted.")
        file_iterators = next_file_iterators


def transform_tfdataset(input_directory, output_directory, num_splits, fraction):
    """
    Transforms a TFRecord dataset by:
      1. Creating a temporary shard folder inside the input_directory.
      2. Splitting every .tfrecord.gz file in the input_directory into num_splits shards.
      3. Randomly rebuilding a new dataset containing fraction * (total records) records.
      4. Saving the rebuilt dataset as one .tfrecord.gz file in output_directory. 
         Its filename is constructed by prefixing the alphabetically first input filename with
         the two-digit integer (fraction * 100) and an underscore.
      5. Deleting the temporary shard folder.
      
    Parameters:
      input_directory (str): Path to the input TFRecord collection (directory).
      output_directory (str): Path to the directory where the output TFRecord will be saved.
      num_splits (int): Number of shards to split each TFRecord into.
      fraction (float): Fraction of total records to sample for the new dataset.
    """
    
    # Create a temporary folder for shards inside the input directory.
    temp_folder = os.path.join(input_directory, "temp_shards")
    os.makedirs(temp_folder, exist_ok=True)
    print(f"Temporary folder created at: {temp_folder}")
    
    # Count total records in the input TFRecord files.
    total_records = 0
    input_files = [f for f in os.listdir(input_directory)
                   if os.path.isfile(os.path.join(input_directory, f)) and f.endswith(".tfrecord.gz")]
    for fname in input_files:
        filepath = os.path.join(input_directory, fname)
        for _ in tf.data.TFRecordDataset(filepath, compression_type="GZIP"):
            total_records += 1
    print(f"Total records found in input dataset: {total_records}")
    
    # Compute the number of records to sample.
    sample_count = int(fraction * total_records)
    print(f"Sampling {sample_count} records (fraction = {fraction}).")
    
    # Split the input TFRecord files into shards (stored in the temporary folder).
    simple_split_tfrecords(input_directory, temp_folder, num_splits)
    
    # Construct the output filename.
    # Use the alphabetically first .tfrecord.gz filename in the input directory.
    if not input_files:
        raise ValueError("No TFRecord files found in the input directory!")
    base_filename = sorted(input_files)[0]
    # Compute the two-digit prefix (e.g., fraction 0.5 -> "50_").
    prefix = f"{int(round(fraction * 100)):02d}_"
    output_filename = prefix + base_filename
    os.makedirs(output_directory, exist_ok=True)
    output_filepath = os.path.join(output_directory, output_filename)
    
    # Write out the randomly shuffled sample of records to a single TFRecord file.
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    writer = tf.io.TFRecordWriter(output_filepath, options=options)
    records_written = 0
    for record in simple_stream_shuffled_records(temp_folder):
        writer.write(record.numpy())
        records_written += 1
        if records_written % 10000 == 0:
            print(f"{records_written} records written so far...")
        if records_written >= sample_count:
            break
    writer.close()
    print(f"Finished writing {records_written} records to {output_filepath}")
    
    # Delete the temporary shards folder to free disk space.
    shutil.rmtree(temp_folder)
    print("Temporary shards have been removed.")    


def dataset_pipeline_help():
    print('''
          1. build_initial_shards function generates 4 tfrecord shards at each shift fed to it.
          I recommend giving a list of 1 window shift.  Currently paths are hardcoded.  
          The first 4*n characters of the shards will be the shift numbers where n is 
          the number of shifts fed to the function.
          
          2. split_tfrecords takes an input and output directory and number of file splits
          and splits each TFRecord in the input_directory (assumed to be gzipped TFRecords)
          into 'num_splits' smaller TFRecords.
          
          3. write_shuffled_records_to_single_tfrecord builds a single big, shuffled, 
          gzip-compressed TFRecord file 
          
          Args:
            input_dir (str): Directory containing the source TFRecord files.
            allowed_indices (list): List of allowed starting indices.
            output_filepath (str): Full path to the output TFRecord file.
            
          4. tvt_split_records test, validate, train splits the big tfrecord into 
          test, validate, and train datasets
          
          5. convert_and_write_tfrecord writes a binary only version of the dataset fed to it
          
          6. Removing background happens when loading the data into the model (see Data_Functions.py)
          ''')
###########################################
# Main Pipeline
###########################################

def main():
    print(
    '''
    Hard coded function that converts GTF and FASTA into 
    tfrecord shards for later construction into a dataset
    
    # Step 1. Trim the genome FASTA file to only include the desired chromosomes.
    input_fasta = os.path.join(DATA_DIR, "chr_genome.fa")
    output_fasta = os.path.join(DATA_DIR, "trim_chr_genome.fa")
    trim_chr_genome(input_fasta, output_fasta)
    
    # Step 2. Load GTF annotations, filter out chrM, and calculate introns.
    annotation_file = os.path.join(DATA_DIR, 'basic_annotations.gtf')
    annotation_data = load_gtf_annotations(annotation_file)
    annotation_data = annotation_data[annotation_data["seqname"] != "chrM"]
    introns = calculate_introns(annotation_data)
    
    trimmed_annotation_data = annotation_data[["seqname", "feature", "cstart", "cend", "strand"]]
    IntronExonDF = pd.concat([trimmed_annotation_data, introns])
    
    # Swap columns for minus strand if needed.
    FixedIntronExonDF = swap_columns_if_needed(IntronExonDF, 'cstart', 'cend')
    Trimmed_Intron_Exon_DF = FixedIntronExonDF[((FixedIntronExonDF["feature"] == "exon") |
                                                 (FixedIntronExonDF["feature"] == "intron"))]
    Trimmed_Intron_Exon_DF = Trimmed_Intron_Exon_DF[["seqname", "feature", "cstart", "cend", "strand"]]
    print(Trimmed_Intron_Exon_DF.sample(10))
    output_csv = os.path.join(DATA_DIR, "FinalIntronExonDF.csv")
    Trimmed_Intron_Exon_DF.to_csv(output_csv, index=False)
    
    # Step 3. Write TFRecord shards using hybrid (multiprocessing + threading) approach.
    my_fasta = os.path.join(DATA_DIR, 'trim_chr_genome.fa')
    my_gtf_df = pd.read_csv(output_csv)
    output_directory = os.path.join(DATA_DIR, "Final_Optimized_TFRecord_Shards")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Example shift for data augmentation (adjust as needed)
    shifts = [3334]
    my_prefix = os.path.join(output_directory, '3334_inex_shard')
    
    write_tfrecord_in_shards_hybrid(
        shard_prefix=my_prefix,
        fasta_file=my_fasta,
        gtf_df=my_gtf_df,
        num_shards=4,
        compression_type="GZIP",
        max_processes=4,
        max_threads_per_process=2,
        chunk_size=5000,
        skip_empty=True,
        shifts=shifts
    )
    ''')


if __name__ == "__main__":
    main()