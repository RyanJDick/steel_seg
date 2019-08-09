import os
import random
import argparse

import numpy as np
import tensorflow as tf
from PIL import Image
import yaml

from dataset.severstal_steel_dataset import load_annotations, rle_to_dense


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_tfrecords(
    img_dir,
    anns_file,
    val_split,
    output_train_dir,
    base_train_tfrecord_filename,
    output_val_dir,
    base_val_tfrecord_filename):

    anns = load_annotations(anns_file)
    anns_list = list(anns.items())
    random.shuffle(anns_list)

    num_train_examples = int(len(anns_list) * (1 - val_split))
    anns_train_list = anns_list[:num_train_examples]
    anns_val_list = anns_list[num_train_examples:]
    print('Split train/validation data. '
          f'Train: {len(anns_train_list)}, Val: {len(anns_val_list)}\n')

    print('Creating train tfrecords...')
    create_tfrecords_from_annotations(
        img_dir,
        anns_train_list,
        output_train_dir,
        base_train_tfrecord_filename
    )

    print('Creating validation tfrecords...')
    create_tfrecords_from_annotations(
        img_dir,
        anns_val_list,
        output_val_dir,
        base_val_tfrecord_filename
    )


def create_tfrecords_from_annotations(
    img_dir,
    anns_list,
    output_dir,
    base_tfrecord_filename,
    examples_per_file=2000):

    os.makedirs(output_dir, exist_ok=True)

    batch_start = 0
    file_index = 0
    while batch_start < len(anns_list):
        batch_end = min(batch_start + examples_per_file, len(anns_list))
        file_path = os.path.join(output_dir, f'{base_tfrecord_filename}{file_index}.tfrecord')
        print(f'Starting batch {batch_start}-{batch_end} out of {len(anns_list)}.')
        print(f'Writing file {file_path}')

        with tf.io.TFRecordWriter(file_path) as writer:
            for i in range(batch_start, batch_end):
                img_name, annotations_dict = anns_list[i]

                # Load image
                img_path = os.path.join(img_dir, img_name)
                img = np.array(Image.open(img_path))
                img_gray = img[:, :, 0] # All channels are the same

                # Load annotations
                dense_anns = []
                for cls in ['1', '2', '3', '4']:
                    dense_ann = rle_to_dense(annotations_dict[cls], img_gray.shape[0], img_gray.shape[1])
                    dense_anns.append(dense_ann)
                annotation_array = np.stack(dense_anns, axis=-1)
                annotation_array.astype(np.uint8)

                # Serialize example
                assert img_gray.dtype == np.uint8
                assert annotation_array.dtype == np.uint8
                feature = {
                    'image':       _bytes_feature(tf.compat.as_bytes(img_gray.tostring())),
                    'annotations': _bytes_feature(tf.compat.as_bytes(annotation_array.tostring()))
                }
                example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

                # Write to TFRecord file
                writer.write(example_proto.SerializeToString())
        batch_start = batch_end
        file_index += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert raw severstal steel data to tfrecord files.')
    parser.add_argument('--config', required=True, help='Path to .yaml config file.')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.load(f)

    create_tfrecords(
        img_dir=cfg['TRAIN_IMAGE_DIR'],
        anns_file=cfg['TRAIN_ANNOTATIONS_FILE'],
        val_split=cfg['VAL_SPLIT'],
        output_train_dir=cfg['TRAIN_TFRECORD_DIR'],
        base_train_tfrecord_filename=cfg['TRAIN_BASE_TFRECORD_FILENAME'],
        output_val_dir=cfg['VAL_TFRECORD_DIR'],
        base_val_tfrecord_filename=cfg['VAL_BASE_TFRECORD_FILENAME'])
