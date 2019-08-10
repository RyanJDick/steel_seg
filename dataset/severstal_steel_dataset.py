import os
import random
import glob
from collections import defaultdict

import yaml
import numpy as np
import tensorflow as tf
from PIL import Image


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class SeverstalSteelDataset():
    def __init__(self, config_path):
        with open(config_path) as f:
            self._cfg = yaml.load(f)

    def rle_to_dense(self, rle, img_height, img_width):
        '''Convert the rle representation of a single class mask to the equivalent dense binary np
        array.
        '''
        if rle is None or rle == '':
            return np.zeros((img_height, img_width), dtype=np.uint8)
        rle_list = rle.strip().split(' ')
        rle_pairs = [(int(rle_list[i]), int(rle_list[i+1])) for i in range(0, len(rle_list), 2)]

        dense_1d_array = np.zeros(img_height * img_width, dtype=np.uint8)
        for rle_start, rle_run in rle_pairs:
            # Subtract 1 from indices because pixel indices start at 1 rather than 0
            dense_1d_array[rle_start - 1:rle_start + rle_run - 1] = 1
        
        # Use Fortran ordering, meaning that the first index changes fastest (sort of unconventional)
        dense_2d_array = np.reshape(dense_1d_array, (img_height, img_width), order='F')
        return dense_2d_array

    def load_annotations(self):
        '''Load all annotations from a file. Returns a dict mapping image filenames to annotations.
        '''
        anns = defaultdict(dict)
        with open(self._cfg['TRAIN_ANNOTATIONS_FILE']) as f:
            for line in f:
                file_name, rle_ann = line.split(',')
                if file_name == 'ImageId_ClassId': # Skip header
                    continue

                img_id, cls_id = file_name.split('_')
                anns[img_id][cls_id] = rle_ann.strip()
        return anns

    def create_tfrecords(self):
        anns = self.load_annotations()
        anns_list = list(anns.items())
        random.shuffle(anns_list)

        num_train_examples = int(len(anns_list) * (1 - self._cfg['VAL_SPLIT']))
        anns_train_list = anns_list[:num_train_examples]
        anns_val_list = anns_list[num_train_examples:]
        print('Split train/validation data. '
            f'Train: {len(anns_train_list)}, Val: {len(anns_val_list)}\n')

        print('Creating train tfrecords...')
        self._create_tfrecords_from_annotations(
            img_dir=self._cfg['TRAIN_IMAGE_DIR'],
            anns_list=anns_train_list,
            output_dir=self._cfg['TRAIN_TFRECORD_DIR'],
            base_tfrecord_filename=self._cfg['TRAIN_BASE_TFRECORD_FILENAME'],
            examples_per_file=self._cfg['EXAMPLES_PER_TFRECORD']
        )

        print('Creating validation tfrecords...')
        self._create_tfrecords_from_annotations(
            img_dir=self._cfg['TRAIN_IMAGE_DIR'],
            anns_list=anns_val_list,
            output_dir=self._cfg['VAL_TFRECORD_DIR'],
            base_tfrecord_filename=self._cfg['VAL_BASE_TFRECORD_FILENAME'],
            examples_per_file=self._cfg['EXAMPLES_PER_TFRECORD']
        )

    def _create_tfrecords_from_annotations(
        self,
        img_dir,
        anns_list,
        output_dir,
        base_tfrecord_filename,
        examples_per_file):

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
                        dense_ann = self.rle_to_dense(
                            annotations_dict[cls], img_gray.shape[0], img_gray.shape[1])
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

    def _build_parse_fn(self):
        def _parse_example(proto):
            keys_to_features = {'image': tf.FixedLenFeature([], tf.string),
                                'annotations': tf.FixedLenFeature([], tf.string)}
    
            # Load one example
            parsed_features = tf.parse_single_example(proto, keys_to_features)
            
            # Turn your saved image string into an array
            img = tf.decode_raw(parsed_features['image'], tf.uint8)
            ann = tf.decode_raw(parsed_features['annotations'], tf.uint8)
            
            return img, ann
        return _parse_example

    def _build_augment_fn(self):
        def _augment_example(img, ann):
            # Concat img and ann so that flips get applied to both
            combined_img_ann = tf.concat([img, ann], axis=-1)
            
            combined_img_ann = tf.image.random_flip_left_right(combined_img_ann)
            combined_img_ann = tf.image.random_flip_up_down(combined_img_ann)
            
            # Split img and ann, we only want to apply colour augmentations to img
            img, ann = tf.split(
                combined_img_ann, num_or_size_splits=[1, self._cfg['NUM_CLASSES']], axis=-1)
            
            img = tf.image.random_brightness(img, max_delta=self._cfg['BRIGHTNESS_MAX_DELTA'])
            img = tf.image.random_contrast(
                img, self._cfg['CONTRAST_LOWER_FACTOR'], self._cfg['CONTRAST_UPPER_FACTOR'])
            return img, ann
        return _augment_example

    def _build_resize_fn(self):
        def _resize_example(img, ann):
            img = tf.reshape(img,
                [self._cfg['IMG_HEIGHT'], self._cfg['IMG_WIDTH'], 1])
            ann = tf.reshape(ann,
                [self._cfg['IMG_HEIGHT'], self._cfg['IMG_WIDTH'], self._cfg['NUM_CLASSES']])
            return img, ann
        return _resize_example

    def create_dataset(self, training):
        '''Create training dataset if training == True, else validation dataset.
        '''
        if training:
            tfrecord_pattern = os.path.join(self._cfg['TRAIN_TFRECORD_DIR'], '*.tfrecord')
        else:
            tfrecord_pattern = os.path.join(self._cfg['VAL_TFRECORD_DIR'], '*.tfrecord')

        tfrecord_files = glob.glob(tfrecord_pattern)
        tfrecord_files.sort()

        #filenames = tf.placeholder(tf.string, shape=[None])
        dataset = tf.data.TFRecordDataset(tfrecord_files)
        
        dataset = dataset.map(self._build_parse_fn(), num_parallel_calls=6)
        dataset = dataset.map(self._build_resize_fn(), num_parallel_calls=6)
        if training:
            dataset = dataset.map(self._build_augment_fn(), num_parallel_calls=6)
            dataset = dataset.shuffle(64)
        dataset = dataset.batch(32)
        # Each training step consumes 1 element (i.e. 1 batch)
        dataset = dataset.prefetch(buffer_size=1)

        #iterator = dataset.make_one_shot_iterator()
        return dataset
