import os
import random
import glob
from collections import defaultdict
from datetime import datetime

import yaml
import json
import numpy as np
import tensorflow as tf
from PIL import Image

from steel_seg.utils import dice_coeff_kaggle, rle_to_dense, visualize_segmentations
from steel_seg.model.unet import postprocess
from steel_seg.dataset.dataset_utils import load_annotations


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_img_list_file_path(tfrecord_dir, base_tfrecord_filename):
    return os.path.join(tfrecord_dir, f'{base_tfrecord_filename}_images.json')


class SeverstalSteelDataset():
    def __init__(self,
                 train_img_dir,
                 train_anns_file,
                 img_height,
                 img_width,
                 num_classes,
                 val_split,
                 batch_size,
                 train_tfrecord_dir,
                 train_base_tfrecord_filename,
                 val_tfrecord_dir,
                 val_base_tfrecord_filename,
                 examples_per_tfrecord,
                 brightness_max_delta,
                 contrast_lower_factor,
                 contrast_upper_factor,
                 patch_size,
                 num_patches_per_image):
        self._train_img_dir = train_img_dir
        self._train_anns_file = train_anns_file

        self._img_height = img_height
        self._img_width = img_width
        self._num_classes = num_classes
        self._val_split = val_split
        self._batch_size = batch_size

        self._train_tfrecord_dir = train_tfrecord_dir
        self._train_base_tfrecord_filename = train_base_tfrecord_filename
        self._val_tfrecord_dir = val_tfrecord_dir
        self._val_base_tfrecord_filename = val_base_tfrecord_filename
        self._examples_per_tfrecord = examples_per_tfrecord

        self._brightness_max_delta = brightness_max_delta
        self._contrast_lower_factor = contrast_lower_factor
        self._contrast_upper_factor = contrast_upper_factor

        self._patch_size = patch_size
        self._num_patches_per_image = num_patches_per_image

        self._anns_dict = load_annotations(self._train_anns_file)

    @classmethod
    def init_from_config(cls, config_path):
        with open(config_path) as f:
            cfg = yaml.load(f)

        print('Warning!!! Sort out the num_patches.')

        return cls(
            train_img_dir=cfg['TRAIN_IMAGE_DIR'],
            train_anns_file=cfg['TRAIN_ANNOTATIONS_FILE'],
            img_height=cfg['IMG_HEIGHT'],
            img_width=cfg['IMG_WIDTH'],
            num_classes=cfg['NUM_CLASSES'],
            val_split=cfg['VAL_SPLIT'],
            batch_size=cfg['SEGMENTATION_BATCH_SIZE'],
            train_tfrecord_dir=cfg['TRAIN_TFRECORD_DIR'],
            train_base_tfrecord_filename=cfg['TRAIN_BASE_TFRECORD_FILENAME'],
            val_tfrecord_dir=cfg['VAL_TFRECORD_DIR'],
            val_base_tfrecord_filename=cfg['VAL_BASE_TFRECORD_FILENAME'],
            examples_per_tfrecord=cfg['EXAMPLES_PER_TFRECORD'],
            brightness_max_delta=cfg['BRIGHTNESS_MAX_DELTA'],
            contrast_lower_factor=cfg['CONTRAST_LOWER_FACTOR'],
            contrast_upper_factor=cfg['CONTRAST_UPPER_FACTOR'],
            patch_size=cfg['PATCH_SIZE'],
            num_patches_per_image=cfg['NUM_PATCHES_PER_IMAGE_TRAIN'],
        )

    def create_tfrecords(self):
        imgs = list(self._anns_dict.keys())
        random.shuffle(imgs)

        # Split train and test, and make sure that a multiple of BATCH_SIZE is stored in each set
        num_train_examples = int(len(imgs) * (1 - self._val_split))
        num_train_examples = num_train_examples - num_train_examples % self._batch_size
        num_val_examples = len(imgs) - num_train_examples
        num_val_examples = num_val_examples - num_val_examples % self._batch_size

        train_imgs = imgs[:num_train_examples]
        val_imgs = imgs[num_train_examples:num_train_examples+num_val_examples]
        print('Split train/validation data. '
            f'Train: {len(train_imgs)}, Val: {len(val_imgs)}\n')

        print('Creating train tfrecords...')
        self._create_tfrecords_from_img_list(
            img_dir=self._train_img_dir,
            img_list=train_imgs,
            output_dir=self._train_tfrecord_dir,
            base_tfrecord_filename=self._train_base_tfrecord_filename,
            examples_per_file=self._examples_per_tfrecord
        )

        print('Creating validation tfrecords...')
        self._create_tfrecords_from_img_list(
            img_dir=self._train_img_dir,
            img_list=val_imgs,
            output_dir=self._val_tfrecord_dir,
            base_tfrecord_filename=self._val_base_tfrecord_filename,
            examples_per_file=self._examples_per_tfrecord
        )

    def _create_tfrecords_from_img_list(
        self,
        img_dir,
        img_list,
        output_dir,
        base_tfrecord_filename,
        examples_per_file):

        os.makedirs(output_dir, exist_ok=True)

        batch_start = 0
        file_index = 0
        while batch_start < len(img_list):
            batch_end = min(batch_start + examples_per_file, len(img_list))
            file_path = os.path.join(
                output_dir, f'{base_tfrecord_filename}{batch_start}-{batch_end}.tfrecord')
            print(f'Processing examples {batch_start}-{batch_end} out of {len(img_list)}.')
            print(f'Writing file {file_path}')

            with tf.io.TFRecordWriter(file_path) as writer:
                for i in range(batch_start, batch_end):
                    img_gray, annotation_array = self.get_example_from_img_name(img_list[i])

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
        summary_file_path = get_img_list_file_path(output_dir, base_tfrecord_filename)
        with open(summary_file_path, 'w') as f:
            json.dump(img_list, f)

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
                combined_img_ann, num_or_size_splits=[1, self._num_classes], axis=-1)

            img = tf.image.random_brightness(img, max_delta=self._brightness_max_delta)
            img = tf.image.random_contrast(
                img, self._contrast_lower_factor, self._contrast_upper_factor)
            return img, ann
        return _augment_example

    def _build_reshape_fn(self):
        def _reshape_example(img, ann):
            img = tf.reshape(img, [self._img_height, self._img_width, 1])
            ann = tf.reshape(ann, [self._img_height, self._img_width, self._num_classes])
            return img, ann
        return _reshape_example

    def _build_convert_to_classification_fn(self):
        def _convert_to_classification_example(img, ann):
            # Binary variable indicating whether there is a mask or not for each class
            ann = tf.reduce_max(ann, axis=(0, 1))
            return img, ann
        return _convert_to_classification_example

    def _build_sample_patches_fn(self, num_patches=4, patch_size=256):
        def _sample_patches(img, ann):
            img_channels = tf.shape(img)[-1]
            ann_channels = tf.shape(ann)[-1]
            combined = tf.concat([img, ann], axis=-1)
            combined_channels = img_channels + ann_channels
            img_patches = []
            ann_patches = []
            for _ in range(num_patches):
                patch = tf.random_crop(combined, [patch_size, patch_size, combined_channels])
                img_patches.append(patch[:, :, :img_channels])
                ann_patches.append(patch[:, :, img_channels:])

            img_patches = tf.stack(img_patches)
            ann_patches = tf.stack(ann_patches)

            return img_patches, ann_patches
        return _sample_patches

    def _build_deterministic_patches_fn(self, patch_size, img_width, img_height):
        def _select_patches(img, ann):
            img_patches = []
            ann_patches = []
            for start_row in range(0, img_height - patch_size + 1, patch_size):
                for start_col in range(0, img_width - patch_size + 1, patch_size):
                    img_patches.append(img[start_row:start_row+patch_size, start_col:start_col+patch_size, :])
                    ann_patches.append(ann[start_row:start_row+patch_size, start_col:start_col+patch_size, :])

            img_patches = tf.stack(img_patches)
            ann_patches = tf.stack(ann_patches)
            return img_patches, ann_patches
        return _select_patches

    def create_dataset(self, dataset_type, use_patches=False, dense_segmentation=True):
        '''Create tf dataset, where dataset_type is either 'training' or 'validation'
        '''
        training = False
        if dataset_type == 'training':
            training = True
            tfrecord_pattern = os.path.join(self._train_tfrecord_dir, '*.tfrecord')
            img_list_file = get_img_list_file_path(
                self._train_tfrecord_dir, self._train_base_tfrecord_filename)
        elif dataset_type == 'validation':
            tfrecord_pattern = os.path.join(self._val_tfrecord_dir, '*.tfrecord')
            img_list_file = get_img_list_file_path(
                self._val_tfrecord_dir, self._val_base_tfrecord_filename)
        else:
            raise ValueError('Unsupported dataset_type.')

        tfrecord_files = glob.glob(tfrecord_pattern)
        tfrecord_files.sort()

        with open(img_list_file) as f:
            img_list = json.load(f)
        num_batches = len(img_list) / self._batch_size

        num_parallel_calls = 6
        dataset = tf.data.TFRecordDataset(tfrecord_files)
        dataset = dataset.map(self._build_parse_fn(), num_parallel_calls=num_parallel_calls)
        dataset = dataset.map(self._build_reshape_fn(), num_parallel_calls=num_parallel_calls)
        if use_patches:
            if training:
                dataset = dataset.map(
                    self._build_sample_patches_fn(num_patches=self._num_patches_per_image, patch_size=self._patch_size),
                    num_parallel_calls=num_parallel_calls)
            else:
                dataset = dataset.map(
                    self._build_deterministic_patches_fn(self._patch_size, self._img_width, self._img_height),
                    num_parallel_calls=num_parallel_calls)
            dataset = dataset.apply(tf.data.experimental.unbatch())
        if training:
            dataset = dataset.map(self._build_augment_fn(), num_parallel_calls=num_parallel_calls)
            dataset = dataset.shuffle(2 * self._batch_size)
        if not dense_segmentation:
            dataset = dataset.map(
                self._build_convert_to_classification_fn(),
                num_parallel_calls=num_parallel_calls)
        dataset = dataset.repeat()
        dataset = dataset.batch(self._batch_size)
        # Each training step consumes 1 element (i.e. 1 batch)
        dataset = dataset.prefetch(buffer_size=3)

        #iterator = dataset.make_one_shot_iterator()
        return dataset, int(num_batches)

    def get_image_list(self, dataset_type):
        if dataset_type == 'training':
            img_list_file = get_img_list_file_path(
                self._train_tfrecord_dir, self._train_base_tfrecord_filename)
        elif dataset_type == 'validation':
            img_list_file = get_img_list_file_path(
                self._val_tfrecord_dir, self._val_base_tfrecord_filename)
        else:
            raise ValueError('Unsupported dataset_type.')

        with open(img_list_file) as f:
            img_list = json.load(f)
        return img_list

    def get_example_from_img_name(self, img_name):
        # Load image
        img_path = os.path.join(self._train_img_dir, img_name)
        img = np.array(Image.open(img_path))
        img_gray = img[:, :, :1] # All channels are the same

        # Load annotations
        img_anns_dict = self._anns_dict.get(img_name, None)
        if img_anns_dict is None:
            return img_gray, None # Test example, no annotations
        dense_anns = []
        for cls in ['1', '2', '3', '4']:
            dense_ann = rle_to_dense(
                img_anns_dict[cls], img_gray.shape[0], img_gray.shape[1])
            dense_anns.append(dense_ann)
        annotation_array = np.stack(dense_anns, axis=-1)
        annotation_array.astype(np.uint8)
        return img_gray, annotation_array
