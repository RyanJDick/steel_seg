import os
from collections import defaultdict
import yaml
import numpy as np
import tensorflow as tf
import random
from PIL import Image
import imgaug.augmenters as iaa

from steel_seg.dataset.dataset_utils import load_annotations
from steel_seg.utils import rle_to_dense, dense_to_rle
from steel_seg.image_augmentation import adjust_brightness_and_contrast


class SeverstalSteelDatasetGenerator(tf.keras.utils.Sequence):
    '''Dataset generator that generates Severstal Steel images/annotations.
    '''
    def __init__(self,
                 img_list,
                 is_training,
                 train_img_dir,
                 train_anns_file,
                 img_height,
                 img_width,
                 num_classes,
                 batch_size,
                 brightness_max_delta,
                 contrast_lower_factor,
                 contrast_upper_factor,
                 balance_classes,
                 max_oversample_rate,
                 dense_annotation=True):
        self._img_list = img_list
        self._is_training = is_training

        self._train_img_dir = train_img_dir
        self._train_anns_file = train_anns_file

        self._img_height = img_height
        self._img_width = img_width
        self._num_classes = num_classes
        self._batch_size = batch_size

        self._brightness_max_delta = brightness_max_delta
        self._contrast_lower_factor = contrast_lower_factor
        self._contrast_upper_factor = contrast_upper_factor

        self._balance_classes = balance_classes
        self._max_oversample_rate = max_oversample_rate

        self._dense_annotation = dense_annotation

        self._anns_dict = load_annotations(self._train_anns_file)

        if self._balance_classes:
            self._examples_by_class = self._build_examples_by_class_dict()

        self._augmenter = self._build_augmenter()

        self._epoch_examples = None
        self.on_epoch_end()

    def _build_examples_by_class_dict(self):
        '''Build dict mapping class names to a list of examples containing that class.
        '''
        examples_by_class = defaultdict(list)

        for img_name, annotations in self._anns_dict.items():
            img_contains_mask = False
            for cls_name, ann in annotations.items():
                if ann != '':
                    img_contains_mask = True
                    try:
                        cls_name = int(cls_name) - 1 # TODO: this is only relevant to severstal data
                    except:
                        pass
                    examples_by_class[cls_name].append(img_name)
            if not img_contains_mask:
                examples_by_class['no_class'].append(img_name)

        return examples_by_class

    def _build_augmenter(self):
        sometimes = lambda aug: iaa.Sometimes(0.95, aug)

        augmenter = iaa.Sequential(
            [
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.Sometimes(0.8,
                    iaa.Sequential(
                        [
                            sometimes(iaa.Affine(
                                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                                rotate=(-5, 5), # rotate by -5 to +5 degrees
                                shear=(-30, 30), # shear by -16 to +16 degrees
                                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                                cval=0, # fill with black pixels
                                mode='constant', # fill with constant value
                            )),
                            sometimes(
                                iaa.OneOf(
                                    [
                                        iaa.ElasticTransformation(alpha=20.0, sigma=5.0),
                                        iaa.PerspectiveTransform(scale=0.07),
                                    ]
                                )
                            ),
                            iaa.Multiply((0.8, 1.2)),
                            iaa.Add((-30, 30)), # change brightness of images (by -30 to 30 of original value)
                        ]
                    )
                ),
            ]
        )
        return augmenter

    def __len__(self):
        '''Number of batches per epoch
        '''
        return int(np.floor(len(self._epoch_examples) / self._batch_size))

    def __getitem__(self, index):
        '''Generate one batch of data
        '''
        img_names = self._epoch_examples[index*self._batch_size:(index+1)*self._batch_size]

        imgs = []
        anns = []
        for img_name in img_names:
            # Load image
            img, ann = self.get_example_from_img_name(img_name)
            imgs.append(img)
            anns.append(ann)

        imgs = np.stack(imgs)
        anns = np.stack(anns)

        if self._is_training:
            imgs, anns = self._apply_augmentations(imgs, anns)

        if not self._dense_annotation:
            anns = np.amax(anns, axis=(1, 2))

        return imgs, anns

    def _apply_augmentations(self, imgs, anns):
        imgs_aug, anns_aug = self._augmenter(images=imgs, segmentation_maps=anns)
        return imgs_aug, anns_aug

    # def _apply_augmentations(self, img, ann):
    #     # Concat img and ann so that flips get applied to both
    #     img_channels = img.shape[-1]
    #     combined = np.concatenate([img, ann], axis=-1)

    #     if random.choice([True, False]):
    #         combined = combined[:, ::-1, :] # Flip left-right

    #     if random.choice([True, False]):
    #         combined = combined[::-1, :, :] # Flip up-down

    #     img = combined[:, :, :img_channels]
    #     ann = combined[:, :, img_channels:]

    #     brightness_delta = random.uniform(-1 * self._brightness_max_delta,
    #                                       self._brightness_max_delta)
    #     contrast_factor = random.uniform(self._contrast_lower_factor,
    #                                      self._contrast_upper_factor)

    #     img = adjust_brightness_and_contrast(img, brightness_delta, contrast_factor)

    #     return img, ann

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

    def on_epoch_end(self):
        '''Select images that will be included in the next epoch,
        oversampling less frequent classes if balance_classes is enabled.
        '''
        if not self._is_training:
            self._epoch_examples = self._img_list.copy()
            return

        if not self._balance_classes:
            self._epoch_examples = self._img_list.copy()
            random.shuffle(self._epoch_examples)
            return

        # Set images_per_class_per_epoch to the number of images in the largest class
        images_per_class_per_epoch = 0
        for class_name, examples in self._examples_by_class.items():
            if len(examples) > images_per_class_per_epoch:
                images_per_class_per_epoch = len(examples)

        epoch_examples = []
        for class_name, examples in self._examples_by_class.items():
            # Figure out appropriate oversample_rate for images containing current class
            num_examples = len(examples)
            oversample_rate = images_per_class_per_epoch / num_examples
            if oversample_rate > self._max_oversample_rate:
                print(f'Oversample rate of {oversample_rate} for class {class_name} is too large.'
                      f'Setting to {self._max_oversample_rate}.')
                oversample_rate = self._max_oversample_rate

            # Add oversampled images to epoch
            random.shuffle(examples)
            for i in range(int(oversample_rate * num_examples)):
                epoch_examples.append(examples[i % num_examples])

        random.shuffle(epoch_examples)
        self._epoch_examples = epoch_examples
