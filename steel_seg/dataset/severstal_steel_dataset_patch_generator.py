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


def get_image_patches(img, patch_size, num_patches_per_image):
    # TODO: only supports changing x position
    h, w, _ = img.shape
    x_step_size = int((w - patch_size) / (num_patches_per_image - 1))
    img_patches = []
    for x in range(0, w - patch_size + 1, x_step_size):
        img_patches.append(img[0:patch_size, x:x+patch_size])
    return img_patches, x_step_size

class SeverstalSteelDatasetPatchGenerator(tf.keras.utils.Sequence):
    '''Dataset generator that generates patches of Severstal Steel images.
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
                 patch_size,
                 num_patches_per_image,
                 balance_classes,
                 max_oversample_rate):
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

        self._patch_size = patch_size
        self._num_patches_per_image = num_patches_per_image

        self._balance_classes = balance_classes
        self._max_oversample_rate = max_oversample_rate

        self._anns_dict = load_annotations(self._train_anns_file)

        self._augmenter = self._build_augmenter()

        if self._balance_classes:
            self._examples_by_class = self._build_examples_by_class_dict()

        # So that we dont have to share images between batches
        assert self._batch_size % self._num_patches_per_image == 0

        self._epoch_examples = None
        self.on_epoch_end()

    def _build_examples_by_class_dict(self):
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

    def __len__(self):
        '''Number of batches per epoch
        '''
        return int(np.floor(
            len(self._epoch_examples) * self._num_patches_per_image / self._batch_size
        ))

    def __getitem__(self, index):
        '''Generate one batch of data
        '''
        imgs_per_batch = int(self._batch_size / self._num_patches_per_image)
        imgs = self._epoch_examples[index*imgs_per_batch:(index+1)*imgs_per_batch]

        img_patches = []
        ann_patches = []
        for img_name, target_class in imgs:
            # Load image
            img, ann = self.get_example_from_img_name(img_name)

            if not self._is_training:
                val_img_patches, _ = get_image_patches(
                    img, self._patch_size, self._num_patches_per_image)
                val_ann_patches, _ = get_image_patches(
                    ann, self._patch_size, self._num_patches_per_image)
                img_patches += val_img_patches
                ann_patches += val_ann_patches
                continue

            if target_class == 'no_class':
                random_patches_per_image = self._num_patches_per_image
            else:
                random_patches_per_image = 0 #int(self._num_patches_per_image / 2)

            # Get random patches from current image
            for _ in range(random_patches_per_image):
                img_patch, ann_patch = self._get_random_patch(img, ann)
                #img_patch, ann_patch = self._apply_augmentations(img_patch, ann_patch)
                img_patches.append(img_patch)
                ann_patches.append(ann_patch)

            # Get heuristic-targeted patches from current image
            for _ in range(self._num_patches_per_image - random_patches_per_image):
                img_patch, ann_patch = self._get_targeted_patch(img, ann, target_class)
                #img_patch, ann_patch = self._apply_augmentations(img_patch, ann_patch)
                img_patches.append(img_patch)
                ann_patches.append(ann_patch)

        if self._is_training:
            patches = list(zip(img_patches, ann_patches))
            random.shuffle(patches)
            img_patches, ann_patches = zip(*patches)

        img_patches = np.stack(img_patches)
        ann_patches = np.stack(ann_patches)

        img_patches, ann_patches = self._apply_augmentations(img_patches, ann_patches)

        return img_patches, ann_patches

    def _get_random_patch(self, img, ann):
        h, w, _ = img.shape
        patch_x = random.randint(0, w - self._patch_size) # inclusive
        patch_y = random.randint(0, h - self._patch_size)
        img_patch = img[patch_y:patch_y+self._patch_size, patch_x:patch_x+self._patch_size]
        ann_patch = ann[patch_y:patch_y+self._patch_size, patch_x:patch_x+self._patch_size]
        return img_patch, ann_patch

    def _get_targeted_patch(self, img, ann, target_class):
        assert ann.dtype == np.uint8
        h, w, _ = img.shape
        masked_pixels = np.where(ann[:, :, target_class] == 1)
        if len(masked_pixels[0]) == 0:
            print('Target class not present in image. '
                  'Something weird is going on, but will return a random patch instead.')
            return self._get_random_patch(img, ann)

        selected_pixel = random.randint(0, len(masked_pixels[0]) - 1) # inclusive
        selected_y = masked_pixels[0][selected_pixel]
        selected_x = masked_pixels[1][selected_pixel]

        # Calculate valid pixel locations of the top left corner of the patch
        min_y = max(0, selected_y - self._patch_size + 1)
        max_y = min(h - self._patch_size, selected_y)
        min_x = max(0, selected_x - self._patch_size + 1)
        max_x = min(w - self._patch_size, selected_x)

        patch_x = random.randint(min_x, max_x) # inclusive
        patch_y = random.randint(min_y, max_y)
        img_patch = img[patch_y:patch_y+self._patch_size, patch_x:patch_x+self._patch_size]
        ann_patch = ann[patch_y:patch_y+self._patch_size, patch_x:patch_x+self._patch_size]

        return img_patch, ann_patch

    def _apply_augmentations(self, imgs, anns):
        imgs_aug, anns_aug = self._augmenter(images=imgs, segmentation_maps=anns)
        return imgs_aug, anns_aug

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
        oversampling less frequent classes.
        '''
        if not self._is_training:
            # If not training, just include all of the images in order
            self._epoch_examples = [(img_name, None) for img_name in self._img_list]
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
                epoch_examples.append((examples[i % num_examples], class_name))

        random.shuffle(epoch_examples)
        self._epoch_examples = epoch_examples
