import os
from collections import defaultdict
import yaml
import numpy as np
import tensorflow as tf
import random
from PIL import Image

from steel_seg.dataset.severstal_steel_dataset import load_annotations
from steel_seg.utils import rle_to_dense, dense_to_rle
from steel_seg.image_augmentation import adjust_brightness_and_contrast


class SeverstalSteelDatasetPatchGenerator(tf.keras.utils.Sequence):
    '''Dataset generator that generates patches of Severstal Steel images.
    This is slower than the SeverstalSteelDataset, which uses tf.data instead.
    The reason for writing a custom generator is to allow for more direct control
    over patch sampling.
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
                 num_patches_per_image):
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

        self._anns_dict = load_annotations(self._train_anns_file)

        self._examples_by_class = self._build_examples_by_class_dict()

        # So that we dont have to share images between batches
        assert self._batch_size % self._num_patches_per_image == 0

        self._epoch_examples = None
        self.on_epoch_end()

    @classmethod
    def init_from_config(cls, config_path, img_list, is_training):
        with open(config_path) as f:
            cfg = yaml.load(f)

        num_patches_per_image = cfg['NUM_PATCHES_PER_IMAGE_TRAIN'] if is_training \
            else cfg['NUM_PATCHES_PER_IMAGE_VAL']

        return cls(
            img_list=img_list,
            is_training=is_training,
            train_img_dir=cfg['TRAIN_IMAGE_DIR'],
            train_anns_file=cfg['TRAIN_ANNOTATIONS_FILE'],
            img_height=cfg['IMG_HEIGHT'],
            img_width=cfg['IMG_WIDTH'],
            num_classes=cfg['NUM_CLASSES'],
            batch_size=cfg['BATCH_SIZE'],
            brightness_max_delta=cfg['BRIGHTNESS_MAX_DELTA'],
            contrast_lower_factor=cfg['CONTRAST_LOWER_FACTOR'],
            contrast_upper_factor=cfg['CONTRAST_UPPER_FACTOR'],
            patch_size=cfg['PATCH_SIZE'],
            num_patches_per_image=num_patches_per_image,
        )

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
                val_img_patches, val_ann_patches = self._get_validation_patches(img, ann)
                img_patches += val_img_patches
                ann_patches += val_ann_patches
                continue

            if target_class == 'no_class':
                random_patches_per_image = self._num_patches_per_image
            else:
                random_patches_per_image = int(self._num_patches_per_image / 2)

            # Get random patches from current image
            for _ in range(random_patches_per_image):
                img_patch, ann_patch = self._get_random_patch(img, ann)
                img_patch, ann_patch = self._apply_augmentations(img_patch, ann_patch)
                img_patches.append(img_patch)
                ann_patches.append(ann_patch)

            # Get heuristic-targeted patches from current image
            for _ in range(self._num_patches_per_image - random_patches_per_image):
                img_patch, ann_patch = self._get_targeted_patch(img, ann, target_class)
                img_patch, ann_patch = self._apply_augmentations(img_patch, ann_patch)
                img_patches.append(img_patch)
                ann_patches.append(ann_patch)

        if self._is_training:
            patches = list(zip(img_patches, ann_patches))
            random.shuffle(patches)
            img_patches, ann_patches = zip(*patches)

        return np.stack(img_patches), np.stack(ann_patches)

    def _get_validation_patches(self, img, ann):
        # TODO: only supports changing x position
        h, w, _ = img.shape
        x_step_size = int((w - self._patch_size) / (self._num_patches_per_image - 1))
        img_patches = []
        ann_patches = []
        for x in range(0, w - self._patch_size + 1, x_step_size):
            img_patches.append(img[0:self._patch_size, x:x+self._patch_size])
            ann_patches.append(ann[0:self._patch_size, x:x+self._patch_size])
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

    def _apply_augmentations(self, img, ann):
        # Concat img and ann so that flips get applied to both
        img_channels = img.shape[-1]
        combined = np.concatenate([img, ann], axis=-1)

        if random.choice([True, False]):
            combined = combined[:, ::-1, :] # Flip left-right

        if random.choice([True, False]):
            combined = combined[::-1, :, :] # Flip up-down

        img = combined[:, :, :img_channels]
        ann = combined[:, :, img_channels:]

        brightness_delta = random.uniform(-1 * self._brightness_max_delta,
                                          self._brightness_max_delta)
        contrast_factor = random.uniform(self._contrast_lower_factor,
                                         self._contrast_upper_factor)

        img = adjust_brightness_and_contrast(img, brightness_delta, contrast_factor)
        return img, ann

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

        # Set images_per_class_per_epoch to the number of images in the largest class
        images_per_class_per_epoch = 0
        for class_name, examples in self._examples_by_class.items():
            if len(examples) > images_per_class_per_epoch:
                images_per_class_per_epoch = len(examples)

        max_oversample_rate = 5
        epoch_examples = []
        for class_name, examples in self._examples_by_class.items():
            # Figure out appropriate oversample_rate for images containing current class
            num_examples = len(examples)
            oversample_rate = images_per_class_per_epoch / num_examples
            if oversample_rate > max_oversample_rate:
                print(f'Oversample rate of {oversample_rate} for class {class_name} is too large.'
                      f'Setting to {max_oversample_rate}.')
                oversample_rate = max_oversample_rate

            # Add oversampled images to epoch
            random.shuffle(examples)
            for i in range(int(oversample_rate * num_examples)):
                epoch_examples.append((examples[i % num_examples], class_name))

        random.shuffle(epoch_examples)
        self._epoch_examples = epoch_examples



#  class DeepQDataGenerator(tf.keras.utils.Sequence):
#     def __init__(self,
#                  base_model,
#                  steel_dataset,
#                  dataset_name,
#                  batch_size,
#                  threshold=0.85,
#                  img_height=256,
#                  img_width=1600,
#                  shuffle=True):
#         self._rle_preds = []
#         self._scores = []
#         self._batch_size = batch_size
#         self._img_height = img_height
#         self._img_width = img_width
#         self._shuffle = shuffle

#          img_list = steel_dataset.get_image_list(dataset_name)
#         for i in range(0, len(img_list), self._batch_size):
#             print(f'Preparing dataset batch {i / self._batch_size} / {len(img_list) / self._batch_size}...')
#             img_batch = []
#             ann_batch = []
#             for img_name in img_list[i:i+self._batch_size]:
#                 img, ann = steel_dataset.get_example_from_img_name(img_name)
#                 img_batch.append(img)
#                 ann_batch.append(ann)
#             img_batch = np.stack(img_batch, axis=0)
#             ann_batch = np.stack(ann_batch, axis=0)

#              y_batch = base_model.predict(img_batch)
#             y_post_batch = postprocess(y_batch, thresh=threshold)
#             print(f'y_post_batch.shape: {y_post_batch.shape}')
#             for i in range(self._batch_size):
#                 rles = []
#                 scores = []
#                 for c in range(y_post_batch.shape[-1]):
#                     # Score if we use the mask
#                     score_mask = dice_coeff_kaggle(y_post_batch[i, :, :, c:c+1],
#                                                    ann_batch[i, :, :, c:c+1])
#                     # Score if we predict empty
#                     score_no_mask = dice_coeff_kaggle(np.zeros_like(y_post_batch[i, :, :, c:c+1]),
#                                                       ann_batch[i, :, :, c:c+1])
#                     scores.append({'mask': score_mask, 'no_mask': score_no_mask})
#                     rles.append(dense_to_rle(y_post_batch[i, :, :, c]))
#                 self._rle_preds.append(rles)
#                 self._scores.append(scores)

#          self.on_epoch_end()

#      def __len__(self):
#         '''Number of batches per epoch'''
#         return int(np.floor(len(self._scores) / self._batch_size))

#      def __getitem__(self, index):
#         '''Generate one batch of data'''
#         indexes = self._indexes[index*self._batch_size:(index+1)*self._batch_size]
#         batch_rles = [self._rle_preds[i] for i in indexes]
#         batch_scores = [self._scores[i] for i in indexes]

#          x_batch = []
#         y_batch = []

#          for i in range(len(batch_rles)):
#             # Create single X from RLE
#             x = []
#             for rle in batch_rles[i]:
#                 x.append(rle_to_dense(rle, self._img_height, self._img_width))
#             x = np.stack(x, axis=-1)

#              # Create single y
#             y = []
#             for score_pair in batch_scores[i]:
#                 y.append([score_pair['mask'], score_pair['no_mask']])
#             y = np.array(y)
#             x_batch.append(x)
#             y_batch.append(y)

#          x_batch = np.stack(x_batch, axis=0)
#         y_batch = np.stack(y_batch, axis=0)
#         return x_batch, y_batch

#      def on_epoch_end(self):
#         '''Shuffle indexes after each epoch'''
#         self._indexes = np.arange(len(self._scores))
#         if self._shuffle == True:
#             np.random.shuffle(self._indexes)