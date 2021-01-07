# This script is intended to be run in Kaggle's kernel environment to generate a submission.

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

import os
import numpy as np # linear algebra
import tensorflow as tf
import keras
from PIL import Image
import cv2

severstal_test_dir = '/kaggle/input/severstal-steel-defect-detection/test_images/'

cls_model_path = '/kaggle/input/mobilenet-cls-model-imgaug-20191020-153642/mobilenet_classification_model_imgaug20191020-153642.h5'
cls_thresh = [0.9, 0.9, 0.7, 0.5]

seg_model_path_1 = '/kaggle/input/resnet-seg-model-imgaug-20191024011145/resnet_seg_model_imgaug_20191024-011145.h5'
seg_model_path_2 = '/kaggle/input/resnet34-fcn-seg-model-imgaug-20191024-204740/resnet34_fcn_seg_model_imgaug_20191024-204740.h5'
seg_thresh = [0.5, 0.5, 0.5, 0.3]

def load_img(img_path):
    img = np.array(Image.open(img_path))
    img_gray = img[:, :, :1] # All channels are the same
    return img_gray

def onehottify(x, n=None, dtype=np.uint8):
    '''1-hot encode x with the max value n (computed from data if n is None).
    '''
    x = np.asarray(x)
    n = np.max(x) + 1 if n is None else n
    return np.eye(n, dtype=dtype)[x]

def postprocess(y, y_cls, thresh=None, cls_thresh=None, min_px_area=500):
    if thresh is None:
        thresh = [0.85, 0.85, 0.85, 0.85]
    if cls_thresh is None:
        cls_thresh = [0.5, 0.5, 0.5, 0.5]

    # TODO: handle batches properly
    batches, height, width, classes = y.shape
    assert batches == 1
    
    # Only allow one class at each pixel
    y_argmax = np.argmax(y, axis=-1)
    y_one_hot = onehottify(y_argmax, y.shape[-1], dtype=int)
    for c in range(classes):
        y_one_hot[:, :, :, c][y[:, :, :, c] < thresh[c]] = 0 # Background
    
    for c in range(classes):
        if y_cls is not None and y_cls[0, c] < cls_thresh[c]:
            y_one_hot[:, :, :, c] = 0
        else:
            num_component, component = cv2.connectedComponents(y_one_hot[0, :, :, c].astype(np.uint8))
            for comp_idx in range(1, num_component):
                comp_mask = (component == comp_idx)
                if comp_mask.sum() < min_px_area:
                    y_one_hot[0, :, :, c][comp_mask] = 0
    return y_one_hot

def dense_to_rle(dense):
    '''Convert the dense np ndarray representation of a single class mask to the equivalent rle
    representation.
    '''
    assert len(dense.shape) == 2
    # Use Fortran (column-major) ordering
    pixels = dense.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def apply_tta(img):
    h, w, c = img.shape # Assert only 3 dimensions
    
    img_flip_h = img[:, ::-1, :]
    img_flip_v = img[::-1, :, :]
    img_flip_hv = img[::-1, ::-1, :]
    
    tta_batch = np.stack([img, img_flip_h, img_flip_v, img_flip_hv])
    return tta_batch

def combine_tta_preds(y_tta):
    dims = len(y_tta.shape)
    if dims == 4:
        y_0 = y_tta[0, :, :, :]
        y_1 = y_tta[1, :, ::-1, :]
        y_2 = y_tta[2, ::-1, :, :]
        y_3 = y_tta[3, ::-1, ::-1, :]

        y = np.stack([y_0, y_1, y_2, y_3])
    elif dims == 2:
        y = y_tta
    y = np.mean(y, axis=0, keepdims=True)
    return y

def store_preds(y_one_hot, img_name, preds):
    for c in range(y_one_hot.shape[-1]):
        preds.append(f'{img_name}_{c+1},{dense_to_rle(y_one_hot[0, :, :, c])}')


# Load model
cls_model = tf.keras.models.load_model(cls_model_path)
seg_model_1 = keras.models.load_model(seg_model_path_1, custom_objects={'tf': tf})
seg_model_2 = keras.models.load_model(seg_model_path_2, custom_objects={'tf': tf})

# Make preds
preds = []
preds.append('ImageId_ClassId,EncodedPixels')
img_names = os.listdir(severstal_test_dir)
img_names.sort()
for i, img_name in enumerate(img_names):
    if i % 100 == 0:
        print(f'Running inference on image {i} / {len(img_names)}')
    img_path = os.path.join(severstal_test_dir, img_name)
    img = load_img(img_path)
    
    #img_batch = np.expand_dims(img, axis=0)
    img_tta_batch = apply_tta(img)

    y_cls = cls_model.predict(img_tta_batch)
    y_cls = combine_tta_preds(y_cls)
    
    all_empty = True
    for c in range(4):
        if y_cls is not None and y_cls[0, c] > cls_thresh[c]:
            all_empty = False
    
    if all_empty:
        y_one_hot = np.zeros((1, 256, 1600, 4), dtype=np.uint8)
    else: 
        y_seg_1 = seg_model_1.predict(img_tta_batch)
        y_seg_1 = combine_tta_preds(y_seg_1)
        
        y_seg_2 = seg_model_2.predict(img_tta_batch)
        y_seg_2 = combine_tta_preds(y_seg_2)
        
        y_seg = (y_seg_1 + y_seg_2) / 2.0
        y_one_hot = postprocess(y_seg, y_cls, seg_thresh, cls_thresh)
    
    store_preds(y_one_hot, img_name, preds)

# Save to file
with open('submission.csv', 'w') as f:
    f.writelines([p + '\n' for p in preds])
