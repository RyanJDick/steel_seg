import numpy as np
import cv2


def dice_coeff_kaggle(y_pred, y_true):
    '''Dice Coefficient metric as defined in the Kaggle competition.
    '''
    y_pred = np.where(y_pred > 0.5, 1, 0)
    
    dice_scores = []
    for i in range(y_pred.shape[-1]):
        y_pred_sum = np.sum(y_pred[:, :, i])
        y_true_sum = np.sum(y_true[:, :, i])
        if y_pred_sum == 0 and y_true_sum == 0:
            dice_scores.append(1.0)
            continue
        intersection = np.sum(y_pred[:, :, i] * y_true[:, :, i])
        dice_scores.append(
            2 * intersection / (y_pred_sum + y_true_sum))
    return np.mean(dice_scores)


def rle_to_dense(rle, img_height, img_width):
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


def visualize_segmentations(img, anns):
    '''Visualize a set of segmentations (ground truth or predicted) on an image.
    '''
    vis_img = img.copy()
    
    colours = [[0, 235, 235], [0, 210, 0], [0, 0, 255], [255, 0, 255]]
    for i in range(4):
        mask = anns[:, :, i]
        if np.any(mask):
            print(f'Class {i}')
        kernel = np.ones((10, 10), np.uint8) 
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        contour_mask = dilated_mask - mask
        for c in range(3):
            vis_img[contour_mask == 1, c] = colours[i][c]
    return vis_img


def onehottify(x, n=None, dtype=float):
    '''1-hot encode x with the max value n (computed from data if n is None).
    '''
    x = np.asarray(x)
    n = np.max(x) + 1 if n is None else n
    return np.eye(n, dtype=dtype)[x]
