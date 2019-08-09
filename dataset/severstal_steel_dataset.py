from collections import defaultdict

import numpy as np

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


def load_annotations(ann_file_name):
    '''Load all annotations from a file. Returns a dict mapping image filenames to annotations.
    '''
    anns = defaultdict(dict)

    with open(ann_file_name) as f:
        for line in f:
            file_name, rle_ann = line.split(',')
            if file_name == 'ImageId_ClassId': # Skip header
                continue

            img_id, cls_id = file_name.split('_')
            anns[img_id][cls_id] = rle_ann.strip()
    return anns
