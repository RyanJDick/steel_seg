import os
from collections import defaultdict
import json
import random

def load_annotations(train_anns_file):
    '''Load all annotations from a file.
    Returns a dict mapping image filenames to annotations.
    '''
    anns = defaultdict(dict)
    with open(train_anns_file) as f:
        for line in f:
            file_name, rle_ann = line.split(',')
            if file_name == 'ImageId_ClassId': # Skip header
                continue

            img_id, cls_id = file_name.split('_')
            anns[img_id][cls_id] = rle_ann.strip()
    return anns


def split_data(imgs, test_split, val_split, batch_size, load_cached=True, cache_dir='data_split'):
    '''Perform train/val/test split. Loads split from cache files if load_cached == True.
    '''

    test_cache_file = os.path.join(cache_dir, 'test.json')
    val_cache_file = os.path.join(cache_dir, 'val.json')
    train_cache_file = os.path.join(cache_dir, 'train.json')

    if load_cached:
        if (os.path.exists(test_cache_file) and 
            os.path.exists(val_cache_file) and
            os.path.exists(train_cache_file)
        ):
            with open(test_cache_file) as f:
                test_imgs = json.load(f)
            
            with open(val_cache_file) as f:
                val_imgs = json.load(f)
            
            with open(train_cache_file) as f:
                train_imgs = json.load(f)
            
            print('Loaded train/test/validation split. '
                  f'Test: {len(test_imgs)}, Val: {len(val_imgs)}, Train: {len(train_imgs)}\n')
            return test_imgs, val_imgs, train_imgs
        else:
            print('Data split cache files missing. Will re-generate split.')
    
    def data_split_to_num_examples(total, split, batch_size):
        '''Split total examples based on split fraction,
        and make sure that the result is a multiple of batch_size.
        '''
        num_split_examples = int(total * split)
        num_split_examples = num_split_examples - (num_split_examples % batch_size)
        return num_split_examples

    # Split train, val, test, and make sure that a multiple of BATCH_SIZE is stored in each set
    num_test_examples = data_split_to_num_examples(len(imgs), test_split, batch_size)
    num_val_examples = data_split_to_num_examples(len(imgs), val_split, batch_size)

    num_train_examples = len(imgs) - num_val_examples - num_test_examples
    num_train_examples = num_train_examples - (num_train_examples % batch_size)

    random.shuffle(imgs)
    start_idx = 0
    test_imgs = imgs[start_idx:start_idx+num_test_examples]

    start_idx += num_test_examples
    val_imgs = imgs[start_idx:start_idx+num_val_examples]

    start_idx += num_val_examples
    train_imgs = imgs[start_idx:start_idx+num_train_examples]

    # Store split for repeatability
    os.makedirs(cache_dir, exist_ok=True)

    with open(test_cache_file, 'w') as f:
        json.dump(test_imgs, f)

    with open(val_cache_file, 'w') as f:
        json.dump(val_imgs, f)

    with open(train_cache_file, 'w') as f:
        json.dump(train_imgs, f)

    print('Split train/test/validation data. '
          f'Test: {len(test_imgs)}, Val: {len(val_imgs)}, Train: {len(train_imgs)}\n')
    return test_imgs, val_imgs, train_imgs
