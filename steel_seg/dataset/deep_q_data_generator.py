import numpy as np
import tensorflow as tf

from steel_seg.utils import dice_coeff_kaggle, rle_to_dense, dense_to_rle


def onehottify(x, n=None, dtype=float):
    '''1-hot encode x with the max value n (computed from data if n is None).
    '''
    x = np.asarray(x)
    n = np.max(x) + 1 if n is None else n
    return np.eye(n, dtype=dtype)[x]


def postprocess(y, thresh):
    '''Only allow one class at each pixel'''
    y_argmax = np.argmax(y, axis=-1)
    y_one_hot = onehottify(y_argmax, y.shape[-1])
    y_one_hot[y < thresh] = 0
    return y_one_hot


class DeepQDataGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 base_model,
                 steel_dataset,
                 dataset_name,
                 batch_size,
                 threshold=0.85,
                 img_height=256,
                 img_width=1600,
                 shuffle=True):
        self._rle_preds = []
        self._scores = []
        self._batch_size = batch_size
        self._img_height = img_height
        self._img_width = img_width
        self._shuffle = shuffle

        img_list = steel_dataset.get_image_list(dataset_name)
        for i in range(0, len(img_list), self._batch_size):
            print(f'Preparing dataset batch {i / self._batch_size} / {len(img_list) / self._batch_size}...')
            img_batch = []
            ann_batch = []
            for img_name in img_list[i:i+self._batch_size]:
                img, ann = steel_dataset.get_example_from_img_name(img_name)
                img_batch.append(img)
                ann_batch.append(ann)
            img_batch = np.stack(img_batch, axis=0)
            ann_batch = np.stack(ann_batch, axis=0)

            y_batch = base_model.predict(img_batch)
            y_post_batch = postprocess(y_batch, thresh=threshold)
            print(f'y_post_batch.shape: {y_post_batch.shape}')
            for i in range(self._batch_size):
                rles = []
                scores = []
                for c in range(y_post_batch.shape[-1]):
                    # Score if we use the mask
                    score_mask = dice_coeff_kaggle(y_post_batch[i, :, :, c:c+1],
                                                   ann_batch[i, :, :, c:c+1])
                    # Score if we predict empty
                    score_no_mask = dice_coeff_kaggle(np.zeros_like(y_post_batch[i, :, :, c:c+1]),
                                                      ann_batch[i, :, :, c:c+1])
                    scores.append({'mask': score_mask, 'no_mask': score_no_mask})
                    rles.append(dense_to_rle(y_post_batch[i, :, :, c]))
                self._rle_preds.append(rles)
                self._scores.append(scores)

        self.on_epoch_end()

    def __len__(self):
        '''Number of batches per epoch'''
        return int(np.floor(len(self._scores) / self._batch_size))

    def __getitem__(self, index):
        '''Generate one batch of data'''
        indexes = self._indexes[index*self._batch_size:(index+1)*self._batch_size]
        batch_rles = [self._rle_preds[i] for i in indexes]
        batch_scores = [self._scores[i] for i in indexes]

        x_batch = []
        y_batch = []

        for i in range(len(batch_rles)):
            # Create single X from RLE
            x = []
            for rle in batch_rles[i]:
                x.append(rle_to_dense(rle, self._img_height, self._img_width))
            x = np.stack(x, axis=-1)

            # Create single y
            y = []
            for score_pair in batch_scores[i]:
                y.append([score_pair['mask'], score_pair['no_mask']])
            y = np.array(y)
            x_batch.append(x)
            y_batch.append(y)

        x_batch = np.stack(x_batch, axis=0)
        y_batch = np.stack(y_batch, axis=0)
        return x_batch, y_batch

    def on_epoch_end(self):
        '''Shuffle indexes after each epoch'''
        self._indexes = np.arange(len(self._scores))
        if self._shuffle == True:
            np.random.shuffle(self._indexes)
