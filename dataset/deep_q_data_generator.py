import numpy as np

from .train import dice_coeff_kaggle

def postprocess(y, thresh):
    # Only allow one class at each pixel
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
                 shuffle=True):
        self._rle_preds = []
        self._scores = []
        self._batch_size = batch_size
        self._shuffle = shuffle

        img_list = steel_dataset.get_image_list(dataset_name)
        for i in range(0, len(img_list), self._batch_size):
            print(f'Preparing dataset batch {i} / {len(img_list) / self._batch_size}...')
            img_batch = []
            ann_batch = []
            for img_name in img_list[i:i+self._batch_size]:
                img, ann = steel_dataset.get_example_from_img_name(ing_name)
                img_batch.append(img)
                ann_batch.append(ann)
            img_batch = np.stack(img_batch, axis=0)
            ann_batch = np.stack(ann_batch, axis=0)

            y_batch = model.predict(img_batch)
            y_post_batch = postprocess(y_batch, thresh=threshold)
            
            for i in range(self._batch_size):
                rles = []
                scores = []
                for c in range(y_post_batch.shape[-1]):
                    # Score if we use the mask
                    score_mask = dice_coeff_kaggle(y_post_batch[i, :, :, c:c+1],
                                                   ann[i, :, :, c:c+1])
                    # Score if we predict empty
                    score_no_mask = dice_coeff_kaggle(np.zeros_like(y_post_batch[i, :, :, c:c+1]),
                                                      ann[i, :, :, c:c+1])
                    scores.append({'mask': score_mask, 'no_mask': score_no_mask))
                    rles.append(dense_to_rle(y_post_batch[i, :, :, c]))
                self._rle_preds.append(rles)
                self._scores.append(scores)

        for img_name in img_list:
            img, ann = steel_dataset.get_example_from_img_name(img_name)
            img_batch = np.expand_dims(img, axis=0)
            y = model.predict(img_batch)
            y_post = postprocess(y, thresh=0.5, upper_thresh=0, num_px_thresh=0)
            for c in range(y_post.shape[-1]):
                rle = dense_to_rle(y_post[0, :, :, c])
                y_list.append(rle)



#######################################
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)