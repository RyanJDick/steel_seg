import os
import multiprocessing

import yaml
from datetime import datetime
import numpy as np
import skopt
import tensorflow as tf
from tensorflow.keras import backend as K

from steel_seg.dataset.severstal_steel_dataset import SeverstalSteelDataset
from steel_seg.model.unet import build_unet_model, postprocess
from steel_seg.utils import dice_coeff_kaggle


def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    
    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def focal_loss(alpha=0.25, gamma=2):
    def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)

        return (tf.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b 

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        logits = tf.log(y_pred / (1 - y_pred))

        loss = focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)

        return tf.reduce_mean(loss)

    return loss

def dice_coef_channel_helper(y_true, y_pred):
    '''Helper funtion used by dice metrics to compute the dice score for a single image and class.
    '''
    y_true_b = K.flatten(y_true) > 0.5
    y_pred_b = K.flatten(y_pred) > 0.5
    y_true_f = tf.to_float(y_true_b)
    y_pred_f = tf.to_float(y_pred_b)
    y_true_i = tf.to_int32(y_true_b)
    y_pred_i = tf.to_int32(y_pred_b)

    def empty():
        return 1.0

    def not_empty():
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))

    y_true_and_y_pred_empty = tf.equal(K.sum(y_true_f) + K.sum(y_pred_f), 0)
    return tf.cond(y_true_and_y_pred_empty, empty, not_empty)

def dice_coef(batch_size, num_classes=4):
    def _dice_coef(y_true, y_pred):
        batch_scores = []
        for b in range(batch_size):
            for i in range(num_classes):
                dice_score = dice_coef_channel_helper(y_true[b, :, :, i], y_pred[b, :, :, i])
                batch_scores.append(dice_score)
        return tf.add_n(batch_scores) / (batch_size * num_classes)
    return _dice_coef


# class DiceCoefByClassAndEmptiness(tf.keras.metrics.Metric):
#     def __init__(
#         self,
#         cls_id,
#         empty_masks_only,
#         batch_size,
#         name='dice_coef_by_class_and_emptiness',
#         **kwargs
#     ):
#         super(DiceCoefByClassAndEmptiness, self).__init__(name=name, **kwargs)
#         self.cls_id = cls_id
#         self.empty_masks_only = empty_masks_only
#         self.batch_size = batch_size
#         self.dice_score_sum = self.add_weight(name='dice_score_sum', initializer='zeros')
#         self.dice_score_count = self.add_weight(name='dice_score_count', initializer='zeros')

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         if sample_weight is not None:
#             raise NotImplementedError("sample_weight not supported by DiceCoefOnEmptyByClass")
#         sample_scores = []
#         sample_weights = []
#         for b in range(self.batch_size):
#             dice_score = dice_coef_channel_helper(
#                 y_true[b, :, :, self.cls_id],
#                 y_pred[b, :, :, self.cls_id])

#             def on_gt_empty():
#                 if self.empty_masks_only:
#                     self.dice_score_sum.assign_add(tf.add_n(batch_scores))
#                     self.dice_score_count.assign_add(len(batch_scores))

#             def on_gt_not_empty():
#                 if not self.empty_masks_only:
#                     self.dice_score_sum.assign_add(tf.add_n(batch_scores))
#                     self.dice_score_count.assign_add(len(batch_scores))

#             tf.cond(gt_is_empty, on_gt_empty, on_gt_not_empty)

#     def result(self):
#         return self.dice_score_sum / self.dice_score_count

class DiceCoefByClassAndEmptiness(tf.keras.metrics.Mean):
    '''
    '''
    def __init__(self,
        cls_id,
        empty_masks_only,
        batch_size,
        name='dice_coef_by_class_and_emptiness',
        **kwargs
    ):
        super(DiceCoefByClassAndEmptiness, self).__init__(name=name, **kwargs)
        self.cls_id = cls_id
        self.empty_masks_only = empty_masks_only
        self.batch_size = batch_size

    def update_state(self, y_true, y_pred, sample_weight=None):
        if sample_weight is not None:
            raise ValueError("sample_weight not supported by DiceCoefByClassAndEmptiness")

        sample_scores = []
        for b in range(self.batch_size):
            sample_scores.append(
                dice_coef_channel_helper(
                    y_true[b, :, :, self.cls_id],
                    y_pred[b, :, :, self.cls_id])
            )
        sample_scores = tf.stack(sample_scores)

        y_true_counts = tf.math.reduce_sum(y_true[:, :, :, self.cls_id], axis=[1, 2])
        sample_weights = tf.clip_by_value(y_true_counts, 0.0, 1.0) # Clip counts greater than 0 to 1.0
        if self.empty_masks_only:
            sample_weights = sample_weights * -1.0 + 1.0 # Flip 0.0 and 1.0 values

        return super(DiceCoefByClassAndEmptiness, self).update_state(
            sample_scores, sample_weight=sample_weights)

def dice_coef_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss_multi_class(y_true, y_pred):
    smooth = 0.0001
    numerator_weight = 10
    numerator = numerator_weight * 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2)) + smooth
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2)) + smooth

    # Was 1 - dice_coeff, but made it numerator_weight - dice_coeff to keep the loss positive
    per_channel_dice_loss = numerator_weight - numerator / denominator
    return tf.reduce_mean(per_channel_dice_loss)

def weighted_binary_crossentropy(beta, from_logits=False):
    def _weighted_binary_crossentropy(target, output):
        if not from_logits:
            # When sigmoid activation function is used for output operation, we
            # use logits from the sigmoid function directly to compute loss in order
            # to prevent collapsing zero when training.
            assert len(output.op.inputs) == 1
            output = output.op.inputs[0]
        return tf.nn.weighted_cross_entropy_with_logits(logits=output, labels=target, pos_weight=beta)
    return _weighted_binary_crossentropy

def class_weighted_binary_crossentropy(cls_weights, from_logits=False):
    def _class_weighted_binary_crossentropy(target, output):
        if not from_logits:
            # When sigmoid activation function is used for output operation, we
            # use logits from the sigmoid function directly to compute loss in order
            # to prevent collapsing zero when training.
            assert len(output.op.inputs) == 1
            output = output.op.inputs[0]

        cls_cross_entropy_losses = []
        for i, cls_weight in enumerate(cls_weights):
            target[:, :, :, i]
            cls_cross_entropy_losses.append(
                tf.nn.weighted_cross_entropy_with_logits(
                    logits=output[:, :, :, i],
                    labels=target[:, :, :, i],
                    pos_weight=cls_weight)
            )

        return sum(cls_cross_entropy_losses)
    return _class_weighted_binary_crossentropy

def pixel_map_weighted_binary_crossentropy(cls_weights, from_logits=False):
    def _pixel_map_weighted_binary_crossentropy(target, output):
        if not from_logits:
            # When sigmoid activation function is used for output operation, we
            # use logits from the sigmoid function directly to compute loss in order
            # to prevent collapsing zero when training.
            assert len(output.op.inputs) == 1
            output = output.op.inputs[0]

        class_weights = tf.constant(np.array(cls_weights, dtype=np.float32))

        weight_map = tf.multiply(target, class_weights) # Multiply class weights at every pixel
        weight_map = tf.reduce_sum(weight_map, axis=-1, keepdims=True) # Sum weights pixelwise
        weight_map += 1.0

        loss_map = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output)
        weighted_loss = tf.multiply(loss_map, weight_map)

        loss = tf.reduce_mean(weighted_loss)
        return loss
    return _pixel_map_weighted_binary_crossentropy

def class_weighted_binary_classification_crossentropy(cls_weights, from_logits=False):
    '''Loss function used by binary mask classifier (NOT dense segmentation)
    '''
    def _class_weighted_binary_classification_crossentropy(target, output):
        if not from_logits:
            # When sigmoid activation function is used for output operation, we
            # use logits from the sigmoid function directly to compute loss in order
            # to prevent collapsing zero when training.
            assert len(output.op.inputs) == 1
            output = output.op.inputs[0]
        class_weights = tf.constant(np.array(cls_weights, dtype=np.float32))

        losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output)
        loss_weights = tf.multiply(target, class_weights) # Weight the positive examples, not the negatives ones
        loss_weights += 1.0
        weighted_losses = tf.multiply(losses, loss_weights)
        loss = tf.reduce_mean(weighted_losses)
        return loss
    return _class_weighted_binary_classification_crossentropy

def binary_accuracy_by_class(cls_id, threshold=0.5):
    def _binary_accuracy_by_class(y_true, y_pred):
        y_true = y_true[:, cls_id]
        y_pred = y_pred[:, cls_id]

        y_true = tf.to_int32(y_true)
        y_pred = tf.to_int32(y_pred > threshold)

        total = tf.shape(y_true)[0]
        correct = tf.reduce_sum(tf.to_int32(tf.math.equal(y_pred, y_true)))

        return tf.math.divide(correct, total)
    fn = _binary_accuracy_by_class
    fn.__name__ += f'_{cls_id}' # Hack to get the cls_id in the metric name
    return fn

def eval(model, dataset, img_list, threshold=0.5):
    dice_coeffs = []
    for img_name in img_list:
        img, ann = dataset.get_example_from_img_name(img_name)
        img_batch = np.expand_dims(img, axis=0)
        y = model.predict(img_batch)
        y_one_hot = postprocess(y, threshold)
        dice_coeffs.append(dice_coeff_kaggle(y_one_hot[0, :, :, :], ann))
    mean_dice_coeff = np.mean(dice_coeffs)
    print(f'Mean dice coeff: {mean_dice_coeff}')
    return mean_dice_coeff


HP_SPACE = [
    skopt.space.Real(0.0001, 0.01, name='learning_rate', prior='log-uniform'),
    skopt.space.Real(5, 100, name='cross_entropy_weight'),
    skopt.space.Integer(1, 8, name='layer_1_features'),
    skopt.space.Integer(1, 16, name='layer_2_features'),
    skopt.space.Integer(1, 32, name='layer_3_features'),
    skopt.space.Integer(1, 64, name='layer_4_features'),
]

@skopt.utils.use_named_args(HP_SPACE)
def train_process(**kwargs):
    queue = multiprocessing.Queue()
    kwargs['queue'] = queue
    p = multiprocessing.Process(target=train, kwargs=kwargs)
    p.start()
    score = queue.get()
    p.join()
    return score

def train(
    learning_rate,
    cross_entropy_weight,
    layer_1_features,
    layer_2_features,
    layer_3_features,
    layer_4_features,
    queue,
    cfg_path='SETTINGS.yaml'):
    '''Train the model, print the progress as training progresses, and return the final dice_coeff
    on the validation set
    '''

    print('Training with configs:')
    print(f'learning_rate: {learning_rate}')
    print(f'cross_entropy_weight: {cross_entropy_weight}')
    print(f'layer_features: {[layer_1_features, layer_2_features, layer_3_features, layer_4_features]}')
    with open(cfg_path) as f:
        cfg = yaml.load(f)

    # May have to add CUPTI path to LD_LIBRARY_PATH
    # export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64
    #os.environ["LD_LIBRARY_PATH"] += os.pathsep + '/usr/local/cuda/extras/CUPTI/lib64'
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true" # Necessary for CUDA 10 or something?
    os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"
    os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1"
    os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_LOSS_SCALING"] = "1"

    dataset = SeverstalSteelDataset.init_from_config(cfg_path)
    train_data, train_batches = dataset.create_dataset(dataset_type='training')
    val_data, val_batches = dataset.create_dataset(dataset_type='validation')

    model = build_unet_model(
        img_height=cfg['IMG_HEIGHT'],
        img_width=cfg['IMG_WIDTH'],
        img_channels=1,
        num_classes=cfg['NUM_CLASSES'],
        num_layers=4,
        activation=tf.keras.activations.elu,
        kernel_initializer='he_normal',
        kernel_size=(3, 3),
        pool_size=(2, 4),
        num_features=[layer_1_features, layer_2_features, layer_3_features, layer_4_features],
        drop_prob=0.5)

    model.summary()

    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate),
              loss=weighted_binary_crossentropy(cross_entropy_weight),#'binary_crossentropy',
              metrics=[tf.keras.metrics.BinaryAccuracy(), dice_coef])#[dice_coef, 'accuracy'])

    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_path = f'checkpoints/cp_{date_str}.ckpt'
    logdir = "logs/" + date_str
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=logdir),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, save_weights_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            patience=4, monitor='val_loss'),
    ]

    results = model.fit(train_data,
                        epochs=200,
                        verbose=2,
                        callbacks=callbacks,
                        validation_data=val_data,
                        steps_per_epoch=train_batches,
                        validation_steps=val_batches,
                        validation_freq=1)

    val_imgs = dataset.get_image_list('validation')
    print('Running evaluation on validation set...')
    mean_dice_coeff = eval(model, dataset, val_imgs)

    K.clear_session() # Clear GPU memory

    queue.put(-1.0 * mean_dice_coeff)

if __name__ == '__main__':
    res = skopt.gp_minimize(
        train_process,              # the function to minimize
        HP_SPACE,              # the bounds on each dimension of x
        acq_func="EI",      # the acquisition function
        n_calls=20,         # the number of evaluations of f 
        n_random_starts=5,  # the number of random initialization points
        random_state=123)   # the random seed

    skopt.dump(res, 'skopt_result.pkl')

#python train.py | tee skopt_logs.txt
