import numpy as np
import tensorflow as tf


# Taken from https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py#L220-L254
# and updfated slightly
def upsample(filters, size, apply_dropout=False):
    """Upsamples an input.
    Conv2DTranspose => Batchnorm => Dropout => Relu
    Args:
        filters: number of filters
        size: filter size
        apply_dropout: If True, adds the dropout layer
    Returns:
        Upsample Sequential Model
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def build_pretrained_unet_model(
                     img_height,
                     img_width,
                     img_channels,
                     num_classes,
                     num_layers=3):


    base_model = tf.keras.applications.MobileNetV2(
        input_shape=[img_height, img_width, 3], include_top=False)

    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',      # 4x4
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
    down_stack.trainable = False

    inputs = tf.keras.layers.Input(shape=[img_height, img_width, 1])
    x = tf.tile(inputs, [1, 1, 1, 3]) # Convert from grayscale to RGB
    x = tf.cast(x, tf.float32) / 128.0 - 1 # Normalize

    # Downsampling through the model
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    up_stack = [
        upsample(512, 3),  # 4x4 -> 8x8
        upsample(256, 3),  # 8x8 -> 16x16
        upsample(128, 3),  # 16x16 -> 32x32
        upsample(64, 3),   # 32x32 -> 64x64
    ]

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    x = tf.keras.layers.Conv2DTranspose(
        num_classes, 3, strides=2,
        padding='same', activation='sigmoid')(x)  #64x64 -> 128x128

    return tf.keras.Model(inputs=inputs, outputs=x)
