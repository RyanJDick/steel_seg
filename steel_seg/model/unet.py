import tensorflow as tf


def build_unet_model(img_height,
                     img_width,
                     img_channels,
                     num_classes,
                     num_layers=3,
                     activation=tf.keras.activations.elu,
                     kernel_initializer='he_normal',
                     kernel_size=(3, 3),
                     pool_size=(2, 2),
                     num_features=[16, 32, 64],
                     drop_prob=0.5):

    assert len(num_features) == num_layers

    inputs = tf.keras.Input(shape=(img_height, img_width, img_channels))

    # Convert to float format
    x = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    # Down-sampling layers
    down_outputs = []
    for layer_idx in range(num_layers):
        x = tf.keras.layers.Conv2D(num_features[layer_idx],
                                   kernel_size,
                                   activation=activation,
                                   kernel_initializer=kernel_initializer,
                                   padding='same')(x)
        x = tf.keras.layers.Dropout(drop_prob)(x)
        x = tf.keras.layers.Conv2D(num_features[layer_idx],
                                   kernel_size,
                                   activation=activation,
                                   kernel_initializer=kernel_initializer,
                                   padding='same')(x)
        down_outputs.append(x)

        # Apply max pooling only if this is not the last down-sampling block
        if layer_idx < num_layers - 1:
            x = tf.keras.layers.MaxPooling2D(pool_size)(x)

    # Up-sampling layers
    for layer_idx in range(num_layers - 2, -1, -1):
        x = tf.keras.layers.Conv2DTranspose(num_features[layer_idx],
                                            pool_size,
                                            strides=pool_size,
                                            padding='same')(x)
        x = tf.keras.layers.concatenate([x, down_outputs[layer_idx]])
        x = tf.keras.layers.Conv2D(num_features[layer_idx],
                                   kernel_size,
                                   activation=activation,
                                   kernel_initializer=kernel_initializer,
                                   padding='same')(x)
        x = tf.keras.layers.Dropout(drop_prob)(x)
        x = tf.keras.layers.Conv2D(num_features[layer_idx],
                            kernel_size,
                            activation=activation,
                            kernel_initializer=kernel_initializer,
                            padding='same')(x)

    # FC Prediction Layer
    outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), kernel_initializer=kernel_initializer, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return model
