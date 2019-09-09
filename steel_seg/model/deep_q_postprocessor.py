import tensorflow as tf

def build_deep_q_model(img_height,
                       img_width,
                       num_classes,
                       num_layers=3,
                       activation=tf.keras.activations.elu,
                       kernel_initializer='he_normal',
                       kernel_size=(3, 3),
                       pool_size=(2, 4),
                       num_features=[4, 8, 16],
                       drop_prob=0.5):
    assert len(num_features) == num_layers

    inputs = tf.keras.Input(shape=(img_height, img_width, num_classes))
    x = inputs
    # Convert to float format
    #x = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

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
        if layer_idx < num_layers - 1:
            x = tf.keras.layers.MaxPooling2D(pool_size)(x)

    x = tf.keras.layers.Flatten()(x)
    class_outputs = [tf.keras.layers.Dense(2, activation=tf.keras.activations.sigmoid)(x) for _ in range(num_classes)]
    outputs = tf.stack(class_outputs, axis=1) # output shape: (batch, class, 2)
    #outputs = tf.keras.layers.Dense(4, activation=tf.keras.activations.softmax)(x)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return model
