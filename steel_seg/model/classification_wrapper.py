import tensorflow as tf

def build_classification_model(
    base_model,
    last_feature_layer,
    num_classes,
    output_layer,
    input_layer='input_1',
    kernel_size=[3, 3],
    num_conv_features=16,
    conv_activation=tf.keras.activations.elu,
    kernel_initializer='he_normal'):

    base_model.trainable = False
    x = base_model.get_layer(last_feature_layer).output
    x = tf.keras.layers.Conv2D(
        num_conv_features,
        kernel_size,
        activation=conv_activation,
        kernel_initializer=kernel_initializer,
        padding='same')(x)
    x = tf.keras.layers.Flatten()(x)
    cls_output = tf.keras.layers.Dense(num_classes, activation=tf.keras.activations.sigmoid, name='classification_output')(x)

    inputs = base_model.get_layer(input_layer).input

    seg_output = base_model.get_layer(output_layer).output
    print(seg_output)

    model = tf.keras.Model(inputs=[inputs], outputs=[seg_output, cls_output])
    return model
