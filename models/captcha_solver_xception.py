import tensorflow as tf

class ModelTools:

  def create_model(n_classes, weights):
    kernel_initializer = tf.keras.initializers.glorot_uniform(seed=1337)
    trained_model = tf.keras.applications.xception.Xception(
        include_top=False,
        weights=weights,
        input_shape=[80, 160, 3],
        pooling='max')

    c1 = tf.keras.layers.Dense(n_classes, activation='softmax', kernel_initializer = kernel_initializer)(trained_model.output)
    c2 = tf.keras.layers.Dense(n_classes, activation='softmax', kernel_initializer = kernel_initializer)(trained_model.output)
    c3 = tf.keras.layers.Dense(n_classes, activation='softmax', kernel_initializer = kernel_initializer)(trained_model.output)
    c4 = tf.keras.layers.Dense(n_classes, activation='softmax', kernel_initializer = kernel_initializer)(trained_model.output)
    c5 = tf.keras.layers.Dense(n_classes, activation='softmax', kernel_initializer = kernel_initializer)(trained_model.output)
    c6 = tf.keras.layers.Dense(n_classes, activation='softmax', kernel_initializer = kernel_initializer)(trained_model.output)

    return tf.keras.Model(inputs=trained_model.input, outputs=[c1, c2, c3, c4, c5, c6])
