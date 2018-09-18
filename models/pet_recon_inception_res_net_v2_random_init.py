import tensorflow as tf

class ModelTools:

  def create_model(n_classes):
    trained_model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(
        include_top=False,
        weights=None,
        input_shape=[100, 100, 3],
        pooling='max')

    kernel_initializer = tf.keras.initializers.glorot_uniform(seed=1337)
    model = tf.keras.Sequential()

    model.add(trained_model)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(n_classes, activation='softmax', kernel_initializer = kernel_initializer))

    return model
