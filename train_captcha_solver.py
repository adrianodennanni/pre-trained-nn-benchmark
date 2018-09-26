import sys
import os
import re
import tensorflow as tf
import numpy as np
from pathlib import Path
sys.path.append('./models')

# tf.enable_eager_execution()

N_CLASSES=36

features = {'label': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            'image': tf.FixedLenFeature([], tf.string)}

# Function for convert label into array of 6 integers
def extract_and_parse(example):
  parsed_example = tf.parse_single_example(example, features)
  image_decoded = tf.image.decode_jpeg(parsed_example['image'], 3)
  image_normalized = tf.image.convert_image_dtype(image_decoded, tf.float32)

  return (image_normalized, parsed_example['label'])

# Get parameters from command line
if(len(sys.argv) != 2):
  print('Usage: python train_captcha_solver.py (pre_trained | random_init)')
  sys.exit()
else:
  mode  = sys.argv[1]

from captcha_solver_xception import ModelTools as model_tools
if mode == 'pre_trained':
  model = model_tools.create_model(N_CLASSES, 'imagenet')
elif mode == 'random_init':
  model = model_tools.create_model(N_CLASSES, None)

TOTAL_EPOCHS = 30
BATCH_SIZE = 16
CHECKPOINT_DIRECTORY = './checkpoints/captcha_solver/{0}'.format(mode)
SAVE_CHECKPOINT_PATH = CHECKPOINT_DIRECTORY + '/{epoch:02d}_{val_dense_acc:.4f}.h5'

def create_dataset(tfrecord_file, batch_size=16):
  dataset = tf.data.TFRecordDataset(tfrecord_file)
  dataset = dataset.map(extract_and_parse)
  dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(1000))
  dataset  = dataset.batch(batch_size)

  iterator = dataset.make_one_shot_iterator()
  images, labels = iterator.get_next()

  return images, tf.split(labels, [1, 1, 1, 1, 1, 1], 1)


train_images, train_labels = create_dataset('./captcha_train.tfrecord', BATCH_SIZE)
validation_images, validation_labels = create_dataset('./captcha_validation.tfrecord', BATCH_SIZE)

# Creates some callbacks to be called each epoch
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    SAVE_CHECKPOINT_PATH,
    save_weights_only=True,
    verbose=1,
    monitor='val_dense_acc',
    save_best_only=True,
    mode='max'
)
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='./logs/captcha_solver/xception_{0}'.format(mode),
    histogram_freq=0,
    batch_size=BATCH_SIZE
)
reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_dense_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6
)

# Loads best weights if avaiable
if Path(CHECKPOINT_DIRECTORY).exists():
  epoch_number_array = []
  val_accuracy_array = []
  file_name_array = []
  for file in os.listdir(CHECKPOINT_DIRECTORY):
    epoch, val_acc = re.search(r'(\d\d)_(\d\.\d{4})\.h5', file).group(1,2)
    epoch_number_array.append(int(epoch))
    val_accuracy_array.append(float(val_acc))
    file_name_array.append(file)

  if len(val_accuracy_array) == 0:
    INITIAL_EPOCH = 0
  else:
    highest_acc = val_accuracy_array.index(max(val_accuracy_array))
    INITIAL_EPOCH = epoch_number_array[highest_acc]
    model_checkpoint_callback.best = val_accuracy_array[highest_acc]
    model.load_weights('./checkpoints/captcha_solver/' + mode + '/' + file_name_array[highest_acc])
else:
  os.makedirs(CHECKPOINT_DIRECTORY)
  INITIAL_EPOCH = 0


model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy']
              )

# Starts training the model
model.fit(train_images,
          train_labels,
          epochs=30,
          verbose=1,
          steps_per_epoch=int(200000/BATCH_SIZE),
          validation_data=(validation_images, validation_labels),
          validation_steps=int(5000/BATCH_SIZE),
          initial_epoch=INITIAL_EPOCH,
          callbacks=[model_checkpoint_callback, tensorboard_callback, reduce_lr_callback]
          )
