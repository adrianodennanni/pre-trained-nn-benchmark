import sys
import os
import re
from pathlib import Path
import tensorflow as tf
import numpy as np
sys.path.append('./models')

# Get parameters from command line
if(len(sys.argv) != 3):
  print('Usage: python train_pet_recon.py model (pre-trained | random-init)')
  sys.exit()
else:
  model_name = sys.argv[1]
  mode  = sys.argv[2]

# Load the corresponding model
if model_name == 'xception':
  if mode == 'pre_trained':
    from pet_recon_xception_pre_trained import ModelTools as model_tools
  elif mode == 'random_init':
    from pet_recon_xception_random_init import ModelTools as model_tools
elif model_name == 'inception_res_net_v2':
  if mode == 'pre_trained':
    from pet_recon_inception_res_net_v2_pre_trained import ModelTools as model_tools
  elif mode == 'random_init':
    from pet_recon_inception_res_net_v2_random_init import ModelTools as model_tools
elif model_name == 'mobile_net':
  if mode == 'pre_trained':
    from pet_recon_mobile_net_pre_trained import ModelTools as model_tools
  elif mode == 'random_init':
    from pet_recon_mobile_net_random_init import ModelTools as model_tools
else:
  print('Model ' + model_name + ' could not be found.')
  sys.exit()

TOTAL_EPOCHS = 30
BATCH_SIZE = 16
N_CLASSES = 3
TRAIN_DATASET_PATH = './pet_dataset/train'
VALIDATION_DATASET_PATH = './pet_dataset/validation'
CHECKPOINT_DIRECTORY = './checkpoints/pet_classifier/{0}_{1}'.format(model_name, mode)
SAVE_CHECKPOINT_PATH = CHECKPOINT_DIRECTORY + '/{epoch:02d}_{val_acc:.4f}.h5'

if not os.path.exists(CHECKPOINT_DIRECTORY):
  os.makedirs(CHECKPOINT_DIRECTORY)

# Declare generators that read from folders
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip=True,
    data_format='channels_last',
    rescale=1. / 255
)

train_batches = train_generator.flow_from_directory(
    batch_size=BATCH_SIZE,
    directory=TRAIN_DATASET_PATH,
    target_size=[100, 100],
    class_mode='categorical',
    shuffle=True
)

val_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    data_format='channels_last',
    rescale=1. / 255
)

val_batches = train_generator.flow_from_directory(
    batch_size=BATCH_SIZE,
    directory=VALIDATION_DATASET_PATH,
    target_size=[100, 100],
    class_mode='categorical'
)

TRAIN_DATASET_SIZE = len(train_batches)
VAL_DATASET_SIZE   = len(val_batches)


# Weighted losses for class equilibrium
unique, counts = np.unique(train_batches.classes, return_counts=True)
class_weigths = dict(zip(unique, np.true_divide(counts.sum(), N_CLASSES*counts)))


# Creates some callbacks to be called each epoch
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    SAVE_CHECKPOINT_PATH,
    save_weights_only=True,
    verbose=1,
    monitor='val_acc',
    save_best_only=True,
    mode='max'
)
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='./logs/pet_classifier/{0}_{1}'.format(model_name, mode),
    histogram_freq=0,
    batch_size=BATCH_SIZE
)
reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6
)

model = model_tools.create_model(N_CLASSES)

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
    model.load_weights('./checkpoints/pet_classifier/' + '{0}_{1}/'.format(model_name, mode) + file_name_array[highest_acc])
else:
  os.makedirs(CHECKPOINT_DIRECTORY)
  INITIAL_EPOCH = 0

# Prepares the model to run
model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'categorical_crossentropy',
              metrics = ['accuracy']
              )

# Starts training the model
model.fit_generator(train_batches,
                    epochs=TOTAL_EPOCHS,
                    verbose=1,
                    steps_per_epoch=TRAIN_DATASET_SIZE,
                    validation_data=val_batches,
                    initial_epoch=INITIAL_EPOCH,
                    validation_steps=VAL_DATASET_SIZE,
                    class_weight=class_weigths,
                    callbacks=[model_checkpoint_callback, tensorboard_callback, reduce_lr_callback]
                    )
