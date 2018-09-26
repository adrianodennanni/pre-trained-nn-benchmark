import sys
import os
import re
sys.path.append('./models')
from pathlib import Path
import tensorflow as tf

# Get parameters from command line
if(len(sys.argv) != 3):
  print('Usage: python test_pet_recon.py model (pre_trained | random_init)')
  sys.exit()
else:
  model_name = sys.argv[1]
  mode  = sys.argv[2]

N_CLASSES = 3

# Load the corresponding model
if model_name == 'xception':
  from pet_recon_xception import ModelTools as model_tools
  if mode == 'pre_trained':
    model = model_tools.create_model(N_CLASSES, 'imagenet')
  elif mode == 'random_init':
    model = model_tools.create_model(N_CLASSES, None)
else:
  print('Model ' + model_name + ' could not be found.')
  sys.exit()

checkpoint_directory = './checkpoints/pet_classifier/{0}_{1}'.format(model_name, mode)

# Loads best weights
if Path(checkpoint_directory).exists():
  epoch_number_array = []
  val_accuracy_array = []
  file_name_array = []
  for file in os.listdir(checkpoint_directory):
    epoch, val_acc = re.search(r'(\d\d)_(\d\.\d{4})\.h5', file).group(1,2)
    epoch_number_array.append(int(epoch))
    val_accuracy_array.append(float(val_acc))
    file_name_array.append(file)

  if len(val_accuracy_array) > 0:
    highest_acc = val_accuracy_array.index(max(val_accuracy_array))
    model.load_weights(checkpoint_directory + '/' + file_name_array[highest_acc])

# Prepares the model to run
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy']
              )

os.system('cls' if os.name == 'nt' else 'clear')

test_pet_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    data_format='channels_last',
    rescale=1./255
)

test_pet_batches = test_pet_generator.flow_from_directory(
    batch_size=1,
    directory='./pet_dataset/test',
    target_size=[100, 100],
    class_mode='categorical'
)

evaluation_pet = model.evaluate_generator(test_pet_batches, verbose=1)

print(model.metrics_names)
print('Pet test dataset: ' + str(evaluation_pet))
