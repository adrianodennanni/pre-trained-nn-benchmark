import sys
import os
import re
sys.path.append('./models')
from pathlib import Path
import tensorflow as tf

# Get parameters from command line
if(len(sys.argv) != 2):
  print('Usage: python test_captcha_solver.py (pre_trained | random_init)')
  sys.exit()
else:
  mode  = sys.argv[1]

N_CLASSES = 36

# Load the corresponding model
from captcha_solver_xception import ModelTools as model_tools
if mode == 'pre_trained':
  model = model_tools.create_model(N_CLASSES, 'imagenet')
elif mode == 'random_init':
  model = model_tools.create_model(N_CLASSES, None)
else:
  print('Model ' + model_name + ' could not be found.')
  sys.exit()

checkpoint_directory = './checkpoints/captcha_solver/{0}'.format(mode)

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

features = {'label': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            'image': tf.FixedLenFeature([], tf.string)}

# Function for convert label into array of 6 integers
def extract_and_parse(example):
  parsed_example = tf.parse_single_example(example, features)
  image_decoded = tf.image.decode_jpeg(parsed_example['image'], 3)
  image_normalized = tf.image.convert_image_dtype(image_decoded, tf.float32)

  return (image_normalized, parsed_example['label'])

def create_dataset(tfrecord_file, batch_size=16):
  dataset = tf.data.TFRecordDataset(tfrecord_file)
  dataset = dataset.map(extract_and_parse)
  dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(1000))
  dataset  = dataset.batch(batch_size)

  iterator = dataset.make_one_shot_iterator()
  images, labels = iterator.get_next()

  return images, tf.split(labels, [1, 1, 1, 1, 1, 1], 1)

# Prepares the model to run
model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy']
              )

test_images, test_labels = create_dataset('./captcha_test.tfrecord', 16)

os.system('cls' if os.name == 'nt' else 'clear')

evaluation_captcha = model.evaluate(test_images, test_labels, verbose=1, steps = int(5000/16))

print(model.metrics_names)
print('Captcha dataset: ' + str(evaluation_captcha))
