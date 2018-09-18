import glob
import re
import tensorflow as tf
from PIL import Image

# Loads a single picture and returns its binary and label
def load_image(filename):
  image = Image.open(filename)
  label = re.search('\/(\w{6})\.jpg', filename).group(1)
  return tf.compat.as_bytes(image.tobytes()), tf.compat.as_bytes(label)

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_to_trf_file(tfr_file_name, images_path):
  writer = tf.python_io.TFRecordWriter(tfr_file_name)
  images_list = glob.glob(images_path+"/*.jpg")
  for image_path in images_list:
    image, label = load_image(image_path)
    feature = {'label': _bytes_feature(label),
               'image': _bytes_feature(image)}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())

  writer.close()

write_to_trf_file('./captcha_train.tfrecord', './captcha_dataset/train')
write_to_trf_file('./captcha_validation.tfrecord', './captcha_dataset/validation')
write_to_trf_file('./captcha_test.tfrecord', './captcha_dataset/test')
