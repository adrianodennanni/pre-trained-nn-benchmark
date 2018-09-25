import glob
import re
import tensorflow as tf
from PIL import Image

char_dict = {
'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18,
'J': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27,
'S': 28, 'T': 29, 'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35
}

# Loads a single picture and returns its binary and label
def load_image(filename):
  with tf.gfile.FastGFile(filename, 'rb') as f:
    image_data = f.read()
  label = re.search('\/(\w{6})\.jpg', filename).group(1)
  label_num = []
  for c in label:
    label_num.append(char_dict[c])
  return tf.compat.as_bytes(image_data), label_num

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_list_feature(list):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=list))

def write_to_trf_file(tfr_file_name, images_path):
  writer = tf.python_io.TFRecordWriter(tfr_file_name)
  images_list = glob.glob(images_path+"/*.jpg")
  for image_path in images_list:
    image, label = load_image(image_path)
    feature = {'label': _int64_list_feature(label),
               'image': _bytes_feature(image)}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())

  writer.close()

write_to_trf_file('./captcha_train.tfrecord', './captcha_dataset/train')
write_to_trf_file('./captcha_validation.tfrecord', './captcha_dataset/validation')
write_to_trf_file('./captcha_test.tfrecord', './captcha_dataset/test')
