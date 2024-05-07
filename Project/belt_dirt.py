import os
from google.colab import drive
import tensorflow as tf
from tensorflow.keras import layers, Sequential, callbacks, metrics
from random import choices, uniform
from matplotlib import pyplot as plt
from tensorflow.keras.utils import save_img, image_dataset_from_directory
from time import time
import numpy as np

if "drive" not in os.listdir():
  drive.mount("/content/drive")
  !unzip -q /content/drive/MyDrive/Dataset_DirtBuildup_BeltConveyor.zip

def process_image(img_file):
  image = tf.io.read_file(img_file)
  image = tf.io.decode_jpeg(image)
  height, width, band = image.shape
  fig, ax = plt.subplots(1, 5)
  ax[2].set_title(img_file.split("/")[-1])
  for i in range(5):
    k = uniform(0.65, 1)
    new_height = int(k * height)
    new_width = int(k * width)

    _image = tf.image.random_crop(image,
                                  size=[new_height, new_width, 3])
    aspect_ratio = uniform(0.75, 1.33)
    new_height = int(new_width * aspect_ratio)
    _image = tf.image.resize(_image,
                             size=(new_height, new_width))

    if uniform(0, 1) >= 0.5:
      _image = tf.image.flip_left_right(_image)
      _image = tf.image.flip_up_down(_image)

    _image = tf.image.resize(_image, size=(224, 224))

    ax[i].imshow(_image/255)
    ax[i].axis("off")

    filename, ext = tuple(img_file.split("."))
    save_img(f"{filename}-{i}.{ext}", _image)
  plt.show()
  os.remove(img_file)

if "01.JPG" in os.listdir("/content/Clean"):
  image_type = ("/content/Clean/", "/content/Dirt Buildup/")
  for img_type in image_type:
    print(img_type.split("/")[-1])
    start = time()
    foldername = img_type
    image_files = os.listdir(foldername)
    l = len(image_files)
    for i, img_file in enumerate(image_files):
      process_image(foldername + img_file)
      if (i+1) % 10 == 0:
        print(f"""
        {i+1}/{l}:
        elapsed: {time() - start: .2f}
        eta: {(time() - start) * (l-i)/i: .2f}
         """)

if "train" not in os.listdir():
  clean = os.listdir("Clean")
  dirty = os.listdir("Dirt Buildup")

  os.mkdir("/content/train")
  os.mkdir("/content/test")

  for t in ("/content/train", "/content/test"):
    os.mkdir(t + "/Clean")
    os.mkdir(t + "/Dirt Buildup")

  clean = os.listdir("/content/Clean")
  dirty = os.listdir("/content/Dirt Buildup")

  test_clean = int(len(clean) * 0.15)
  test_dirty = int(len(dirty) * 0.15)

  for _ in range(test_clean):
    filename = choices(clean, k=1)[0]
    os.rename("/content/Clean/" + filename, "/content/test/Clean/" + filename)
    clean.remove(filename)

  for _ in range(test_dirty):
    filename = choices(dirty, k=1)[0]
    os.rename("/content/Dirt Buildup/" + filename, "/content/test/Dirt Buildup/" + filename)
    dirty.remove(filename)

  os.rename("/content/Clean", "/content/train/Clean")
  os.rename("/content/Dirt Buildup", "/content/train/Dirt Buildup")

train = image_dataset_from_directory("train")
train.batch(32).prefetch(tf.data.AUTOTUNE)
test = image_dataset_from_directory("test")
test.batch(32).prefetch(tf.data.AUTOTUNE)

effnetb0 = tf.keras.applications.efficientnet.EfficientNetB0(include_top=False)
effnetb0.trainable = False
for layer in effnetb0.layers[-10: ]:
  layer.trainable = True

model = Sequential([
    layers.RandomRotation(0.08333,
                          input_shape=(256, 256, 3),
                          name="augmentation_layer"),
    effnetb0,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation="sigmoid")
], name="dirt_detector")

model.summary()

early_stop = callbacks.EarlyStopping(monitor="val_accuracy",
                                     mode="max",
                                     patience=6)

model.compile(loss="binary_crossentropy",
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy",
                       metrics.Precision(),
                       metrics.Recall(),
                       metrics.AUC()])

model.fit(train,
          validation_data=test,
          epochs=20,
          callbacks=[early_stop])

model.evaluate(test)


image = tf.keras.utils.load_img("/content/test/Clean/03-0.JPG")
image_arr = tf.keras.utils.img_to_array(image)
image_arr = tf.image.resize(image_arr, size=[256, 256])
image_arr = np.reshape(image_arr, (1, 256, 256, 3))
model.predict([image_arr])[0, 0]

