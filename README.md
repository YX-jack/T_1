> 所有的实验说明都放在这个README里面哦~ o(*￣▽￣*)ブ

# 实验一 安装相关程序

>安装一个小小的 Android Studio

![Android_Studio.jpg](./image/Android_Studio.jpg)



> 接着安装 Anaconda

![Anaconda.jpg](./image/Anaconda.jpg)



> 再安装 Jupyter

![jupyter.jpg](./image/jupyter.jpg)



> 最后安装 VS Code！

![](./image/vscode.jpg)



# 实验二 安卓应用实践

#### 实现第一个Kotlin应用

> 实现第一个手机框架界面

![](./image/2.jpg)



> 添加按钮并设置水平垂直依赖，实现计数

![](./image/21.jpg)

> 完成两个页面的跳转关联

![](./image/22.jpg)

#### 构建CameraX应用

> 搭建初始框架

![](./image/23.jpg)

> 实现摄像头的拍照功能

![](./image/24.jpg)

> 添加拍照成功通知

![](./image/25.jpg)

> 查看相片

![](./image/26.jpg)

> 添加摄像功能

![](./image/27.jpg)

# 实验三 Notebook基础实践

>  读入数据集

![](./image/31.jpg)

> 使用直方图显示一下利润按照年份的分布情况

![](./image/32.jpg)

> 使用matplotlib进行画图，显示利润走势

![](./image/33.jpg)

> 同理查看收入走势

![](./image/34.jpg)

> 查看不同公司之间的收入和利润差距

![](./image/35.jpg)

# 实验四 实现图像分类APP

> 将训练完成的APP导入手机，检测玫瑰的分类可能性如下

![](./image/41.jpg)

> 检测雏菊的分类可能性如下

![](./image/42.jpg)

# 实验五 TensorFlow训练石头剪刀布数据集



```python
import os

import numpy as np

import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader

import matplotlib.pyplot as plt


```

## 获取数据


```python

image_path = tf.keras.utils.get_file(
      'flower_photos.tgz',
      'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
      extract=True)
image_path = os.path.join(os.path.dirname(image_path), 'flower_photos')

```

## 运行示例

1.加载数据集，并将数据集分为训练数据和测试数据。


```python
data = DataLoader.from_folder(image_path)
train_data, test_data = data.split(0.9)

```

    INFO:tensorflow:Load image with size: 3670, num_label: 5, labels: daisy, dandelion, roses, sunflowers, tulips.


    INFO:tensorflow:Load image with size: 3670, num_label: 5, labels: daisy, dandelion, roses, sunflowers, tulips.

2.训练Tensorflow模型

```python
model = image_classifier.create(train_data)

```

    INFO:tensorflow:Retraining the models...


    INFO:tensorflow:Retraining the models...


    Model: "sequential_1"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     hub_keras_layer_v1v2_1 (Hub  (None, 1280)             3413024   
     KerasLayerV1V2)                                                 
                                                                     
     dropout_1 (Dropout)         (None, 1280)              0         
                                                                     
     dense_1 (Dense)             (None, 5)                 6405      
                                                                     
    =================================================================
    Total params: 3,419,429
    Trainable params: 6,405
    Non-trainable params: 3,413,024
    _________________________________________________________________
    None
    Epoch 1/5
    103/103 [==============================] - 53s 496ms/step - loss: 0.8643 - accuracy: 0.7709
    Epoch 2/5
    103/103 [==============================] - 51s 493ms/step - loss: 0.6500 - accuracy: 0.8962
    Epoch 3/5
    103/103 [==============================] - 50s 487ms/step - loss: 0.6196 - accuracy: 0.9138
    Epoch 4/5
    103/103 [==============================] - 50s 488ms/step - loss: 0.6015 - accuracy: 0.9275
    Epoch 5/5
    103/103 [==============================] - 50s 488ms/step - loss: 0.5896 - accuracy: 0.9336

3.评估模型

```python
loss, accuracy = model.evaluate(test_data)

```

    12/12 [==============================] - 8s 463ms/step - loss: 0.6284 - accuracy: 0.9101

4.导出Tensorflow Lite模型

```python
model.export(export_dir='.')

```



# TensorFlow训练石头剪刀布数据集

```python
rock_dir = os.path.join('D:/rps/rock')
paper_dir = os.path.join('D:/rps/paper')
scissors_dir = os.path.join('D:/rps/scissors')

print('total training rock images:', len(os.listdir(rock_dir)))
print('total training paper images:', len(os.listdir(paper_dir)))
print('total training scissors images:', len(os.listdir(scissors_dir)))

rock_files = os.listdir(rock_dir)
print(rock_files[:10])

paper_files = os.listdir(paper_dir)
print(paper_files[:10])

scissors_files = os.listdir(scissors_dir)
print(scissors_files[:10])

```

1.检测数据集的解压结果，打印相关信息。

```python
import os
```


```python
rock_dir = os.path.join('D:/rps/rock')
paper_dir = os.path.join('D:/rps/paper')
scissors_dir = os.path.join('D:/rps/scissors')

print('total training rock images:', len(os.listdir(rock_dir)))
print('total training paper images:', len(os.listdir(paper_dir)))
print('total training scissors images:', len(os.listdir(scissors_dir)))

rock_files = os.listdir(rock_dir)
print(rock_files[:10])

paper_files = os.listdir(paper_dir)
print(paper_files[:10])

scissors_files = os.listdir(scissors_dir)
print(scissors_files[:10])
```

    total training rock images: 840
    total training paper images: 840
    total training scissors images: 840
    ['rock01-000.png', 'rock01-001.png', 'rock01-002.png', 'rock01-003.png', 'rock01-004.png', 'rock01-005.png', 'rock01-006.png', 'rock01-007.png', 'rock01-008.png', 'rock01-009.png']
    ['paper01-000.png', 'paper01-001.png', 'paper01-002.png', 'paper01-003.png', 'paper01-004.png', 'paper01-005.png', 'paper01-006.png', 'paper01-007.png', 'paper01-008.png', 'paper01-009.png']
    ['scissors01-000.png', 'scissors01-001.png', 'scissors01-002.png', 'scissors01-003.png', 'scissors01-004.png', 'scissors01-005.png', 'scissors01-006.png', 'scissors01-007.png', 'scissors01-008.png', 'scissors01-009.png']

2.各打印石头剪刀布训练集图片

```python
%matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

pic_index = 2

next_rock = [os.path.join(rock_dir, fname) 
                for fname in rock_files[pic_index-2:pic_index]]
next_paper = [os.path.join(paper_dir, fname) 
                for fname in paper_files[pic_index-2:pic_index]]
next_scissors = [os.path.join(scissors_dir, fname) 
                for fname in scissors_files[pic_index-2:pic_index]]

for i, img_path in enumerate(next_rock+next_paper+next_scissors):
  #print(img_path)
  img = mpimg.imread(img_path)
  plt.imshow(img)
  plt.axis('Off')
  plt.show()
```


​    ![](./image/52.jpg)    ![](./image/53.jpg)    ![](./image/54.jpg)

3.调用TensorFlow的keras进行数据模型的训练和评估

```python
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

TRAINING_DIR = "E:/rps/"
training_datagen = ImageDataGenerator(
      rescale = 1./255,
	    rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

VALIDATION_DIR = "E:/rps-test-set/"
validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(150,150),
	class_mode='categorical',
  batch_size=126
)

validation_generator = validation_datagen.flow_from_directory(
	VALIDATION_DIR,
	target_size=(150,150),
	class_mode='categorical',
  batch_size=126
)

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])


model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit(train_generator, epochs=25, steps_per_epoch=20, validation_data = validation_generator, verbose = 1, validation_steps=3)

model.save("rps.h5")

```

    Found 2520 images belonging to 3 classes.
    Found 372 images belonging to 3 classes.
    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv2d (Conv2D)             (None, 148, 148, 64)      1792      
                                                                     
     max_pooling2d (MaxPooling2D  (None, 74, 74, 64)       0         
     )                                                               
                                                                     
     conv2d_1 (Conv2D)           (None, 72, 72, 64)        36928     
                                                                     
     max_pooling2d_1 (MaxPooling  (None, 36, 36, 64)       0         
     2D)                                                             
                                                                     
     conv2d_2 (Conv2D)           (None, 34, 34, 128)       73856     
                                                                     
     max_pooling2d_2 (MaxPooling  (None, 17, 17, 128)      0         
     2D)                                                             
                                                                     
     conv2d_3 (Conv2D)           (None, 15, 15, 128)       147584    
                                                                     
     max_pooling2d_3 (MaxPooling  (None, 7, 7, 128)        0         
     2D)                                                             
                                                                     
     flatten (Flatten)           (None, 6272)              0         
                                                                     
     dropout (Dropout)           (None, 6272)              0         
                                                                     
     dense (Dense)               (None, 512)               3211776   
                                                                     
     dense_1 (Dense)             (None, 3)                 1539      
                                                                     
    =================================================================
    Total params: 3,473,475
    Trainable params: 3,473,475
    Non-trainable params: 0
    _________________________________________________________________
    Epoch 1/25
    20/20 [==============================] - 71s 3s/step - loss: 1.5301 - accuracy: 0.3528 - val_loss: 1.0804 - val_accuracy: 0.5591
    ....
    Epoch 25/25
    20/20 [==============================] - 58s 3s/step - loss: 0.1172 - accuracy: 0.9603 - val_loss: 0.0227 - val_accuracy: 1.0000

4.完成模型训练之后，我们绘制训练和验证结果的相关信息。

```python
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()
```

   ![](./image/51.jpg)

> 完结撒花 ✿✿ヽ(°▽°)ノ✿
