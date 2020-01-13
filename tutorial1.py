from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

# 채널 차원을 추가
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
x_train.shape
x_test.shape

# tf.data를 사용하여 데이터를 섞고 배치를 만든다.
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(32)


# 모델구조를 만든다.
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')
    
    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

model = MyModel()

# 훈련에 필요한 옵티마이저와 손실함수를 정한다.
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# 모델의 손실과 성능을 측정할 지표를 선택한다. 에포크가 진행되는동안
# 선택된 지표를 바탕으로 최종 결과를 출력한다.
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')
''' 행렬의 평균을 계산해주는 함수.
m = tf.keras.metrics.Mean()
m.update_state([[1,2,3],[4,5,6]], sample_weight=[[1,1,1],[0,0,0]])
m.result().numpy()
'''
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_acc')

# tf.GradientTape를 사용하여 훈련한다.

'''@tf.funtion
TF 2.0 버전은 즉시 실행 (eager execution)의 편리함과 
TF 1.0의 성능을 합쳤습니다. 이러한 결합의 중심에는 tf.function 이 있는데, 
이는 파이썬 문법의 일부를 이식 가능하고 높은 성능의 텐서플로 그래프 코드로 변환시켜준다.
tf.function을 함수에 붙여줄 경우, 여전히 다른 일반 함수들처럼 사용할 수 있다. 
하지만 그래프 내에서 컴파일 되었을 때는 더 빠르게 실행하고, GPU나 TPU를 사용해서
 작동하고, 세이브드모델(SavedModel)로 내보내는 것이 가능해진다.
'''

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients =tape.gradient(loss, model.trainable_variables)
    # trainable_variables 그 시점에서의 가중치를 나타냄.
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)

EPOCH = 5

for epoch in range(EPOCH):
    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)
    
    template = '에포크: {}, 손실: {}, 정확도: {}, 테스트 손실: {}, 테스트 정확도: {}'
    print(template.format(epoch+1, 
    train_loss.result(),
    train_accuracy.result()*100,
    test_loss.result(),
    test_accuracy.result()*100))