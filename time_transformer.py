from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow_datasets as tfds
import tensorflow as tf
import time
import numpy as numpy
import matplotlib.pyplot as plt
import numpy as np

tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus((
    en.numpy() for pt, en in train_examples), target_vocab_size=2**13)
tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (pt.numpy() for pt, en in train_examples), target_vocab_size=2**13)


sample_string  = 'Transformer is awesome.'
tokenized_string = tokenizer_en.encode(sample_string)
print('Tokenized string is {}'.format(tokenized_string))

original_string = tokenizer_en.decode(tokenized_string)
print('The original string: {}'.format(original_string))

input_vocab_size = tokenizer_pt.vocab_size + 2
d_model = 512

embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)

embedding(tokenized_string)

train_dataset = train_examples.map(tf_encode)
train_dataset = train_dataset.filter(filter_max_length)
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE,padded_shapes=([None],[None]))
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
train_dataset
for (batch, (inp, tar)) in enumerate(train_dataset):
    print(batch, inp )
    break
embedding(inp)[0].shape


''' ship data (Simple RNN)'''
import pandas as pd
org = pd.read_csv(r'C:\Users\YongTaek\Desktop\씨벤티지\ship_original.csv')
org = org.drop(org.columns[0], axis=1)
org = org.drop('Timestamp',axis=1)

plt.plot(org.iloc[:,22])
# Slip 이상치 제거 보류
# org = org.drop(org[org.iloc[:,22]<-40].index,axis=0) 

# 수치형 자료에서 숫자 아닌 행 제거
org[org[org.columns[3]].isin(['-'])]
org = org.drop(org[org[org.columns[3]].isin(['-'])].index)
org[org[org.columns[4]].isin(['-'])]
org = org.drop(org[org[org.columns[4]].isin(['-'])].index)


# reindex를 하면, Slip에 결측치가 생긴다. 왜?? 그럴까?
org.Slip.isnull().sum()
# org = org.reindex(range(len(org)))
org.Slip.isnull().sum() # 158개

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(org)
org[org.columns[3:5]] = org[org.columns[3:5]].astype(np.float64)
org.info()
np_org = scaler.transform(org)
np_org.shape


plt.hist(np_org[:,0])
plt.plot(np_org[22,:])

plt.plot(np_org[:,22])


TRAIN_SPLIT = round(len(np_org) * 0.7)

def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices,:], (history_size, 38, 1)))
        labels.append(dataset[i+target_size])
    return np.array(data), np.array(labels)

x_train_uni, y_train_uni = univariate_data(np_org, 0, TRAIN_SPLIT, 20, 0)
x_val, y_val = univariate_data(np_org, TRAIN_SPLIT, None, 20, 0)

y_train_uni.shape[0] + y_val.shape[0]

def create_time_steps(length):
    return list(range(-length, 0))

''' 첫 번째 데이터 세트를 생각한다면 , window의 다음 값을 예측해야 한다.
그것을 X로 표시한다. '''
def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10,
            label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future+5)*2])
    plt.xlabel('Time-Step')
    return plt
show_plot([x_train_uni[0, :, 22], y_train_uni[0, 22]], 0, 'Sample Example')


''' 하지만 전 시계열 target 값들의 평균보다 더 좋아야한다. base라인 설정''' 
def baseline(history):
    return np.mean(history)
show_plot([x_train_uni[0, :, 22], y_train_uni[0, 22], baseline(x_train_uni[0,:,22])], 0,
           'Baseline Prediction Example')

''' RNN 
비교 모델로 바닐라 RNN에서 LSTM을 사용한 모델로 선택할 것이다.'''
BATCH_SIZE = 256
# 맨처음의 10000개를 가져와서 그거를 셔플하고 나머지는 그대로 순서대로
# classifier의 경우 분류개수보다 많게해야 학습이 잘이루어진다. label이 잘 안섞여있으면
# 치중해서 가중치 업데이트를 하기 때문에 잘 안됨.
BUFFER_SIZE = 10000

train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni[:, :, 22], y_train_uni[:, 22]))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_univariate = tf.data.Dataset.from_tensor_slices((x_val[:, :, 22], y_val[:, 22]))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

''' LSTM에 데이터 입력 형태가 필요하다 따라서 그렇게 바꿔주는 과정 '''
simple_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(30, input_shape=(x_train_uni.shape[-3], x_train_uni.shape[-1])),
    tf.keras.layers.Dense(1)])

simple_lstm_model.compile(optimizer='adam', loss='mae')

''' 모델의 출력을 확인하기 위해 샘플 예측을 만들어보자 ''''
for x, y in val_univariate.take(1):
    print(simple_lstm_model.predict(x).shape)

EVALUATION_INTERVAL = 200
EPOCHS = 30
simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
steps_per_epoch=EVALUATION_INTERVAL, 
validation_data=val_univariate, validation_steps=50)


for x, y in val_univariate.take(1):
    print(x[0], y[0])

for x, y in val_univariate.take(3):
    plot = show_plot([x[0].numpy(), y[0].numpy(),
    simple_lstm_model.predict(x)[0]], 0, 'Simple LSTM model')
    plot.show()

for i,j in train_univariate:
    print(i[0], j[0])
    break
for x, y in train_univariate.take(1):
    print(x.shape, simple_lstm_model.predict(x).shape)


''' 전문가처럼 코딩 짜보기 ''' 
from tensorflow.keras import Model
from tensorflow.keras.layers import Input

''' 이 두가지 확인 필요'''
# 1. 내 생각에는 MYModel이 Model로 인식되는 것 같다.
# 2. Layer라고 생각하면 Layer를 넣으면 된다  

class MYModel(Model):
    def __init__(self):
        super(MYModel, self).__init__()
        self.lstm = tf.keras.layers.LSTM(24)
        self.Dense = tf.keras.layers.Dense(1)
    def call(self, x):
        x = self.lstm(x)
        return self.Dense(x)

model = MYModel()

## 인풋이 train의 인풋과 확인하여 위의 모델이 잘 작동하는지 확인
inp = Input(shape=(20,1,))
inp
x = tf.keras.layers.LSTM(24)(inp)
output = tf.keras.layers.Dense(1)(x)

# 모델의 input을 거친 것과 그대로 진행했을 때의 아웃풋이 동일
model(inp)
output

# 파라미터 수와 층들 확인
model.summary()

''' 케라스 컴파일 버전'''
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='mae',
              metrics=['mae'])
model.fit(train_univariate, epochs=EPOCHS,
steps_per_epoch=EVALUATION_INTERVAL, 
validation_data=val_univariate, validation_steps=50)




''' 케라스 Gradient Version'''

'''compile 및 fit으로 맞출 때에는 shuffle batch repeat을 썼는데

 '''
train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni[:, :, 22], y_train_uni[:, 22]))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

val_univariate = tf.data.Dataset.from_tensor_slices((x_val[:, :, 22], y_val[:, 22]))
val_univariate = val_univariate.batch(BATCH_SIZE)

class MYModel(Model):
    def __init__(self):
        super(MYModel, self).__init__()
        self.lstm = tf.keras.layers.LSTMCell(units=24)
        self.Dense = tf.keras.layers.Dense(1)
    def call(self, x):
        batch_size, seq_length, _= tf.shape(inp)
        state = self.lstm.zero_state(batch_size=batch_size, dtype=tf.float32)
        for t in range(seq_length.numpy()):
            output, state = self.lstm(inp[:, t, :], state)
            print(output, state)
        x = self.lstm(x)
        return self.Dense(x)



# train과 val step 정의
loss_object = tf.keras.losses.MeanAbsoluteError()
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.MeanAbsoluteError()
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.MeanAbsoluteError()

@tf.function
def train_step(inp, targets):
    with tf.GradientTape() as tape:
        predictions = model(inp)
        loss = loss_object(targets, predictions)

    gradients = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(zip(gradients, model.variables))

    train_loss(loss)
    train_accuracy(targets, predictions)

@tf.function
def test_step(inp, targets):
    predictions = model(inp)
    t_loss = loss_object(targets, predictions)

    test_loss(t_loss)
    test_accuracy(targets, predictions)

EPOCHS = 200
for epoch in range(EPOCHS):
    for inp, targets in train_univariate:
        train_step(inp, targets)
    for test_inp, test_targets in val_univariate:
        test_step(test_inp, test_targets)
    
    template = 'EPOCHS: {}, LOSS: {}, train_MSE: {}, val_MSE: {}'
    print( template.format(epoch+1,
    train_loss.result(),
    train_accuracy.result()*100,
    test_loss.result(),
    test_accuracy.result()*100))
