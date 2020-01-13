''' 텐서와 연산 '''
from __future__ import absolute_import, division, print_function

import tensorflow as tf

# 텐서 : 다차원 배열.
print(tf.add(1, 2))
print(tf.add([1,2],[3,4]))
print(tf.square(5))
print(tf.reduce_sum([1,2,3]))

# 연산자 오버로딩(overloading) 지원
print(tf.square(2) + tf.square(3))

# 각각의 tf.Tensor는 크기와 데이터 타입을 가지고 있다.
x = tf.matmul([[1]],[[2,3]])
print(x)
print(x.shape)
print(x.dtype)

# numpy 와 Tensor간 변환은 자유로운데 Tensor는 GPU메모리에 저장될 수 있으나
# numpy는 호스트 메모리에만 저장된다.

import numpy as np

ndarray = np.ones([3,3])

print("텐서플로 연산은 자동적으로 넘파이 배열을 텐서로 변환한다.")
tensor = tf.multiply(ndarray,42)
print(tensor)

print("그리고 넘파이 연산은 자동적으로 텐서를 넘파이 배열로 변환한다.")
print(np.add(tensor, 1))

print(".numpy() 메서드는 텐서를 넘파이 배열로 변환한다.")
print(tensor.numpy())


''' GPU 가속 '''
x = tf.random.uniform([3,3])

print("GPU 사용이 가능한가 : ")
print(tf.test.is_gpu_available())

print("텐서가 GPU #0에 있는가 : ")
print(x.device.endswith('GPU:0'))



# 장치이름 : Tensor.device는 텐서를 구성하고 있는 호스트 장치의 풀네임을 제공한다.
# 이러한 이름은 프로그램이 실행중인 호스트의 네트워크 주소 및 해당 호스트 내의 장치와 같은
# 많은 세부정보를 인코딩하고, 이것은 분산 작업에서 필요한다.
# N번째 GPU가 있으면 문자열은 GPU:<N>으로 나온다.

# replacement는 장치에 할당하는 것.
import time 

def time_matmul(x):
    start = time.time()
    for loop in range(10):
        tf.matmul(x,x)
    
    result = time.time()-start
    print("10 loops: {0:2f}ms".format(1000*result))

# Cpu에서 강제 실행
print("On CPU:")
with tf.device("CPU:0"):
    x = tf.random.uniform([1000, 1000])
    assert x.device.endswith("CPU:0")
    time_matmul(x)

# GPU #0이 이용가능하다면 GPU #0에서 강제 실행한다.
if tf.test.is_gpu_available():
    print("On GPU:")
    with tf.device("GPU:0"): # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
        x = tf.random.uniform([1000,1000])
        assert x.device.endswith("GPU:0")
        time_matmul(x)


'''데이터 셋
모델의 데이터를 제공하기 위한 파이프라인을 구축하기 위한
tf.data.Dataset API를 사용
tf.data.Dataset API는 모델을 훈련시키고 평가 루프를 제공할, 간단하고
재사용 가능한 모듈로부터 복잡한 입력 파이프라인을 구축하기 위해 사용됨.'''

# 소스 데이터셋 생성
ds_tensors = tf.data.Dataset.from_tensor_slices([1,2,3,4,5,6])

# csv 파일을 생성
import tempfile
_, filename = tempfile.mkstemp()

with open(filename, 'w') as f:
    f.write("""Line 1
    Line 2
    Line 3
    """)
ds_file = tf.data.TextLineDataset(filename)

# 변환 적용
ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)
ds_file = ds_file.batch(2)

# 반복
print('ds_tensors 요소 : ')
for x in ds_tensors:
    print(x)

print('\nds_file 요소 : ')
for x in ds_file:
    print(x)    