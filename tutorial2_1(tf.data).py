''' tf.data : TensorFlow 입력 파이프 라인 빌드 !!!자세히!!!
tf.data API는 단순하고 재사용 가능한 조각에서 복잡한 입력 파이프 라인을 구축 할 수 있다.
예를 들어, 이미지 모델의 파이프 라인은 분산 파일의 시스템의 파일에서 데이터를 집계하고
각 이미지이에 임의의 섭동을 적용하여 무작위로 선택한 이미지를 배치를 위해 병합하여 학습
할 수 있다. 텍스트 모델의 파이프 라인에는 원시 텍스트 데이터에서 심볼을 추출하고 이를
룩업 테이블이 있는 식별자를 포함시키는 것으로 변환하고 길이가 다른 시퀀스를 일괄 처리하는
것이 포함될 수 있다. '''

# 데이터 소스 Dataset은 메모리 또는 하나 이상의 파일에 저장된 데이터를 구성
# 데이터 변환은 하나 이상의 tf.data.Dataset개체에서 집합을 구성한다.

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.set_printoptions(precision=4)

''' Basic mechanics : 입력 파이프 라인을 만들려면 데이터 소스로 시작해야한다.
예를들어 Dataset 메모리의 데이터를 구성하려면 
tf.data.Dataset.from_tensors() or tf.data.Dataset.from_tensor_slices()를 사용할 수 있다.
또는 입력 데이터가 권장 TFRecord 형식으로 저장된 경우 사용할 수 있다.: tf.data.TFRecordDataset().

당신은 일단 Dataset개체를 가질 수 있다. 그것을 tf.data.Dataset의 방법으로 연결함으로써 
그것을 new Dataset으로 변형할 수 있다. 예를들어 너는 요소 당 변환을 Dataset.map()을 
적용을 할 수 있다. 그리고 다중 요소 변환은 Dataset.batch()를 사용하면 된다.
'''
# Dataset 객체는 반복문 사용이 가능하다. 
dataset = tf.data.Dataset.from_tensor_slices([8, 3, 0, 8, 2, 1])
dataset

for elem in dataset:
    print(elem.numpy())

# 또는 next를 사용하여 iter 요소를 사용 하고 소비 후 next로 넘긴다.
it = iter(dataset)
print(next(it).numpy())

# 또는 reduce 변환을 사용하여 데이터 세트 요소를 사용할 수 있어 모든 요소를 줄여 단일 결과를 생성할 수 있다.
# 다음은 reduce변환으로 모든 정수의 합계를 계산한다.
print(dataset.reduce(0, lambda state, value: state + value).numpy())

''' Dataset 구조
데이터 세트 각가은 동일한 (중첩) 구조를 가지며, 구조의 각 구성 요소에는
Tensor, SparseTensor, RaggedTensor, TensorArray 또는 Dataset을 포함하여 
tf.Typespec이 표시할 수 있는 구조가 있다.  Dataset.element_spec 속성을 사용하면 각 요소 구성 요소의 유형을 검사할 수 있다.
이 속성은 단일 구성요소, 구성 요소 튜프 또는 구성 요소의 중첩튜플일 수 있는
요소의 구조와 일치하는 tf.TypeSpec개체의 중첩 구조를 반환한다.'''

# 예는 다음과 같다.
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([4, 10]))
dataset1.element_spec

dataset2 = tf.data.Dataset.from_tensor_slices(
    (tf.random.uniform([4]),
    tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)))
dataset2.element_spec

dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
dataset3.element_spec

# sparse tensor를 포함하는 Dataset
dataset4 = tf.data.Dataset.from_tensors(tf.SparseTensor(indices=[[0,0],[1,2]],values=[1,2], dense_shape=[3,4]))
dataset4.element_spec
tf.sparse.to_dense(tf.SparseTensor(indices=[[0,0],[1,2]],values=[1,2], dense_shape=[3,4]))
dataset4.element_spec.value_type

# Dataset변환은 임의의 구조의 데이터 세트를 지원한다. 각 요소에 함수를 적용하는 
# Dataset.map(), Dataset.filter() 변환을 사용할 때 요소 구조는 함수의 인수를 결정한다.
dataset1 = tf.data.Dataset.from_tensor_slices(
    tf.random.uniform([4,10],minval=1, maxval=10, dtype=tf.int32))
dataset1

for z in dataset1:
    print(z.numpy())

dataset2 = tf.data.Dataset.from_tensor_slices(
    (tf.random.uniform([4]),
    tf.random.uniform([4,100],maxval=100, dtype=tf.int32)))
dataset2

dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
dataset3

for a, (b,c) in dataset3:
    print('shape: {a.shape}, {b.shape}, {c.shape}'.format(a=a,b=b,c=c))

# Numpy 배열 소비
# 모든 입력데이터가 메모리에 맞는 경우 데이터를 생성하는 가장 간단한 바방법은 데이터 
# Dataset를 tf.Tensor객체로 변환하고 사용하는 Dataset.from_tensor_slices()

train, test = tf.keras.datasets.fashion_mnist.load_data()

images, labels = train
images = images/255

dataset = tf.data.Dataset.from_tensor_slices((images, labels))
dataset

# Python generators 소비
# tf.data.Dataset으로서 쉽게 수집할 수 있는 다른 보통 데이터 소스는 python generator이다.
def count(stop):
    i = 0
    while i<stop:
        yield i
        i += 1
for n in count(5):
    print(n)


# Dataset.from_generator 생성자는 완전한 기능에 파이썬 generator를 함수의
# tf.data.Dataset 으로 변환한다.
# 생성자는 반복자가 아닌 입력 가능 항목을 입력을 사용한다. 이를 통해 생성기가 끝에 도달
# 하면 다시 시작할 수 있다. 그것은 선택 가능한 args인수를 취하는데, 이는 호출 가능한 인수로 전달된다.


# tf.data는 tf.Graph를 내부로 빌드하기 때문에 output_types이 요구된다. 
# 그리고 graph edges는 tf.dtype을 요구한다.
ds_counter = tf.data.Dataset.from_generator(count, args=[25], output_types=tf.int32, output_shapes=(),)

for count_batch in ds_counter.repeat().batch(10).take(10):
    print(count_batch.numpy())

''' output_shapes는 요구되지 않지만 tensorflow 작업이 알 수 없는 rank와 tensor를 지원하지
않는 고도의 시스템에서 권장된다. 만약 특정 축이 unknown이거나 variable 이라면, output_shapes
는 None으로 설정하여라 '''

''' output_shapes와 output_types는 같은 nesting rules를 따른다 다른 dataset methods와 같이'''

# 다음은 두 가지 측면을 모두 보여주는 예제 생성기이다. 두 배열은 길이가 알려지지 않은 벡터
def gen_series():
    i = 0
    while True:
        size = np.random.randint(0, 10)
        yield i, np.random.normal(size=(size,))
        i += 1
for i, series in gen_series():
    print(i, ":", str(series))
    if i > 5 :
        break

# 첫 번째 output은 int23, 두 번째 output은 float32.
# 첫 번째 item은 scalar, sahpe()이고 두 번째는 unknown 길이의 vector, shape(None,)이다.
ds_series = tf.data.Dataset.from_generator(
    gen_series,
    output_types=(tf.int32, tf.float32),
    output_shapes=((), (None,)))
ds_series
ds_series

# 이제 일반화하여 tf.data.Dataset을 사용할 수 있다. 가변 모양으로 데이터 집합을 일괄처리
# 할 때에는 Dataset.padded.batch를 사용해야한다.
''' 아무래도 padded_batch는 shape에 맞게 ()안에 output타입에 맞게 지정해줘야한다.'''
ds_series_batch = ds_series.shuffle(20).padded_batch(10, padded_shapes=(() ,([None])))
ids, sequence_batch = next(iter(ds_series_batch))
print(ids.numpy())

print(sequence_batch.numpy())

'''실생활 예제를 보려면 preprocessing.image.ImageDataGenerator로 싸여진
tf.data.Datset을 시도해봐라'''

# 먼저 데이터 다운로드
flowers = tf.keras.utils.get_file(
    'flower_photos',
    'http://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar = True)

# image.ImageDataGenerator 만들기
img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=20)
img_gen

from IPython.display import display
from PIL import Image
images, labels = next(img_gen.flow_from_directory(flowers))

print(images.dtype, images.shape)
print(labels.dtype, labels.shape)

ds = tf.data.Dataset.from_generator(
    img_gen.flow_from_directory, args=[flowers],
    output_types=(tf.float32, tf.float32),
    output_shapes=([32,256,256,3], [32,5])
)

''' TFRecord data 소비 
tf.data API는 메모리에 맞지 않는 큰 데이터 세트를 처리할 수 있도록 다양한 파일 형식을 
지원한다. 예를들어, TFRecord 파일 형식은 많은 TensorFlow 애플리케이션이 데이터 학습에
사용하는 간단한 레코드 지향 이진 형식이다. 이 tf.data.TFRecordDataset클래스를 사용하면
입력 파이프 라인의 일부로 하나 이상의 TFRecord파일 내용을 스트리밍 할 수 있다.
'''
# 다음은 FNSS(French Street Name Signs)의 테스트 파일을 사용하는 예이다.
# 두 파일에서 예제의 모든것을 읽어 데이터 셋 생성
fsns_test_file = tf.keras.utils.get_file("fsns.tfrec", "https://storage.googleapis.com/download.tensorflow.org/data/fsns-20160927/testdata/fsns-00000-of-00001")

# TFREcordDataset이니셜 라이저의 filenames인수는 문자열, list of strings or 
# tf.Tensor of strings일 수 있다. 그러므로 만약 train 및 validation목적으로 두
# 파일 세트가 있는 경우 파일 이름을 입력 인수로 사용하여 데이터 세트를 생성하는 
# 팩토리 메소드를 작성할 수 있습니다.
dataset = tf.data.TFRecordDataset(filenames = [fsns_test_file])
dataset

# 많은 Tensorflow 프로젝트 tf.train.Example은 TRFecord파일에서 직렬화 된 레코드를 
# 사용한다. 검사하기 전에 이를 해독해야 한다.
raw_example = next(iter(dataset))
parsed = tf.train.Example.FromString(raw_example.numpy())

parsed.features.feature['image/text']

# https://www.tensorflow.org/tutorials/load_data/tfrecord 자세한거 참고.


''' 텍스트 데이터 소비
많은 데이터 세트가 하나 이상의 텍스트 파일로 배포된다. tf.data.TextLineDataset은
하나 이상의 텍스트 파일에서 라인을 추출 할 수 있는 쉬운 방법을 제공한다. 하나 이상의 파일
이름이 주어지면 TextLineDataset은 해당 파일의 행 당 하나의 문자열 값 요소를 생성한다.'''

directory_url = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
file_names = ['cowper.txt', 'derby.txt', 'butler.txt']
file_paths = [
    tf.keras.utils.get_file(file_name, directory_url + file_name)
    for file_name in file_names
]

dataset = tf.data.TextLineDataset(file_paths)

# 첫 번째 파일의 처음 몇줄 보기
for line in dataset.take(5):
    print(line.numpy())

# 파일에서 줄을 바꾸려면 Data.interleave를 사용하라 이렇게 하면 파일을 쉽게 섞을 수 
# 있다. 각 번역의 첫 번재, 두 번째, 세 번째 줄은 다음과 같다.
file_ds = tf.data.Dataset.from_tensor_slices(file_paths)
lines_ds = file_ds.interleave(tf.data.TextLineDataset, cycle_length=3)

for i, line in enumerate(lines_ds.take(9)):
    if i % 3 ==0:
        print()
    print(line.numpy())

''' 기본적으로, TextLineDataset은 각 파일의 모든 줄을 산출한다. 예를들어 파일이 헤더 줄로
시작 하거나 주석이 포함 된 경우 작동하지 않을 수 있다. Dataset.skip() 이나 Dataset.filter()
로 이 줄을 제거할 수 있다.'''

# 여기서 첫 번째 줄을 건너 뛰고 survivors만 찾기 위해 필터링한다.

titanic_file = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")
titanic_lines = tf.data.TextLineDataset(titanic_file)

for line in titanic_lines.take(10):
    print(line.numpy())

def survived(line):
    return tf.not_equal(tf.strings.substr(line, 0, 1), "0")
# tf.strings.substr(line, 0, 1)은 0부터 1번째 스트링 가져오기
survivors = titanic_lines.skip(1).filter(survived)

for line in survivors.take(10):
    print(line.numpy())


''' CSV 데이터 소비 
CSV 파일 형식은 표 형식의 데이터를 일반 텍스트로 저장하는 데 널리 사용되는 형식이다.'''
# 예는 다음과 같다.
titanic_file = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")
df = pd.read_csv(titanic_file, index_col=None)
df.head()

# 데이터가 메모리에 맞는 경우 동일한 Dataset.from_tensor_slices방법이 사전에서 작동하
# 여 이 데이터를 쉽게 가져올 수 있다.
titanic_slices = tf.data.Dataset.from_tensor_slices(dict(df))

for feature_batch in titanic_slices.take(1):
    for key, value in feature_batch.items():
        print(" {!r:20s}: {}".format(key, value))

# 보다 확장 가능한 접근 방식은 필요에 따라 디스크에서 로드하는 것이다.

# 이 tf.data모듈은 RFC 4180을 준수하는 하나 이상의 CSV파일에서 레코드를 추출하는 방법을 제공.
# 이 experimental.make_csv_dataset기능은 CSV파일 세트를 읽기 위한 고급 인터페이스이다.
# 열 형식 유추와 일괄 처리 및 셔플링과 같은 많은 다른 기능을 지우너하여 사용이 간편하다.
titanic_batchs = tf.data.experimental.make_csv_dataset(
    titanic_file, batch_size=4,
    label_name='survived')

for feature_batch, label_batch in titanic_batchs.take(1):
    print("'survived': {}".format(label_batch))
    print("features:")
    for key, value in feature_batch.items():
        print(" {!r:20s}: {}".format(key, value))

# columns의 부분집합이 필요할 때, select_columns을 사용할 수 있다.
titanic_batchs = tf.data.experimental.make_csv_dataset(
    titanic_file, batch_size=4,
    label_name="survived", select_columns=['class', 'fare', 'survived'])
    
for feature_batch, label_batch in titanic_batchs.take(1):
    print("'survived': {}".format(label_batch))
    for key, value in feature_batch.items():
        print(" {!r:20s}: {}".format(key, value))

# 더 낮은 level의 experimental.CsvDataset class가 존재한다. 열 형식 유추를 지원
# 하지 않고, 대신 각 열의 유형을 지정해야한다.
titanic_types = [tf.int32, tf.string, tf.float32, tf.int32, tf.int32, tf.float32, tf.string, tf.string, tf.string, tf.string]
dataset = tf.data.experimental.CsvDataset(titanic_file, titanic_types, header=True)

for line in dataset.take(10):
    print([item.numpy() for item in line])

# 일부 열이 비어있는 경우 낮은 수준의 인터페이스를 사용하면 열 유형 대신 기본 값을 제공할 수 있다.
%%writefile missing.csv
1,2,3,4
,2,3,4
1,,3,4
1,2,,4
1,2,3,
,,,


# missing values가 있는 네개의 float columns 각각의 파일을 두 CSV의 기록으로 부터
# 모든 값을 로드하여 dataset을 생성한다.
record_defaults = [999,999,999,999]
dataset = tf.data.experimental.CsvDataset("missing.csv", record_defaults)
dataset = dataset.map(lambda *items: tf.stack(items))
dataset

for line in dataset:
    print(line.numpy())


'''파일 세트 소비
파일 세트로 분배 된 많은 데이터 세트가 있으며 여기서 각 파일은 예이다.'''
flowers_root = tf.keras.utils.get_file(
    'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True)
flowers_root = pathlib.Path(flowers_root)

# 루트 디렉토리에는 각 클래스에 대한 디렉토리가 있다.
for item in flowers_root.glob("*"):
    print(item.name)

list_ds = tf.data.Dataset.list_files(str(flowers_root/'*/*'))
for f in list_ds.take(5):
    print(f.numpy())

# tf.io.read_file 함수를 사용하여 데이터를 읽고 경로에서 레이블을 추출하여 (image, label)
# 쌍을 리턴하라
def process_path(file_path):
    label = tf.strings.split(file_path, '\\')[-2]
    return(tf.io.read_file(file_path), label)
labeled_ds = list_ds.map(process_path)

for image_raw, label_text in labeled_ds.take(1):
    print(repr(image_raw.numpy()[:100]))
    print()
    print(label_text.numpy())


''' 배치 데이터 세트 요소
간단한 배치
가장 간단한 배치 형식은 n데이터 집합의 연속 요소를 단일 요소로 쌓는다. 
Dataset.batch()변환은 요소의 각 구성요소에 적용되는 tf.stack() 연산자와 동일한
제약 조건을 사용하여 정확하게 수행합니다. 즉, 각 구성요소 i의 경우 모든 요소의 모양이
동일한 텐서가 있어야한다. '''

inc_dataset = tf.data.Dataset.range(100)
dec_dataset = tf.data.Dataset.range(0, -100, -1)
dataset = tf.data.Dataset.zip((inc_dataset, dec_dataset))
batched_dataset = dataset.batch(4)

for batch in batched_dataset.take(4):
    print([arr.numpy() for arr in batch])

# tf.data shape정보를 전파하려고 시도하는 동안 Dataset.batch 마지막 배치가 가득 차지
# 않을 수 있으므로 기본 설정으로 배치크기를 알 수 없다. None을 주목!
batched_dataset

# drop_remainder인수를 사용하여 마지막 배치를 무시하고 전체 모양 전파를 얻어라
batched_dataset = dataset.batch(7, drop_remainder=True)
batched_dataset


''' 패딩이 있는 배치 텐서
위의 레시피는 모두 같은 크기의 텐서에 적용된다. 그러나 많은 모델(ex 시퀀스 모델)은
다양한 크기(ex 다른 길이의 시퀀스)를 가질 수 있는 입력 데이터로 작동한다. 이 경우를 처리
하기 위해 Dataset.padded_batch변환을 사용하면 채워질 수 있는 하나 이상의 치수를 정하여
다른 모양의 텐서를 배치할 수 있다.'''
#  Dataset.padded_batch 변환을 사용하면 각 구성 요소의 각 차원에 대해 서로 다른 패딩을 
# 설정할 수 있으며 가변 길이 (위 예제에서 None으로 표시됨) 또는 상수 길이 일 수 있습니다. 
# 패딩 값을 무시할 수도 있습니다. 기본값은 0입니다.
dataset = tf.data.Dataset.range(100)
dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
dataset = dataset.padded_batch(4, padded_shapes=(None,))

for batch in dataset.take(2):
    print(batch.numpy())
    print()



''' Training Workflows
Processing multiple epochs 
tf.data API는 같은 데이터의 여러 epoch를 처리하는 두가지 방법을 제공한다.

여러 epoch에서 데이터 집합을 반복하는 가장 간단한 방법은 Dataset.repeat()변환을
사용하는 것이다. 먼저 taitanic데이터 세트를 만들어보자'''
titanic_file = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")
titanic_lines = tf.data.TextLineDataset(titanic_file)

def plot_batch_sizes(ds):
    batch_sizes = [batch.shape[0] for batch in ds]
    plt.bar(range(len(batch_sizes)), batch_sizes)
    plt.xlabel('Batch number')
    plt.ylabel('Batch size')

# Dataset.repeat() 인수 없이 transform을 적용하면 입력이 무제한 반복된다.
# Dataset.repeat 변환은 한 에포크의 끝과 다음 에포크의 시작을 알리지 않고 인수를 연결합니다.
# 이 때문에 Dataset.repeat 이후에 적용된 Dataset.batch는 에포크 경계를 넘어서는 배치를 생성합니다.
titanic_batches = titanic_lines.repeat(3).batch(128)
plot_batch_sizes(titanic_batches)

# 명확한 epoch 분리가 필요한 경우 Dataset.batch를 반복전에 두어라
titanic_batches = titanic_lines.batch(128).repeat(3)
plot_batch_sizes(titanic_batches)

# 각 epoch의 끝에서 사용자 정의 계산 (통계 수집)을 수행하려면 각 epoch에서 데이터 세트
# 반복을 다시 시작하는 것이 가장 간단하다.
epochs = 3
dataset = titanic_lines.batch(128)

for epoch in range(epochs):
    for batch in dataset:
        print(batch.shape)
    print("End of epoch: ", epoch)



''' 입력 데이터를 임의로 섞기
Dataset.shuffle()변형은 고젖ㅇ 된 크기 버퍼를 유지하고 해당 버퍼로부터 임의로 다음요소를
선택한다.'''
# 효과를 볼 수 있도록 데이터 세트에 색인을 추가하라
lines = tf.data.TextLineDataset(titanic_file)
counter = tf.data.experimental.Counter()

dataset = tf.data.Dataset.zip((counter, lines))
dataset = dataset.shuffle(buffer_size=100)
dataset = dataset.batch(20)
dataset

# buffer_size가 100이고 배치 크기가 20이므로 첫 번째 배치에는 120보다 
# 큰 인덱스를 가진 요소가 없습니다.
n, line_batch = next(iter(dataset))
print(n.numpy())



# Dataset.batch와 마찬가지로 Dataset.repeat와 관련된 순서가 중요합니다.

# Dataset.shuffle은 셔플 버퍼가 비워 질 때까지 에포크의 끝을 알리지 않습니다.
# 따라서 반복하기 전에 셔플을하면 다음 시대로 이동하기 전에 한 epoch의 모든 요소가 표시됩니다
dataset = tf.data.Dataset.zip((counter, lines))
shuffled = dataset.shuffle(buffer_size=100).batch(10).repeat(2)

print("Here are the item ID's near the epoch boundary: \n")
for n, line_batch in shuffled.skip(60).take(5):
    print(n.numpy())

shuffle_repeat = [n.numpy().mean() for n, line_batch in shuffled]
plt.plot(shuffle_repeat, label='shuffle().repeat()')
plt.ylabel("Mean item ID")
plt.legend()

# 그러나 셔플이 반복되기 전에 반복이 에포크 경계를 혼합한다.
dataset = tf.data.Dataset.zip((counter, lines))
shuffled = dataset.repeat(2).shuffle(buffer_size=100).batch(10)

print("Here are the item Id's near the epoch boundary : \n")
for n, line_batch in shuffled.skip(55).take(15):
    print(n.numpy)

repeat_shuffle = [n.numpy().mean() for n, line_batch in shuffled]

plt.plot(shuffle_repeat, label="shuffle().repeat()")
plt.plot(repeat_shuffle, label="repeat().shuffle()")
plt.ylabel("Mean item ID")
plt.legend()


'''
Dataset.map (f) 변환은 주어진 함수 f를 입력 데이터 세트의 각 요소에 적용하여 
새 데이터 세트를 생성합니다. 함수형 프로그래밍 언어의 목록 (및 기타 구조)에 일반적으로 
적용되는 map () 함수를 기반으로합니다. 함수 f는 입력에서 단일 요소를 나타내는 
tf.Tensor 객체를 가져 와서 새 데이터 세트에서 단일 요소를 나타내는 tf.Tensor 
객체를 반환합니다. 구현시 표준 TensorFlow 작업을 사용하여 한 요소를 다른 요소로
변환합니다.

이 섹션에서는 Dataset.map () 사용 방법에 대한 일반적인 예를 다룹니다.

이미지 데이터 디코딩 및 크기 조정
실제 이미지 데이터에서 신경망을 학습 할 때 종종 크기가 다른 이미지를 일반 크기로 
변환하여 고정 크기로 배치 될 수 있도록해야합니다.

꽃 파일 이름 데이터 세트를 다시 작성하라.'''

list_ds = tf.data.Dataset.list_files(str(flowers_root/'*/*'))

# 데이터 세트 요소를 조작하는 함수를 작성하라
# 파일에서 이미지를 읽고 밀도가 높은 텐서로 디코딩 한 다음 고정 된 모양으로 크기를 조정합니다.
def parse_image(filename):
    parts = tf.strings.split(filename, '\\')
    label = parts[-2]

    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [128,128])
    return image, label

# 작동하는지 테스트
file_path = next(iter(list_ds))
image, label = parse_image(file_path)

def show(image, label):
    plt.figure()
    plt.imshow(image)
    plt.title(label.numpy().decode('utf-8'))
    plt.axis('off')

show(image, label)

# 데이터 세트에 매핑
images_ds = list_ds.map(parse_image)

for image, label in images_ds.take(2):
  show(image, label)


  ''' 임의의 파이썬 로직 적용
성능상의 이유로 가능할 때마다 데이터를 사전 처리하기 위해 TensorFlow 조작을 사용하십시오. 
그러나 입력 데이터를 구문 분석 할 때 외부 Python 라이브러리를 호출하는 것이 
유용한 경우가 있습니다. Dataset.map () 변환에서 tf.py_function () 연산을 
사용할 수 있습니다.

예를 들어, 임의 회전을 적용하려는 경우 tf.image 모듈에는 tf.image.rot90 만 있으므로 
이미지 확대에는 그다지 유용하지 않습니다.'''

# tf.py_function을 시연하려면 scipy.ndimage.rotate 함수를 대신 사용해보십시오.
import scipy.ndimage as ndimage

def random_rotate_image(image):
    image = ndimage.rotate(image, np.random.uniform(-30, 30), reshape=False)
    return image
image, label = next(iter(images_ds))
image = random_rotate_image(image)
show(image, label)

#  이 함수를 Dataset.map과 함께 사용하려면 Dataset.from_generator와 동일한 경고가
#  적용됩니다. 함수를 적용 할 때 반환 모양과 유형을 설명해야합니다.
def tf_random_rotate_image(image, label):
    im_shape = image.shape
    [image, ] = tf.py_function(random_rotate_image, [image], [tf.float32])
    image.set_shape(im_shape)
    return image, label

rot_ds = images_ds.map(tf_random_rotate_image)
for image, label in rot_ds.take(2):
    show(image, label)


'''tf.Example프로토콜 버퍼 메시지 구문 분석
많은 입력 파이프 라인 tf.train.Example이 TFRecord 형식에서 프로토콜 버퍼
 메시지를 추출 합니다. 각 tf.train.Example레코드에는 하나 이상의 "기능"이 포함되며
  입력 파이프 라인은 일반적으로 이러한 기능을 텐서로 변환합니다.'''
fsns_test_file = tf.keras.utils.get_file("fsns.tfrec", "https://storage.googleapis.com/download.tensorflow.org/data/fsns-20160927/testdata/fsns-00000-of-00001")
dataset = tf.data.TFRecordDataset(filenames = [fsns_test_file])
dataset

# tf.train.Example 프로토 타입을 사용하여 tf.data.Dataset 외부에서 데이터를 이해할 수 있습니다.
raw_example = next(iter(dataset))
parsed = tf.train.Example.FromString(raw_example.numpy())

feature = parsed.features.feature
raw_img = feature['image/encoded'].bytes_list.value[0]
img = tf.image.decode_png(raw_img)
plt.imshow(img)
plt.axis('off')
_ = plt.title(feature['image/text'].bytes_list.value[0])

raw_example = next(iter(dataset))

def tf_parse(eg):
    example = tf.io.parse_example(
        eg[tf.newaxis], {
            'image/encoded': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
          'image/text': tf.io.FixedLenFeature(shape=(), dtype=tf.string)
      })
    return example['image/encoded'][0], example['image/text'][0]
img, txt = tf_parse(raw_example)
print(txt.numpy())
print(repr(img.numpy()[:20]), '...')

decoded = dataset.map(tf_parse)
decoded

image_batch, text_batch = next(iter(decoded.batch(10)))
image_batch.shape



'''
시계열 윈도우
종단 간 시계열 예는 시계열 예측을 참조하십시오.

시계열 데이터는 종종 시간 축을 그대로 유지하여 구성됩니다.

간단한 Dataset.range를 사용하여 다음을 보여줍니다.
'''

range_ds = tf.data.Dataset.range(100000)
# 일반적으로 이러한 종류의 데이터를 기반으로하는 모델은 인접한 시간 조각을 원할 것입니다.
# 가장 간단한 방법은 데이터를 일괄 처리하는 것입니다.
# batch 사용 
batches = range_ds.batch(10, drop_remainder=True)
for batch in batches.take(5):
    print(batch.numpy())

# 또는 한 단계 씩 밀집된 예측을하기 위해 기능과 레이블을 한 단계 씩 서로 이동할 수 있습니다.
def dense_1_step(batch):
    # Shift features and labels on step relative to each other.
    return batch[:-1], batch[1:]

predict_dense_1_step = batches.map(dense_1_step)

for features, label in predict_dense_1_step.take(3):
    print(features.numpy(), "=>", label.numpy())

# 고정 오프셋 대신 전체 창을 예측하려면 배치를 두 부분으로 나눌 수 있습니다.
batches = range_ds.batch(15, drop_remainder=True)

def label_next_5_steps(batch):
    return(batch[:-5], batch[-5:])
predict_5_steps = batches.map(label_next_5_steps)

for features, label in predict_5_steps.take(3):
    print(features.numpy(), "=>", label.numpy())

# Dataset.flat_map 메소드는 데이터 세트의 데이터 세트를 가져와 단일 데이터 세트로 병합 할 수 있습니다.
feature_length = 10
label_length = 5

features = range_ds.batch(feature_length, drop_remainder=True)
labels = range_ds.batch(feature_length).skip(1).map(lambda labels: labels[:-5])

predict_5_steps = tf.data.Dataset.zip((features, labels))

for features, label in predict_5_steps.take(3):
    print(features.numpy(), " => ", label.numpy())


'''
Using Window
Dataset.batch를 사용하는 동안 더 세밀한 제어가 필요한 상황이 있습니다. 
Dataset.window 메서드는 완전한 제어를 제공하지만주의가 필요합니다. 
Datasets의 Dataset을 반환합니다. 자세한 내용은 데이터 세트 구조를 참조하십시오.
'''
window_size = 5

windows = range_ds.window(window_size, shift=1)

for sub_ds in windows.take(5):
    print(sub_ds)

# 이 Dataset.flat_map메소드는 데이터 세트의 데이터 세트를 가져와 단일 데이터 세트로 병합 할 수 있습니다.
for x in windows.flat_map(lambda x: x).take(30):
    print(x.numpy(), end=' ')


# 거의 모든 경우에 먼저 .batch 데이터 세트를 원할 것입니다 .
def sub_to_batch(sub):
    return sub.batch(window_size, drop_remainder=True)

for example in windows.flat_map(sub_to_batch).take(5):
    print(example.numpy())

# 이제 shift인수가 각 윈도우 의 이동량을 제어 한다는 것을 알 수 있습니다 .
# 이것을 합치면 다음 함수를 작성할 수 있습니다.

def make_window_dataset(ds, window_size=5, shift=1, stride=1):
    windows = ds.window(window_size, shift=shift, stride=stride)
    windows = windows.flat_map(sub_to_batch)
    return windows
ds = make_window_dataset(range_ds, window_size=10, shift = 5, stride=3)
for example in ds.take(10):
    print(example.numpy())

# 그런 다음 이전과 같이 라벨을 쉽게 추출 할 수 있습니다.
dense_labels_ds = ds.map(dense_1_step)

for inputs,labels in dense_labels_ds.take(3):
    print(inputs.numpy(), "=>", labels.numpy())