from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow_datasets as tfds
import tensorflow as tf
import time
import numpy as numpy
import matplotlib.pyplot as plt
import numpy as np

examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                               as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']

train_examples

tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus((
    en.numpy() for pt, en in train_examples), target_vocab_size=2**13)
tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (pt.numpy() for pt, en in train_examples), target_vocab_size=2**13)

sample_string  = 'Transformer is awesome.'
tokenized_string = tokenizer_en.encode(sample_string)
print('Tokenized string is {}'.format(tokenized_string))

original_string = tokenizer_en.decode(tokenized_string)
print('The original string: {}'.format(original_string))

# 토큰화는 단어가 사전에 없는 경우 문자열을 하위 단어로 분리하여 인코딩함.
for ts in tokenized_string:
    print('{} ----> {}'.format(ts,tokenizer_en.decode([ts])))

BUFFER_SIZE = 20000
BATCH_SIZE = 64

def encode(lang1, lang2):
    lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(
        lang1.numpy()) +[tokenizer_pt.vocab_size+1]

    lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
        lang2.numpy()) +[tokenizer_en.vocab_size+1]
    
    return lang1, lang2

def tf_encode(pt, en):
    result_pt, result_en = tf.py_function(encode, [pt, en], [tf.int64, tf.int64])
    result_pt.set_shape([None])
    result_en.set_shape([None])
    return result_pt, result_en

MAX_LENGTH = 40

def filter_max_length(x, y, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length,
    tf.size(y) <= max_length)

train_dataset = train_examples.map(tf_encode)
train_dataset = train_dataset.filter(filter_max_length)
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE,padded_shapes=([None],[None]))
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
train_dataset

val_dataset = val_examples.map(tf_encode)
val_dataset = val_dataset.filter(filter_max_length).padded_batch(BATCH_SIZE,padded_shapes=([None],[None]))

pt_batch, en_batch = next(iter(val_dataset))
pt_batch, en_batch

# position encoding
def get_angles(pos, i, d_model):
    angle_rates = 1/np.power(10000, (2*(i//2))/ np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
    np.arange(d_model)[np.newaxis, :],
    d_model)

    # 2i의 어레이 인덱스에 sin 적용
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # 2i+1의 어레이 인덱스에 cos 적용
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding =angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

pos_encoding = positional_encoding(50, 512)
print(pos_encoding.shape)

import matplotlib.pyplot as plt
plt.pcolormesh(pos_encoding[0], cmap='RdBu')
plt.xlabel('Depth')
plt.xlim((0,512))
plt.ylabel('Position')
plt.colorbar()
plt.show()


# masking
# 토큰을 마스크한다.
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq,0), tf.float32)

    # padding을 추가하기 위해서 extra dimensions을 추가한다. attention logits.을 위함
    return seq[:, tf.newaxis, tf.newaxis, :] # (batch_size, 1, 1, seq_len)

x = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
create_padding_mask(x)
'''
미리보기 마스크는 미래 토큰을 순서대로 마스킹하는 데 사용됩니다. 다시 말해, 
마스크는 어떤 항목을 사용하지 않아야하는지 나타냅니다.
이는 세 번째 단어를 예측하기 위해 첫 번째와 두 번째 단어 만 사용됨을 의미합니다. 
네 번째 단어를 예측하는 것과 마찬가지로 첫 번째, 두 번째 및 세 번째 단어 만 사용됩니다.'''

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size,size)), -1, 0)
    return mask # (seq_len, seq_len)

x = tf.random.uniform((1, 3))
x.numpy()
temp = create_look_ahead_mask(x.shape[1])
temp

def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v는 선행 dimension과 일치해야합니다.
    k, v는 일치하는 두번째 차원을 가져야한다. i.e.:seq_len_k = seq_len_v
    마스크는 그 type에 따라 shape가 다르다. (padding or look ahead)
    하지만 추가하려면 broadcastable 해야한다.'''

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask : (..., seq_len_q, seq_len_k)로 Broadcast가능한 shape의 Tensor 기본값은 None

    Returns:
    output, attention_weights
    """
    matmul_qk = tf.matmul(q, k, transpose_b =True) # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # Scaled된 tensor에 마스크 추가
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax는 마지막 축 (seq_len k)에서 정규화되므로 점수가 1이됩니다.
    attention_weights = tf.nn.softmax(scaled_attention_logits)

    output = tf.matmul(attention_weights, v) # (..., seq_len_q, depth_v)
    return output, attention_weights


# 소프트 맥스 정규화가 K에서 수행 될 때, 그 값은 Q에 주어진 중요도를 결정합니다.
# 출력은주의 가중치와 V (값) 벡터의 곱을 나타냅니다. 이렇게하면 집중하려는 
# 단어가 그대로 유지되고 관련이없는 단어가 플러시됩니다.
def print_out(q, k, v):
    temp_out, temp_attn = scaled_dot_product_attention(
        q,k,v,None)
    print('Attention weights are:')
    print(temp_attn)
    print('Output is')
    print(temp_out)

np.set_printoptions(suppress=True)

temp_k = tf.constant([[10,0,0],
[0,10,0],
[0,0,10],
[0,0,10]], dtype=tf.float32) # (4,3)

temp_v = tf.constant([[1,0],
[10,0],
[100,5],
[1000,6]],dtype=tf.float32) #(4,2)

# 이 '쿼리'는 두 번째 '키'와 정렬되므로 두 번째 '값'이 반환됩니다.
temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32) # (1,3)
print_out(temp_q, temp_k, temp_v)

# 이 쿼리는 반복되는 키 (3 및 4)와 정렬되므로 모든 관련 값이 평균화됩니다.
temp_q = tf.constant([[0, 0, 10]], dtype=tf.float32)  # (1, 3)
print_out(temp_q, temp_k, temp_v)

# 이 쿼리는 첫 번째와 두 번째 키와 동일하게 정렬되므로 값이 평균화됩니다.
temp_q = tf.constant([[10, 10, 0]], dtype=tf.float32)  # (1, 3)
print_out(temp_q, temp_k, temp_v)

# 모든 쿼리를 함께 전달해보면.
temp_q = tf.constant([[0, 0, 10], [0, 10, 0], [10, 10, 0]], dtype=tf.float32)  # (3, 3)
print_out(temp_q, temp_k, temp_v)


# Multi head attention
''' 
각 멀티 헤드주의 블록에는 3 개의 입력이 있습니다. Q (쿼리), K (키), V (값) 이들은 선형
(고밀도) 레이어를 통해 여러 헤드로 분할됩니다.
위에서 정의한 scaled_dot_product_attention은 각 헤드에 적용됩니다 
(효율성을 위해 브로드 캐스트). 주의 단계에서 적절한 마스크를 사용해야합니다. 
그런 다음 각 헤드에 대한주의 출력이 연결되고 (tf.transpose 및 tf.reshape를 사용하여)
최종 밀도 계층을 통과합니다.
하나의 단일주의 헤드 대신 Q, K 및 V는 모델이 서로 다른 표현 공간의 다른 위치에있는 정보에
공동으로 참석할 수있게하므로 여러 헤드로 분할됩니다. 분할 후 각 헤드의 치수가 감소하므로 
전체 계산 비용은 전체 치수로 단일 헤드주의와 동일합니다.'''
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """마지막 차원을 (num_heads, 깊이)로 나눕니다.
        모양이 (batch_size, num_heads, seq_len, depth)가되도록 결과를 바꿉니다."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0,2,1,3])

    def call(self, v,k,q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q) # (batch_size, seq_len, d_model)
        k = self.wk(k) # (batch_size, seq_len, d_model)
        v = self.wv(v) # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size) # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size) # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size) # (batch_size, num_heads, seq_len_v, depth)
        
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q,k,v,mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0,2,1,3])
        # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention) # (batch_size, seq_len_q, d_model)
        return output, attention_weights
'''시험해볼 MultiHeadAttention 레이어를 만듭니다. 시퀀스의 각 위치 (y)에서 
MultiHeadAttention은 시퀀스의 다른 모든 위치에서 8 개의주의 헤드를 모두 실행하여
 각 위치에서 동일한 길이의 새로운 벡터를 반환합니다.
'''

temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
y = tf.random.uniform((1, 60 ,512)) # (batch_size, encoder_sequence, d_model)
out, attn = temp_mha(y, k=y, q=y, mask=None)
out.shape, attn.shape

'''
Point wise feed forward network
포인트 단위 피드 포워드 네트워크는 ReLU 활성화를 통해 완전히 연결된 두 개의 레이어로
구성됩니다.
'''
def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'), # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model) # (batch_size, seq_len, d_model)
    ])

sample_ffn = point_wise_feed_forward_network(512, 2048)
sample_ffn(tf.random.uniform((64,50,512))).shape


''' 인코더 및 디코더
각 인코더 계층은 다음과 같은 하위 계층으로 구성된다.
1. 멀티 헤드 어텐션 (패딩 마스크 포함)
2. Point wise feed forward network
이 서브 계층들 각가은 그 주위에 잔차 연결을 갖고 있기 때문에 계층 정규화를 갖는다.
잔차 연결은 딥 네트워크에서 사라지는 그래디언트 문제를 방지한다.
각 하위 계층의 출력은 LayerNorm (x + Sublayer (x))입니다. 
정규화는 d_model (마지막) 축에서 수행됩니다. 변압기에는 N 개의 엔코더 레이어가 있습니다.
'''

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask) # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1) # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output) # (batch_size, input_seq_len, d_model)
        return out2

sample_encoder_layer = EncoderLayer(512,8,2048)

sample_encoder_layer_output = sample_encoder_layer(
    tf.random.uniform((64,43,512)),False, None)

sample_encoder_layer_output.shape

''' 디코더 레이어

각 디코더 계층은 하위 계층으로 구성됩니다.

마스킹 된 멀티 헤드어텐션 (미리보기 마스크 및 패딩 마스크 포함)
멀티 헤드 어텐션 (패딩 마스크 포함). V (값) 및 K (키)는 인코더 출력을 입력으로받습니다. 
Q (쿼리)는 마스크 된 다중 헤드주의 서브 레이어로부터 출력을 수신합니다.

현명한 피드 포워드 네트워크
이들 서브 계층들 각각은 그 주위에 잔류 연결을 갖고이어서 계층 정규화를 갖는다.
 각 하위 계층의 출력은 LayerNorm (x + Sublayer (x))입니다. 
 정규화는 d_model (마지막) 축에서 수행됩니다.

변압기에는 N 개의 디코더 레이어가 있습니다.

Q가 디코더의 첫 번째주의 블록에서 출력을 수신하고 K가 인코더 출력을 수신함에 따라 
어텐션 가중치는 인코더의 출력을 기반으로 디코더의 입력에 주어진 중요성을 나타냅니다. 
다시 말해, 디코더는 엔코더 출력을보고 자체 출력에 자체 참석하여 다음 단어를 예측합니다.
스케일 도트 제품주의 섹션에서 위의 데모를 참조하십시오
'''

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, 
    look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output,
        out1, padding_mask) # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1) # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2) # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2) # (batch_size, target_seq_len, d_model)
        return out3, attn_weights_block1, attn_weights_block2

sample_decoder_layer = DecoderLayer(512, 8, 2048)

sample_decoder_layer_output, _, _ = sample_decoder_layer(
    tf.random.uniform((64,50,512)), sample_encoder_layer_output,
    False, None, None)

sample_decoder_layer_output.shape # (batch_size, target_seq_len, d_model)


'''
Encoder
인코더는 다음으로 구성됩니다.

1. 입력 임베딩
2. 위치 인코딩
3. N 개의 인코더 층
입력은 위치 인코딩과 합쳐진 임베딩을 통해 이루어집니다. 
이 합산의 출력은 인코더 계층에 대한 입력입니다. 인코더의 출력은 디코더에 대한 입력입니다.
'''

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
    maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
        self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
        for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding
        x = self.embedding(x) # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training = training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x # (batch_size, input_seq_len, d_model)

sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8,
dff=2048, input_vocab_size=8500, maximum_position_encoding=10000)

temp_input = tf.random.uniform((64,62), dtype=tf.int64, minval=0, maxval=200)

sample_encoder_output = sample_encoder(temp_input, training=False, mask=None)

print(sample_encoder_output.shape) # (batch_size, input_seq_len, d_model)


'''디코더
다음으로 Decoder구성됩니다.

1. 출력 임베딩
2. 위치 인코딩
3. N 디코더 레이어
타겟은 위치 인코딩과 합산 된 임베딩을 통해 이루어진다. 이 합산의 출력은 디코더 층에 
대한 입력이다. 디코더의 출력은 최종 선형 계층에 대한 입력입니다.
'''

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, 
    maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
        for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
    
    def call(self, x, enc_output, training, 
    look_ahead_mask, padding_mask):
        
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x) # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
            look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights

sample_decoder = Decoder(num_layers=2, d_model=512, num_heads=8,
dff=2048, target_vocab_size=8000, maximum_position_encoding=5000)

temp_input = tf.random.uniform((64,26), dtype=tf.int64, minval=0, maxval=200)

output, attn =sample_decoder(temp_input,
enc_output=sample_encoder_output,
training=False,
look_ahead_mask=None,
padding_mask=None)

output.shape, attn['decoder_layer2_block2'].shape


'''
Transformer 만들기
Transformer는 인코더, 디코더 및 최종 선형 계층으로 구성됩니다. 
디코더의 출력은 선형 레이어에 대한 입력이며 출력이 반환됩니다.
'''

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
    target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
        input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
        target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask,
    look_ahead_mask, dec_padding_mask):
        
        enc_output = self.encoder(inp, training, enc_padding_mask) # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output) # (batch_size, tar_seq_len, target_vocab_size)
        return final_output, attention_weights


sample_transformer = Transformer(
    num_layers=2, d_model=512, num_heads=8, dff=2048, 
    input_vocab_size=8500, target_vocab_size=8000, 
    pe_input=10000, pe_target=6000)

temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)

fn_out, _ = sample_transformer(temp_input, temp_target, training=False, 
                               enc_padding_mask=None, 
                               look_ahead_mask=None,
                               dec_padding_mask=None)

fn_out.shape  # (batch_size, tar_seq_len, target_vocab_size)


'''
하이퍼 파라미터 설정
이 예제를 작고 비교적 빠르게 유지하기 위해 num_layers, d_model 및 
dff 의 값이 줄었습니다.

Transformer 기본 모델에 사용 된 값은 다음과 같습니다. num_layers = 6, 
d_model = 512 , dff = 2048 입니다. 다른 모든 버전의 Transformer에 대해서는 
논문 을 참조하십시오 .
'''

num_layers = 4
d_model = 128
dff = 512
num_heads = 8

input_vocab_size = tokenizer_pt.vocab_size + 2
target_vocab_size = tokenizer_en.vocab_size + 2
dropout_rate = 0.1

'''
옵티 마이저
논문 의 공식에 따라 맞춤형 학습 속도 스케줄러와 함께 
Adam 최적화 프로그램을 사용하십시오 .
'''

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
epsilon=1e-9)


temp_learning_rate_schedule = CustomSchedule(d_model)

plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
plt.ylabel('Learning Rate')
plt.xlabel('Train Step')

'''
손실 및 지표
대상 시퀀스가 ​​채워 지므로 손실을 계산할 때 패딩 마스크를 적용하는 것이 중요합니다.
'''

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction="none")

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real,0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

''' Training and CheckPoint '''
transformer = Transformer(num_layers, d_model, num_heads, dff,
input_vocab_size, target_vocab_size, pe_input=input_vocab_size,
pe_target=target_vocab_size, rate=dropout_rate)

def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)
    
    # 디코더의 두 번째 Attention 블록에 사용됩니다.
    # 이 패딩 마스크는 인코더 출력을 마스킹하는 데 사용됩니다.
    dec_padding_mask = create_padding_mask(inp)

    # 디코더의 첫 번째 Attention 블록에 사용됩니다.
    # 디코더가 수신 한 입력에서 미래 토큰을 채우고 마스킹하는 데 사용됩니다.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return enc_padding_mask, combined_mask, dec_padding_mask


# 검사 점 경로와 검사 점 관리자를 만듭니다. 모든 n에포크 마다 체크 포인트를 저장하는 데 사용됩니다 .
checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')

EPOCHS =20 

''' @ tf.function은 train_step을 더 빠른 실행을 위해 TF 그래프로 추적 컴파일합니다.
 이 함수는 인수 텐서의 정확한 모양에 특화되어 있습니다. 가변 시퀀스 길이 또는 
 가변 배치 크기 (마지막 배치가 더 작음)로 인한 재 추적을 피하려면 
 input_signature를 사용하여보다 일반적인 모양을 지정하십시오.
 '''
train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]
@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
  
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
  
    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp, 
                                     True, 
                                     enc_padding_mask, 
                                     combined_mask, 
                                     dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
  
    train_loss(loss)
    train_accuracy(tar_real, predictions)

# 입력 언어로는 포르투갈 언어가 사용되고 영어는 대상 언어이다.

for epoch in range(EPOCHS):
    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()

    # inp -> portuguese, tar -> english

    for (batch, (inp, tar)) in enumerate(train_dataset):
        train_step(inp, tar)
        if batch % 50 == 0 :
            print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                epoch + 1, batch, train_loss.result(), train_accuracy.result()))
    
    if (epoch + 1) % 5 ==0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch+1,
        ckpt_save_path))

    print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
    train_loss.result(),
    train_accuracy.result()))
    print('time taken for 1 epoch: {} secs\n'.format(time.time() - start))

'''
평가
다음 단계는 평가에 사용됩니다.

포르투갈어 토크 나이저 (tokenizer_pt)를 사용하여 입력 문장을 인코딩하십시오. 
또한 시작 및 종료 토큰을 추가하여 입력이 모델이 학습 한 것과 동일하도록하십시오. 엔코더 입력입니다.
디코더 입력은 시작 토큰 == tokenizer_en.vocab_size입니다.
패딩 마스크와 미리보기 마스크를 계산하십시오.
그런 다음 디코더는 인코더 출력과 자체 출력 (자기주의)을보고 예측을 출력합니다.
마지막 단어를 선택하고 해당 단어의 argmax를 계산하십시오.
예측 된 단어를 디코더 입력에 연결하여 디코더로 전달합니다.
이 접근법에서, 디코더는 예측 된 이전 단어에 기초하여 다음 단어를 예측한다.
'''

def evaluate(inp_sentence):
    start_token = [tokenizer_pt.vocab_size]
    end_token = [tokenizer_pt.vocab_size +1]
    # inp 문장은 포르투갈어이므로 시작 및 끝 토큰 영어 시작 토큰을 추가합니다.
    inp_sentence = start_token + tokenizer_pt.encode(inp_sentence) + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)
    
    # 대상이 영어이므로 변환기의 첫 번째 단어는 영어 시작 토큰이어야합니다.
    decoder_input = [tokenizer_en.vocab_size]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input,
        output,
        False, 
        enc_padding_mask,
        combined_mask,
        dec_padding_mask)

        # seq_len 차원에서 마지막 단어를 선택하십시오
        predictions = predictions[:, -1:, :] # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # predicted_id가 종료 토큰과 같은 경우 결과를 반환
        if predicted_id == tokenizer_en.vocab_size+1:
            return tf.squeeze(output, axis=0), attention_weights
        
        # 예측 된 id를 입력으로서 디코더에 제공되는 출력에 연결한다.
        output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0), attention_weights

def plot_attention_weights(attention, sentence, result, layer):
    fig = plt.figure(figsize=(16,8))

    sentence = tokenizer_pt.encode(sentence)

    attention = tf.squeeze(attention[layer], axis=0)

    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head+1)

        # plot the attention weights
        ax.matshow(attention[head][:-1, :], cmap='viridis')

        fontdict = {'fontsize': 10}
        
        ax.set_xticks(range(len(sentence)+2))
        ax.set_yticks(range(len(result)))

        ax.set_ylim(len(result)-1.5, -0.5)

        ax.set_xticklabels(
            ['<start>']+[tokenizer_en.decode([i]) for i in sentence] + ['<end>'],
            fontdict=fontdict, rotation=90)
        
        ax.set_yticklabels([tokenizer_en.decode([i]) for i in result
        if i < tokenizer_en.vocab_size],
        fontdict=fontdict)

        ax.set_xlabel('Head {}'.format(head + 1))
    plt.tight_layout()
    plt.show()

def translate(sentence, plot=''):
    result, attention_weights = evaluate(sentence)

    predicted_sentence = tokenizer_en.decode([i for i in result
    if i < tokenizer_en.vocab_size])

    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(predicted_sentence))

    if plot:
        plot_attention_weights(attention_weights, sentence, result, plot)

translate("este é um problema que temos que resolver.")
print ("Real translation: this is a problem we have to solve .")

translate("os meus vizinhos ouviram sobre esta ideia.")
print ("Real translation: and my neighboring homes heard about this idea .")

translate("vou então muito rapidamente partilhar convosco algumas histórias de algumas coisas mágicas que aconteceram.")
print ("Real translation: so i 'll just share with you some stories very quickly of some magical things that have happened .")

translate("este é o primeiro livro que eu fiz.", plot='decoder_layer4_block2')
print ("Real translation: this is the first book i've ever done.")
