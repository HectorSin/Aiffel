import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ERROR 메시지만 출력하도록 설정
# GPU 설정 로그 억제
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# from tensorflow.keras.models import load_model
from transformer.Models import transformer, PositionalEncoding, Decoder
from transformer.Layers import DecoderLayer
from transformer.SubLayers import MultiHeadAttention
from train import preprocess_sentence, tokenizer, START_TOKEN, END_TOKEN, MAX_LENGTH
import tensorflow_datasets as tfds
from translator import sentence_generation
from translator import real_time_translation

tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file('tokenizer/tokenizer')
model_location = 'model/'

model = '202406211514_20epoch_transformer.h5'

model_name = model_location + model

# 저장된 모델 불러오기
loaded_model = tf.keras.models.load_model(model_name,
                                          custom_objects={'PositionalEncoding': PositionalEncoding,
                                                          'MultiHeadAttention': MultiHeadAttention,
                                                          # 'EncoderLayer': EncoderLayer,
                                                          'DecoderLayer': DecoderLayer,
                                                          # 'Encoder': Encoder,
                                                          'Decoder': Decoder})


sentence_generation('지금까지 어디 있었어?', loaded_model, tokenizer)

# real_time_translation(loaded_model, tokenizer)