import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ERROR 메시지만 출력하도록 설정
# GPU 설정 로그 억제
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# from tensorflow.keras.models import load_model
from transformer.Models import transformer, PositionalEncoding, Encoder, Decoder
from transformer.Layers import EncoderLayer, DecoderLayer
from transformer.SubLayers import MultiHeadAttention
from train import preprocess_sentence, tokenizer, START_TOKEN, END_TOKEN, MAX_LENGTH
import tensorflow_datasets as tfds


tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file('tokenizer/tokenizer')
model_location = 'model/'

model = 'transformer_25epoch_202406201639.h5'

model_name = model_location + model

# 저장된 모델 불러오기
loaded_model = tf.keras.models.load_model(model_name,
                                          custom_objects={'PositionalEncoding': PositionalEncoding,
                                                          'MultiHeadAttention': MultiHeadAttention,
                                                          'EncoderLayer': EncoderLayer,
                                                          'DecoderLayer': DecoderLayer,
                                                          'Encoder': Encoder,
                                                          'Decoder': Decoder})

def decoder_inference(sentence, model, tokenizer):
  sentence = preprocess_sentence(sentence)

  # 입력된 문장을 정수 인코딩 후, 시작 토큰과 종료 토큰을 앞뒤로 추가.
  # ex) Where have you been? → [[8331   86   30    5 1059    7 8332]]
  sentence = tf.expand_dims(
      START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

  # 디코더의 현재까지의 예측한 출력 시퀀스가 지속적으로 저장되는 변수.
  # 처음에는 예측한 내용이 없음으로 시작 토큰만 별도 저장. ex) 8331
  output_sequence = tf.expand_dims(START_TOKEN, 0)

  # 디코더의 인퍼런스 단계
  for i in range(MAX_LENGTH):
    # 디코더는 최대 MAX_LENGTH의 길이만큼 다음 단어 예측을 반복합니다.
    predictions = model(inputs=[sentence, output_sequence], training=False)
    predictions = predictions[:, -1:, :]

    # 현재 예측한 단어의 정수
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    # 만약 현재 예측한 단어가 종료 토큰이라면 for문을 종료
    if tf.equal(predicted_id, END_TOKEN[0]):
      break

    # 예측한 단어들은 지속적으로 output_sequence에 추가됩니다.
    # 이 output_sequence는 다시 디코더의 입력이 됩니다.
    output_sequence = tf.concat([output_sequence, predicted_id], axis=-1)

  return tf.squeeze(output_sequence, axis=0)

def sentence_generation(sentence, model, tokernizer):
  # 입력 문장에 대해서 디코더를 동작 시켜 예측된 정수 시퀀스를 리턴받습니다.
  prediction = decoder_inference(sentence, model, tokenizer)

  # 정수 시퀀스를 다시 텍스트 시퀀스로 변환합니다.
  predicted_sentence = tokenizer.decode(
      [i for i in prediction if i < tokenizer.vocab_size])

  # print('나 : {}'.format(sentence))
  print('챗봇 : {}'.format(predicted_sentence))

  return predicted_sentence

def real_time_translation(model, tokenizer):
  print("종료를 원하시면 '종료'를 입력해주세요.")
  while True:
    sentence = input("나 : ")
    if sentence == '종료':
      break
    sentence_generation(sentence, model, tokenizer)