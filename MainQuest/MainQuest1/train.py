import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ERROR 메시지만 출력하도록 설정
# GPU 설정 로그 억제
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
import tensorflow_datasets as tfds
import datetime
import re
from transformer.Models import transformer
from transformer.Loss import loss_function

# 하이퍼파라미터
MAX_SAMPLES = 50000
MAX_LENGTH = 40
BATCH_SIZE = 64
BUFFER_SIZE = 20000

# 전처리 함수
def preprocess_sentence(sentence):
  # 입력받은 sentence의 양쪽 공백을 제거
  sentence = sentence.strip()

  # 단어와 구두점(punctuation) 사이의 거리를 만듭니다.
  # 예를 들어서 "저는 학생입니다." => "저는 학생 입니다 ."와 같이
  # 학생과 마침표 사이에 거리를 만듭니다.
  sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
  sentence = re.sub(r'[" "]+', " ", sentence)
  # (한글, ".", "?", "!", ",")를 제외한 모든 문자를 공백인 ' '로 대체합니다.
  sentence = re.sub(r"[^가-힣?.!,]+", " ", sentence)
  sentence = sentence.strip()
  return sentence

# 데이터 로드 및 전처리
path_to_dataset = os.path.join(os.getcwd(),'data\ChatbotData.csv')

def load_conversation():
    # 논문에서 초기에 라벨이 없는 데이터 셋을 받는다고 하여 라벨을 제거하고 진행
    inputs = []
    with open(path_to_dataset, 'rt', encoding='UTF8') as file:
        lines = file.readlines()
        for line in lines[1:]:
            parts = line.split(',')
            # inputs.append(preprocess_sentence(parts[0]))
            # outputs.append(preprocess_sentence(parts[1]))
            inputs.append(preprocess_sentence(parts[1]))

            if len(inputs) >= MAX_SAMPLES:
                return inputs
    return inputs

# 데이터를 로드하고 전처리하여 질문을 questions, 답변을 answers에 저장합니다.
# questions, answers = load_conversation()
inputs = load_conversation()

# 질문과 답변 데이터셋에 대해서 Vocabulary 생성
tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(inputs, target_vocab_size=2**13)

# 시작 토큰과 종료 토큰에 고유한 정수를 부여합니다.
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

# 시작 토큰과 종료 토큰을 고려하여 +2를 하여 단어장의 크기를 산정합니다.
VOCAB_SIZE = tokenizer.vocab_size + 2



# 정수 인코딩, 최대 길이를 초과하는 샘플 제거, 패딩
def tokenize_and_filter(inputs):
  tokenized_inputs = []
  
  for sentence in inputs:
    # 정수 인코딩 과정에서 시작 토큰과 종료 토큰을 추가
    sentence = START_TOKEN + tokenizer.encode(sentence) + END_TOKEN
    # sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN

    # 최대 길이 40 이하인 경우에만 데이터셋으로 허용
    if len(sentence) <= MAX_LENGTH:
      tokenized_inputs.append(sentence)
      #tokenized_outputs.append(sentence2)
  
  # 최대 길이 40으로 모든 데이터셋을 패딩
  tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
  #tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
  #    tokenized_outputs, maxlen=MAX_LENGTH, padding='post')
  
  return tokenized_inputs

inputs = tokenize_and_filter(inputs)

# 디코더는 이전의 target을 다음의 input으로 사용합니다.
# 이에 따라 outputs에서는 START_TOKEN을 제거하겠습니다.
"""
dataset = tf.data.Dataset.from_tensor_slices((
    {
        # 'inputs': questions,
        # 'dec_inputs': answers[:, :-1]
        'dec_inputs': answers[:, :-1]
    },
    {
        'outputs': answers[:, 1:]
    },
))
"""
dataset = tf.data.Dataset.from_tensor_slices((inputs, inputs))
dataset = dataset.cache()
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

# 하이퍼파라미터
VOCAB_SIZE = 8000  # 예시 값
NUM_LAYERS = 6  # 예시 값
D_MODEL = 256  # 예시 값
NUM_HEADS = 8  # 예시 값
UNITS = 512  # 예시 값
DROPOUT = 0.15  # 예시 값
EPOCHS = 20  # 예시 값



if __name__ == '__main__':

    model = transformer(
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        units=UNITS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dropout=DROPOUT
    )

    # 모델 컴파일
    learning_rate = 1e-5
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                # loss = loss_function,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    # 모델 요약 정보 출력
    model.summary()

    # 데이터셋 사용하여 모델 학습
    model.fit(dataset, epochs=EPOCHS)
    time = datetime.datetime.now().strftime('%Y%m%d%H%M')
    model_name = f'model/{time}_{EPOCHS}epoch_transformer.h5'

    model.save(model_name)