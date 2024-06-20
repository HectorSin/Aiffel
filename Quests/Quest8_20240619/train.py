import tensorflow as tf
import tensorflow_datasets as tfds
import datetime
import os
import re
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformer.Models import transformer, PositionalEncoding, Encoder, Decoder
from transformer.Layers import EncoderLayer, DecoderLayer
from transformer.SubLayers import MultiHeadAttention

# 사용할 샘플의 최대 개수
MAX_SAMPLES = 50000

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
    inputs, outputs = [], []
    with open(path_to_dataset, 'rt', encoding='UTF8') as file:
        lines = file.readlines()
        for line in lines[1:]:
            parts = line.split(',')
            inputs.append(preprocess_sentence(parts[0]))
            outputs.append(preprocess_sentence(parts[1]))

            if len(inputs) >= MAX_SAMPLES:
                return inputs, outputs
    return inputs, outputs

# 데이터를 로드하고 전처리하여 질문을 questions, 답변을 answers에 저장합니다.
questions, answers = load_conversation()

# 질문과 답변 데이터셋에 대해서 Vocabulary 생성
tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(questions + answers, target_vocab_size=2**13)

# 시작 토큰과 종료 토큰에 고유한 정수를 부여합니다.
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

# 시작 토큰과 종료 토큰을 고려하여 +2를 하여 단어장의 크기를 산정합니다.
VOCAB_SIZE = tokenizer.vocab_size + 2

# 샘플의 최대 허용 길이 또는 패딩 후의 최종 길이
MAX_LENGTH = 40

# 정수 인코딩, 최대 길이를 초과하는 샘플 제거, 패딩
def tokenize_and_filter(inputs, outputs):
  tokenized_inputs, tokenized_outputs = [], []
  
  for (sentence1, sentence2) in zip(inputs, outputs):
    # 정수 인코딩 과정에서 시작 토큰과 종료 토큰을 추가
    sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
    sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN

    # 최대 길이 40 이하인 경우에만 데이터셋으로 허용
    if len(sentence1) <= MAX_LENGTH and len(sentence2) <= MAX_LENGTH:
      tokenized_inputs.append(sentence1)
      tokenized_outputs.append(sentence2)
  
  # 최대 길이 40으로 모든 데이터셋을 패딩
  tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
  tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_outputs, maxlen=MAX_LENGTH, padding='post')
  
  return tokenized_inputs, tokenized_outputs

questions, answers = tokenize_and_filter(questions, answers)

BATCH_SIZE = 64
BUFFER_SIZE = 20000

# 디코더는 이전의 target을 다음의 input으로 사용합니다.
# 이에 따라 outputs에서는 START_TOKEN을 제거하겠습니다.
dataset = tf.data.Dataset.from_tensor_slices((
    {
        'inputs': questions,
        'dec_inputs': answers[:, :-1]
    },
    {
        'outputs': answers[:, 1:]
    },
))

dataset = dataset.cache()
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

# 하이퍼파라미터
VOCAB_SIZE = 10000  # 예시 값
NUM_LAYERS = 4  # 예시 값
D_MODEL = 256  # 예시 값
NUM_HEADS = 8  # 예시 값
UNITS = 512  # 예시 값
DROPOUT = 0.1  # 예시 값


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
    learning_rate = 1e-4
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    # 모델 요약 정보 출력
    model.summary()

    # 데이터셋 사용하여 모델 학습
    EPOCHS = 60  # 예시 값
    model.fit(dataset, epochs=EPOCHS)
    time = datetime.datetime.now().strftime('%Y%m%d%H%M')
    model_name = f'model/transformer_{EPOCHS}epoch_{time}.h5'

    model.save(model_name)