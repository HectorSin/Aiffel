from bs4 import BeautifulSoup 
import re
import pandas as pd
import numpy as np
import os
import tensorflow as tf 
from tensorflow.python.client import device_lib
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

contractions = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                           "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                           "you're": "you are", "you've": "you have"}

def get_data(link):
    data = pd.read_csv(link,  encoding='iso-8859-1')
    return data

def data_extract(data):
    data = data.lower()
    data = data.dropna()
    data = data.drop_duplicates()
    return data

def data_preprocessing(data, remove_stopwords = True):
    data = data.lower() # 텍스트 소문자화
    data = BeautifulSoup(data, "lxml").text # <br />, <a href = ...> 등의 html 태그 제거
    data = re.sub(r'\([^)]*\)', '', data) # 괄호로 닫힌 문자열 (...) 제거 Ex) my husband (and myself!) for => my husband for
    data = re.sub('"','', data) # 쌍따옴표 " 제거
    data = ' '.join([contractions[t] if t in contractions else t for t in data.split(" ")]) # 약어 정규화
    data = re.sub(r"'s\b","", data) # 소유격 제거. Ex) roland's -> roland
    data = re.sub("[^a-zA-Z]", " ", data) # 영어 외 문자(숫자, 특수문자 등) 공백으로 변환
    data = re.sub('[m]{2,}', 'mm', data) # m이 3개 이상이면 2개로 변경. Ex) ummmmmmm yeah -> umm yeah
    
    # 불용어 제거 (Text)
    if remove_stopwords:
        tokens = ' '.join(word for word in data.split() if not word in stopwords.words('english') if len(word) > 1)
    # 불용어 미제거 (Summary)
    else:
        tokens = ' '.join(word for word in data.split() if len(word) > 1)
    return tokens

def data_work(link, text, summary):
    data = get_data(link)
    data = data_extract(data)
    data = data_preprocessing(data)
    clean_text = []
    for t in data[text]:
        clean_text.append(data_preprocessing(t, remove_stopwords = True))
    clean_summary = []
    for t in data[summary]:
        clean_summary.append(data_preprocessing(t, remove_stopwords = False))
    data['clean_text'] = clean_text
    data['clean_summary'] = clean_summary
    data.dropna(axis=0, inplace=True)
    return data

def max_length(data, text_max_len=45, summary_max_len=12, ratio=0.2):
    data = data[data['text'].apply(lambda x: len(x.split()) <= text_max_len)]
    data = data[data['headlines'].apply(lambda x: len(x.split()) <= summary_max_len)]
    # add sostoken and eostoken
    data['decoder_input'] = data['headlines'].apply(lambda x : 'sostoken '+ x)
    data['decoder_target'] = data['headlines'].apply(lambda x : x + ' eostoken')
    return data


def data_encoding(data, text_max_len=45, summary_max_len=12, ratio=0.2, threshold = 6):
    encoder_input = np.array(data['text']) # 인코더의 입력
    decoder_input = np.array(data['decoder_input']) # 디코더의 입력
    decoder_target = np.array(data['decoder_target']) # 디코더의 레이블
    # encoder_input과 크기와 형태가 같은 순서가 섞인 정수 시퀀스 만들기
    indices = np.arange(encoder_input.shape[0])
    np.random.shuffle(indices)
    encoder_input = encoder_input[indices]
    decoder_input = decoder_input[indices]
    decoder_target = decoder_target[indices]
    
    n_of_val = int(len(encoder_input)*ratio) # 검증 데이터의 수
    encoder_input_train = encoder_input[:-n_of_val]
    decoder_input_train = decoder_input[:-n_of_val]
    decoder_target_train = decoder_target[:-n_of_val]

    encoder_input_test = encoder_input[-n_of_val:]
    decoder_input_test = decoder_input[-n_of_val:]
    decoder_target_test = decoder_target[-n_of_val:]

    src_tokenizer = Tokenizer() # 토크나이저 정의
    src_tokenizer.fit_on_texts(encoder_input_train) # 입력된 데이터로부터 단어 집합 생성

    src_vocab = 8000
    src_tokenizer = Tokenizer(num_words=src_vocab) # 단어 집합의 크기를 8,000으로 제한
    src_tokenizer.fit_on_texts(encoder_input_train) # 단어 집합 재생성

    # 텍스트 시퀀스를 정수 시퀀스로 변환
    encoder_input_train = src_tokenizer.texts_to_sequences(encoder_input_train) 
    encoder_input_test = src_tokenizer.texts_to_sequences(encoder_input_test)

    tar_tokenizer = Tokenizer()
    tar_tokenizer.fit_on_texts(decoder_input_train)

    total_cnt = len(tar_tokenizer.word_index) # 단어의 수
    rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
    total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
    rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

    # 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
    for key, value in tar_tokenizer.word_counts.items():
        total_freq = total_freq + value

        # 단어의 등장 빈도수가 threshold보다 작으면
        if(value < threshold):
            rare_cnt = rare_cnt + 1
            rare_freq = rare_freq + value

    tar_vocab = 2000
    tar_tokenizer = Tokenizer(num_words=tar_vocab) 
    tar_tokenizer.fit_on_texts(decoder_input_train)
    tar_tokenizer.fit_on_texts(decoder_target_train)

    # 텍스트 시퀀스를 정수 시퀀스로 변환
    decoder_input_train = tar_tokenizer.texts_to_sequences(decoder_input_train) 
    decoder_target_train = tar_tokenizer.texts_to_sequences(decoder_target_train)
    decoder_input_test = tar_tokenizer.texts_to_sequences(decoder_input_test)
    decoder_target_test = tar_tokenizer.texts_to_sequences(decoder_target_test)

    drop_train = [index for index, sentence in enumerate(decoder_input_train) if len(sentence) == 1]
    drop_test = [index for index, sentence in enumerate(decoder_input_test) if len(sentence) == 1]

    encoder_input_train = [sentence for index, sentence in enumerate(encoder_input_train) if index not in drop_train]
    decoder_input_train = [sentence for index, sentence in enumerate(decoder_input_train) if index not in drop_train]
    decoder_target_train = [sentence for index, sentence in enumerate(decoder_target_train) if index not in drop_train]

    encoder_input_test = [sentence for index, sentence in enumerate(encoder_input_test) if index not in drop_test]
    decoder_input_test = [sentence for index, sentence in enumerate(decoder_input_test) if index not in drop_test]
    decoder_target_test = [sentence for index, sentence in enumerate(decoder_target_test) if index not in drop_test]

    encoder_input_train = pad_sequences(encoder_input_train, maxlen=text_max_len, padding='post')
    encoder_input_test = pad_sequences(encoder_input_test, maxlen=text_max_len, padding='post')
    decoder_input_train = pad_sequences(decoder_input_train, maxlen=summary_max_len, padding='post')
    decoder_target_train = pad_sequences(decoder_target_train, maxlen=summary_max_len, padding='post')
    decoder_input_test = pad_sequences(decoder_input_test, maxlen=summary_max_len, padding='post')
    decoder_target_test = pad_sequences(decoder_target_test, maxlen=summary_max_len, padding='post')
    return encoder_input_train, encoder_input_test, decoder_input_train, decoder_target_train, decoder_input_test, decoder_target_test