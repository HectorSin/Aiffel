import tensorflow as tf

def loss_function(y_true, y_pred):
    # y_true: 실제 레이블 (정수 인덱스)
    # y_pred: 모델의 출력 (logits)
    
    # 1. 로그 소프트맥스 계산
    log_softmax = tf.nn.log_softmax(y_pred, axis=-1)
    
    # 2. 실제 레이블의 로그 확률 추출
    # tf.gather를 사용하여 실제 레이블의 로그 확률을 추출
    log_probs = tf.gather(log_softmax, y_true, batch_dims=-1, axis=-1)
    
    # 3. 평균 손실 계산 (음의 로그 확률의 평균)
    loss = -tf.reduce_mean(log_probs)
    
    return loss
