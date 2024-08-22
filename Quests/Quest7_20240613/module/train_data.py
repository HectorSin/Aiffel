from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import save_model
from data_preprocessing import data_work, max_length, data_encoding
from model import encoder

def train_data(text_max_len=45, summary_max_len=12, ratio=0.2, threshold = 7, embedding_dim=128, hidden_size=256, src_vocab=8000, tar_vocab=2000):
    data = data_work('data/news_summary_more.csv', 'text', 'headlines')
    data = max_length(data, text_max_len, summary_max_len)
    encoder_input_train, encoder_input_test, decoder_input_train, decoder_target_train, decoder_input_test, decoder_target_test = data_encoding(data, text_max_len, summary_max_len, ratio, threshold)
    model = encoder(embedding_dim, hidden_size, text_max_len, src_vocab, tar_vocab)
    
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
    es = EarlyStopping(monitor='val_loss', patience=2, verbose=1)

    history = model.fit(x=[encoder_input_train, decoder_input_train], y=decoder_target_train,
                        validation_data=([encoder_input_test, decoder_input_test], decoder_target_test),
                        batch_size=256, 
                        callbacks=[es], 
                        epochs=50)
    # Save the model
    save_model(model, data/'model.h5')

