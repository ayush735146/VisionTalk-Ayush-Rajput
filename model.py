"""CNN + LSTM merge model.
BY AYUSH RAJPUT
"""
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Concatenate

def build_merge_model(vocab_size, max_length, feature_dim=4096, embedding_dim=256, lstm_units=256):
    image_input = Input(shape=(feature_dim,), name="image_features")
    image_branch = Dense(256, activation="relu", name="image_dense")(image_input)
    image_branch = Dropout(0.5, name="image_dropout")(image_branch)

    text_input = Input(shape=(max_length,), name="caption_sequence")
    text_branch = Embedding(vocab_size, embedding_dim, mask_zero=True, name="word_embedding")(text_input)
    text_branch = LSTM(lstm_units, name="caption_lstm")(text_branch)

    merged = Concatenate(name="multimodal_merge")([image_branch, text_branch])
    merged = Dense(256, activation="relu", name="fusion_dense")(merged)
    output = Dense(vocab_size, activation="softmax", name="next_word")(merged)

    model = Model(inputs=[image_input, text_input], outputs=output, name="VisionTalk_CNN_LSTM")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
