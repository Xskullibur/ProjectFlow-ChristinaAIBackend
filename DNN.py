from keras.layers import Input, Subtract, Dense, Lambda
from keras.models import Model, Sequential
import keras.backend as K
from keras import layers


def build_siamese_network(encoder, input_shape):
    input_1 = Input(input_shape)
    input_2 = Input(input_shape)

    encoded_1 = encoder(input_1)
    encoded_2 = encoder(input_2)

    # caculate the eucliden distance
    embedded_distance = Subtract()([encoded_1, encoded_2])
    embedded_distance = Lambda(
        lambda x: K.sqrt(K.mean(K.square(x), axis=-1, keepdims=True))
    )(embedded_distance)

    output = Dense(1, activation='sigmoid')(embedded_distance)

    siamese = Model(inputs=[input_1, input_2], outputs=output)

    return siamese


def build_network(dropout=0.05):
    encoder = Sequential()

    # input layer
    encoder.add(layers.Conv1D(128, 32, padding='same', activation='relu'))
    encoder.add(layers.BatchNormalization())
    encoder.add(layers.SpatialDropout1D(dropout))
    encoder.add(layers.MaxPool1D(4, 4))

    # second layer convolution
    encoder.add(layers.Conv1D(128 * 2, 3, padding='same', activation='relu'))
    encoder.add(layers.BatchNormalization())
    encoder.add(layers.SpatialDropout1D(dropout))
    encoder.add(layers.MaxPool1D())

    # third layer convolution
    encoder.add(layers.Conv1D(128 * 3, 3, padding='same', activation='relu'))
    encoder.add(layers.BatchNormalization())
    encoder.add(layers.SpatialDropout1D(dropout))
    encoder.add(layers.MaxPool1D())

    # fourth layer convolution
    encoder.add(layers.Conv1D(128 * 4, 3, padding='same', activation='relu'))
    encoder.add(layers.BatchNormalization())
    encoder.add(layers.SpatialDropout1D(dropout))
    encoder.add(layers.MaxPool1D())

    encoder.add(layers.GlobalMaxPool1D())
    encoder.add(layers.Dense(64))

    return encoder
