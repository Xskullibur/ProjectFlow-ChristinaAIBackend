from keras.models import model_from_json, Model
from keras.layers import Input
import sys
from os import path
import os
import numpy as np
from scipy import stats

import librosa
from scipy.io import wavfile
from VoiceDataset import pad

################################
# Defined global configurable  #
################################
model_file = 'model.json'
model_weight_file = 'model.h5'
datasets_path = r'G:\NYP\EADP\EADP AI\Testing'
sr = 16000
nsec = 3
length = sr * nsec


def to_index(arr):
    unique = np.unique(arr)
    return [np.where(unique == a)[0][0] for a in arr]


def get_speakers():
    speakers = [f for f in os.listdir(datasets_path) if path.isdir(path.join(datasets_path, f))]
    return zip(speakers, range(len(speakers)))

##############
# Load Model #
##############


def load_model():
    if path.exists(model_file) and path.exists(model_weight_file):
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights("model.h5")
        print("Loaded model from disk")
        return model
    else:
        raise FileNotFoundError('Error: model.json or model.h5 is not found!')


def get_encoder(model):
    inputs = Input(shape=(length, 1))
    outputs = model.layers[2](inputs)
    network = Model(inputs=inputs, outputs=outputs)
    network.compile(loss='mse', optimizer='adam')
    return network


def load_wav(file):
    #_sr, data = wavfile.read(file)
    data, _sr = librosa.load(file, sr=sr)
    librosa.output.write_wav('misc/test2.wav', data, sr)
    # wav file must be in correct sample rate
    if _sr != sr:
        raise Exception('Error: input file {} is not in correct sample rate of {} but is in {}'.format(file, sr, _sr))
    else:
        return _sr, data


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def predict_embedding():

    siamese = load_model()
    network = get_encoder(siamese)

    speakers_embedding = []
    speakers = []
    for speaker in os.listdir(datasets_path):
        speaker_path = path.join(datasets_path, speaker)
        if path.isdir(speaker_path):
            for root, folders, files in os.walk(speaker_path):
                for f in files:
                    if f.endswith('.wav') or f.endswith('.WAV'):
                        _sr, wav_data = load_wav(path.join(root, f))
                        datas = [np.array(pad(data, int(_sr * nsec))) for data in
                                        list(chunks(wav_data, length))]
                        for data in datas:
                            embedding = network.predict(data.reshape(-1, length, 1))
                            speakers_embedding.append(embedding[0])
                            speakers.append(speaker)
    return speakers_embedding, speakers


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

if __name__ == '__main__':
    speaker_embedding, speakers = predict_embedding()
    np_speaker = np.array(to_index(speakers))

    fig = plt.figure(1, figsize=(4, 3))
    fig.suptitle('PCA')
    plt.clf()

    pca = PCA(n_components=3)
    pca.fit(speaker_embedding)
    X = pca.transform(speaker_embedding)

    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    for name, label in list(get_speakers()):
        ax.text3D(X[np_speaker == label, 0].mean(),
                  X[np_speaker == label, 1].mean() + 1.5,
                  X[np_speaker == label, 2].mean(), name,
                  horizontalalignment='center',
                  bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=np_speaker, cmap=plt.cm.nipy_spectral,
               edgecolor='k')

    fig = plt.figure(2, figsize=(4, 3))
    fig.suptitle('PCA Variances percentage')
    print('Variances: {}'.format(pca.explained_variance_))
    total_variances = np.sum(pca.explained_variance_)
    pcs = pca.explained_variance_ / total_variances
    plt.bar([1, 2, 3], pcs)
    plt.xticks([1, 2, 3])

    # fig = plt.figure(3, figsize=(4, 3))
    # tsne = TSNE(n_components=2, perplexity=40, n_iter=1000)
    # tsne_results = tsne.fit_transform(speaker_embedding)
    #
    # plt.scatter(tsne_results[:, 0], tsne_results[:, 1])

    plt.show()

