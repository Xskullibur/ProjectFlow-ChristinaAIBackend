from keras.models import model_from_json
import sys
from os import path
import numpy as np
from scipy import stats

from scipy.io import wavfile
import librosa
from VoiceDataset import pad

################################
# Defined global configurable  #
################################
model_file = 'model.json'
model_weight_file = 'model-5.h5'
sr = 4000
nsec = 3
length = sr * nsec

##############
# Load Model #
##############


def load_model():
    if path.exists(model_file) and path.exists(model_weight_file):
        json_file = open(model_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(model_weight_file)
        print("Loaded model from disk")
        return model
    else:
        raise FileNotFoundError('Error: model.json or model.h5 is not found!')


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


def predict(input_1, input_2):
    input_1_sr, input_1_data = load_wav(input_1)
    input_2_sr, input_2_data = load_wav(input_2)

    # pad if length is not correct
    input_1_data = [np.array(pad(data, int(input_1_sr*nsec))) for data in list(chunks(input_1_data, length))]
    input_2_data = [np.array(pad(data, int(input_2_sr*nsec))) for data in list(chunks(input_2_data, length))]

    # load model
    siamese = load_model()

    # predict
    predictions = []
    for data_1 in input_1_data:
        for data_2 in input_2_data:
            prediction = siamese.predict([data_1.reshape(-1, length, 1), data_2.reshape(-1, length, 1)])[0][0]
            predictions.append(1.0 - prediction)

    mean = np.mean(predictions)
    median = np.median(predictions)
    mode = stats.mode(predictions)
    var = np.var(predictions)
    std = np.sqrt(var)

    # print results
    print('==============')
    print('= Prediction =')
    print('==============')
    print('Mean: {}'.format(mean))
    print('Median: {}'.format(median))
    print('Mode: {}'.format(mode))
    print('Variance: {}'.format(var))
    print('Std.: {}'.format(std))


if __name__ == '__main__':
    # check arguments is greater then 2
    if len(sys.argv) > 2:
        input_wav_file1 = str(sys.argv[1])
        input_wav_file2 = str(sys.argv[2])

        predict(input_wav_file1, input_wav_file2)

    elif len(sys.argv) == 0:
        print('Usage: python Predict.py <input_wav_file1> <input_wav_file2>')

    else:
        print('Error: Expected two arguments')
