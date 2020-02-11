from keras.models import model_from_json
from VoiceDataset import VoiceDataset
from os import path
from scipy.io import wavfile


################################
# Defined global configurable  #
################################
model_file = 'model.json'
model_weight_file = 'model.h5'
sr = 16000
nsec = 3
length = sr * nsec
datasets_path = r'G:\NYP\EADP\EADP AI'

##############
# Load Model #
##############

network = None
siamese = None

if path.exists(model_file) and path.exists(model_weight_file):
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    siamese = model_from_json(loaded_model_json)
    # load weights into new model
    siamese.load_weights("model.h5")
    print("Loaded model from disk")
else:
    raise FileNotFoundError('Error: model.json or model.h5 is not found!')

################
# Prepare data #
################

test = VoiceDataset(datasets_path, 'Testing', nsec=nsec, stochastic=False)

print('====================')
print('Predicting same pair')
print('====================')

same_pair = test.get_same_pair(1)

input_1, speaker_1 = test[same_pair[0][0]]
input_2, speaker_2 = test[same_pair[0][1]]
# save inputs files as wav for debugging
wavfile.write('misc/same_speaker1.wav', sr, input_1)
wavfile.write('misc/same_speaker2.wav', sr, input_2)

print('Testing against speaker {} with speaker {}'.format(speaker_1, speaker_2))

prediction = siamese.predict([input_1.reshape(-1, length, 1), input_2.reshape(-1, length, 1)])[0][0]

assert prediction > 0.9, 'Prediction is not accurate, predicted value: {}'.format(prediction)
print('Prediction score: {}'.format(prediction))

print('=====================')
print('Predicting wrong pair')
print('=====================')
wrong_pair = test.get_wrong_pair(1)

input_1, speaker_1 = test[wrong_pair[0][0]]
input_2, speaker_2 = test[wrong_pair[0][1]]
# save inputs files as wav for debugging
wavfile.write('misc/wrong_speaker1.wav', sr, input_1)
wavfile.write('misc/wrong_speaker2.wav', sr, input_2)

print('Testing against speaker {} with speaker {}'.format(speaker_1, speaker_2))

prediction = siamese.predict([input_1.reshape(-1, length, 1), input_2.reshape(-1, length, 1)])[0][0]

assert prediction < 0.1, 'Prediction is not accurate, predicted value: {}'.format(prediction)
print('Prediction score: {}'.format(prediction))
