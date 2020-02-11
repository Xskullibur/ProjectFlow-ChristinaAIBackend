from keras.optimizers import Adam
from keras.models import model_from_json
from DNN import build_network, build_siamese_network
from VoiceDataset import VoiceDataset
from os import path

################################
# Defined global configurable  #
################################
model_file = 'model.json'
model_weight_file = 'model.h5'
sr = 4000
nsec = 3
length = sr * nsec
datasets_path = r'G:\NYP\EADP\EADP AI'

###################
# Create datasets #
###################
train = VoiceDataset(datasets_path, 'Training', nsec=nsec, sr=sr)
train_generator = (batch for batch in train.get_yield_batch_data(2))

test = VoiceDataset(datasets_path, 'Testing', nsec=nsec, sr=sr)
test_generator = (batch for batch in test.get_yield_batch_data(2))

################
# Define model #
################

network = None
siamese = None

# check if model file exists then we load the model instead of creating new one
if path.exists(model_file) and path.exists(model_weight_file):
    ##############
    # Load Model #
    ##############
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    siamese = model_from_json(loaded_model_json)
    # load weights into new model
    siamese.load_weights("model.h5")
    print("Loaded model from disk")
else:
    network = build_network(dropout=0.0)
    siamese = build_siamese_network(network, (length, 1))

# plot_model(network, show_shapes=True, to_file=os.getcwd() + '/plots/network.png')
# plot_model(siamese, show_shapes=True, to_file=os.getcwd() + '/plots/siamese.png')

opt = Adam(clipnorm=1)
siamese.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
print(siamese.summary())

import multiprocessing
from keras.callbacks import LambdaCallback

# save temporary models
filepath = "models/model-{}.h5"
checkpoint = LambdaCallback(on_epoch_end=lambda epochs, logs: siamese.save_weights(filepath.format(epochs)))
callbacks_list = [checkpoint]

siamese.fit_generator(
    generator=train_generator,
    steps_per_epoch=100,
    epochs=50,
    validation_data=test_generator,
    validation_steps=100,
    callbacks=callbacks_list
    # workers=multiprocessing.cpu_count(),
    # use_multiprocessing=True
)

##############
# Save Model #
##############
# serialize model to JSON
model_json = siamese.to_json()
with open(model_file, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
siamese.save_weights(model_weight_file)
print("Saved model to disk")
