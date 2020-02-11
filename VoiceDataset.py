import os
from scipy.io import wavfile
import librosa
from tqdm import tqdm
import pandas as pd
import numpy as np

import pickle
from scipy.io import wavfile
cache_file = 'misc/index-{}.pickle'
import time
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed
class VoiceDataset:

    def __init__(self, path, subfolder, sr=16000, nsec=3, stochastic=True, pad=True, cache=True, preprocessing=None):
        self._path = path
        self._nsec = nsec
        self._sr = sr
        self._fragment_length = int(nsec * sr)
        self._stochastic = stochastic
        self._pad = pad
        self._preprocessing = preprocessing

        index = None

        if os.path.exists(cache_file.format(subfolder)) and cache:
            pickle_in = open(cache_file.format(subfolder), 'rb')
            index = pickle.load(pickle_in)
            pickle_in.close()
        if index is None:
            index = self.index_path(subfolder)
            if cache:
                pickle_out = open(cache_file.format(subfolder), 'wb')
                pickle.dump(index, pickle_out)
                pickle_out.close()

        self._df = pd.DataFrame(index)
        self._df = self._df.assign(index=np.arange(len(self._df)))

        self._datasetid_to_filepath = self._df.to_dict()['path']
        self._datasetid_to_speaker_id = self._df.to_dict()['speaker_id']

    def get_speaker(self, speaker):
        return self._df[self._df['speaker_id'] == speaker]

    def get_same_pair(self, size):
        random_sample = pd.merge(self._df.sample(size*2),
                                    self._df,
                                    on='speaker_id'
                                    ).sample(size)[['speaker_id', 'index_x', 'index_y']]
        random_sample_zip = zip(random_sample['index_x'].values.tolist(), random_sample['index_y'].values.tolist())

        return list(random_sample_zip)

    def get_wrong_pair(self, size):
        random_sample = self._df.sample(size)
        random_sample_from_other_speakers = self._df[~self._df['speaker_id'].isin(
            random_sample['speaker_id'])].sample(size)

        random_sample_zip = zip(random_sample['index'].values, random_sample_from_other_speakers['index'].values)

        return list(random_sample_zip)

    def get_yield_batch_data(self, size):
        while True:
            yield self.get_batch_data(size)
    @timeit
    def get_batch_data(self, size):

        same_pairs = self.get_same_pair(size // 2)

        input_1_same = np.stack([self[in1][0] for (in1, in2) in same_pairs])
        input_2_same = np.stack([self[in2][0] for (in1, in2) in same_pairs])

        wrong_pairs = self.get_wrong_pair(size // 2)

        input_1_wrong = np.stack([self[in1][0] for (in1, in2) in wrong_pairs])
        input_2_wrong = np.stack([self[in2][0] for (in1, in2) in wrong_pairs])

        input_1 = np.vstack([input_1_same, input_1_wrong])[:, :, np.newaxis]
        input_2 = np.vstack([input_2_same, input_2_wrong])[:, :, np.newaxis]

        outputs = np.append(np.zeros(size // 2), np.ones(size // 2))[:, np.newaxis]

        return [input_1, input_2], outputs

    def __getitem__(self, index):
        wav, sr = librosa.load(self._datasetid_to_filepath[index], sr=self._sr)

        fragment_length = int(sr * self._nsec)

        # Choose a random sample of the file
        if self._stochastic:
            fragment_start_index = np.random.randint(0, max(len(wav) - fragment_length, 1))
        else:
            fragment_start_index = 0

        instance = wav[fragment_start_index:fragment_start_index + fragment_length]
        # Check if need padding
        if self._pad:
            instance = pad(instance, fragment_length, self._stochastic)

        if self._preprocessing:
            instance = self._preprocessing(instance)
        label = self._datasetid_to_speaker_id[index]
        return instance, label

    def index_path(self, subfolder):
        voices = []

        print('Currently indexing file')

        total_files_len = 0

        for root, folders, files in os.walk(os.path.join(self._path, subfolder)):
            total_files_len += len([f for f in files if f.endswith('.wav') or f.endswith('.WAV')])
        progress_bar = tqdm(total=total_files_len)

        for speaker in os.listdir(os.path.join(self._path, subfolder)):
            speaker_path = os.path.join(self._path, subfolder, speaker)
            for root, folders, files in os.walk(speaker_path):
                total_files_len += len(files)
                for f in files:
                    if f.endswith('.wav') or f.endswith('.WAV'):
                        progress_bar.update(1)
                        sr, wav = wavfile.read(os.path.join(root, f))
                        voices.append({
                            'speaker_id': speaker,
                            'id': f,
                            'path': os.path.join(root, f),
                            'sample_rate': sr,
                            'length': len(wav),
                            'duration (s)': len(wav) / sr
                        })

        progress_bar.close()
        return voices


def pad(instance, fragment_length, stochastic=False):
    # Check for required length and pad if necessary
    if len(instance) < fragment_length:
        less_timesteps = fragment_length - len(instance)
        if stochastic:
            # Stochastic padding, ensure instance length == self.fragment_length by appending a random number of 0s
            # before and the appropriate number of 0s after the instance
            less_timesteps = fragment_length - len(instance)

            before_len = np.random.randint(0, less_timesteps)
            after_len = less_timesteps - before_len

            instance = np.pad(instance, (before_len, after_len), 'constant')
            return instance
        else:
            # Deterministic padding. Append 0s to reach self.fragment_length
            instance = np.pad(instance, (0, less_timesteps), 'constant')
            return instance
    else:
        return instance
