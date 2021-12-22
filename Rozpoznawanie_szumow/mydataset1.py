import os
import glob
from tqdm import tqdm
import librosa
import h5py
import soundfile as sf
import random
import numpy as np
import tensorflow as tf
import sys


noise = 'Noise'
clean_speech_file = 'Clean/corpora_train_bt.hdf5'


class Dataset:
    def __init__(self, clean_speech_file, noise):
        self.noise = self.load_noise(noise)
        self.clean = self.clean_batch(self.noise, clean_speech_file)

    def load_noise(self, name):
        noise_path = name
        noise_names = os.listdir(noise_path)

        noise_files_with_lists = {}

        for noise in noise_names:
            cur_path = noise_path + "/" + noise
            folders_names = os.listdir(cur_path)

            for folder_name in folders_names:
                cur_path_2 = cur_path + "/" + folder_name
                label = folder_name + "_" + noise

                current_list = []

                for file in glob.glob(cur_path_2 + "/*.wav"):
                    current_list.append(file)

                noise_files_with_lists[label] = current_list

        print("--------------Load noise from waves-------------------------")
        lists_noise = self.load_signal_from_wavs(noise_files_with_lists)
        return lists_noise

    def load_signal_from_wavs(self, lists):
        signals = dict()
        for k in lists.keys():
            print('Loading ' + k)
            current_list = lists[k]
            current_signals = []
            for f in tqdm(current_list):
                current_signal, _ = librosa.load(f, sr=44100)  # czy ustawić duration sygnału + jakies inne parametry ??
                current_signals.append(current_signal)
            signals[k] = current_signals
        return signals

    def clean_batch(self, noise,  clean):
        train_samples = val_samples = test_samples = 0
        for name in noise.keys():
            if name.startswith('TRAIN'):
                train_samples = train_samples + len(noise[name])
            elif name.startswith('VAL'):
                val_samples = val_samples + len(noise[name])
            elif name.startswith('TEST'):
                test_samples = test_samples + len(noise[name])

        clean_files_with_lists = {}

        c_train = h5py.File('Clean/clean_train.h5', 'w')
        c_val = h5py.File('Clean/clean_val.h5', 'w')
        c_test = h5py.File('Clean/clean_test.h5', 'w')

        with h5py.File(clean, 'r') as f:
            counter = 0
            for name in f.keys():
                if counter < train_samples:
                    c_train.create_dataset(name, data=f[name][...])
                elif counter < (train_samples + val_samples):
                    c_val.create_dataset(name, data=f[name][...])
                elif counter < (train_samples + val_samples + test_samples):
                    c_test.create_dataset(name, data=f[name][...])
                else:
                    clean_files_with_lists['Clean_train'] = c_train
                    clean_files_with_lists['Clean_val'] = c_val
                    clean_files_with_lists['Clean_test'] = c_test
                    break
                counter = counter + 1

        lists_clean = self.load_lists_clean(clean_files_with_lists)
        return lists_clean

    def load_lists_clean(self, files_with_lists):
        lists = dict()
        for name in files_with_lists:
            current_list = []
            for signal in files_with_lists[name]:
                signal_from_h5_file = files_with_lists[name][signal][...]  # signal from h5 file
                current_list.append(signal_from_h5_file)
            lists[name] = current_list
        #self.signals_to_wave(lists)
        return lists

    def signals_to_wave(self, signals):
        for name in signals.keys():
            for i in range(len(signals[name])):
                sf.write('Clean/Wav_signals/{}_{}.wav'.format(name, i), signals[name][i], samplerate=8000)

    def batch_creator(self, mode, batch_size):
        train_length = 1000


        # Noise
        noise_signal = []
        noise_names = []
        labels = []
        seed = 36
        #seed = random.randrange(0, 100)
        n_idx = 0
        for name in self.noise.keys():
            if name.startswith(mode):
                noise_signal += self.noise[name]
                for x in range(len(self.noise[name])):
                    labels.append(n_idx)
                    noise_names.append(name[(len(mode)+1):])
                n_idx += 1

        random.seed(seed)
        random.shuffle(noise_signal)

        random.seed(seed)
        random.shuffle(noise_names)

        random.seed(seed)
        random.shuffle(labels)

        #print(labels)
        #print(noise_names)
        #print(len(noise_signal[15]))


        #kodowanie kategorialne

        one_hot_train_labels = self.to_one_hot(labels, n_idx)


        sample_idx = 0

        while sample_idx < len(noise_signal):

            mixed_signals_spectrograms = np.empty((batch_size, 2, 256, train_length), dtype=np.float32)
            clean_signals_spectrograms = np.empty((batch_size, 2, 256, train_length), dtype=np.float32)
            weights = []

            mixed_signals_spectrograms.fill(0.0)
            clean_signals_spectrograms.fill(0.0)
            #weights.fill(0.0)
            #print(len(noise_signal[15]))

            for batch_idx in range(batch_size):
                mixed_spec = self.mix_signals(noise_signal[sample_idx+batch_idx])
                #sf.write('Mixed_signals/' + mode + '/signal_{}+{}.wav'.format((batch_idx+sample_idx), noise_names[batch_idx+sample_idx]), mixed,
                         #samplerate=8000)
                #print(mixed_spec.shape)
                if mixed_spec.shape[1] > train_length:
                    mixed_spec = mixed_spec[:, :train_length]
                #print(mixed_spec.shape)
                #start_idx = train_length//2 - mixed_spec.shape[1]//2
                start_idx = 0
                end_idx = start_idx + mixed_spec.shape[1]

                mixed_signals_spectrograms[batch_idx, 0, :, start_idx:end_idx] = np.real(mixed_spec)
                mixed_signals_spectrograms[batch_idx, 1, :, start_idx:end_idx] = np.imag(mixed_spec)
                #clean_signals_spectrograms[batch_idx, 0, :, start_idx:end_idx] = np.real(clean_spec)
                #clean_signals_spectrograms[batch_idx, 1, :, start_idx:end_idx] = np.imag(clean_spec)
                weights.append(noise_names[sample_idx+batch_idx])

            n_categories = one_hot_train_labels[sample_idx:(sample_idx+batch_size), :]
            sample_idx += batch_size
            yield mixed_signals_spectrograms, clean_signals_spectrograms, n_categories, weights# generator

    def to_one_hot(self, labels, dimension):
        results = np.zeros((len(labels), dimension))
        for idx, label in enumerate(labels):
            results[idx, label] = 1
        return results

    def mix_signals(self, n_signal):
        #x = int(len(n_signal)/2)

        #start_idx = random.randint(0, x)
        #n_signal = n_signal[start_idx: start_idx+x]

        #print(n_signal)


                  #RMS
        n_signal_rms = np.sqrt(np.mean(n_signal**2.0))

        #snr = 0.0
        #g = 10 ** (snr / 20.0) * n_signal_rms / c_signal_rms    #wzmocnienie
        g = 0.0


        noise_spec = self.spectrogram(n_signal)

        #mixed_signal = g*c_signal + n_signal
        #mixed_spec = clean_spec + noise_spec
        mixed_spec = noise_spec
        mixed_signal = n_signal
        #print(mixed_spec.shape)

        return mixed_spec

    def spectrogram(self, signal):
        sr = 44100
        n_fft = 512
        hop_length = int(sr*0.01)
        win_length = int(sr*0.025)
        window_fn = tf.signal.hann_window


        signal = np.pad(signal, int(n_fft // 2), mode='reflect') #odbicie (początek + koniec) jest po to aby po istft i stft wyszło to samo

        signal = tf.signal.stft(signal, fft_length=n_fft, frame_step=hop_length,
                                frame_length=win_length, window_fn=window_fn)

        signal = signal.numpy().T[:256] / 40.0  # 256, połowa, widmo odbite; dzielenie przez 40 normalizacja (-1,1)

        return signal


if __name__ == "__main__":

    obiekt = Dataset(clean_speech_file, noise)
    for batch in obiekt.batch_creator('TRAIN', 10):
        i, o, labels, w = batch
        print(labels)
        print(w)
        print('-------')
        print(i.shape)
    print('koniec')


