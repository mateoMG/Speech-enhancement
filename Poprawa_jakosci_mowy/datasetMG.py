import os
import numpy as np
import random
import h5py
import glob
from tqdm import tqdm
import yaml
import sys
from multiprocessing import Process, JoinableQueue
import librosa

from evaluation.reco import testsdr

class Dataset:
    def __init__(self, paths, train_batch_size, network):
        self.train_batch_size = train_batch_size
        self.SNR = 0.0
        self.network = network

        self.stft_pars = yaml.load(open('./cfg/stft_pars.yaml', 'r'))

        #self.train_starts = self.read_file_with_starts(paths['train_starts_file'])
        #self.test_starts = self.read_file_with_starts(paths['test_starts_file'])

        self.noise_type = 'airport'
        self.noise = self.load_noise(paths['noise'])
        self.noise_train, self.noise_val, self.noise_test = self.noise_split(self.noise, self.noise_type)

        self.train_data = self.read_hdf5(paths['train_hdf5_file'], self.noise_train) #, self.train_starts)

        #self.val_data = self.read_hdf5(paths['train_hdf5_file'], self.noise_val) Też tak można tylko osobny plik dla val_clean_speech

        random.shuffle(self.train_data)
        self.val_data = self.train_data[:len(self.train_data)//10]
        del self.train_data[:len(self.train_data)//10]

        self.test_data = self.read_hdf5(paths['test_hdf5_file'], self.noise_test)

        self.print_data_stats(mode='TRAIN')
        self.print_data_stats(mode='VAL')
        self.print_data_stats(mode='TEST')
        sys.stdout.flush()

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

    def noise_split(self, noise, type):
        noise_train = []
        noise_val = []
        noise_test = []

        for name in noise.keys():
            if name.startswith('TRAIN'):
                #if name[6:] == type:
                noise_train = noise[name]
            if name.startswith('VAL'):
                if name[4:] == type:
                    noise_val = noise[name]
            if name.startswith('TEST'):
                if name[5:] == type:
                    noise_test = noise[name]

        return noise_train, noise_val, noise_test

    def read_hdf5(self, filename, noise):
        items = list()
        stft_pars = self.stft_pars

        with h5py.File(filename, 'r') as f:
            counter = 0
            for signal in f.keys():
                item = dict()
                #We are converting to db scale and also make rough normalization to range near -1:1
                x = f[signal][...]
                n = noise[counter]

                start_idx = random.randint(0, len(n) - len(x))
                n = n[start_idx: start_idx + len(x)]

                rms_x = np.sqrt(np.mean(x**2.0))
                rms_n = np.sqrt(np.mean(n**2.0))

                gx=10.0**(self.SNR/20.0)*rms_n/rms_x

                x *= gx
                y = x + n
                
                x_complex = librosa.core.stft(x, hop_length=stft_pars['hop_length'], win_length=stft_pars['win_length'], window=stft_pars['window'], n_fft=stft_pars['n_fft'])
                n_complex = librosa.core.stft(n, hop_length=stft_pars['hop_length'], win_length=stft_pars['win_length'], window=stft_pars['window'], n_fft=stft_pars['n_fft'])
                y_complex = librosa.core.stft(y, hop_length=stft_pars['hop_length'], win_length=stft_pars['win_length'], window=stft_pars['window'], n_fft=stft_pars['n_fft'])

                item['x_mag'] = np.abs(x_complex)
                item['x_phase'] = np.angle(x_complex)

                item['n_mag'] = np.abs(n_complex)
                
                item['y_mag'] = np.abs(y_complex) #val['y_mag'][()]
                item['y_phase'] = np.angle(y_complex) #val['y_phase'][()]

                item['input1'] = np.real(y_complex)[:256]/40.0
                item['input2'] = np.imag(y_complex)[:256]/40.0

                item['output1'] = np.real(x_complex)[:256]/40.0
                item['output2'] = np.imag(x_complex)[:256]/40.0

                items.append(item)
                if counter < len(noise)-1:
                    counter = counter + 1
                else:
                    counter = 0
        return items

    def read_file_with_starts(self, filename):
        lines = open(filename, 'rt').readlines()
        starts = dict()
        for l in lines:
            k, v = l.strip().split()
            starts[k] = int(v)
        return starts
    
    def get_mode_data(self, mode):
        if mode == 'TRAIN':
            data = self.train_data
        elif mode == 'VAL':
            data = self.val_data
        else:
            data = self.test_data

        return data

    def print_data_stats(self, mode):
        data = self.get_mode_data(mode)

        print("{} stats: ".format(mode))
        print("   num items: {}".format(len(data)))

        min_lenght = 10000000000
        max_lenght = 0
        max_y = 0
        min_y = 100000
        sdr_original_y = 0.0
        sdr_perfect_x = 0.0
        for item in data:
            min_lenght = min(min_lenght, item['input1'].shape[1])
            max_lenght = max(max_lenght, item['input1'].shape[1])

            max_y = max(max_y, item['input1'].max())
            min_y = min(min_y, item['input1'].min())

            sdr_original_y += testsdr(item['y_mag'], item['y_phase'], item['x_mag'], item['x_phase'])



            S_r = item['output1']
            S_i = item['output2']

            Y_r = item['input1']
            Y_i = item['input2']

            M_r = (Y_r*S_r + Y_i*S_i)/(Y_r**2.0 + Y_i**2.0+1e-5)
            M_i = (Y_r*S_i - Y_i*S_r)/(Y_r**2.0 + Y_i**2.0+1e-5)

            K = 10.0
            C = 0.1

            M_r = K * (1.0-np.exp(-C*M_r))/(1.0+np.exp(-C*M_r))
            M_i = K * (1.0-np.exp(-C*M_i))/(1.0+np.exp(-C*M_i))

            M_r = np.clip(M_r, -10.0 + 1e-6, 10.0 - 1e-6)
            M_i = np.clip(M_i, -10.0 + 1e-6, 10.0 - 1e-6)
            M_r = -(1.0/C)*np.log((K-M_r)/(K+M_r))
            M_i = -(1.0/C)*np.log((K-M_i)/(K+M_i))

            x_pred = (M_r*Y_r - M_i*Y_i) + 1j*(M_r*Y_i + M_i*Y_r)
            x_pred *= 40.0



            x_mag = item['y_mag'].copy()
            x_mag[:256] = np.abs(x_pred)

            x_pha = item['y_phase'].copy()
            x_pha[:256] = np.angle(x_pred)

            sdr_perfect_x += testsdr(x_mag, x_pha, item['x_mag'], item['x_phase'])

        print("   min max length: {} {}".format(min_lenght, max_lenght))
        print("   min max spectrogram value: {} {}".format(min_y, max_y))
        print("   average sdr of original y: {}".format(sdr_original_y/len(data)))
        print("   average sdr if net out is perfect: {}".format(sdr_perfect_x/len(data)))
        print("")

    def batch_creator(self, mode):
        data = self.get_mode_data(mode)
        batch_size = self.train_batch_size if mode == 'TRAIN' else 1
        train_length = 668
        if mode == 'TRAIN':
            random.shuffle(data)
            train_length = self.network.train_length()

        inputs = np.empty((batch_size, 2, self.network.num_freq(), train_length), dtype=np.float32)
        weights = np.empty((batch_size, 2, self.network.num_freq(), train_length), dtype=np.float32)
        targets = np.empty((batch_size, 2, self.network.num_freq(), train_length), dtype=np.float32)
        original_data = None
        
        sample_idx = 0
        while sample_idx + batch_size <= len(data):
            inputs.fill(0.0)
            weights.fill(0.0)
            targets.fill(0.0)
            for batch_idx in range(batch_size):
                sample = data[sample_idx + batch_idx]
                if sample['input1'].shape[1] > train_length:
                    start_idx_from = random.randint(0, sample['input1'].shape[1] - train_length)
                    stop_idx_from = start_idx_from + train_length
                    start_idx_to = 0
                    stop_idx_to = train_length
                else:
                    start_idx_from = 0
                    stop_idx_from = sample['input1'].shape[1]
                    start_idx_to = (train_length - sample['input1'].shape[1])//2
                    stop_idx_to = start_idx_to + sample['input1'].shape[1]
                assert(sample['input1'].shape[0] == self.network.num_freq())

                inputs[batch_idx, 0, :, start_idx_to:stop_idx_to] = sample['input1'][:, start_idx_from:stop_idx_from]
                inputs[batch_idx, 1, :, start_idx_to:stop_idx_to] = sample['input2'][:, start_idx_from:stop_idx_from]

                weights[batch_idx, :, :, start_idx_to:stop_idx_to] = 1.0

                S_r = sample['output1'][:, start_idx_from:stop_idx_from]
                S_i = sample['output2'][:, start_idx_from:stop_idx_from]

                Y_r = sample['input1'][:, start_idx_from:stop_idx_from]
                Y_i = sample['input2'][:, start_idx_from:stop_idx_from]

                M_r = (Y_r*S_r + Y_i*S_i)/(Y_r**2.0 + Y_i**2.0+1e-5)
                M_i = (Y_r*S_i - Y_i*S_r)/(Y_r**2.0 + Y_i**2.0+1e-5)

                K = 10.0
                C = 0.1

                M_r = K * (1.0-np.exp(-C*M_r))/(1.0+np.exp(-C*M_r))
                M_i = K * (1.0-np.exp(-C*M_i))/(1.0+np.exp(-C*M_i))

                targets[batch_idx, 0, :, start_idx_to:stop_idx_to] = M_r
                targets[batch_idx, 1, :, start_idx_to:stop_idx_to] = M_i

                if mode != 'TRAIN':
                    original_data = (sample['x_mag'],
                                     sample['x_phase'],
                                     sample['y_mag'],
                                     sample['y_phase'],
                                     start_idx_to)

            sample_idx += batch_size
            yield inputs, targets, weights, original_data


if __name__ == "__main__":
    data_files = {'train_hdf5_file': 'Clean/corpora_train_bt.hdf5',
                  'noise': 'Noise',
                                                            }
    obiekt = Dataset(data_files, 10,10)
    print(obiekt.noise_train[5])
    print(len(obiekt.noise_val))
    print(len(obiekt.noise_test))

    print('koniec')
