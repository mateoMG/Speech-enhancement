import sys
import os

import h5py
import yaml
import numpy as np
import cPickle as pickle

import scipy

from reco import testsdr
from stoi2 import stoi as teststoi
from stoi2 import metrics
import librosa

import vqmetrics


def read_hdf5(filename, noise, starts, stft_pars):
    SNR = 0.0
    items = list()
    with h5py.File(filename, 'r') as f:
        n_items = len(f.keys())
        #n_items = 20
        for mix_idx in range(n_items):
            item = dict()
            #We are converting to db scale and also make rough normalization to range near -1:1
            uttid = 'utt_' + str(mix_idx)
            x = f[uttid][...]
            n = noise[starts[uttid]:starts[uttid]+x.size]

            rms_x = np.sqrt(np.mean(x**2.0))
            rms_n = np.sqrt(np.mean(n**2.0))

            gx=10.0**(SNR/20.0)*rms_n/rms_x

            x *= gx
            y = x + n

            x_complex = librosa.core.stft(x, hop_length=stft_pars['hop_length'], win_length=stft_pars['win_length'], window=stft_pars['window'], n_fft=stft_pars['n_fft'])
            n_complex = librosa.core.stft(n, hop_length=stft_pars['hop_length'], win_length=stft_pars['win_length'], window=stft_pars['window'], n_fft=stft_pars['n_fft'])
            y_complex = librosa.core.stft(y, hop_length=stft_pars['hop_length'], win_length=stft_pars['win_length'], window=stft_pars['window'], n_fft=stft_pars['n_fft'])

            item['x_mag'] = np.abs(x_complex)
            item['x_phase'] = np.angle(x_complex)

            item['n_mag'] = np.abs(n_complex)
            item['n_phase'] = np.angle(n_complex)

            item['y_mag'] = np.abs(y_complex) #val['y_mag'][()]
            item['y_phase'] = np.angle(y_complex) #val['y_phase'][()]

            item['input1'] = np.real(y_complex)[:256]/40.0
            item['input2'] = np.imag(y_complex)[:256]/40.0

            item['output1'] = np.real(x_complex)[:256]/40.0
            item['output2'] = np.imag(x_complex)[:256]/40.0

            items.append(item)
    return items

def read_file_with_starts(filename):
    lines = open(filename, 'rt').readlines()
    starts = dict()
    for l in lines:
        k, v = l.strip().split()
        starts[k] = int(v)
    return starts




def write_wav(path, y, sr, norm=False):
    """Output a time series as a .wav file

    Note: only mono or stereo, floating-point data is supported.
        For more advanced and flexible output options, refer to
        `soundfile`.

    Parameters
    ----------
    path : str
        path to save the output wav file

    y : np.ndarray [shape=(n,) or (2,n), dtype=np.float]
        audio time series (mono or stereo).

        Note that only floating-point values are supported.

    sr : int > 0 [scalar]
        sampling rate of `y`

    norm : boolean [scalar]
        enable amplitude normalization.
        For floating point `y`, scale the data to the range [-1, +1].

    Examples
    --------
    Trim a signal to 5 seconds and save it back

    >>> y, sr = librosa.load(librosa.util.example_audio_file(),
    ...                      duration=5.0)
    >>> librosa.output.write_wav('file_trim_5s.wav', y, sr)

    See Also
    --------
    soundfile.write
    """

    # Validate the buffer.  Stereo is okay here.
    #util.valid_audio(y, mono=False)

    # normalize
    # if norm and np.issubdtype(y.dtype, np.floating):
    #     wav = util.normalize(y, norm=np.inf, axis=None)
    # else:
    #     wav = y

    wav = y
    # Check for stereo
    if wav.ndim > 1 and wav.shape[0] == 2:
        wav = wav.T

    # Save
    scipy.io.wavfile.write(path, sr, wav)

def get_default_pars():
    pars = {'properties_file': '../properties.yaml'}
    return pars


if __name__ == '__main__':
    if len(sys.argv) == 1:
        pars = get_default_pars()
    elif len(sys.argv) == 2:
        pars = yaml.load(open(sys.argv[1], 'r'))

    output_tag = open('../exp.tag', 'rt').read()

    SNR=0.0
    


    paths = yaml.load(open(pars['properties_file'], 'r'))

    data_files = {'train_hdf5_file' : os.path.join(paths['data_path'], 'corpora_train.hdf5'),
                  'test_hdf5_file' : os.path.join(paths['data_path'], 'corpora_test.hdf5'),
                  'train_noise_file' : os.path.join(paths['data_path'], 'noise_tr.wav'),
                  'test_noise_file' : os.path.join(paths['data_path'], 'noise_tst.wav'),
                  'train_starts_file' : os.path.join(paths['data_path'], 'starts_tr.txt'),
                  'test_starts_file' : os.path.join(paths['data_path'], 'starts_tst.txt')}


    stft_pars = yaml.load(open('../cfg/stft_pars.yaml', 'rt'))
    noise, _ = librosa.load(data_files['test_noise_file'], stft_pars['sr'])
    starts = read_file_with_starts(data_files['test_starts_file'])


    TEST_HDF5_FILE = data_files['test_hdf5_file']
    OUTPUT_FILE = os.path.join(paths['output_path'], output_tag,  "predictions.pkl")
    DUMP_PATH = os.path.join(paths['output_path'], output_tag)

    if not os.path.exists(DUMP_PATH):
        os.makedirs(DUMP_PATH)


    mixtures = read_hdf5(TEST_HDF5_FILE, noise, starts, stft_pars)
    predictions = pickle.load(open(OUTPUT_FILE, "rb"))

    assert(len(mixtures) == len(predictions))

    maxv = np.iinfo(np.int16).max

    sdr_old = 0.0
    stoi = 0.0
    sdr = 0.0
    sir = 0.0
    sar = 0.0
    pesq = 0.0
    for idx in range(len(predictions)):

        x_mag_pred = mixtures[idx]['y_mag'].copy()
        x_mag_pred[:256] = np.abs(predictions[idx])

        x_phase_pred = mixtures[idx]['y_phase'].copy()
        x_phase_pred[:256] = np.angle(predictions[idx])


        #sdr_old += testsdr(x_mag_pred, x_phase_pred,
        #           mixtures[idx]['x_mag'], mixtures[idx]['x_phase'])

        complex_noisy = mixtures[idx]['y_mag']*np.exp(1j*mixtures[idx]['y_phase'])
        noisy_wave = librosa.core.istft(complex_noisy, hop_length=80, win_length=200, window='hann')
        filename_noisy = os.path.join(DUMP_PATH, "{}_3_noisy.wav".format(str(idx+1).zfill(3)))
        write_wav(filename_noisy, (noisy_wave*maxv).astype(np.int16), 8000)
        
        complex_clean = x_mag_pred*np.exp(1j*x_phase_pred)
        clean_wave = librosa.core.istft(complex_clean, hop_length=80, win_length=200, window='hann')
        clean_wave /= 1.1*np.max(np.abs(clean_wave))

        filename_clean = os.path.join(DUMP_PATH, "{}_1_clean.wav".format(str(idx+1).zfill(3)))
        write_wav(filename_clean, (clean_wave*maxv).astype(np.int16), 8000)

        complex_original = mixtures[idx]['x_mag']*np.exp(1j*mixtures[idx]['x_phase'])
        original_wave = librosa.core.istft(complex_original, hop_length=80, win_length=200, window='hann')
        original_wave /= 1.1*np.max(np.abs(original_wave))

        filename_orig = os.path.join(DUMP_PATH, "{}_2_original.wav".format(str(idx+1).zfill(3)))
        write_wav(filename_orig, (original_wave*maxv).astype(np.int16), 8000)

        complex_noise = mixtures[idx]['n_mag']*np.exp(1j*mixtures[idx]['n_phase'])
        noise_wave = librosa.core.istft(complex_noise, hop_length=80, win_length=200, window='hann')

        stoi += teststoi(clean_wave, original_wave, 8000)

        pesq_idx = vqmetrics.pesq(filename_orig, filename_clean, 8000, 'pesq')[0]

        sd, si, sa = metrics(original_wave, clean_wave, noise_wave)
        sdr += sd
        sir += si
        sar += sa
        pesq += pesq_idx

    #print "SDR:       {}".format(sdr_old/len(predictions))
    print "SDR (alt): {}".format(sdr/len(predictions))
    print "SIR:       {}".format(sir/len(predictions))
    print "SAR:       {}".format(sar/len(predictions))
    print "STOI:      {}".format(stoi/len(predictions))
    print "PESQ:      {}".format(pesq/len(predictions))
