import numpy as np
import librosa
import h5py
import yaml

def configure(pars):
    default_pars={'n_filters' : 40,
                  'n_fft' :128,
                  'hop_length' : 64,
                  'win_length' : 128}

    if pars is None:
        pars=default_pars
    else:
        for k in default_pars.keys():
            if not (k in pars.keys()):
                pars.update({k : default_pars[k]})
    return pars
    
def filter_bank (n_fft, nfilt, sample_rate):
    low_freq_mel  = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))        # Convert Hz to Mel
    mel_points    = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)   # Equally spaced in Mel scale
    hz_points     = (700 * (10**(mel_points / 2595) - 1))                 # Convert Mel to Hz
    bin           = np.floor((n_fft + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(n_fft / 2 + 1))))

    
    for m in range(1, nfilt + 1):

        f_m_minus = int(bin[m - 1])         # left
        f_m       = int(bin[m])             # center
        f_m_plus  = int(bin[m + 1])         # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    fbank = fbank.T
    #fbank /= (fbank**2).sum(axis=0)[None, :]
           
    return fbank

def reconstruct(X_mag, X_phase, pars=None):
    pars = configure(pars)
    # M = filter_banks(pars['n_fft'],
    #                  pars['n_filt'],
    #                  pars['sample_rate'])
    X_bar = X_mag*np.exp(1j*X_phase)
    x_bar = librosa.core.istft(X_bar, hop_length=pars['hop_length'], win_length=pars['win_length'])
    return x_bar

def compsdr(sig_est, sig_true):
    sig_est = sig_est[:, None]
    sig_true = sig_true[:, None]
    Psig_est = (sig_true.T.dot(sig_est)/sig_true.T.dot(sig_true))*sig_true
    distortion = ((sig_est-Psig_est)**2).sum()
    signal = (Psig_est**2).sum()
    SDR = 10*np.log10(signal/distortion)
    return SDR

def testsdr(X_bar_mag, Y_phase, X_mag, X_phase, pars=None):
    reconstructed = reconstruct(X_bar_mag, Y_phase, pars)
    true = reconstruct(X_mag, X_phase, pars)
    n_samples = np.min([reconstructed.size, true.size])
    reconstructed = reconstructed[:n_samples]
    true = true[:n_samples]
    SDR = compsdr(reconstructed, true)
    return SDR

def domel(x_mag, y_mag, n_mag, M):
    LC = 5
    x_mel = M.T.dot(x_mag)
    y_mel = M.T.dot(y_mag)
    n_mel = M.T.dot(n_mag)
    snr = 20.0*np.log10(x_mel/n_mel)
    ibm = (snr>LC)*1
    return x_mel, y_mel, n_mel, ibm

def unmel(y_mel, M, use_pinv=True):
    if use_pinv:
        y_bar_mag = np.linalg.pinv(M.T).dot(y_mel)
    else:
        y_bar_mag = M.dot(y_mel)
    return y_bar_mag

def test_tools():
    f = h5py.File('/home/silas/corpora/mixtures_te.hdf5')
    stft_pars = yaml.load(f.attrs['stft_pars'])
    x_mag = f['mix_1']['x_mag'][...]
    x_phase = f['mix_1']['x_phase'][...]
    y_mag = f['mix_1']['y_mag'][...]
    y_phase = f['mix_1']['y_phase'][...]
    n_mag = f['mix_1']['n_mag']
    
    SDR = testsdr(y_mag, y_phase, x_mag, x_phase, pars=stft_pars)
    print(SDR)

    M = filter_bank(stft_pars['n_fft'], 60, stft_pars['sr'])
    
    
    x_mel, y_mel, n_mel, ibm = domel(x_mag, y_mag, n_mag, M)

    plt.subplot(4, 1, 1)
    plt.imshow(x_mel, interpolation='nearest', aspect='auto')
    plt.subplot(4, 1, 2)
    plt.imshow(y_mel, interpolation='nearest', aspect='auto')
    plt.subplot(4, 1, 3)
    plt.imshow(n_mel, interpolation='nearest', aspect='auto')
    plt.subplot(4, 1, 4)
    plt.imshow(ibm, interpolation='nearest', aspect='auto')

    
    SDR = testsdr(unmel(y_mel, M), y_phase, x_mag, x_phase, pars=stft_pars)
    print(SDR)
