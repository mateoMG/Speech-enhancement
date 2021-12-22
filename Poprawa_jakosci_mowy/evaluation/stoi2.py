import numpy as np
import scipy.signal
import librosa

def metrics(x, y, n):
    x = x[:, None]
    y = y[:, None]
    n = n[:, None]

    s = np.hstack([x, n])
    
    s_target = x*(x.T.dot(y))/(x.T.dot(x))
    P_s_y = (s.dot(np.linalg.inv(s.T.dot(s)))).dot(s.T.dot(y))
    #P_s_y = (s.dot(np.linalg.pinv(s.T.dot(s)))).dot(s.T.dot(y))
    #P_s_y = (s.dot(np.linalg.inv(s.T.dot(s)+np.eye(2)*1e-5))).dot(s.T.dot(y))
    
    e_interf = P_s_y-s_target
    e_artif = y-P_s_y
    SDR = 10.0*np.log10(np.sum(s_target**2.0)/np.sum((e_interf + e_artif)**2.0))
    SIR = 10.0*np.log10(np.sum(s_target**2.0)/np.sum(e_interf**2.0))
    SAR = 10.0*np.log10(np.sum((s_target+e_interf)**2.0)/np.sum(e_artif**2.0))
    return SDR, SIR, SAR
    
def stdft(x, N, K, N_fft):
    frames = np.arange(0, len(x)-N, K)
    x_stdft = np.zeros((len(frames), N_fft), dtype=np.complex)
    w = scipy.signal.hann(N)

    for i in xrange(len(frames)):
        ii = slice(frames[i], frames[i]+N)
        x_stdft[i] = np.fft.fft(x[ii]*w, N_fft)

    return x_stdft


def remove_silent_frames(x, y, rng, N, K):
    frames = np.arange(0, len(x)-N, K)
    w = scipy.signal.hann(N)
    msk = np.zeros(len(frames))

    for j in xrange(len(frames)):
        jj = slice(frames[j], frames[j]+N)
        msk[j] = 20*np.log10(np.linalg.norm(x[jj]*w)/np.sqrt(N))

    msk = msk-np.max(msk)+rng > 0.0

    count = 0

    x_sil = np.zeros(len(x))
    y_sil = np.zeros(len(y))

    for j in xrange(len(frames)):
        if msk[j]:
            jj_i = slice(frames[j], frames[j]+N)
            jj_o = slice(frames[count], frames[count]+N)
            x_sil[jj_o] += x[jj_i]*w
            y_sil[jj_o] += y[jj_i]*w
            count += 1
    x_sil = x_sil[:jj_o.stop]
    y_sil = y_sil[:jj_o.stop]

    return x_sil, y_sil

def taa_corr(x, y):
    xn = x-np.mean(x)
    xn /= np.sqrt(np.sum(xn**2))
    yn = y-np.mean(y)
    yn /= np.sqrt(np.sum(yn**2))
    rho = np.sum(xn*yn)
    return rho

def thirdoct(fs, N_fft, num_bands, mn):
    f = np.linspace(0, fs, N_fft+1)
    f = f[:(N_fft/2)]
    k = np.arange(num_bands)
    cf = 2.0**(k/3.0)*float(mn)
    fl = np.sqrt((2.0**(k/3.0)*mn)*2.0**((k-1)/3.0)*float(mn))
    fr = np.sqrt((2.0**(k/3.0)*mn)*2.0**((k+1)/3.0)*float(mn))
    # print num_bands
    # print len(f)
    # print fl
    # print fr
    # print cf
    A = np.zeros((num_bands, len(f)))

    for i in xrange(len(cf)):
        b = np.argmin((f-fl[i])**2.0)
        fl[i] = f[b]
        fl_ii = b

        b = np.argmin((f-fr[i])**2.0)
        fr[i] = f[b]
        fr_ii = b
        A[i, fl_ii:fr_ii] = 1.0
    #rnk = np.sum(A, axis=1)
    #num_bands = np.where(np.logical_and((rnk[1:] >= rnk[:-1]), (rnk[1:] != 0)!=0))[0][-1]
    #A = A[:num_bands, :]
    #cf = cf[:num_bands]

    return A, cf




def stoi(x, y, sr_signal):
    assert(len(x)==len(y))

    sr = 10000
    N_frame = 256

    K = 512
    J = 15
    mn = 150

    H, _ = thirdoct(sr, K, J, mn)
    N = 30
    Beta = -15
    dyn_range = 40

    if sr_signal != sr:
        x = scipy.signal.resample(x, int((float(len(x))/float(sr_signal))*float(sr)))
        y = scipy.signal.resample(y, int((float(len(y))/float(sr_signal))*float(sr)))

    x, y = remove_silent_frames(x, y, dyn_range, N_frame, N_frame/2)
    #print 'silent frames'
    #print len(x)


    x_hat = stdft(x, N_frame, N_frame/2, K)
    y_hat = stdft(y, N_frame, N_frame/2, K)

    x_hat = x_hat[:, :(K/2)].T
    y_hat = y_hat[:, :(K/2)].T

    #X = np.zeros((J, x_hat.shape[1]))
    #Y = np.zeros((J, y_hat.shape[1]))

    X = np.sqrt(H.dot(np.abs(x_hat)**2.0))
    Y = np.sqrt(H.dot(np.abs(y_hat)**2.0))

    c = 10.0**(-Beta/20.0)

    d_interm = np.zeros((J, X.shape[1]-N+1))
    #J = 12 
    for m in xrange(N-1, X.shape[1]):
        X_seg = X[:, (m-N+1):m+1]
        Y_seg = Y[:, (m-N+1):m+1]

        alpha = np.sqrt(np.sum(X_seg**2.0, axis=1, keepdims=True)/np.sum(Y_seg**2.0, axis=1, keepdims=True))
        aY_seg = Y_seg*alpha    

        for j in xrange(J):
            Y_prime = np.minimum(aY_seg[j, :], X_seg[j, :]+X_seg[j, :]*c)
            d_interm[j, m-N] = taa_corr(X_seg[j, :].T, Y_prime.T)

    d = np.mean(d_interm)
    return d
# for i in xrange(x_hat.shape[1]):
#     X[:, i] = np.sqrt(H.dot(np.abs(x_hat[:, i])**2))
#     Y[:, i] = np.sqrt(H.dot(np.abs(y_hat[:, i])**2))


def test_stoi():
    x, sr_signal = librosa.load('./18174a.wav', sr=None)
    y, _ = librosa.load('./noisy.wav', sr=None)
    n = y-x
    
    print('x min {} max {}'.format(np.min(x), np.max(x)))
    print('y min {} max {}'.format(np.min(y), np.max(y)))

    d = stoi(x, y, sr_signal)
    print('STOI: {}'.format(d))
    SDR, SIR, SAR = metrics(x, y, n)
    print('SDR {} SIR {} SAR {}'.format(SDR, SIR, SAR))
    
