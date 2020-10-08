import scipy.io.wavfile
import scipy.signal
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def ln_GLRT(s_array:np.ndarray, x_array:np.ndarray) -> np.ndarray:
    """[summary]

    Args:
        s_array (np.ndarray): Template to detect
        x_array (np.ndarray): Signal to analyze

    Returns:
        np.ndarray: Natural logarithm of the generalized likelihood ratio test
    """
    assert s_array.ndim == x_array.ndim == 1
    N = s_array.shape[0]

    GLRT_out = []

    for n_0 in tqdm(range(x_array.shape[0] - N)):
        x_array_truncate = x_array[n_0:n_0+N]

        A_MLE = np.sum(np.multiply(s_array,x_array_truncate)) / np.sum(np.square(s_array))

        sigma2_0_MLE = np.average(np.square(x_array_truncate))
        
        sigma2_1_MLE = np.average(np.square(x_array_truncate - (A_MLE * s_array)))

        GLRT_out.append( (N/2.0) * (np.log(sigma2_0_MLE) -  np.log(sigma2_1_MLE)) )
    
    return np.array(GLRT_out)

def multipass_lnGLRT(s_array:np.ndarray, x_array:np.ndarray, nbr_pass:int, treshold:float) -> np.ndarray:

    s_array_size = s_array.shape[0]
    template_array = s_array

    for i in range(nbr_pass):
        print("i = ", i)

        signal_lnGLRT = ln_GLRT(template_array, x_array)
        
        positions_detected, properties = scipy.signal.find_peaks(signal_lnGLRT, threshold=treshold, distance = s_array_size)

        template_array = np.mean([x_array[pos : pos + s_array_size] for pos in positions_detected ], axis=0)
    
    return signal_lnGLRT, positions_detected


def draw_signal_analysis(ax, template_URL:str, signal_URL:str):

    # Read wav files
    template_samplerate, template_WAV = scipy.io.wavfile.read(template_URL)
    signal_samplerate, signal_WAV = scipy.io.wavfile.read(signal_URL)
    assert template_samplerate == signal_samplerate

    # Get Likelihood ratio
    signal_lnGLRT, positions_detected = multipass_lnGLRT(template_WAV, signal_WAV, nbr_pass=1, treshold=20)
    print("Positions detected:", positions_detected/signal_samplerate)

    # Draw plot
    signal_timeline = np.linspace(0, signal_WAV.shape[0]/signal_samplerate, signal_WAV.shape[0])
    signal_lnGLRT_timeline = np.linspace(0, signal_lnGLRT.shape[0]/signal_samplerate, signal_lnGLRT.shape[0])

    ax_bis = ax.twinx()

    ax.plot(signal_timeline, signal_WAV, color='grey', alpha=.3, linewidth=.2)
    ax_bis.plot(signal_lnGLRT_timeline, signal_lnGLRT, color='#62a2f5', linewidth=1.5)

    ax.set_xlabel(r'Time $t$ $[s]$')
    ax_bis.set_ylabel(r'$\ln(L_G(t))$', color='#1f6dd1')
    ax.set_ylabel(r'Analyzed signal')

if __name__ == "__main__":
    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    draw_signal_analysis(ax, './audio_files/call_nature.wav', './audio_files/nature.wav')
    plt.show()