import scipy.io.wavfile
import matplotlib.pyplot as plt
import numpy as np

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

    for n_0 in range(x_array.shape[0] - N):
        x_array_truncate = x_array[n_0:n_0+N]

        A_MLE = np.sum(np.multiply(s_array,x_array_truncate)) / np.sum(np.square(s_array))

        sigma2_0_MLE = np.average(np.square(x_array_truncate))
        
        sigma2_1_MLE = np.average(np.square(x_array_truncate - (A_MLE * s_array)))

        GLRT_out.append( (N/2.0) * (np.log(sigma2_0_MLE) -  np.log(sigma2_1_MLE)) )
    
    return np.array(GLRT_out)


def draw_signal_analysis(ax, template_URL:str, signal_URL:str):

    # Read wav files
    template_samplerate, template_WAV = scipy.io.wavfile.read(template_URL)
    signal_samplerate, signal_WAV = scipy.io.wavfile.read(signal_URL)

    # Get Likelihood ratio
    signal_lnGLRT = ln_GLRT(template_WAV, signal_WAV)

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
    draw_signal_analysis(ax, './audio_files/call.wav', './audio_files/heavy_noise.wav')
    plt.show()