import scipy.io.wavfile
import scipy.signal
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pathlib
import os


CURRENT_PATH = pathlib.Path(__file__).parent.absolute()

def ln_GLRT(s_array:np.ndarray, x_array:np.ndarray) -> np.ndarray:
    """ Compute the value of natural logarithm of the generalized likelihood ratio along the signal x_array using the template s_array.

    Args:
        s_array (np.ndarray): Template to detect
        x_array (np.ndarray): Signal to analyze

    Returns:
        np.ndarray: Natural logarithm of the generalized likelihood ratio test
    """
    assert s_array.ndim == x_array.ndim == 1
    N = s_array.shape[0]

    GLRT_out = []

    print('\n## Generalized Likelihood Ration Computation ##')

    for n_0 in tqdm(range(x_array.shape[0] - N)):
        x_array_truncate = x_array[n_0:n_0+N]

        A_MLE = np.sum(np.multiply(s_array,x_array_truncate)) / np.sum(np.square(s_array))

        sigma2_0_MLE = np.average(np.square(x_array_truncate))
        
        sigma2_1_MLE = np.average(np.square(x_array_truncate - (A_MLE * s_array)))

        GLRT_out.append( (N/2.0) * (np.log(sigma2_0_MLE) -  np.log(sigma2_1_MLE)) )
    
    return np.array(GLRT_out)


def create_video(template_URL:pathlib.Path, signal_URL:pathlib.Path, frame_rate:int, name:str):
    """ Generate a sequence of frames to visualize the evolution of the generalized likelihood ratio in real-time.
        The user can overlay the image sequence on the video from which the signal comes from.

    Args:
        template_URL (pathlib.Path): Path to the audio file (.wav) used as template for the computation of the generalized likelihood ratio
        signal_URL (pathlib.Path): Path to the audio file (.wav) used as signal for the computation of the generalized likelihood ratio
        frame_rate (int): Number of frame generated per second of audio file
        name (str): Name of the image sequence
    """

    saving_path = CURRENT_PATH / 'video' / 'python_export'

    assert saving_path.exists()
    assert template_URL.is_file()
    assert signal_URL.is_file()

    # Read wav files
    template_samplerate, template_WAV = scipy.io.wavfile.read(template_URL)
    signal_samplerate, signal_WAV = scipy.io.wavfile.read(signal_URL)
    assert template_samplerate == signal_samplerate

    #signal_WAV = signal_WAV[:signal_samplerate*5]

    signal_size = signal_WAV.shape[0]
    signal_min = signal_WAV.min()
    signal_max = signal_WAV.max()

    # Get Likelihood ratio
    signal_lnGLRT = ln_GLRT(template_WAV, signal_WAV)
    signal_lnGLRT_size = signal_lnGLRT.shape[0]
    signal_lnGLRT_max = signal_lnGLRT.max()

    # Draw plot
    signal_timeline = np.linspace(0, signal_size/signal_samplerate, signal_size)
    signal_lnGLRT_timeline = np.linspace(0, signal_lnGLRT_size/signal_samplerate, signal_lnGLRT_size)

    # Initialize graph
    fig, ax = plt.subplots(figsize=(10,4))
    ax_bis = ax.twinx()
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    # Create frames
    timestamps = np.arange(0, signal_lnGLRT_size, int(signal_samplerate/frame_rate))
    saving_path /= name
    if not saving_path.exists():
        os.mkdir(saving_path)

    print('\n## Frames generetion ##')
    for i in tqdm(range(timestamps.shape[0])):
        t = timestamps[i]
        ax.clear()
        ax_bis.clear()
        ax.axis('off')
        ax_bis.axis('off')
        ax_bis.set_xlim(0, signal_size)
        ax_bis.set_ylim(0, signal_lnGLRT_max)
        ax.set_xlim(0, signal_size)
        ax.set_ylim(signal_min, signal_max*2)
        
        ax.plot(signal_WAV[:t], color='grey', alpha=.7, linewidth=.5)
        ax_bis.plot(signal_lnGLRT[:t], color='#9500ff', linewidth=1.)

        fig.savefig(saving_path / (name + '_{}.png'.format(i)), transparent=True, dpi=192, pad_inches=0.)


if __name__ == "__main__":
    plt.style.use('ggplot')

    template_path = CURRENT_PATH / 'audio_files' / 'template' / 'call_2.wav'
    signal_path = CURRENT_PATH / 'audio_files' / 'signals' / 'nature_1.wav'

    create_video(template_path, signal_path, 30, 'nature1_call2')