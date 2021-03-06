[![Video real-time analysis](./.github/markdown/SwainsonCall.gif)](https://youtu.be/NUYM1yvVPls "Swainson's Thrush call")

[**Watch full video with audio here!**](https://youtu.be/NUYM1yvVPls)

**For a better visualization of this notebook, please use [NbViewer](https://nbviewer.jupyter.org/github/ArthurFDLR/SwainsonsThrush-detector/blob/main/SwainsonsTrush-detector.ipynb) or [Google Colab](https://colab.research.google.com/github/ArthurFDLR/SwainsonsThrush-detector/blob/main/SwainsonsTrush-detector.ipynb#scrollTo=s0ywcXzcjI7H).**

![GitHub](https://img.shields.io/github/license/ArthurFDLR/SwainsonsThrush-detector)
[![Linting](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


# Swainson's Thrush call detector


The **GLRT** (Generalized Likelihood Ratio Test) is a general procedure for composite testing problems. The basic idea
is to compare the likelihood of two hypotheses <img src="https://render.githubusercontent.com/render/math?math=H_0"> and <img src="https://render.githubusercontent.com/render/math?math=H_1"> for a given sequence of input values. This notebook propose an implementation of this procedure for the detection of Swainson's Thrush call

* <img src="https://render.githubusercontent.com/render/math?math=H_0">: The signal is only noise.
* <img src="https://render.githubusercontent.com/render/math?math=H_1">: The signal contains Swainson's Thrush pit call.



## Audio arrays import

The Generalized Likelihood Ratio is computed all along a recording <img src="https://render.githubusercontent.com/render/math?math=\{x(n)\}_{n = 0 \ \dots \ N-1}"> where the Swainson's Thrush pit call can be heard. Peaks in the Generalized Likelihood Ratio curve indicates the presence of a call that matches the template <img src="https://render.githubusercontent.com/render/math?math=\{s(n)\}_{n = 0 \ \dots \ N-1}">. The template signal is a simple filtered recording of the pit call. Audio samples are imported from [this](https://youtu.be/0LNtk5OVssQ) superb recording of a Swainson's Thrush in the wild.

### Audio file formats

- File format: WAV Audio (.wav)
- Sample rate: 22050 Hz
- Encoding: 64-bit float
- Recording channel: Mono

### Import


```python
import os
import pathlib
import IPython.display as ipd
import scipy.io.wavfile

repository_path = pathlib.Path(".")
template_path = repository_path / "audio_files" / "template" / "call_2.wav"
signal_path = repository_path / "audio_files" / "signals" / "nature_1.wav"

template_samplerate, template_WAV = scipy.io.wavfile.read(template_path)
signal_samplerate, signal_WAV = scipy.io.wavfile.read(signal_path)
assert template_samplerate == signal_samplerate
```


```python
ipd.Audio(template_WAV, rate=template_samplerate)
```


*Audio display not supported in Markdowns, please use [NbViewer](https://nbviewer.jupyter.org/github/ArthurFDLR/SwainsonsThrush-detector/blob/main/SwainsonsTrush-detector.ipynb) or [Google Colab](https://colab.research.google.com/github/ArthurFDLR/SwainsonsThrush-detector/blob/main/SwainsonsTrush-detector.ipynb#scrollTo=s0ywcXzcjI7H).*



```python
ipd.Audio(signal_WAV, rate=signal_samplerate)
```


*Audio display not supported in Markdowns, please use [NbViewer](https://nbviewer.jupyter.org/github/ArthurFDLR/SwainsonsThrush-detector/blob/main/SwainsonsTrush-detector.ipynb) or [Google Colab](https://colab.research.google.com/github/ArthurFDLR/SwainsonsThrush-detector/blob/main/SwainsonsTrush-detector.ipynb#scrollTo=s0ywcXzcjI7H).*



## Implementation of the GLRT

We will have a sequence of problems at each time <img src="https://render.githubusercontent.com/render/math?math=n_0"> with the
following hypotheses:

<img src="https://render.githubusercontent.com/render/math?math=\begin{array}{ll}%20H_0:%20x(n)%20=%20w(n)\\%20H_1:%20x(n)%20=%20w(n)%20+%20A%20\%20s(n%20-%20n_0)%20\end{array}%20\quad%20,%20\%20n=n_0,%20%20\dots,%20%20n_0%20+%20N%20-%201">



<img src="https://render.githubusercontent.com/render/math?math=w"> is supposed to be white Gaussian noise of unknown variance <img src="https://render.githubusercontent.com/render/math?math=\sigma^2">. Furthermore, <img src="https://render.githubusercontent.com/render/math?math=A \neq 0"> is an unknow scale factor.

Given our framework, the expression of the GLRT is

<img src="https://render.githubusercontent.com/render/math?math=L_G(x) = \frac{p(x,\widehat{A},\widehat{\sigma_1^2},H_1)}{p(x,\widehat{\sigma_0^2},H_0)} = \left(\frac{\widehat{\sigma_0^2}}{\widehat{\sigma_1^2}}\right)^{\frac{N}{2}}">


where <img src="https://render.githubusercontent.com/render/math?math=\widehat{A}">, <img src="https://render.githubusercontent.com/render/math?math=\widehat{\sigma_0^2}"> and <img src="https://render.githubusercontent.com/render/math?math=\widehat{\sigma_1^2}"> are the maximum likelihood estimates of <img src="https://render.githubusercontent.com/render/math?math=A">, <img src="https://render.githubusercontent.com/render/math?math=\sigma_0^2"> and <img src="https://render.githubusercontent.com/render/math?math=\sigma_1^2">:

<img src="https://render.githubusercontent.com/render/math?math=\begin{array}{ll}%20\widehat{A}%20=%20\frac{\sum^{n_0+N-1}_{n=n_0}%20x(n)%20s(n-n_0)}{\sum^{n_0+N-1}_{n=n_0}s^2(n-n_0)}%20\\%20\widehat{\sigma_0^2}%20=%20\frac{1}{N}%20\sum^{n_0+N-1}_{n=n_0}%20x^2(n)%20\\%20\widehat{\sigma_1^2}%20=%20\frac{1}{N}%20\sum^{n_0+N-1}_{n=n_0}%20(x(n)%20-%20\widehat{A}%20s(n-n_0))^2%20\end{array}">

We will compute the natural logarithm of the GLRT to avoid float overflows.


```python
import numpy as np
from tqdm import tqdm


def ln_GLRT(s_array: np.ndarray, x_array: np.ndarray) -> np.ndarray:
    """Compute the value of natural logarithm of the generalized likelihood ratio along the signal x_array using the template s_array.

    Args:
        s_array (np.ndarray): Template to detect
        x_array (np.ndarray): Signal to analyze

    Returns:
        np.ndarray: Natural logarithm of the generalized likelihood ratio test
    """
    assert s_array.ndim == x_array.ndim == 1
    N = s_array.shape[0]

    GLRT_out = []

    for n_0 in tqdm(range(x_array.shape[0] - N), desc="GLR Computation"):
        x_array_truncate = x_array[n_0 : n_0 + N]

        A_MLE = np.sum(np.multiply(s_array, x_array_truncate))
        A_MLE /= np.sum(np.square(s_array))

        sigma2_0_MLE = np.average(np.square(x_array_truncate))

        sigma2_1_MLE = np.average(np.square(x_array_truncate - (A_MLE * s_array)))

        GLRT_out.append((N / 2.0) * (np.log(sigma2_0_MLE) - np.log(sigma2_1_MLE)))

    return np.array(GLRT_out)


signal_lnGLRT = ln_GLRT(template_WAV, signal_WAV)
```

    GLR Computation: 100%|██████████| 977456/977456 [00:34<00:00, 28395.60it/s]
    

# Results

We can now detect Swainson's Thrush pit calls for values exceding a certain treshold. 


```python
treshold = 20.0

import matplotlib.pyplot as plt

plt.style.use("ggplot")
fig, ax = plt.subplots()

signal_timeline = np.linspace(
    0, signal_WAV.shape[0] / signal_samplerate, signal_WAV.shape[0]
)
signal_lnGLRT_timeline = np.linspace(
    0, signal_lnGLRT.shape[0] / signal_samplerate, signal_lnGLRT.shape[0]
)

ax_bis = ax.twinx()

ax.plot(signal_timeline, signal_WAV, color="grey", alpha=0.3, linewidth=0.2)
ax_bis.plot(signal_lnGLRT_timeline, signal_lnGLRT, color="#9500ff", linewidth=1.0)
ax_bis.hlines(
    treshold,
    0,
    signal_lnGLRT.shape[0] / signal_samplerate,
    colors="#9a65bf",
    linestyles="dashdot",
    label="Treshold",
)
ax_bis.legend()

ax.set_xlabel(r"$t$ $[s]$")
ax_bis.set_ylabel(r"$\ln(L_G(t))$", color="#9500ff")
ax_bis.set_yscale("log")
ax_bis.set_ylim([0.5, ax_bis.get_ylim()[1]])
ax.set_ylabel(r"Analyzed signal")

plt.show()
```


    
![plot](.github/markdown/SwainsonsTrush-detector_16_0.png)
    



```python
import pandas as pd
import scipy.signal

timestamps_detected, properties = scipy.signal.find_peaks(
    signal_lnGLRT, threshold=treshold, distance=template_WAV.shape[0]
)
timestamps_detected = timestamps_detected / signal_samplerate
pd.DataFrame(timestamps_detected, columns=["Timestamps"])

```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Timestamps</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.733333</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6.428571</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11.882222</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14.100952</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14.799546</td>
    </tr>
    <tr>
      <th>5</th>
      <td>17.386712</td>
    </tr>
    <tr>
      <th>6</th>
      <td>20.091701</td>
    </tr>
    <tr>
      <th>7</th>
      <td>22.646712</td>
    </tr>
    <tr>
      <th>8</th>
      <td>25.350340</td>
    </tr>
    <tr>
      <th>9</th>
      <td>28.432018</td>
    </tr>
    <tr>
      <th>10</th>
      <td>31.054921</td>
    </tr>
    <tr>
      <th>11</th>
      <td>33.758549</td>
    </tr>
    <tr>
      <th>12</th>
      <td>36.840499</td>
    </tr>
    <tr>
      <th>13</th>
      <td>39.933923</td>
    </tr>
  </tbody>
</table>
</div>


