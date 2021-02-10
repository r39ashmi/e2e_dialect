"""
@author: Rashmi Kethireddy
"""
import numpy as np
def trimf(x,params):
    f1=params[0];
    f2=params[1];
    f3=params[2];
    fb=np.zeros(np.size(x));
    
    if f1 != f2:
        index = np.where((f1 < x) & (x < f2),True,False)
        fb[index] = (x[index]-f1)*(1/(f2-f1));
    if f2 != f3:
        index = np.where((f2 < x) & (x < f3),True,False);
        fb[index] = (f3-x[index])*(1/(f3-f2));

    fb[x == f2] = 1;
    return fb

def mel_filter_bank(fs: int,
                    window_width: int,
                    n_filt: int = 40) -> (np.ndarray, np.ndarray):
    freq=(fs/2)*np.linspace(0,1,int(window_width/2)+1);
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (fs / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_filt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
    filterbank=np.zeros((n_filt,int(window_width/2)+1));
    for i in range(0,n_filt):
       params=[hz_points[i],hz_points[i+1],hz_points[i+2]];
       filterbank[i,:]=trimf(freq,params);
    return hz_points[1:n_filt + 1], filterbank

