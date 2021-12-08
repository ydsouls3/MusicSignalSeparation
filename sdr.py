import sys
import random
import soundfile as sf
import numpy as np
import math
from scipy import signal as sg
from scipy import stats
import matplotlib.pyplot as plt

def sdr(refdft,estdft,indft):
   refadft = np.abs(refdft)
   estadft = np.abs(estdft)
   inadft = np.abs(indft)
   time = refdft.shape[1]
   freq = refdft.shape[0]
   sdr = 0
   for i in range(time):
       bunshi = 0
       bunbo = 0
       for j in range(freq):
           bunshi = bunshi + (estadft[j,i]-inadft[j,i])**2
           bunbo = bunbo + (refadft[j,i] - estadft[j,i])**2   
       if bunbo < 1e-10:
           bunbo = 1e-10    
       a = bunshi/bunbo
       if np.isnan(a):
           a = 1e-10 
       if a < 1e-10:
           a = 1e-10    
       sdr = sdr + 10*math.log10(a)
   return sdr/time



frame_length = 0.04    #STFT window width (second) [Default]0.04
frame_shift = 0.02     #STFT window shift (second) [Default]0.02

#reference data
data, Fs = sf.read(sys.argv[1])

if data.ndim == 2:
    refdata = 0.5*data[:, 0] + 0.5*data[:, 1]
else:
    refdata = data
"""
pxx, freq, bins, t = plt.specgram(refdata,Fs = Fs)
plt.xlabel("Time[s]", fontsize=12)
plt.ylabel("freqency[Hz]", fontsize=12)
plt.show()
"""

#estimate date
data, Fs = sf.read(sys.argv[2])

if data.ndim == 2:
    estdata = 0.5*data[:, 0] + 0.5*data[:, 1]
else:
    estdata = data    


FL = round(frame_length * Fs)
FS = round(frame_shift * Fs)
OL = FL - FS    

reffreqs, reftimes, refdft = sg.stft(refdata, fs=Fs, window='hamm', nperseg=FL, noverlap=OL)
estfreqs, esttimes, estdft = sg.stft(estdata, fs=Fs, window='hamm', nperseg=FL, noverlap=OL)

#others 
otherdft = np.zeros(refdft.shape)
for n in range(3, len(sys.argv)):
 data, Fs = sf.read(sys.argv[n])

 if data.ndim == 2:
    interdata = 0.5*data[:, 0] + 0.5*data[:, 1]
 else:
    interdata = data    

 infreqs, intimes, indft = sg.stft(interdata, fs=Fs, window='hamm', nperseg=FL, noverlap=OL)

 otherdft += indft

"""
print(refdft.shape)
print(estdft.shape)
"""

sdr = sdr(refdft,estdft,indft)
print(sdr)
