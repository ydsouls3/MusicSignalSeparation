import sys
import function
import numpy as np
from matplotlib import pyplot as plt

path = sys.argv[1]                  
data, samplerate = function.wavload(path)   
if data.ndim == 2:
    data = 0.5*data[:, 0] + 0.5*data[:, 1]
else:
    data = data
x = np.arange(0, len(data)) / samplerate    

#FFT
Fs = 4096                                  
overlap = 75                               

time_array, N_ave, final_time = function.ov(data, samplerate, Fs, overlap)

time_array, acf = function.hanning(time_array, Fs, N_ave)

fft_array, fft_mean, fft_axis = function.fft_ave(time_array, samplerate, Fs, N_ave, acf)

fft_array = fft_array.T

# グラフ描画
fig = plt.figure()
ax1 = fig.add_subplot(111)

im = ax1.imshow(fft_array, \
                vmin = -10, vmax = 60,
                extent = [0, final_time, 0, samplerate], \
                aspect = 'auto',\
                cmap = 'jet')

cbar = fig.colorbar(im)
cbar.set_label('SPL [dBA]')

ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Frequency [Hz]')

ax1.set_xticks(np.arange(0, 120, 2))
ax1.set_yticks(np.arange(0, 20000, 1000))
ax1.set_xlim(0, 8)
ax1.set_ylim(0, 8000)

plt.show()
plt.close()