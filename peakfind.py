import os
import numpy as np
import scipy.signal as sps
from scipy.io import wavfile
import matplotlib.pyplot as plt
from tqdm import tqdm
from constants import sr, people, sentences, counts

people=['aaditya']

data=dict()
for p in people:
	base = '/'.join(['Recordings', p])
	pd = data.setdefault(p, dict())
	for f in os.listdir(base):
		if f.endswith('.wav') and not f.startswith('template'):
			pd.setdefault(f.split('-')[0], dict()).setdefault('raw_data', []).append(wavfile.read('/'.join([base, f]))[1])

# {'aaditya': {'1': {'raw_data': [0, 0, waveform], 'peaks': [locs]}, '2': ..}}

for p in data:
	for si in data[p]:
		for wav in data[p][si]['raw_data']:
			peaks,_ = sps.find_peaks(wav, distance=int(0.07*sr), height=10)
			data[p][si].setdefault('peaks', []).append(peaks)
			print(p, si, len(peaks), counts[si])

to_plot = [('aaditya','1'), ('aaditya','2')]#, ('aaditya-old','3'), ('william','1')]
fig,axs = plt.subplots(len(to_plot),1)
for i,pair in enumerate(to_plot):
	axs[i].plot(data[pair[0]][pair[1]]['raw_data'][0], linewidth=0.2)
	peaks = data[pair[0]][pair[1]]['peaks'][0]
	last_peak_height = data[pair[0]][pair[1]]['raw_data'][0][peaks[-1]]
	if pair[0] != 'william':
		print('hi')
		axs[i].set_ylim(-last_peak_height*3, last_peak_height*3)
		axs[i].plot(peaks, [last_peak_height*2.5]*len(peaks), 'v')
	else:
		axs[i].plot(peaks, [max(data[pair[0]][pair[1]]['raw_data'][0])*1.5]*len(peaks), 'v')
	# else:
	# 	axs[i].set_ylim(-max(data[pair[0]][pair[1]]['raw_data'][0]), max(data[pair[0]][pair[1]]['raw_data'][0]))
	
# axs[1].plot(data['aaditya']['2']['raw_data'][0], linewidth=0.2)
# axs[0].plot(data['aaditya']['2']['peaks'][0], max(data['aaditya']['2']['raw_data'][0])*1.5, 'o')
# axs[2].plot(data['william']['1']['raw_data'][0], linewidth=0.2)
# axs[0].plot(data['william']['1']['peaks'][0], max(data['william']['1']['raw_data'][0])*1.5, 'o')
plt.show()
