import os
import numpy as np
import scipy.signal as sps
from scipy.io import wavfile
import matplotlib.pyplot as plt
from tqdm import tqdm
from constants import sr, people, sentences, counts

def max_finder(signal, numpeaks, poss_set):#threshold=5, mindist=int(0.025*sr)):
	distances = np.array([len(signal)]*len(signal))
	# peaks,_ = sps.find_peaks(sig, distance = mindist)
	for i in tqdm(range(len(signal))):
		# if abs(signal[i]) < threshold:
		# 	distances[i] = -1
		# 	continue
		if i not in poss_set:
			distances[i]=-1
		for j in range(1, max(len(signal)-i, i)):
			if j < 0:
				print('BAD')
			if i-j >= 0:
				if abs(signal[i-j]) >= abs(signal[i]):
					distances[i] = j
					break
			if i+j < len(signal):
				if abs(signal[i+j]) > abs(signal[i]):
					distances[i] = j
					break
	# print(sorted(distances)[-10:])
	# print(distances[5650:5700])
	# print(distances[5691])
	# sig = np.array(signal)
	# sig[distances<mindist]=0
	# return np.argsort(np.abs(sig))[-numpeaks:]
	return np.argsort(distances)[-numpeaks:]#, distances


people=['aaditya']

data=dict()
for p in people:
	base = '/'.join(['Recordings', p])
	pd = data.setdefault(p, dict())
	for f in os.listdir(base):
		if f.endswith('.wav') and not f.startswith('template'):
			pd.setdefault(f.split('-')[0], dict()).setdefault('raw_data', []).append(wavfile.read('/'.join([base, f]))[1])

# {'aaditya': {'1': {'raw_data': [[trial1], [trial2]], 'peaks': [locs]}, '2': ..}}

for p in data:
	for si in data[p]:
		for wav in data[p][si]['raw_data']:
			peaks,_ = sps.find_peaks(np.abs(wav), distance = int(0.03*sr), prominence=20)
			peaks=np.intersect1d(peaks, np.where(np.abs(wav)>0)[0])
			print(peaks)
			# peaks = max_finder(wav, counts[si], )
			data[p][si].setdefault('peaks', []).append(peaks)
			print(p, si, len(peaks), counts[si])

to_plot = [('aaditya','1'), ('aaditya','2')]#, ('aaditya-old','3'), ('william','1')]
fig,axs = plt.subplots(len(to_plot),1)
if len(to_plot) == 1:
	axs = [axs]
for i,pair in enumerate(to_plot):
	signal = data[pair[0]][pair[1]]['raw_data'][0]
	axs[i].plot(signal, linewidth=0.2)
	peaks = data[pair[0]][pair[1]]['peaks'][0]
	last_peak_height = data[pair[0]][pair[1]]['raw_data'][0][peaks[-1]]
	# if pair[0] != 'william':
	# 	print('hi')
		# axs[i].set_ylim(-last_peak_height*3, last_peak_height*3)
	axs[i].plot(peaks, abs(np.array(data[pair[0]][pair[1]]['raw_data'][0])[peaks])*1.2, 'v')
	# for pk in peaks:
	# 	axs[i].plot([max(0,pk-distances[pk]), min(len(signal),pk+distances[pk])], [abs(signal[pk])*1.2]*2, c='r')
	# else:
	# 	axs[i].plot(peaks, [max(data[pair[0]][pair[1]]['raw_data'][0])*1.5]*len(peaks), 'v')
	# else:
	# 	axs[i].set_ylim(-max(data[pair[0]][pair[1]]['raw_data'][0]), max(data[pair[0]][pair[1]]['raw_data'][0]))
	
# axs[1].plot(data['aaditya']['2']['raw_data'][0], linewidth=0.2)
# axs[0].plot(data['aaditya']['2']['peaks'][0], max(data['aaditya']['2']['raw_data'][0])*1.5, 'o')
# axs[2].plot(data['william']['1']['raw_data'][0], linewidth=0.2)
# axs[0].plot(data['william']['1']['peaks'][0], max(data['william']['1']['raw_data'][0])*1.5, 'o')
plt.show()
