import os
import numpy as np
import scipy.signal as sps
from scipy.io import wavfile
import pandas
import matplotlib.pyplot as plt
from tqdm import tqdm
from constants import sr#, people, sentences, counts

keyboards=['aaditya']

data=dict()
for p in keyboards:
	base = '/'.join(['Recordings', p])
	pd = data.setdefault(p, [])
	for f in os.listdir(base):
		if f.endswith('0.wav') and not f.startswith('template'):
			df_times = pandas.read_csv('/'.join([base, f[:-4]+'.csv']))
			raw_times = df_times[df_times.columns[1]].values/1000
			raw_char_lifts = df_times[df_times.columns[2]].values
			indices = []
			chars = []
			wav = wavfile.read('/'.join([base, f]))[1]
			# peaks,_ = sps.find_peaks(np.abs(wav), distance=2*sr, threshold=50)
			# print(peaks/sr)
			shift=None
			for i, num in enumerate(wav):
				if abs(num) > 300:
					shift = i
					break
			print(shift/sr)
			raw_times = raw_times + shift/sr
			for i in range(len(raw_times)):
				if i == 0:
					start = int(0.02*sr)
				else:
					start = int((raw_times[i-1]+0.02)*sr)
				end = int((raw_times[i]-0.02)*sr)
				if end <= start:
					continue
				chars.append(raw_char_lifts[i])
				indices.append(np.argmax(wav[start:end])+start)
			print(len(chars),len(indices), len(raw_times))
			plt.plot(np.arange(len(wav))/sr,wav, linewidth=0.2)
			# for i,idx in enumerate(indices):
			# 	plt.plot(idx/sr, wav[idx]+10, 'v',c='k')
			# 	char = '_' if chars[i]==' ' else chars[i]
			# 	plt.text(idx/sr, wav[idx]+20, char, fontsize=12, horizontalalignment='center')
			for i, rawt in enumerate(raw_times):
				plt.axvline(rawt, c='k', linewidth=0.2)
				char = '_' if raw_char_lifts[i]==' ' else raw_char_lifts[i]
				plt.text(rawt, wav[int(rawt*sr)]+20, char, fontsize=12, c='r', horizontalalignment='center')
			plt.show()

# {'aaditya': {'1': {'raw_data': [[trial1], [trial2]], 'peaks': [locs]}, '2': ..}}