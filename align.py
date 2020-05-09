import os
import numpy as np
import scipy.signal as sps
from scipy.io import wavfile
import pandas
import matplotlib.pyplot as plt
from tqdm import tqdm
from constants import sr#, people, sentences, counts

keyboards=['lauren']


# spec for dataset:
# dictionary mapping characters to lists of raw audios (which will be lists of integers)
# Filter out first N characters (maybe 20ish) from all audio clips
# Filter out any windows that do not contain any samples with magnitude greater than M=10
# Filter out characters before and after shifts (and shifts)
# if time between characters is less than some threshold, cut both of them and the ones before and after the range of fast ones
# all lowercase
# make all of these configurable so we can just call a "generate dataset" method or something

def generate_dataset(audioFile, raw_char_lifts, indices):
	data = {}
	for char in "abcdefghijklmnopqrstuvwxyz":
		data[char] = []
	for specialChar in ["capslock", " ", ".", "comma", "backspace", '\'', 'enter']:
		data[specialChar] = []

	# total = 0

	# take out the first 20-ish chars
	for i in range(20, len(indices) - 1):
		start = indices[i] 
		end = indices[i + 1] 

		# shortest time range I saw was 1400? so made this 1500 but adjustable
		if end - start >= 1500:
			snip = np.array(wav[start : end])
			
			# uncomment to see which snippets have low max vals
			# if np.amax(np.absolute(snip)) < 15:
			# 	print(snip)
			# 	print(np.amax(snip), raw_char_lifts[i], i)

			# max val not 15 - this only cuts out like 30-50 snippets 
			if np.amax(np.absolute(snip)) > 15:
				# total += 1
				a, b, c = raw_char_lifts[i - 1: i + 2]
				if 'Shift' not in set([a, b, c]) and b.lower() in data:
					data[b.lower()].append(snip)
	
	for s in data:
		avg = 0
		for snip in data[s]:
			avg += len(snip)
		if len(data[s]) != 0:
			print("char, count, avg length: ", s, len(data[s]), avg/len(data[s]))

	# print("total: ", total)
	return data


fig_count = 1
data=dict()
for p in keyboards:
	base = '/'.join(['Recordings', p])
	pd = data.setdefault(p, [])
	for f in os.listdir(base):
		if f.endswith('.wav') and not f.startswith('template'):
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
			plt.figure(1)
			plt.plot(np.arange(len(wav))/sr,wav, linewidth=0.2)
			# for i,idx in enumerate(indices):
			# 	plt.plot(idx/sr, wav[idx]+10, 'v',c='k')
			# 	char = '_' if chars[i]==' ' else chars[i]
			# 	plt.text(idx/sr, wav[idx]+20, char, fontsize=12, horizontalalignment='center')
			for i, rawt in enumerate(raw_times):
				plt.axvline(rawt, c='k', linewidth=0.2)
				char = '_' if raw_char_lifts[i]==' ' else raw_char_lifts[i]
				plt.text(rawt, wav[int(rawt*sr)]+20, char, fontsize=12, c='r', horizontalalignment='center')
			# plt.show()
			generate_dataset(wav, raw_char_lifts, indices)



# {'aaditya': {'1': {'raw_data': [[trial1], [trial2]], 'peaks': [locs]}, '2': ..}}