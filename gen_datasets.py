import os
import numpy as np
import scipy.signal as sps
from scipy.io import wavfile
import pandas
import matplotlib.pyplot as plt
from tqdm import tqdm
from constants import sr#, people, sentences, counts
import pickle as pkl

# spec for dataset:
# dictionary mapping characters to lists of raw audios (which will be lists of integers)
# Filter out first N characters (maybe 20ish) from all audio clips
# Filter out any windows that do not contain any samples with magnitude greater than M=15
# Filter out characters before and after shifts (and shifts)
# if time between characters is less than some threshold, cut both of them and the ones before and after the range of fast ones
# all lowercase in dictionary
# make all of these configurable so we can just call a "generate dataset" method or something

def generate_dataset(audioFile, raw_char_lifts, indices):
	data = {}
	for char in "abcdefghijklmnopqrstuvwxyz":
		data[char] = []
	for specialChar in ["capslock", " ", ".", "comma", "backspace", '\'', 'enter']:
		data[specialChar] = []

	# total = 0

	# take out the first 20-ish chars
	thrown_out_too_short = 0
	full_window_too_short = 0
	thrown_out_shift = 0
	thrown_out_no_peak = 0
	accepted = 0
	for i in range(20, len(indices) - 2):
		start = indices[i-1] 
		end = indices[i+1] 

		if 'Shift' in raw_char_lifts[i-1:i+2]:
			thrown_out_shift +=1 
			continue

		# Check for min char diff
		mindiff = 0.01*sr #10ms
		if indices[i] - indices[i-1] < mindiff or indices[i+1] - indices[i] < mindiff:
			thrown_out_too_short += 1
			continue

		if end - start < 0.06*sr:
			full_window_too_short += 1
			continue

		snip = np.array(wav[start:end])
		if np.max(np.absolute(snip)) < 15:
			thrown_out_no_peak += 1
			continue
		# shortest time range I saw was 1400? so made this 1500 but adjustable
		if raw_char_lifts[i] in data:
			accepted += 1
			data[raw_char_lifts[i].lower()].append(snip)

		# if end - start >= 1500:
		# 	snip = np.array(wav[start:end])
			
		# 	# uncomment to see which snippets have low max vals
		# 	# if np.amax(np.absolute(snip)) < 15:
		# 	# 	print(snip)
		# 	# 	print(np.amax(snip), raw_char_lifts[i], i)

		# 	# max val not 15 - this only cuts out like 30-50 snippets 
		# 	if np.max(np.absolute(snip)) > 15:
		# 		# total += 1
		# 		a, b, c = raw_char_lifts[i - 1: i + 2]
		# 		if 'Shift' not in set([a, b, c]) and b.lower() in data:
		# 			data[b.lower()].append(snip)
	
	for s in data:
		avg = 0
		for snip in data[s]:
			avg += len(snip)
		if len(data[s]) != 0:
			print("char, count, avg length: ", s, len(data[s]), avg/len(data[s]))

	print("Too short char spaces", thrown_out_too_short)
	print("Next to Shift", thrown_out_shift)
	print("Full window too short", full_window_too_short)
	print("No peak", thrown_out_no_peak)
	print("Gathered samples", accepted)

	# print("total: ", total)
	return data


fig_count = 1
keyboards=['william','lauren']
data=dict()
for p in keyboards:
	base = '/'.join(['Recordings', p])
	pd = data.setdefault(p, [])
	for f in os.listdir(base):
		if f.endswith('.wav') and not f.startswith('template'):
			if os.path.isfile(p+f[:-4]+'.pkl'):
				print("Skipping", p+f[:-4]+'.pkl')
				continue
			df_times = pandas.read_csv('/'.join([base, f[:-4]+'.csv']))
			raw_times = df_times[df_times.columns[1]].values/1000
			raw_char_lifts = df_times[df_times.columns[2]].values
			wav = wavfile.read('/'.join([base, f]))[1]
			# peaks,_ = sps.find_peaks(np.abs(wav), distance=2*sr, threshold=50)
			# print(peaks/sr)
			shift=None
			for i, num in enumerate(wav):
				if abs(num) > 500:
					shift = i
					break
			print(shift/sr)
			raw_times = raw_times + shift/sr
			indices = [int(rawt*sr) for rawt in raw_times]
			plt.figure(1)

			uptill = min(200,indices[-1])
			plt.plot(np.arange(len(wav))[:indices[uptill]]/sr, wav[:indices[uptill]], linewidth=0.2)
			# for i,idx in enumerate(indices):
			# 	plt.plot(idx/sr, wav[idx]+10, 'v',c='k')
			# 	char = '_' if chars[i]==' ' else chars[i]
			# 	plt.text(idx/sr, wav[idx]+20, char, fontsize=12, horizontalalignment='center')
			for i, rawt in enumerate(raw_times[:uptill]):
				plt.axvline(rawt, c='k', linewidth=0.2)
				char = '_' if raw_char_lifts[i]==' ' else raw_char_lifts[i]
				plt.text(rawt, wav[int(rawt*sr)]+20, char, fontsize=12, c='r', horizontalalignment='center')
			plt.show()
			data = generate_dataset(wav, raw_char_lifts, indices)
			with open(p+f[:-4]+'.pkl', 'wb') as f:
				pkl.dump(data, f)



# {'aaditya': {'1': {'raw_data': [[trial1], [trial2]], 'peaks': [locs]}, '2': ..}}