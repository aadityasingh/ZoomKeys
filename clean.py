import os
import numpy as np
import scipy.signal as sps
from scipy.io import wavfile
import matplotlib.pyplot as plt
from tqdm import tqdm

for p in ['william']:
	fig, axs = plt.subplots(2,1,sharex=True)
	base = '/'.join(['Recordings', p])
	f = '/'.join([base,'template.m4a'])
	newf = '/'.join([base,'template.wav'])
	if not os.path.isfile(newf):
		os.system('ffmpeg -i '+f+' '+newf)
	template = wavfile.read(newf)[1]
	for i, num in enumerate(template):
		if num > 10:
			template = template[i:]
			break

	for i, num in enumerate(reversed(template)):
		if num > 20:
			template = template[:len(template)-i-1]
			break
	axs[0].plot(template, linewidth=0.2)
	# axs[1].plot(np.abs(sps.hilbert(template)), linewidth=0.2)
	i=1
	for f in os.listdir(base):
		if f.endswith('.m4a'):
			if f.startswith('template'):
				continue
			tempf = 'temporary.wav'
			newf = f[:-4]+'.wav'
			if not os.path.isfile('/'.join([base,newf])):
				os.system('ffmpeg -i '+'/'.join([base,f]) + ' '+'/'.join([base,tempf]))
			else:
				continue
			sr, contents = wavfile.read('/'.join([base, tempf]))
			axs[i].plot(contents, linewidth=0.2)
			# vals = 
			# print(len(contents))
			# print(len(vals))
			# axs[i+1].plot(np.abs(sps.hilbert(contents)))
			cross_corr = sps.correlate(np.abs(sps.hilbert(contents[:len(contents)//2])), np.abs(sps.hilbert(template)))
			# Code for doing the template before and after
			# peaks, _ = sps.find_peaks(cross_corr,distance=len(cross_corr)//2) 
			# assert len(peaks) == 2
			# for peak in peaks:
			# 	axs[i].plot(range(peak-len(template),peak), template)
			# wavfile.write('/'.join([base,newf]), sr, contents[peaks[0]:peaks[1]-len(template)-int(0.05*sr)])
			# Code for doing the template before:
			start = int(np.argmax(cross_corr)+0.05*sr)
			axs[i].plot(range(start-len(template),start), template)
			wavfile.write('/'.join([base,newf]), sr, contents[start:])
			os.system('rm '+'/'.join([base,tempf]))
			i = i+1
plt.show()
# for p in data:
# 	for si in data[p]:
# 		for wav in data[p][si]['raw_data']:
# 			peaks,_ = find_peaks(wav, distance=int(0.06*sr), height=10)
# 			data[p][si].setdefault('peaks', []).append(peaks)
# 			print(p, si, len(peaks), counts[si])

# to_plot = [('aaditya','1'), ('aaditya','2'), ('aaditya','3')]#, ('william','1')]
# fig,axs = plt.subplots(len(to_plot),1)
# for i,pair in enumerate(to_plot):
# 	axs[i].plot(data[pair[0]][pair[1]]['raw_data'][0], linewidth=0.2)
# 	peaks = data[pair[0]][pair[1]]['peaks'][0]
# 	last_peak_height = data[pair[0]][pair[1]]['raw_data'][0][peaks[-1]]
# 	axs[i].set_ylim(-last_peak_height*3, last_peak_height*3)
# 	axs[i].plot(peaks, [last_peak_height*2.5]*len(peaks), 'v')
# axs[1].plot(data['aaditya']['2']['raw_data'][0], linewidth=0.2)
# axs[0].plot(data['aaditya']['2']['peaks'][0], max(data['aaditya']['2']['raw_data'][0])*1.5, 'o')
# axs[2].plot(data['william']['1']['raw_data'][0], linewidth=0.2)
# axs[0].plot(data['william']['1']['peaks'][0], max(data['william']['1']['raw_data'][0])*1.5, 'o')
# plt.show()
