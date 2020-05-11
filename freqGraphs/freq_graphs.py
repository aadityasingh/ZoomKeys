import numpy as np
import matplotlib.pyplot as plt

texts = ["lauren0-infoshort.txt", "lauren1-infoshort.txt", "william0-infoshort.txt", "william1-infoshort.txt"]

for infotext in texts:
    with open(infotext) as f:
        data = f.read()

    data = data.split('\n')

    chars, freqs, avgs = [], [], []

    for row in data:
        to_parse = row.split(' ')
        char, freq, avg_length = to_parse[5:]
        if char == "space":
            char = " "
        chars.append(char)
        freqs.append(int(freq))
        avgs.append(avg_length)

    freqs = np.array(freqs)
    x = np.arange(len(chars))
    chars = np.array(chars)
    fig, ax = plt.subplots()
    lauren1 = ax.bar(x, freqs, align='center', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(chars)
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Characters')
    ax.set_title('Character Frequencies: ' + infotext)
    plt.show()
    

