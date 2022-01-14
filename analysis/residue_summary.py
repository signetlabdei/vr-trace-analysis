import numpy as np
import matplotlib.pyplot as plt
import itertools as it

### Plotting parameters ###
index = 4
videos = ['mc', 'ge_cities', 'ge_tour', 'vp']
rates = np.arange(10, 51, 10)
fpss = [30, 60]

### Import results ###
data = np.load('fullresults.npz')
    
const = data['arr_0']
slope = data['arr_1']
resdist = data['arr_2']
singledist = data['arr_3']
videodist = data['arr_4']
ratedist = data['arr_5']
framedist = data['arr_6']
nodist = data['arr_12']

### Plotting loop ###
labels = list(it.product(rates, fpss))
for v in range(len(videos)):
    x = np.arange(len(labels))
    fig, ax = plt.subplots()
    width = 0.15  # the width of the bars
    
    resbar = ax.bar(x - 5/2 * width, resdist[v, :, :, index].flatten(), width, label='Trace model')
    singlebar = ax.bar(x - 3/2 * width, singledist[v, :, :, index].flatten(), width, label='General model')
    videobar = ax.bar(x - 1/2 * width, videodist[v, :, :, index].flatten(), width, label='Video model')
    ratebar = ax.bar(x + 1/2 * width, ratedist[v, :, :, index].flatten(), width, label='Rate model')
    framebar = ax.bar(x + 3/2 * width, framedist[v, :, :, index].flatten(), width, label='Rate model')
    nobar = ax.bar(x + 5/2 * width, nodist[v, :, :, index].flatten(), width, label='No model')
    ax.set_ylabel('Error (kB)')
    ax.set_title(videos[v] + ', percentile residue')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    fig.tight_layout()

plt.show()
