import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from matplotlib import cm

parser = argparse.ArgumentParser()
parser.add_argument("file", type=str)
parser.add_argument("--fields", nargs='+', default=['loss'])
parser.add_argument("--labels", nargs='+', default=None)
parser.add_argument("-e", "--end", default=1, type=int)
parser.add_argument("-b", "--beta", default=1, type=float)
parser.add_argument("-s", "--step", default=1, type=int)
parser.add_argument("--start", default=0, type=int)
parser.add_argument("--title", default='Plot')
args = parser.parse_args()

n = args.end - args.start
colors = cm.get_cmap('hsv', n + 1)([i / n for i in range(n)])
linestyles = ['-', '--', '-.', ':', '']

def smooth(x):
    y = np.zeros(x.shape)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = y[i-1] * (1-args.beta) + x[i] * args.beta
    return y

data = pd.read_csv(args.file)
for i in range(args.start, args.end, args.step):
    color = colors[i - args.start]
    for j, field in enumerate(args.fields):
        label = None if args.labels == None or j != 0 else "-".join([str(data[label.format(i)][0]) for label in args.labels])
        y = smooth(data[field.format(i)])
        plt.plot(y, label=label, color=color, linestyle=linestyles[j])
if args.labels != None:
    plt.legend()

plt.title(args.title)
plt.show()

