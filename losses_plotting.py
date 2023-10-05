import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser('Losses Plotting', 'Used to plot many losses files together.')
    parser.add_argument('path')
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('save_file')

    return parser.parse_args()

def get_moving_average(x, beta):
    y = np.zeros(x.shape)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = y[i-1] * (1-beta) + beta * x[i]
    return y


def main():
    args = get_args()

    files = [os.path.join(args.path, f) for f in os.listdir(args.path) if f.endswith('.csv')]
    for file in files:
        name = os.path.basename(file).removesuffix('.csv')
        data = pd.read_csv(file)['loss'].values
        data = get_moving_average(data, args.beta)
        plt.plot(data, label=name)
    plt.legend()
    plt.savefig(args.save_file)


if __name__ == '__main__':
    main()