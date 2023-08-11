import argparse
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("file", type=str)
args = parser.parse_args()

losses = pd.read_csv(args.file)
plt.plot(losses)
plt.show()

