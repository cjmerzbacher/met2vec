import pandas as pd
import argparse
import os

from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser('Dataset Mixer', 'Used to mix many csv dataset files together')
    parser.add_argument('folder', help='The folder the datasets to be mixed will be read from.')
    parser.add_argument('-o', '--output', help='The folder the mixed files will be stored in.')
    parser.add_argument('-n', type=int, help='Number of dataset files to write (defulat = nr files in).')
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.folder, 'mixed')
    return args

def main():
    args = get_args()
    dataset_files = [os.path.join(args.folder, f) for f in os.listdir(args.folder) if f.endswith('.csv')]
    if args.n == None:
        args.n = len(dataset_files)

    columns = set()
    with tqdm(dataset_files, 'Reading Column Names') as t:
        for df in t:
            columns = columns.union(pd.read_csv(df, nrows=0).columns)
            t.set_postfix({'Found' : len(columns)})
    columns = list(columns)

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    for i in tqdm(range(len(dataset_files)), 'Reading datasets', position=0):
        dataset = pd.read_csv(dataset_files[i])
        dataset.drop(columns=dataset.columns[0], inplace=True)

        for mix in tqdm(range(args.n), 'Writing mix files', position=1):
            sample = dataset.sample(int(len(dataset) / (args.n - mix)))
            dataset.drop(sample.index, inplace=True)

            mix_df = pd.concat([pd.DataFrame(columns=columns), sample])
            mix_df.fillna(0, inplace=True)
            mix_path = os.path.join(args.output, f'mix_{mix}.csv')
            if i == 0:
                mix_df.to_csv(mix_path)
            else:
                mix_df.to_csv(mix_path, mode='a', header=False)


if __name__ == '__main__':
    main()