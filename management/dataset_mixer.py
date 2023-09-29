import pandas as pd
import argparse
import os

from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser('Dataset Mixer', 'Used to mix many csv dataset files together')
    parser.add_argument('folder', help='The folder the datasets to be mixed will be read from.')
    parser.add_argument('-o', '--output', help='The folder the mixed files will be stored in.')
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.folder, 'mixed')
    return args

def main():
    args = get_args()
    dataset_files = [os.path.join(args.folder, f) for f in os.listdir(args.folder) if f.endswith('.csv')]
    dataset_taken_rows = [set() for _ in range(len(dataset_files))]

    columns = set()
    with tqdm(dataset_files, 'Reading Column Names') as t:
        for df in t:
            columns = columns.union(pd.read_csv(df, nrows=0).columns)
            t.set_postfix({'Found' : len(columns)})
    columns = list(columns)

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    for mix in tqdm(range(len(dataset_files)), 'Creating mixed datasets', position=0):
        mix_df = pd.DataFrame(columns=columns)
        for i in tqdm(range(len(dataset_files)), 'Reading datasets', position=1):
            dataset = pd.read_csv(dataset_files[i])

            n_rows = len(dataset)
            dataset.drop(dataset_taken_rows[i], axis=0, inplace=True)
            if i != len(dataset_files) - 1:
                dataset = dataset.sample(int(n_rows / len(dataset_files)))

            dataset_taken_rows[i] = dataset_taken_rows[i].union(dataset[dataset.columns[0]].values)
            dataset.drop(columns=dataset.columns[0], inplace=True)
            mix_df = pd.concat([mix_df, dataset])

        mix_df.to_csv(os.path.join(args.output, f'mix_{mix}.csv'))
            

    






if __name__ == '__main__':
    main()