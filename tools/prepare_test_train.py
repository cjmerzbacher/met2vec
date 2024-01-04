import sys
import os

parent_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(parent_dir)

import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("source_folder", help="The folder that will be split.")
parser.add_argument("n", type=int, nargs='+', help="Ns for train_test_split.")
parser.add_argument("target_folder", help="Folder that samples will be created in.")

args = parser.parse_args()

TRAIN_FOLDER = "train"
TEST_FOLDER = "test"

source_folder = args.source_folder
n = args.n
target_folder = args.target_folder

print(f"Coppying Files from {source_folder}")
files = sorted([file for file in os.listdir(source_folder) if file.endswith(".csv")])

if not os.path.exists(target_folder):
    sys.makedirs(target_folder)

if len(n) == 1:
    target_folders = [target_folder]
else:
    target_folders = [os.path.join(target_folder, f"{i}") for i in range(len(n))]

def cp_files(files, from_, to_):
    for file in files:
        src = os.path.join(from_, file)
        dst = os.path.join(to_, file)
        shutil.copyfile(src, dst)

for n, target_folder in zip(target_folders, n):
    train_files = files[:files.n]
    test_files = files[files.n:]

    print(f"{len(files)} files found.") 

    print("    -> {len(train_files)} train, {len(test_files)} test")

    train_folder = os.path.join(target_folder, TRAIN_FOLDER)
    test_folder = os.path.join(target_folder, TEST_FOLDER)

    print("Copying files.")

    cp_files(train_files, source_folder, train_folder)
    cp_files(test_folder, source_folder, test_folder)


