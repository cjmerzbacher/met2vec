#!/bin/bash

# This is a script to set up and run the reprisentative_sample_testing.
#
# $1 - A folder containing many samples from the same GEM
# $2 - A folder where many samples will have their split forms.
# $3 - The split numbers that will be used "a b c ... z"
# $4 - The tuner folder that will be used.

read -a splits <<< "$3"


echo "Samples folder    : $1"
echo "Split folder      : $2"
echo "Splits            : $3 (${#splits[@]})"
echo "Tuner folder      : $4"
echo "Epochs (testing)  : $5"
echo ""
echo "Current commit:"
git rev-parse HEAD

sleep .5

echo ""
echo "1) Preparing $2 with splits from $1..."
echo ""
rm -rf "$2*"
python ./tools/prepare_test_train.py $1 $3 $2

echo ""
echo "2) Preparing tuner at $4 ..."

python tuner.py $4 setup
python tuner.py $4 add dataset --dataset $3 --root_folder $2 --type str

echo ""
echo "Tuner Status:"
python tuner.py $4 status

echo ""
echo "3) Running tuner..."
echo ""

python tuner.py $4 run "python trainer.py -e $5 -b 128 -s $5 --n_emb 48 --n_lay 4 --lr 0.0004 --lrelu_slope 0.1 --model_folder $1 -n 65536 --join inner --test_size 4096"

echo ""
echo "Done."

