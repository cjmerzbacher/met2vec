# Met2Vec
This library implements a variational autoencoder (VAE), specifically altered to interface and load flux vector sampled from the null space of GSMs. The COBRApy library is used to interface with these GSMs.

## Dependencies
This project was origionally created using the python `venv` utility, for setup information see [this page](https://docs.python.org/3/library/venv.html). Once `venv` is set up run:
```
python -m pip install requirements.txt
```

## Organization of the code
The main files implementing the VAE and dataset can be found in the root folder.

- `fluxDataset.py` contains the implementation of `FluxDataset` which manages loading flux samples generated tih `fluxSampler.py`. It cannot be run independently.
- `fluxFile` contains the implementation of `FluxFile` class which acts as a wrapper around the flux samples files and manager renaming columns etc.
- `fluxModel/py` contains the implementation of `FluxModel` class which acts as a wrapper around a GSM model (which are assumed to be stored in some `./gems/` folder.)
- `fluxSampler.py` is a python script which acts as a wrapper around the COBRApy implementation of `OptGpSampler` and `ACHR`.
- `trainer.py` is a python script that is used to train VAEs. 
- `vae.py` constraints the implementation of the `FluxVAE` class implementing the main VAE functionality along side some extra functionality specific to loading samples from different GSM sources consistently.
- `vaeTrainer.py` contains the implementation of the `VAETrainer` class which `trainer.py` is running.
- `./tools/` contains some additional python scripts used to experiment with the trained VAEs.
    - `ari_score.py` implements a script which automatically runs K-Means then compares its results some original clustering.
    - `extract_fluxes.py` implements a script used to extract specific fluxes from a flux sample file.
    - `flux_processor.py` implements a script allowing a specific flux dataset to be loaded, passed to different stages of some trained VAE and then additionally applies some transformation e.g. t-SNE or PCA.
    - `gmm_classifier.py` allows data to be loaded then fits a GMM model to predict cell type, the classifier is then applied to some test data.
    - `kmeans_cluster.py` allows fluxes to be loaded and possibly passed to some VAE stage, it then applies K-means clustering saving the result.
    - `loss_scoring.py` implements a script allowing multiple losses.csv (in the format output by `trainer.py`) to be combined and averaged.
    - `nearest_centroid.py` allows fluxes to be loaded to some VAE stage and then trains a nearest_centroid classifier to predict some original label, the classifier is then applied to some test data.
    - `prepare_test_train.py` implements a script which automatically splits a collection of samples from `fluxSampler.py` into a test set and many train sets.
- `./misc/` contains common functionality used in `./tools/` and `./` (root). It is kept here to make code organization easier.
    - `ari.py` contains functionality related to `ari_score.py`
    - `classifier.py` contains functionality related to both classification scripts.
    - `constants.py` contains constants used throughout the codebase.
    - `fluxDataset.py` contains functionality related to then `FluxDataset` class.
    - `io.py` contains some general I/O functionality used in different scripts.
    - `kmeans.py` contains some functionality related to `kmeans_cluster.py`
    - `parsing.py` contains many different `argparse` parsers used in many different scripts.
    - `vae.py` contains functionality related to loading a `FluxVAE` and interacting with it.


## Setting up a dataset
The scripts implemented here should work with any GSMs which can be loaded by COBRApy. To set upt a new dataset called `new_dataset` create a folder `new_dataset` with a subfolder `gems`. The `gems` subfolder should contain any GSM files. Passing `new_dataset` folder to  `fluxSampler.py` will popular the main folder with flux samples. A `FluxDataset` can not be set up loading form this folder. It will crease `.pkl` files and some other `.json` files to cache some information in order to speed up interaction.

- **Recommended:** The `./data/` folder was used during the project to store all data not to be saved to the repo and is already included in the `.gitignore` 

## Training a VAE
Once a dataset has been setup `trainer.py` can be run. Using the `-h` tag will reveal all arguments.
```
python trainer.py -h
```
The only required arguments are `-d` (dataset folder) and a final `main_folder` which the VAE and a `losses.csv` file will be saved to. As an example:
```
python trainer.py -d ./data/samples/new_dataset/ -e 32 --lr 0.0001 ./data/models/test/
```
will train a new VAE for 32 epochs on a dataset stored at `./data/samples/new_dataset/` using a learning rate of `0.0001`. The VAE will be saved along with its losses and some files logging which parameter were used to `./data/models/test`.

