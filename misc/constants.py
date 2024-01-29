# Plotting
PLOT_CONFIG_PATH = '.plotconfig.json'
LABEL_CONFIG = 'lable_config'

# VAE Stages
PRE = 'pre'
EMB = 'emb'
REC = 'rec'
VAE_STAGES = [PRE, EMB, REC]

# VAE Config
ARGS_PATH = 'args.json'
LOSSES_PATH = 'losses.csv'

# VAE Loss Names
LOSS = 'loss'
R_LOSS = 'reconstruction_loss'
D_LOSS = 'divergence_loss'
T_LOSS = 'test_loss'

# Prep
TSNE = 'tsne'
PCA = 'pca'
NONE = 'none'

PREPS = [TSNE, PCA, NONE]

# Flux Dataset Management
GEM_FOLDER = 'gems'
RENAME_DICT_FILE = ".renaming.json"
JOIN_FILE = ".join.json"
PKL_FOLDER = ".pkl"
MODELS_PKL_FILE = "models.pkl"
DEFAULT_DATASET_SIZE = 65536
DEFAULT_TEST_SIZE = 2048
PKL_FOLDER = '.pkl'
TRAIN_FOLDER = 'train'

LABEL = 'label'
FILE_N = 'file_n'
SAMPLE_N = 'sample_n'
FILE_NAME = 'file_name'
SOURCE_COLUMNS = [LABEL, FILE_N, SAMPLE_N, FILE_NAME]

# Dataset Joins
INNER = 'inner'
OUTER = 'outer'
DATASET_JOINS = [INNER, OUTER]

# Clustering
KMEANS_C = "kmeans_cluster"
ORIGIONAL = "origional"
CLUSTERING = "clustering"