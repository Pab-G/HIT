import os

package_directory = os.path.dirname(os.path.abspath(__file__))

################## To edit
# Dataset
packaged_data_folder = "/home/yulong/pvbg-thesis/HIT/body_models"

# Trained HIT models
trained_models_folder = os.path.join(
    package_directory, "../pretrained"
)  # folder to save the trained models
pretrained_male_smpl = os.path.join(
    package_directory, "../pretrained/pretrained_male_smpl.ckpt"
)
pretrained_female_smpl = os.path.join(
    package_directory, "../pretrained/pretrained_female_smpl.ckpt"
)
# pretrained_female_mri = os.path.join(
#    package_directory,
#    "../pretrained/hit_female/ckpts/model-epoch=1479-val_accuracy=0.708701.ckpt",
# )

smplx_models_path = "/home/yulong/pvbg-thesis/HIT/body_models"  # folder containing the smplx models, to download from https://smpl-x.is.tue.mpg.de/downloads

# Training logging
wandb_entity = (
    "pablo-vonbaum-technical-university-of-munich"  # wandb account or team to log to
)
wandb_project_name = "hit"  # wandb project to log to
##################


# assets
v2p = os.path.join(
    package_directory, "assets/v2p.pkl"
)  # file that for each smpl vertex, gives the corresponding part

n_chunks_test = 10  # number of chunks to split the test set into. Increase to reduce memory usage when caching the dataset
n_chunks_train = 20  # Same for the train set

# Logger for training
logger = "wandb"  # only wandb is supported
