# HIT - Bone Specialist

This repository is an extension of the original [HIT](https://github.com/MarilynKeller/HIT) codebase, designed to specialize in predicting the indivudal 3D bone locations within the human body. 

# Installation 

1. Follow HIT's installation instructions in the [HIT repository](https://github.com/MarilynKeller/HIT). Including the installation of their checkpoints and body models.


2. Download the datasets from from [On predicting 3D bone locations inside the human body](https://3dbones.is.tue.mpg.de/). 

3. Install our checkpoint from [PabsDa/HIT-Bone-Specialist](https://huggingface.co/PabsDa/HIT-Bone-Specialist/tree/main)


# Structure

```
HIT/
├── hit/                          # Core Python package
│   ├── configs/                  # Hydra configs (training, network architecture)
│   ├── model/                    # Model implementations
│   │   ├── hit_model.py          
│   │   ├── network.py            
│   │   ├── deformer.py           
│   │   ├── mysmpl.py             
│   │   ├── generator.py          
│   │   └── broyden.py            
│   ├── training/                 # Training loop, losses, metrics, dataloaders
│   ├── utils/                    # Rendering, slicing, SMPL utilities, figures
│   ├── smpl/smplx/               # SMPL-X body model code (LBS, vertex ops)
│   ├── external/leap/            # LEAP body model integration
│   └── assets/                   # Vertex-to-part mapping (v2p.pkl)
│
├── demos/                        # Inference, data loading & evaluation
│   ├── infer_smpl.py             #   Infer tissues from SMPL parameters
│   ├── load_data.py              #   Data loading example
│   └── eval.py                   #   Our evaluation script 
│
├── body_models/                  # SMPL body models (male/female/neutral .pkl)
├── mri_bones_release_v2/         # Point-cloud dataset (train/validation/test splits)
├── hit_dataset_v1.0/             # Original HIT dataset
├── pretrained/                   # Trained model checkpoints
│   ├── hit_male_ckpt/            # original HIT checkpoint male
│   ├── hit_female_ckpt/          # original HIT checkpoint female
│   ├── male_specialist_ckpt/     # our specialist male
│   ├── female_specialist_ckpt/   # our specialist female
│   ├── pretrained_male_smpl.ckpt 
│   └── pretrained_female_smpl.ckpt
│
├── output/                       # Generated results & evaluations
│   ├── female_CR260152/          # Demo results for female example
│   │   ├── AT_mesh.obj  
│   │   ├── ..
│   │   └── Tibia_Fibula_mesh.obj
│   └── male_GKF_TSneu/           # Demo results for male example
│       ├── AT_mesh.obj
│       ├── ..
│       └── Tibia_Fibula_mesh.obj
│
├── requirements.txt
├── setup.py
├── extract_v1_smpl_lookup.py     #  Script to create the lookup table for v2p mapping
├── hit-env.yml                   #  Our conda enviorment file
└── LICENSE.txt
```




# Train:
For training the original HIT model, please refer to the original [HIT](https://github.com/MarilynKeller/HIT) repository. 

To train our bone specialist model, you can use the following command:
```shell
PYTHONPATH=. python hit/train.py exp_name=<NAME> smpl_cfg.gender=<GENDER> train_cfg.to_train=occ wdboff=<True/False> overfit_style=posed
```


# Evaluation & Inference:

## To extract mesh from a target SMPL:

### To infer from a SMPL file:

```shell
PYTHONPATH=. python demos/infer_smpl.py --exp_name=<EXP_NAME> --to_infer smpl_file --target_body <PATH_TO_SMPL>.pkl
```

### For a generic demo using the SMPL template and varying the shape parameters (betas):

```shell
PYTHONPATH=. python demos/infer_smpl.py --exp_name=<EXP_NAME> --to_infer smpl_template --betas <b1> <b2>
```

## Evaluation: 

For training the original HIT model, please refer to the original [HIT](https://github.com/MarilynKeller/HIT) repository. 

For evaluating our bone specialist model, you can use the following commands:

### Specialist only evaluation: 
```shell
PYTHONPATH=. python demos/eval.py --specialist_exp <SPECIALIST>  --gender <GENDER> --eval_classification
```

### HIT + Specialist evaluation: 
```shell
PYTHONPATH=. python demos/eval.py --specialist_exp <SPECIALIST>  --gender <GENDER> --eval_full_pipeline
```

### Point to mesh distances: 
```shell
PYTHONPATH=. python demos/eval.py --specialist_exp <SPECIALIST> --gender <GENDER>
```

# Acknowledgments
This code is built on top of the original HIT codebase, which can be found at [MarilynKeller/HIT](https://github.com/MarilynKeller/HIT). We thank the original authors for their work and for making their code publicly available.


