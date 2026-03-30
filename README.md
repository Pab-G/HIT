# HIT

## Train

```shell
#python hit/train.py exp_name=hit_female smpl_cfg.gender=female  run_eval=True wdboff=True


PYTHONPATH=. python hit/train.py     exp_name=posed-no-noise     smpl_cfg.gender=
male     train_cfg.to_train=occ     wdboff=False overfit_style=posed

```

#####
Male:
PYTHONPATH=. python demos/infer_smpl.py --exp_name=SMALL --to_infer smpl_file --target_body /home/yulong/pvbg-thesis/HIT/mri_bones_release_v2/test/male/GKF_TSneu*/mri_smpl.pkl
Female:
PYTHONPATH=. python demos/infer_smpl.py --exp_name=SMALL --to_infer smpl_file --target_body /home/yulong/pvbg-thesis/HIT/mri_bones_release_v2/test/female/CR260152/mri_smpl.pkl
#####

## Evaluate:

```shell
PYTHONPATH=. python demos/infer_smpl.py --exp_name=WHATEVER --to_infer smpl_template --betas 0.64 0.19
```

Try this one: Female:

```shell
PYTHONPATH=. python demos/infer_smpl.py --exp_name=WHATEVER --to_infer smpl_template --betas 0.4650 -0.0454
```

Try this one: Male:

```shell
PYTHONPATH=. python demos/infer_smpl.py --exp_name=OPUS_2 --to_infer smpl_template --betas -1.1172, 0.2070
```


# Installation 
1. Follow HIT's installation instructions in the [HIT repository](https://github.com/MarilynKeller/HIT). Including the installation of their checkpoints and body models.


2. Download the datasets from from [On predicting 3D bone locations inside the human body] (https://3dbones.is.tue.mpg.de/). 

3. Install our checkpoint from [PabsDa/HIT-Bone-Specialist](https://huggingface.co/PabsDa/HIT-Bone-Specialist/tree/main)

# Usage:

Demo: 
```shell
PYTHONPATH=. python demos/infer_smpl.py --exp_name=WHATEVER --to_infer smpl_template --betas 0.4650 -0.0454
```



# Acknowledgments

```
@inproceedings{keller2024hit,
  title = {{HIT}: Estimating Internal Human Implicit Tissues from the Body Surface},
  author = {Keller, Marilyn and Arora, Vaibhav and Dakri, Abdelmouttaleb and Chandhok, Shivam and Machann, J{\"u}rgen and Fritsche, Andreas and Black, Michael J. and Pujades, Sergi},
  booktitle = {IEEE/CVF Conf.~on Computer Vision and Pattern Recognition (CVPR)},
  pages = {3480--3490},
  month = jun,
  year = {2024},
  month_numeric = {6}
}
```


