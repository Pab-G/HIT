# HIT

## Metrics

Generate the test metrics (Accuracy, IOU, Dice score)

```shell
#python hit/train.py exp_name=hit_female smpl_cfg.gender=female  run_eval=True wdboff=True
PYTHONPATH=. python hit/train.py     exp_name=hit_female_multibone     smpl_cfg.gender=female     train_cfg.to_train=occ     wdboff=True
```

# Acknowledgments

We thank the authors of the [COAP](https://github.com/markomih/COAP) and [gDNA](https://github.com/xuchen-ethz/gdna) for their codebase. HIT is built on top of these two projects.
We also thank Soubhik Sanyal for his help on the project.

# Citation

If you use this code, please cite the following paper:

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

## Contact

For more questions, please contact hit@tue.mpg.de

This code repository in the provided [License](LICENSE.txt).
For the licensing of the retrained models and the dataset, please refer to the HIT project page [https://hit.is.tue.mpg.de/].
