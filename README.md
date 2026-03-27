# Multi-label feature selection with high-level semantic label relationships based on fuzzy rough sets (HSR-MLFS)
This repository is associated with "Multi-label feature selection with high-level semantic label relationships based on fuzzy rough sets".
## Data Preparation
We use csv file to store data.
## Train
The feature ranking is obtained according to the following code.
```
model = HSR_MLFS()
model.fit(X_train, y_train)
attribute = model.train()
```
## Citation
```
@article{ChenCL25,
  author       = {Liangzhou Chen and Mingjie Cai and Qingguo Li},
  title        = {Multi-label feature selection with high-level semantic label relationships
                  based on fuzzy rough sets},
  journal      = {Fuzzy Sets Syst.},
  volume       = {510},
  pages        = {109368},
  year         = {2025},
}
```
