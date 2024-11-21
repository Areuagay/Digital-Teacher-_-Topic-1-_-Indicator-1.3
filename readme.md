### Optimizing Reciprocal Rank with Bayesian Average for Improved Next Item Recommendation

---
This is the implementation of the paper "Optimizing Reciprocal Rank with Bayesian Average for Improved Next Item Recommendation".


#### Configure the environment

---
- Cuda: 10.1
- Python: Python 3.7.9
- Pytorch: 1.13.1+cu117'
- dgl: dgl-cu111
- torch_geometric: 2.3.1

---
#### Datasets
- xing
- reddit

---
### Run and test
- set config.py no_train=False; model='HGGNN'; data='xing/reddit'

- run main.py 
