# FlowOT

Official implementation of the work "Computing high-dimensional optimal transport by flow neural networks"




## Installation

```bash
conda env create -f environment.yml
conda activate FlowOT
```


## Usage

1. Get the flow refinment given initial flows (stored in `checkpoints/`)
```bash
python main_FlowOT.py
```

The trajectory of the flow refinement is shown below:
![Trajectory](trajectory.gif)

2. (Optional) Get the infinitesimal DRE given the flow refinment
```bash
python main_infinitesimal_DRE.py
```

## Citation

```
@inproceedings{xu2025computing,
    title={Computing high-dimensional optimal transport by flow neural networks},
    author={Chen Xu and Xiuyuan Cheng and Yao Xie},
    booktitle={The 28th International Conference on Artificial Intelligence and Statistics},
    year={2025},
    url={https://openreview.net/forum?id=oEWYNesvRJ}
}
```
