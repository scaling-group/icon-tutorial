## In-Context Operator Networks

This folder contains a simple implementation of autoregressive ICON in [Fine-Tune Language Models as Multi-Modal Differential Equation Solvers](https://arxiv.org/abs/2308.05061) and [PDE Generalization of In-Context Operator Networks: A Study on 1D Scalar Nonlinear Conservation Laws](https://www.sciencedirect.com/science/article/pii/S0021999124006272). It's an improved version of the original encoder-decoder ICON model in [In-Context Operator Learning with Data Prompts for Differential Equation Problems](https://www.pnas.org/doi/10.1073/pnas.2310142120). The code for data generation of conservation laws is also included.

The code is implemented using PyTorch and aims to provide a starting point for researchers exploring the potential of ICON in solving PDEs. For simplicity, the model is designed to process only data/numerical prompts, without supporting textual prompts. The code is structured to be clean and easy to extend. Additionally, a simple implementation using Jax within Jupyter notebooks is available in the `tutorial-jax` folder.

Please see [our website](https://scaling-group.github.io/research/) for latest updates on ICON and other projects. 

## Environment

`conda env create -f conda_env.yaml`


## Run the code
See scripts in `scripts` folder. Please navigate to the root folder of the project before running the scripts.

Currently the code has errors when using multiple GPUs. It seems that the main issue is that the model takes as input a customized data type instead of tensors, which is not compatible with DataParallel. Will fix this in the future.

## Reference:

```
@article{yang2023context,
  title={In-context operator learning with data prompts for differential equation problems},
  author={Yang, Liu and Liu, Siting and Meng, Tingwei and Osher, Stanley J},
  journal={Proceedings of the National Academy of Sciences},
  volume={120},
  number={39},
  pages={e2310142120},
  year={2023},
  publisher={National Acad Sciences}
}

@article{yang2023FineTune,
  title={Fine-Tune Language Models as Multi-Modal Differential Equation Solvers},
  author={Yang, Liu and Liu, Siting and Osher, Stanley J},
  journal={arXiv preprint arXiv:2308.05061},
  year={2023}
}

@article{yang2024pde,
  title={{PDE} Generalization of In-Context Operator Networks: A Study on {1D} Scalar Nonlinear Conservation Laws},
  author={Yang, Liu and Osher, Stanley J},
  journal = {Journal of Computational Physics},
  volume = {519},
  pages = {113379},
  year = {2024},
  issn = {0021-9991},
  doi = {https://doi.org/10.1016/j.jcp.2024.113379},
  url = {https://www.sciencedirect.com/science/article/pii/S0021999124006272},
}

@article{cao2024vicon,
  title={VICON: Vision In-Context Operator Networks for Multi-Physics Fluid Dynamics Prediction},
  author={Cao, Yadi and Liu, Yuxuan and Yang, Liu and Yu, Rose and Schaeffer, Hayden and Osher, Stanley},
  journal={arXiv preprint arXiv:2411.16063},
  year={2024}
}
```