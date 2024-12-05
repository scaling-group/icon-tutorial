## In-Context Operator Networks

This folder contains a simple implementation of ICON in [Fine-Tune Language Models as Multi-Modal Differential Equation Solvers](https://arxiv.org/abs/2308.05061) and [PDE Generalization of In-Context Operator Networks: A Study on 1D Scalar Nonlinear Conservation Laws](https://www.sciencedirect.com/science/article/pii/S0021999124006272). For simplicity, the model doesn't textual prompts and only takes data/numerical prompts. It also contains the data generation of conservation laws.

I am trying to make the code clean and easy to extend. It aims to provide a starting point for researchers to explore the potential of ICON in solving PDEs.

## Environment

`conda env create -f conda_env.yaml`


## Run the code
See scripts in `scripts` folder. Please navigate to the root folder of the project before running the scripts.


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

```