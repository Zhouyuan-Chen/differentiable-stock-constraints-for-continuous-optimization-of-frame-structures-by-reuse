# Differentiable Stock Constraints for Continuous Optimization of Frame Structures by Reuse

![Teaser](figs/teaser.jpg)

This repository contains the official prototype implementation of the paper

**Differentiable Stock Constraints for Continuous Optimization of Frame Structures by Reuse**

by [Zhouyuan Chen](https://zhouyuan-chen.github.io/), [Marzia Riso](https://marzia-riso.github.io/), [Berend Baas](https://berendbaas.nl/), [Adriana Schulz](https://www.computationaldesign.group/adriana), and [Adrien Bousseau](https://www-sop.inria.fr/members/Adrien.Bousseau/).

## Description

The repository provides the core optimization framework together with several Jupyter notebooks demonstrating the main components of the method. Feel free to explore the notebooks and adapt the code for your own experiments. If you like it, please consider citing our work :)

- **`/top_opt_framework`** — Based on the codebase accompanying the paper [Neural Reparameterization Improves Structural Optimization](https://github.com/google-research/neural-structural-optimization) by Hoyer et al., with modifications for this project.
- **`basis_func_comparison.ipynb`** — Comparison of different basis functions.
- **`beams.ipynb`** — Generates the frame structure under beam stock constraints. **This is the main codebase for the project**; if you want to extend our work, we suggest starting here.
- **`teaser.ipynb`** — Generates the teaser figure.
- **`arc.ipynb`** — Generates the frame structure under arc stock constraints.

## Installation

```bash
# Clone the repository
git clone https://gitlab.inria.fr/zhochen/differentiable-stock-constraints-for-continuous-optimization-of-frame-structures-by-reuse.git
cd differentiable-stock-constraints-for-continuous-optimization-of-frame-structures-by-reuse

# (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

Launch Jupyter and open any of the notebooks:

```bash
jupyter notebook
```

To reproduce the main results, start with **`beams.ipynb`**. The other notebooks (`arc.ipynb`, `basis_func_comparison.ipynb`, `teaser.ipynb`) can be run independently to reproduce their respective figures and comparisons.

## Example

![Example](figs/example.png)
![Example animation](figs/example_1mb.gif)

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{chen2026differentiable,
  author = {Chen, Zhouyuan and Riso, Marzia and Baas, Berend and Schulz, Adriana and Bousseau, Adrien},
  title = {Differentiable Stock Constraints for Continuous Optimization of Frame Structures by Reuse},
  booktitle = {Proceedings of the 11th ACM Symposium on Computational Fabrication},
  series = {SCF '26},
  year = {2026},
  month = {dec},
  location = {Tokyo, Japan},
  pages = {1--13},
  articleno = {8},
  numpages = {13},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  isbn = {979-8-4007-2821-1},
  doi = {10.1145/3828611.3845645},
  url = {https://doi.org/10.1145/3828611.3845645}
}
```

## Acknowledgments

This work was supported by the ANR-NSF grant NaturalCAD (ANR-23-CE94-0003, NSF 2315354), the Inria CircleD Exploratory Action, which benefits from the support of the La Poste Group patron of the Inria Foundation, and by the Circularity 4.0 for Aerospace funded by TU Delft France Initiative (TDFI).

This project builds on [Neural Reparameterization Improves Structural Optimization](https://github.com/google-research/neural-structural-optimization) by Hoyer et al. We thank the authors for making their code available.

## License

This project is licensed under the MIT License.

Copyright (c) 2026 Zhouyuan Chen and contributors
