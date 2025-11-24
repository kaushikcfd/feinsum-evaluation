## Installation

- Install requirements by creating the conda env as:
  - `conda env create -f conda-env.yml`


## HOWTO: Run DG-kernel suite

```console
$ # Face mass kernels
$ python ifj_fe_fej_to_ei.py
$ # Local divergence computations
$ python xre_rij_xej_to_ei.py
$ # Local gradient computations
$ python xre_rij_ej_to_xei.py
```


## HOWTO: Run TCCG benchmark suite

```console
$ python tccg_suite_with_linear_comb.py
```

## HOWTO: Reproduce Canonicalization costs

```console
$ cd feinsum_evaluation
$ python canonicalization_costs.py
```