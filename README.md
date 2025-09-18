## Installation

- Install `feinsum`:
  - `pip install git+https://github.com/kaushikcfd/feinsum`
- Install `arraycontext`:
  - `pip install git+https://github.com/inducer/arraycontext@batched_einsum_actx`
- Install `pytato`:
  - `pip install git+https://github.com/inducer/pytato`
- Install `loopy`:
  - `pip install git+https://github.com/kaushikcfd/loopy`
- Install `pyopencl`:
  - `pip install pyopencl`
- Install `jax`:
  - For CUDA-devtoolkit-11:
    - `pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`
  - For CUDA-devtoolkit-12:
    - `pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`
- Install `feinsum_evaluation`
  - `pip install -e  .`

## HOWTO: Reproduce Canonicalization costs

```console
$ cd feinsum_evaluation
$ python canonicalization_costs.py
```

## HOWTO: Run DG-kernel suite

```console
$ cd feinsum_evaluation
$ # Face mass kernels
$ python ifj_fe_fej_to_ei.py --actxs "jax:jit,pytato:batched_einsum" \
    --batches "1,3,6,19" \
    --ni 4 \
    --nj 3 \ # (See Fig 2.a for all combinations)

$ # Local divergence computations
$ python xre_rij_xej_to_ei.py --actxs "jax:jit,pytato:batched_einsum" \
    --batches "1,3,6" \
    --ni 4 \ # (See Fig 2.b for all combinations)

$ # Local gradient computations
$ python xre_rij_ej_to_xei.py --actxs "jax:jit,pytato:batched_einsum" \
    --batches "1,3,5" \
    --ni 4 \ # (See Fig 2.c for all combinations)
```
