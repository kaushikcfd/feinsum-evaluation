import argparse
import numpy as np
from arraycontext import (ArrayContext, ArrayT, tag_axes, PyOpenCLArrayContext,
                          PytatoJAXArrayContext, EagerJAXArrayContext)
from bidict import bidict
from tabulate import tabulate

from typing import Sequence, Type
from pytools.obj_array import make_obj_array
from feinsum_evaluation.metadata import NamedAxis
from feinsum_evaluation.utils import (get_actx_t_priority, instantiate_actx_t,
                                      get_wallclock_time,
                                      BatchedEinsumPytatoPyOpenCLArrayContext)


def kernel(actx: ArrayContext,
           us: Sequence[ArrayT],
           diff_mat: ArrayT,
           jac: ArrayT) -> np.array:
    diff_mat = tag_axes(actx,
                        {0: NamedAxis("ambient_dim"),
                         1: NamedAxis("dof"),
                         2: NamedAxis("dof")},
                        diff_mat)
    jac = tag_axes(actx,
                   {0: NamedAxis("topo_dim"),
                    1: NamedAxis("ambient_dim"),
                    2: NamedAxis("element")},
                   jac)

    us = [tag_axes(actx,
                   {0: NamedAxis("element"),
                    1: NamedAxis("dof")},
                   u)
          for u in us]

    sub_results = [actx.einsum("xre,rij,ej->xei",
                               jac, diff_mat, u)
                   for u in us]

    return make_obj_array(sub_results)


def get_nel(ni: int):
    if ni == 4:
        return 200_000
    elif ni == 10:
        return 200_000
    elif ni == 20:
        return 100_000
    elif ni == 35:
        return 80_000
    else:
        raise NotImplementedError()


def main(*,
         actx_ts: Sequence[Type[ArrayContext]],
         batches: Sequence[int],
         ni: int) -> None:

    timings = np.empty((len(batches), len(actx_ts)), dtype=np.float64)
    nel = get_nel(ni)

    # sorting `actx_ts` to run JAX related operations at the end as they only
    # free the device memory atexit
    for iactx_t, actx_t in sorted(enumerate(actx_ts),
                                  key=lambda k: get_actx_t_priority(k[1])):
        actx = instantiate_actx_t(actx_t)
        for ibatch, n in enumerate(batches):
            ref_mat = actx.from_numpy(np.random.rand(3, ni, ni))
            jac = actx.from_numpy(np.random.rand(3, 3, nel))
            compiled_knl = actx.compile(lambda *args: kernel(actx, *args))

            wallclock_time = get_wallclock_time(
                compiled_knl,
                (
                    make_obj_array([
                        actx.from_numpy(np.random.rand(nel, ni))
                        for _ in range(n)]),
                    ref_mat,
                    jac,
                ),
            )

            timings[ibatch, iactx_t] = wallclock_time

    table = [["",
              *[_NAME_TO_ACTX_CLASS.inv[actx_t]
                for actx_t in actx_ts]]]
    for ibatch, n in enumerate(batches):
        table.append([str(n)] + [f"{timings[ibatch, iactx_t]:.4f}"
                                 for iactx_t in range(len(actx_ts))])

    print(tabulate(table, tablefmt="fancy_grid"))


_NAME_TO_ACTX_CLASS = bidict({
    "pyopencl": PyOpenCLArrayContext,
    "jax:nojit": EagerJAXArrayContext,
    "jax:jit": PytatoJAXArrayContext,
    "pytato:batched_einsum": BatchedEinsumPytatoPyOpenCLArrayContext,
})

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="xre_rij_ej_to_xei.py",
        description="Run batched xre_rij_ej_to_xei benchmarks for arraycontexts",
    )

    parser.add_argument("--actxs", metavar="A", type=str,
                        help=("comma separated integers representing the"
                              " array context types"
                              " to run the benchmark with (for ex."
                              " 'pyopencl,jax:jit,pytato:batched_einsum')"),
                        required=True,)

    parser.add_argument("--batches", metavar="N", type=str,
                        help=("comma separated integers representing the"
                              " #batches to run"
                              " to run the benchmark with (for ex."
                              " '3, 4, 8, 16')"),
                        required=True,)

    parser.add_argument("--ni", type=int,
                        help="loop-length of `i`.",
                        required=True,)

    args = parser.parse_args()
    main(batches=[int(k.strip()) for k in args.batches.split(",")],
         actx_ts=[_NAME_TO_ACTX_CLASS[k] for k in args.actxs.split(",")],
         ni=args.ni)
