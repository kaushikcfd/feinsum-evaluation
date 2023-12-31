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
           flux_terms_p: Sequence[ArrayT],
           flux_terms_n: Sequence[ArrayT],
           ref_mat: ArrayT,
           jac: ArrayT) -> np.array:
    ref_mat = tag_axes(actx,
                       {0: NamedAxis("voldof"),
                        1: NamedAxis("face"),
                        2: NamedAxis("facedof")},
                       ref_mat)
    jac = tag_axes(actx,
                   {0: NamedAxis("face"),
                    1: NamedAxis("element")},
                   jac)

    flux_terms_p = [tag_axes(actx,
                             {0: NamedAxis("face"),
                              1: NamedAxis("element"),
                              2: NamedAxis("facedof")},
                             flux)
                    for flux in flux_terms_p]

    flux_terms_n = [tag_axes(actx,
                             {0: NamedAxis("face"),
                              1: NamedAxis("element"),
                              2: NamedAxis("facedof")},
                             flux)
                    for flux in flux_terms_n]

    sub_results = [
        actx.einsum("ifj,fe,fej->ei",
                    ref_mat, jac, 0.5 * (flux_n + flux_p))
        for flux_p, flux_n in zip(flux_terms_p, flux_terms_n, strict=True)
    ]

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
         ni: int,
         nj: int) -> None:

    timings = np.empty((len(batches), len(actx_ts)), dtype=np.float64)
    nel = get_nel(ni)

    # sorting `actx_ts` to run JAX related operations at the end as they only
    # free the device memory atexit
    for iactx_t, actx_t in sorted(enumerate(actx_ts),
                                  key=lambda k: get_actx_t_priority(k[1])):
        actx = instantiate_actx_t(actx_t)
        for ibatch, n in enumerate(batches):
            ref_mat = actx.from_numpy(np.random.rand(ni, 4, nj))
            jac = actx.from_numpy(np.random.rand(4, nel))
            compiled_knl = actx.compile(lambda *args: kernel(actx, *args))

            wallclock_time = get_wallclock_time(
                compiled_knl,
                (
                    make_obj_array([
                        actx.from_numpy(np.random.rand(4, nel, nj))
                        for _ in range(n)]),
                    make_obj_array([
                        actx.from_numpy(np.random.rand(4, nel, nj))
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
        prog="ifj_fe_fej_to_ei.py",
        description="Run batched ifj_fe_fej_to_ei benchmarks for arraycontexts",
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

    parser.add_argument("--nj", type=int,
                        help="loop-length of `j`.",
                        required=True,)

    args = parser.parse_args()
    main(batches=[int(k.strip()) for k in args.batches.split(",")],
         actx_ts=[_NAME_TO_ACTX_CLASS[k] for k in args.actxs.split(",")],
         ni=args.ni,
         nj=args.nj)
