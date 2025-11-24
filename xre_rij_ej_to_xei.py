import gc
import itertools
from collections.abc import Mapping
from time import time

import feinsum as fnsm
import loopy as lp
import numpy as np
import pyopencl as cl
import pyopencl.tools as cl_tools
from arraycontext import (
    ArrayContext,
    ArrayT,
    NumpyArrayContext,
    PytatoJAXArrayContext,
)
from pytools.obj_array import ObjectArray1D, new_1d
from tabulate import tabulate

N_WARMUP_ROUNDS = 3
N_MIN_ROUNDS = 10
VARIANTS = [1, 2, 3, 4]
BATCHES = [1, 3, 5]


def get_dims_for_variant(variant: int) -> Mapping[str, int]:
    if variant == 1:
        return {"i": 4, "j": 4, "e": 200_000, "x": 3, "r": 3}
    elif variant == 2:
        return {"i": 10, "j": 10, "e": 200_000, "x": 3, "r": 3}
    elif variant == 3:
        return {"i": 20, "j": 20, "e": 100_000, "x": 3, "r": 3}
    elif variant == 4:
        return {"i": 35, "j": 35, "e": 80_000, "x": 3, "r": 3}
    else:
        raise ValueError(f"variant must be one of {{1, 2, 3, 4}}, got {variant}.")


def f(
    J: ArrayT,
    D: ArrayT,
    us: ObjectArray1D[ArrayT],
    actx: ArrayContext,
) -> ObjectArray1D[ArrayT]:
    return new_1d(
        [
            actx.einsum("xre,rij,ej->xei", J, D, u)
            for u in us
        ]
    )


def get_nflops_for_f(variant: int, b: int) -> int:
    import opt_einsum as op

    dims = get_dims_for_variant(variant)
    J_shape = (dims["x"], dims["r"], dims["e"])
    D_shape = (dims["r"], dims["i"], dims["j"])
    u_shape = (dims["e"], dims["j"])
    _, path_info = op.contract_path(
        "xre,rij,ej->xei",
        fnsm.array("J", J_shape),
        fnsm.array("D", D_shape),
        fnsm.array("u", u_shape),
        optimize="optimal",
    )
    return int(path_info.opt_cost) * b


def untransformed_loopy_kernel(
    variant: int,
    b: int,
) -> lp.TranslationUnit:
    from loopy.symbolic import parse

    dims = get_dims_for_variant(variant)

    insns = [
        lp.Assignment(
            parse(f"out_{ib}[x, e, i]"),
            parse(
                "sum([r, j], J_subst(x, r, e) * D_subst(r, i, j)"
                f" * u_{ib}_subst(e, j))"
            ),
            within_inames=frozenset({"x", "e", "i"}),
        )
        for ib in range(b)
    ]

    substs = [
        lp.SubstitutionRule(
            "J_subst", ["d_0", "d_1", "d_2"], parse("J[d_0, d_1, d_2]")
        ),
        lp.SubstitutionRule(
            "D_subst", ["d_0", "d_1", "d_2"], parse("D[d_0, d_1, d_2]")
        ),
        *[
            lp.SubstitutionRule(
                f"u_{ib}_subst",
                ["d_0", "d_1"],
                parse(f"u_{ib}[d_0, d_1]"),
            )
            for ib in range(b)
        ],
    ]

    t_unit = lp.make_kernel(
        f"{{ [x,r,e,i,j] : 0<=e<{dims['e']} and 0<=x<{dims['x']} and "
        f"0<=r<{dims['r']} and 0<=j<{dims['j']} and 0<=i<{dims['i']} }}",
        insns + substs,
        [
            lp.GlobalArg(
                "J",
                dtype=np.float64,
                shape=lp.auto,
            ),
            lp.GlobalArg(
                "D",
                dtype=np.float64,
                shape=lp.auto,
            ),
            lp.GlobalArg(
                ",".join(f"u_{ib}" for ib in range(b)),
                dtype=np.float64,
                shape=lp.auto,
            ),
            lp.GlobalArg(
                ",".join(f"out_{ib}" for ib in range(b)),
                dtype=np.float64,
                shape=lp.auto,
            ),
        ],
        lang_version=(2018, 2),
    )
    return t_unit


def get_loopy_program_for_facemass(
    variant: int, b: int, queue: cl.CommandQueue
) -> lp.TranslationUnit:
    t_unit = untransformed_loopy_kernel(variant, b)
    einsum, _ = fnsm.get_a_matched_einsum(t_unit, long_dim_length=1_000)
    facts_in_feinsum_db = fnsm.query(einsum, queue.device, err_if_no_results=True)
    best_query = max(
        facts_in_feinsum_db,
        key=lambda q: sum(q.giga_op_info.values()) / q.runtime_in_sec,
    )
    t_unit = best_query.transform(t_unit)
    return t_unit


def measure_flop_rate_for_f_w_feinsum() -> Mapping[tuple[int, int], float]:
    import pyopencl.array as cla

    ctx = cl.create_some_context()
    cq = cl.CommandQueue(ctx)
    if cq.device.name != "NVIDIA TITAN V":
        raise RuntimeError("This script is only for evauating on a TITAN V.")

    from numpy.random import default_rng

    rng = default_rng(0)
    actx_np = NumpyArrayContext()
    measured_gflops: dict[tuple[int, int], float] = {}

    for variant, b in itertools.product(VARIANTS, BATCHES):
        dims = get_dims_for_variant(variant)
        J_np = rng.random((dims["x"], dims["r"], dims["e"]))
        D_np = rng.random((dims["r"], dims["i"], dims["j"]))
        u_nps = [rng.random((dims["e"], dims["j"])) for _ in range(b)]

        u_cls = [cla.to_device(cq, u_np) for u_np in u_nps]
        D_cl = cla.to_device(cq, D_np)
        J_cl = cla.to_device(cq, J_np)
        out_cls = [
            cla.empty(cq, (dims["x"], dims["e"], dims["i"]), dtype=np.float64)
            for _ in range(b)
        ]
        allocator = cl_tools.MemoryPool(cl_tools.ImmediateAllocator(cq))

        compiled_f = get_loopy_program_for_facemass(variant, b, cq).executor(
            cq,
            J_cl,
            D_cl,
            *u_cls,
            *out_cls,
            entrypoint=None,
            allocator=allocator,
        )

        for i in range(N_WARMUP_ROUNDS):
            compiled_f(
                cq,
                J=J_cl,
                D=D_cl,
                **{f"u_{ib}": u_cl for ib, u_cl in enumerate(u_cls)},
                **{f"out_{ib}": out_cl for ib, out_cl in enumerate(out_cls)},
                allocator=allocator,
            )
            if i == 0:
                out_nps = f(
                    J_np,
                    D_np,
                    new_1d(u_nps),
                    actx=actx_np,
                )
                for out_np, out_cl in zip(out_nps, out_cls, strict=True):
                    np.testing.assert_allclose(out_np, out_cl.get())

        total_time = 0
        total_rounds = 0
        while total_time < 2:
            cq.finish()
            t_start = time()
            for _ in range(N_MIN_ROUNDS):
                compiled_f(
                    cq,
                    J=J_cl,
                    D=D_cl,
                    **{f"u_{ib}": u_cl for ib, u_cl in enumerate(u_cls)},
                    **{f"out_{ib}": out_cl for ib, out_cl in enumerate(out_cls)},
                    allocator=allocator,
                )
            cq.finish()
            t_end = time()
            total_time += t_end - t_start
            total_rounds += N_MIN_ROUNDS

        avg_time = total_time / total_rounds
        ngflops = get_nflops_for_f(variant, b) * 1e-9
        measured_gflops[variant, b] = ngflops / avg_time
        print(f"Done within {variant = }, {b = } with Feinsum.")

    return measured_gflops


def measure_flop_rate_for_f_w_jax() -> Mapping[tuple[int, int], float]:
    from functools import partial

    import jax
    from numpy.random import default_rng

    jax.config.update("jax_enable_x64", True)

    actx = PytatoJAXArrayContext()
    actx_np = NumpyArrayContext()
    rng = default_rng(0)
    measured_gflops: dict[tuple[int, int], float] = {}

    for variant, b in itertools.product(VARIANTS, BATCHES):
        dims = get_dims_for_variant(variant)
        J_np = rng.random((dims["x"], dims["r"], dims["e"]))
        D_np = rng.random((dims["r"], dims["i"], dims["j"]))
        u_nps = new_1d([rng.random((dims["e"], dims["j"])) for _ in range(b)])

        J_actx = actx.from_numpy(J_np)
        D_actx = actx.from_numpy(D_np)
        u_actxs = actx.from_numpy(u_nps)

        compiled_f = actx.compile(partial(f, actx=actx))

        for i in range(N_WARMUP_ROUNDS):
            out_actxs = compiled_f(
                J_actx,
                D_actx,
                u_actxs,
            )
            if i == 0:
                out_nps = f(
                    J_np,
                    D_np,
                    u_nps,
                    actx=actx_np,
                )
                for out_np, out_actx in zip(out_nps, out_actxs, strict=True):
                    np.testing.assert_allclose(out_np, actx.to_numpy(out_actx))

        total_time = 0
        total_rounds = 0
        while total_time < 2:
            gc.collect()
            t_start = time()
            for _ in range(N_MIN_ROUNDS):
                compiled_f(
                    J_actx,
                    D_actx,
                    u_actxs,
                )
            t_end = time()
            total_time += t_end - t_start
            total_rounds += N_MIN_ROUNDS

        avg_time = total_time / total_rounds
        ngflops = get_nflops_for_f(variant, b) * 1e-9
        measured_gflops[variant, b] = ngflops / avg_time
        print(f"Done within {variant = }, {b = } with JAX.")

    return measured_gflops


def main():
    feinsum_gflops = measure_flop_rate_for_f_w_feinsum()
    gc.collect()
    jax_gflops = measure_flop_rate_for_f_w_jax()
    gc.collect()

    table = [
        [
            id_,
            jax_gflops[id_],
            feinsum_gflops[id_],
            feinsum_gflops[id_] / jax_gflops[id_],
        ]
        for id_ in sorted(itertools.product(VARIANTS, BATCHES))
    ]

    print(
        tabulate(
            table,
            headers=[
                "ibenchmark",
                "JAX GFLOPS",
                "Feinsum GFLOPS",
                "Feinsum speedup",
            ],
            tablefmt="fancy",
        )
    )


if __name__ == "__main__":
    main()
