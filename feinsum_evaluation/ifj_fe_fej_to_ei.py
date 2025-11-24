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
BATCHES = [3, 4, 5, 6, 19]


def get_dims_for_variant(variant: int) -> Mapping[str, int]:
    if variant == 1:
        return {"i": 4, "e": 200_000, "f": 4, "j": 3}
    elif variant == 2:
        return {"i": 10, "e": 200_000, "f": 4, "j": 6}
    elif variant == 3:
        return {"i": 20, "e": 100_000, "f": 4, "j": 10}
    elif variant == 4:
        return {"i": 35, "e": 80_000, "f": 4, "j": 15}
    else:
        raise ValueError(f"variant must be one of {{1, 2, 3, 4}}, got {variant}.")


def f(
    ref_mat: ArrayT,
    jac: ArrayT,
    flux_terms_p: ObjectArray1D[ArrayT],
    flux_terms_n: ObjectArray1D[ArrayT],
    actx: ArrayContext,
) -> ObjectArray1D[ArrayT]:
    return new_1d(
        [
            actx.einsum("ifj,fe,fej->ei", ref_mat, jac, 0.5 * (flux_n + flux_p))
            for flux_p, flux_n in zip(flux_terms_p, flux_terms_n, strict=True)
        ]
    )


def get_nflops_for_f(variant: int, b: int) -> int:
    import opt_einsum as op
    from pytools import product

    dims = get_dims_for_variant(variant)
    D_shape = (dims["i"], dims["f"], dims["j"])
    J_shape = (dims["f"], dims["e"])
    u_shape = (dims["f"], dims["e"], dims["j"])
    _, path_info = op.contract_path(
        "ifj,fe,fej->ei",
        fnsm.array("D", D_shape),
        fnsm.array("J", J_shape),
        fnsm.array("u", u_shape),
        optimize="optimal",
    )
    return (int(path_info.opt_cost) + 2 * product(u_shape)) * b


def untransformed_loopy_kernel(
    variant: int,
    b: int,
) -> lp.TranslationUnit:
    from loopy.symbolic import parse

    dims = get_dims_for_variant(variant)

    insns = [
        lp.Assignment(
            parse(f"out_{ib}[e, i]"),
            parse(
                "sum([f, j], D_subst(i, f, j) * J_subst(f, e)"
                f" * flux_{ib}_subst(f, e, j))"
            ),
            within_inames=frozenset({"e", "i"}),
        )
        for ib in range(b)
    ]

    substs = [
        lp.SubstitutionRule(
            "D_subst", ["d_0", "d_1", "d_2"], parse("D[d_0, d_1, d_2]")
        ),
        lp.SubstitutionRule("J_subst", ["d_0", "d_1"], parse("J[d_0, d_1]")),
        *[
            lp.SubstitutionRule(
                f"flux_{ib}_subst",
                ["d_0", "d_1", "d_2"],
                parse(f"0.5*(fp_{ib}[d_0, d_1, d_2] + fn_{ib}[d_0, d_1, d_2])"),
            )
            for ib in range(b)
        ],
    ]

    t_unit = lp.make_kernel(
        f"{{ [e,f,i,j] : 0<=e<{dims['e']} and 0<=f<{dims['f']} and 0<=i<{dims['i']}"
        f" and 0<=j<{dims['j']} }}",
        insns + substs,
        [
            lp.GlobalArg(
                "D",
                dtype=np.float64,
                shape=lp.auto,
            ),
            lp.GlobalArg(
                "J",
                dtype=np.float64,
                shape=lp.auto,
            ),
            lp.GlobalArg(
                ",".join(f"fn_{ib}" for ib in range(b)),
                dtype=np.float64,
                shape=lp.auto,
            ),
            lp.GlobalArg(
                ",".join(f"fp_{ib}" for ib in range(b)),
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
        f_n_nps = [rng.random((dims["f"], dims["e"], dims["j"])) for _ in range(b)]
        f_p_nps = [rng.random((dims["f"], dims["e"], dims["j"])) for _ in range(b)]
        D_np = rng.random((dims["i"], dims["f"], dims["j"]))
        J_np = rng.random((dims["f"], dims["e"]))

        f_n_cls = [cla.to_device(cq, f_n_np) for f_n_np in f_n_nps]
        f_p_cls = [cla.to_device(cq, f_n_np) for f_n_np in f_p_nps]
        D_cl = cla.to_device(cq, D_np)
        J_cl = cla.to_device(cq, J_np)
        out_cls = [
            cla.empty(cq, (dims["e"], dims["i"]), dtype=np.float64) for _ in range(b)
        ]
        allocator = cl_tools.MemoryPool(cl_tools.ImmediateAllocator(cq))

        compiled_f = get_loopy_program_for_facemass(variant, b, cq).executor(
            cq,
            D_cl,
            J_cl,
            *f_n_cls,
            *f_p_cls,
            *out_cls,
            entrypoint=None,
            allocator=allocator,
        )

        for i in range(N_WARMUP_ROUNDS):
            compiled_f(
                cq,
                D=D_cl,
                J=J_cl,
                **{f"fn_{ib}": f_n_cl for ib, f_n_cl in enumerate(f_n_cls)},
                **{f"fp_{ib}": f_p_cl for ib, f_p_cl in enumerate(f_p_cls)},
                **{f"out_{ib}": out_cl for ib, out_cl in enumerate(out_cls)},
                allocator=allocator,
            )
            if i == 0:
                out_nps = f(
                    D_np,
                    J_np,
                    new_1d(f_n_nps),
                    new_1d(f_p_nps),
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
                    D=D_cl,
                    J=J_cl,
                    **{f"fn_{ib}": f_n_cl for ib, f_n_cl in enumerate(f_n_cls)},
                    **{f"fp_{ib}": f_p_cl for ib, f_p_cl in enumerate(f_p_cls)},
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
        f_n_nps = [rng.random((dims["f"], dims["e"], dims["j"])) for _ in range(b)]
        f_p_nps = [rng.random((dims["f"], dims["e"], dims["j"])) for _ in range(b)]
        D_np = rng.random((dims["i"], dims["f"], dims["j"]))
        J_np = rng.random((dims["f"], dims["e"]))

        f_n_actxs = new_1d([actx.from_numpy(f_n_np) for f_n_np in f_n_nps])
        f_p_actxs = new_1d([actx.from_numpy(f_n_np) for f_n_np in f_p_nps])
        D_actx = actx.from_numpy(D_np)
        J_actx = actx.from_numpy(J_np)

        compiled_f = actx.compile(partial(f, actx=actx))

        for i in range(N_WARMUP_ROUNDS):
            out_actxs = compiled_f(
                D_actx,
                J_actx,
                f_n_actxs,
                f_p_actxs,
            )
            if i == 0:
                out_nps = f(
                    D_np,
                    J_np,
                    new_1d(f_n_nps),
                    new_1d(f_p_nps),
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
                    D_actx,
                    J_actx,
                    f_n_actxs,
                    f_p_actxs,
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
