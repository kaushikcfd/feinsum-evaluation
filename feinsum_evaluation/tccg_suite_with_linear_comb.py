import gc
from time import time

import feinsum as fnsm
import loopy as lp
import numpy as np
import pyopencl as cl
import pyopencl.tools as cl_tools
from arraycontext import (
    Array,
    ArrayContext,
    NumpyArrayContext,
    PytatoJAXArrayContext,
)
from tabulate import tabulate

N_WARMUP_ROUNDS = 3
N_MIN_ROUNDS = 10
IBENCHMARKS = list(range(1, 49))


def deconstruct_tccg_benchmark(
    ibenchmark,
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]:
    import feinsum.utils

    tensor_contraction = feinsum.utils.get_tccg_benchmark(ibenchmark)
    ((A, B),) = tensor_contraction.args
    A_shape = A.shape
    B_shape = B.shape
    C_shape = tensor_contraction.shape
    assert all(isinstance(dim, int) for dim in A_shape)
    assert all(isinstance(dim, int) for dim in B_shape)
    assert all(isinstance(dim, int) for dim in C_shape)
    return tensor_contraction.get_subscripts(), (A_shape, B_shape, C_shape)


def f(
    alpha1: float,
    b1: float,
    A: Array,
    alpha2: float,
    b2: float,
    B: Array,
    *,
    actx: ArrayContext,
    ibenchmark: int,
) -> Array:
    spec, (a_shape, b_shape, _) = deconstruct_tccg_benchmark(ibenchmark)
    assert a_shape == A.shape
    assert b_shape == B.shape
    assert A.dtype == np.float64 and B.dtype == np.float64
    return actx.einsum(spec, alpha1 * A + b1, alpha2 * B + b2)


def get_untransformed_loopy_program_for_tccg(ibenchmark: int) -> lp.TranslationUnit:
    from feinsum.codegen.loopy import _get_isl_basic_set
    from feinsum.utils import get_tccg_benchmark
    einsum = get_tccg_benchmark(ibenchmark)
    a_subst_index = ", ".join(
        f"_{idim}" for idim in range(len(einsum.arg_to_shape["A"]))
    )
    b_subst_index = ", ".join(
        f"_{idim}" for idim in range(len(einsum.arg_to_shape["B"]))
    )
    out_idxs = ", ".join(einsum.out_idx_set)
    sum_idxs = ", ".join(einsum.sum_indices)
    in_idx1, in_idx2 = [", ".join(in_idx_set) for in_idx_set in einsum.in_idx_sets]
    t_unit = lp.make_kernel(
        [_get_isl_basic_set(einsum.index_to_dim_length)],
        f"""
        subst_A({a_subst_index}) := alpha1 * A[{a_subst_index}] + b1
        subst_B({b_subst_index}) := alpha2 * B[{b_subst_index}] + b2

        out[{out_idxs}] = sum([{sum_idxs}], subst_A({in_idx1}) * subst_B({in_idx2}))
        """,
        [
            lp.ValueArg("alpha1", dtype=np.float64),
            lp.ValueArg("b1", dtype=np.float64),
            lp.GlobalArg("A", dtype=np.float64, shape=lp.auto),
            lp.ValueArg("alpha2", dtype=np.float64),
            lp.ValueArg("b2", dtype=np.float64),
            lp.GlobalArg("B", dtype=np.float64, shape=lp.auto),
            lp.GlobalArg("out", dtype=np.float64, shape=lp.auto),
        ],
        lang_version=(2018, 2),
    )
    return t_unit


def get_loopy_program_for_tccg(
    ibenchmark: int, queue: cl.CommandQueue
) -> lp.TranslationUnit:
    t_unit = get_untransformed_loopy_program_for_tccg(ibenchmark)
    einsum, _ = fnsm.get_a_matched_einsum(t_unit, long_dim_length=10_000)
    facts_in_feinsum_db = fnsm.query(
        einsum, queue.device, err_if_no_results=True
    )
    best_query = max(
        facts_in_feinsum_db,
        key=lambda q: sum(q.giga_op_info.values()) / q.runtime_in_sec,
    )
    t_unit = best_query.transform(t_unit)
    return t_unit


def get_nflops_for_f(ibenchmark) -> int:
    """
    Returns the number of FLOPS for the operations in :func:`f`.
    """
    import feinsum.utils
    from feinsum.measure import _get_giga_ops_from_einsum
    from pytools import product

    _, (A_shape, B_shape, _) = deconstruct_tccg_benchmark(ibenchmark)
    flops_in_tc = (
        _get_giga_ops_from_einsum(feinsum.utils.get_tccg_benchmark(ibenchmark))[
            np.dtype(np.float64)
        ]
        * 1e9
    )
    return 2 * product(A_shape) + 2 * product(B_shape) + flops_in_tc


def get_n_footprint_bytes_for_f(ibenchmark) -> int:
    """
    Returns the number of FLOPS for the operations in :func:`f`.
    """
    from pytools import product

    _, (A_shape, B_shape, C_shape) = deconstruct_tccg_benchmark(ibenchmark)
    return (product(A_shape) + product(B_shape) + product(C_shape)) * 8


def get_roofline_flop_rate_for_f() -> tuple[float, ...]:
    """
    Returns a :class:`tuple`, ``flop_rate``, of 48 floating point values.
    ``flop_rate[i]`` corresponds to the measured GFLOPS corresponding to the
    computation of :func:`f` with the argument ``ibenchmark`` as ``i``.  During
    the computation of the We assume the device here is the Nvidia Titan V.
    """
    roofline_gflops: list[float] = []
    for ibenchmark in IBENCHMARKS:
        ngflops = get_nflops_for_f(ibenchmark) * 1e-9
        ngbytes = get_n_footprint_bytes_for_f(ibenchmark) * 1e-9
        npeak_gflops = 6144  # GFLOPS
        npeak_bw = 652.8  # GB/s
        roofline_gflops.append(
            ngflops / max(ngflops / npeak_gflops, ngbytes / npeak_bw),
        )
    return tuple(roofline_gflops)


def measure_flop_rate_for_f_w_jax() -> tuple[float, ...]:
    """
    Returns a :class:`tuple`, ``flop_rate``, of 48 floating point values.
    ``flop_rate[i]`` corresponds to the measured GFLOPS corresponding to the
    computation of :func:`f` with the argument ``ibenchmark`` as ``i``.
    """
    from functools import partial

    import jax

    jax.config.update("jax_enable_x64", True)
    actx = PytatoJAXArrayContext()

    from numpy.random import default_rng

    rng = default_rng(0)
    alpha1 = rng.random(()).item()
    alpha2 = rng.random(()).item()
    b1 = rng.random(()).item()
    b2 = rng.random(()).item()
    actx_np = NumpyArrayContext()
    measured_gflops: list[float] = []

    for ibenchmark in IBENCHMARKS:
        f_instance = partial(f, actx=actx, ibenchmark=ibenchmark)
        compiled_f = actx.compile(f_instance)
        _, (A_shape, B_shape, _) = deconstruct_tccg_benchmark(ibenchmark)
        A_np = rng.random(A_shape, dtype=np.float64)
        B_np = rng.random(B_shape, dtype=np.float64)

        A_actx = actx.from_numpy(A_np)
        B_actx = actx.from_numpy(B_np)

        out_actx = compiled_f(alpha1, b1, A_actx, alpha2, b2, B_actx)
        out_np = f(
            alpha1, b1, A_np, alpha2, b2, B_np, actx=actx_np, ibenchmark=ibenchmark
        )
        np.testing.assert_allclose(out_np, actx.to_numpy(out_actx))

        for _ in range(N_WARMUP_ROUNDS - 1):
            compiled_f(alpha1, b1, A_actx, alpha2, b2, B_actx)

        total_time = 0
        total_rounds = 0
        while total_time < 2:
            gc.collect()
            t_start = time()
            for _ in range(N_MIN_ROUNDS):
                compiled_f(alpha1, b1, A_actx, alpha2, b2, B_actx)
            t_end = time()
            total_time += t_end - t_start
            total_rounds += N_MIN_ROUNDS

        avg_time = total_time / total_rounds
        ngflops = get_nflops_for_f(ibenchmark) * 1e-9
        measured_gflops.append(ngflops / avg_time)
        print(f"Done within {ibenchmark = } with {actx}.")
    return tuple(measured_gflops)


def measure_flop_rate_for_f_w_feinsum() -> tuple[float, ...]:
    """
    Returns a :class:`tuple`, ``flop_rate``, of 48 floating point values.
    ``flop_rate[i]`` corresponds to the measured GFLOPS corresponding to the
    computation of :func:`f` with the argument ``ibenchmark`` as ``i``.
    """
    import pyopencl.array as cla
    ctx = cl.create_some_context()
    cq = cl.CommandQueue(ctx)
    if cq.device.name != "NVIDIA TITAN V":
        raise RuntimeError("This script is only for evauating on a TITAN V.")

    from numpy.random import default_rng

    rng = default_rng(0)
    alpha1 = rng.random(()).item()
    alpha2 = rng.random(()).item()
    b1 = rng.random(()).item()
    b2 = rng.random(()).item()
    actx_np = NumpyArrayContext()
    measured_gflops: list[float] = []

    for ibenchmark in IBENCHMARKS:
        _, (A_shape, B_shape, out_shape) = deconstruct_tccg_benchmark(ibenchmark)
        A_np = rng.random(A_shape, dtype=np.float64)
        B_np = rng.random(B_shape, dtype=np.float64)

        A_actx = cla.to_device(cq, A_np)
        B_actx = cla.to_device(cq, B_np)
        out_actx = cla.empty(cq, out_shape, dtype=np.float64)
        allocator = cl_tools.MemoryPool(cl_tools.ImmediateAllocator(cq))

        compiled_f = get_loopy_program_for_tccg(ibenchmark, cq).executor(
            cq, alpha1, b1, A_actx, alpha2, b2, B_actx, out=out_actx,
            allocator=allocator,
        )

        for i in range(N_WARMUP_ROUNDS):
            compiled_f(
                cq,
                alpha1=alpha1,
                b1=b1,
                A=A_actx,
                alpha2=alpha2,
                b2=b2,
                B=B_actx,
                out=out_actx,
                allocator=allocator,
            )
            if i == 0:
                out_np = f(
                    alpha1,
                    b1,
                    A_np,
                    alpha2,
                    b2,
                    B_np,
                    actx=actx_np,
                    ibenchmark=ibenchmark,
                )
                np.testing.assert_allclose(out_np, out_actx.get())

        total_time = 0
        total_rounds = 0
        while total_time < 2:
            cq.finish()
            t_start = time()
            for _ in range(N_MIN_ROUNDS):
                compiled_f(
                    cq,
                    alpha1=alpha1,
                    b1=b1,
                    A=A_actx,
                    alpha2=alpha2,
                    b2=b2,
                    B=B_actx,
                    out=out_actx,
                    allocator=allocator,
                )
            cq.finish()
            t_end = time()
            total_time += t_end - t_start
            total_rounds += N_MIN_ROUNDS

        avg_time = total_time / total_rounds
        ngflops = get_nflops_for_f(ibenchmark) * 1e-9
        measured_gflops.append(ngflops / avg_time)
        print(f"Done within {ibenchmark = } with Feinsum.")
    return tuple(measured_gflops)


def main():
    feinsum_gflops = measure_flop_rate_for_f_w_feinsum()
    gc.collect()
    jax_gflops = measure_flop_rate_for_f_w_jax()
    gc.collect()
    roofline_gflops = get_roofline_flop_rate_for_f()

    table = np.empty((len(IBENCHMARKS), 5))
    table[:, 0] = IBENCHMARKS
    table[:, 1] = jax_gflops
    table[:, 2] = feinsum_gflops
    table[:, 3] = roofline_gflops
    table[:, 4] = table[:, 2] / table[:, 1]

    print(
        tabulate(
            table,
            headers=["ibenchmark", "JAX GFLOPS", "Feinsum GFLOPS",
                     "Roofline GFLOPS", "Feinsum speedup"],
            tablefmt="fancy",
        )
    )


if __name__ == "__main__":
    main()
