import gc
from time import time

import feinsum as fnsm
import islpy as isl
import loopy as lp
import numpy as np
import pytato as pt
from arraycontext import (
    Array,
    ArrayContext,
    NumpyArrayContext,
    PytatoJAXArrayContext,
    PytatoPyOpenCLArrayContext,
)
from tabulate import tabulate

N_WARMUP_ROUNDS = 3
N_MIN_ROUNDS = 10


class FeinsumArrayContext(PytatoPyOpenCLArrayContext):
    def transform_dag(self, dag: pt.DictOfNamedArrays) -> pt.DictOfNamedArrays:
        # Step 1. Materialize einsum/reduction outputs.
        # ---------------------------------------------
        def materialize_all_einsums_or_reduces(expr: pt.Array):
            if isinstance(expr, pt.Einsum) or (
                isinstance(expr, pt.IndexLambda) and expr.var_to_reduction_descr
            ):
                return expr.tagged(pt.tags.ImplStored())
            else:
                return expr

        dag = pt.transform.map_and_copy(dag, materialize_all_einsums_or_reduces)

        # Step 2. Make all pt.einsum/pt.reduction inputs as substitutions
        # ---------------------------------------------------------------
        def implement_einsum_reduction_inputs_as_substs(expr):
            from immutables import Map
            from pytato.target.loopy import ImplSubstitution

            if isinstance(expr, pt.Einsum):
                return pt.Einsum(
                    expr.access_descriptors,
                    tuple(arg.tagged(ImplSubstitution()) for arg in expr.args),
                    expr.redn_axis_to_redn_descr,
                    tags=expr.tags,
                    axes=expr.axes,
                )
            elif isinstance(expr, pt.IndexLambda) and expr.var_to_reduction_descr:
                return pt.IndexLambda(
                    expr.expr,
                    expr.shape,
                    expr.dtype,
                    Map(
                        {
                            name: bnd.tagged(ImplSubstitution())
                            for name, bnd in expr.bindings.items()
                        }
                    ),
                    expr.var_to_reduction_descr,
                    tags=expr.tags,
                    axes=expr.axes,
                )
            else:
                return expr

        dag = pt.transform.map_and_copy(
            dag,
            implement_einsum_reduction_inputs_as_substs,
        )

        return dag

    def transform_loopy_program(self, t_unit: lp.TranslationUnit):
        knl = t_unit.default_entrypoint
        assert len(knl.instructions) == 1

        # {{{ Put all basic sets into one domain. (easy for loopy transformations.)

        intersected_domain = knl.domains[0]
        for dom in knl.domains[1:]:
            if set(intersected_domain.get_var_dict()) & set(dom.get_var_dict()):
                raise RuntimeError("Intersecting all domains does not work.")

            intersected_domain, dom = isl.align_two(intersected_domain, dom)
            intersected_domain = intersected_domain & dom

        knl = knl.copy(domains=[intersected_domain])
        t_unit = t_unit.with_kernel(knl)

        # }}}

        einsum, _ = fnsm.get_a_matched_einsum(t_unit, long_dim_length=10_000)
        facts_in_feinsum_db = fnsm.query(
            einsum, self.queue.device, err_if_no_results=True
        )
        best_query = max(
            facts_in_feinsum_db,
            key=lambda q: sum(q.giga_op_info.values()) / q.runtime_in_sec,
        )
        t_unit = best_query.transform(t_unit)
        return t_unit


def deconstruct_tccg_benchmark(
    ibenchmark,
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    import feinsum.utils

    tensor_contraction = feinsum.utils.get_tccg_benchmark(ibenchmark)
    ((A, B),) = tensor_contraction.args
    A_shape = A.shape
    B_shape = B.shape
    assert all(isinstance(dim, int) for dim in A_shape)
    assert all(isinstance(dim, int) for dim in B_shape)
    return tensor_contraction.get_subscripts(), (A_shape, B_shape)


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
    spec, (a_shape, b_shape) = deconstruct_tccg_benchmark(ibenchmark)
    assert a_shape == A.shape
    assert b_shape == B.shape
    assert A.dtype == np.float64 and B.dtype == np.float64
    return actx.einsum(spec, alpha1 * A + b1, alpha2 * B + b2)


def get_nflops_for_f(ibenchmark) -> int:
    """
    Returns the number of FLOPS for the operations in :func:`f`.
    """
    import feinsum.utils
    from feinsum.measure import _get_giga_ops_from_einsum
    from pytools import product

    _, (A_shape, B_shape) = deconstruct_tccg_benchmark(ibenchmark)
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

    _, (A_shape, B_shape) = deconstruct_tccg_benchmark(ibenchmark)
    return (product(A_shape) + product(B_shape)) * 8


def create_array_context(which: str):
    if which == "numpy":
        return NumpyArrayContext()
    elif which == "jax":
        import jax

        jax.config.update("jax_enable_x64", True)
        return PytatoJAXArrayContext()
    elif which == "feinsum":
        import pyopencl as cl
        import pyopencl.tools as cl_tools

        ctx = cl.create_some_context()
        cq = cl.CommandQueue(ctx)
        allocator = cl_tools.MemoryPool(cl_tools.ImmediateAllocator(cq))
        if cq.device.name != "NVIDIA TITAN V":
            raise RuntimeError("This measurement script is written for Titan V.")

        return FeinsumArrayContext(cq, allocator)
    else:
        raise ValueError()


def sync_actx(actx: ArrayContext):
    if isinstance(actx, NumpyArrayContext):
        return
    elif isinstance(actx, PytatoJAXArrayContext):
        return
    elif isinstance(actx, PytatoPyOpenCLArrayContext):
        actx.queue.finish()
        return
    else:
        raise ValueError(f"Unkown actx: {actx}.")


def measure_flop_rate_for_f(actx: ArrayContext) -> tuple[float, ...]:
    """
    Returns a :class:`tuple`, ``flop_rate``, of 48 floating point values.
    ``flop_rate[i]`` corresponds to the measured GFLOPS corresponding to the
    computation of :func:`f` with the argument ``ibenchmark`` as ``i``.
    """
    from functools import partial

    from numpy.random import default_rng

    rng = default_rng(0)
    alpha1 = rng.random(()).item()
    alpha2 = rng.random(()).item()
    b1 = rng.random(()).item()
    b2 = rng.random(()).item()
    actx_np = create_array_context("numpy")
    measured_gflops: list[float] = []

    for ibenchmark in range(1, 49):
        f_instance = partial(f, actx=actx, ibenchmark=ibenchmark)
        compiled_f = actx.compile(f_instance)
        _, (A_shape, B_shape) = deconstruct_tccg_benchmark(ibenchmark)
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
            sync_actx(actx)
            t_start = time()
            for _ in range(N_MIN_ROUNDS):
                compiled_f(alpha1, b1, A_actx, alpha2, b2, B_actx)
            sync_actx(actx)
            t_end = time()
            total_time += t_end - t_start
            total_rounds += N_MIN_ROUNDS

        avg_time = total_time / total_rounds
        ngflops = get_nflops_for_f(ibenchmark) * 1e-9
        measured_gflops.append(ngflops / avg_time)
        print(f"Done within {ibenchmark = } with {actx}.")
    return tuple(measured_gflops)


def get_roofline_flop_rate_for_f() -> tuple[float, ...]:
    """
    Returns a :class:`tuple`, ``flop_rate``, of 48 floating point values.
    ``flop_rate[i]`` corresponds to the measured GFLOPS corresponding to the
    computation of :func:`f` with the argument ``ibenchmark`` as ``i``.  During
    the computation of the We assume the device here is the Nvidia Titan V.
    """
    roofline_gflops: list[float] = []
    for ibenchmark in range(1, 49):
        ngflops = get_nflops_for_f(ibenchmark) * 1e-9
        ngbytes = get_n_footprint_bytes_for_f(ibenchmark) * 1e-9
        npeak_gflops = 6144  # GFLOPS
        npeak_bw = 652.8  # GB/s
        roofline_gflops.append(
            ngflops / max(ngflops / npeak_gflops, ngbytes / npeak_bw),
        )
    return tuple(roofline_gflops)


def main():
    feinsum_gflops = measure_flop_rate_for_f(create_array_context("feinsum"))
    gc.collect()
    jax_gflops = measure_flop_rate_for_f(create_array_context("jax"))
    gc.collect()
    roofline_gflops = get_roofline_flop_rate_for_f()

    table = np.empty((48, 3))
    table[:, 0] = jax_gflops
    table[:, 1] = feinsum_gflops
    table[:, 2] = roofline_gflops

    print(
        tabulate(
            table,
            headers=["JAX GFLOPS", "Feinsum GFLOPS", "Roofline GFLOPS"],
            tablefmt="fancy",
        )
    )


if __name__ == "__main__":
    main()
