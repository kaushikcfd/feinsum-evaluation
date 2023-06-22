from arraycontext import (
    ArrayContext, PyOpenCLArrayContext,
    PytatoPyOpenCLArrayContext,
    EagerJAXArrayContext, PytatoJAXArrayContext,
    BatchedEinsumPytatoPyOpenCLArrayContext as BaseBatchedEinsumPytatoPyOpenCLArrayContext  # noqa: E501
)
from feinsum_evaluation.metadata import NamedAxis
from typing import Any, Callable, Optional, Type, Tuple
from time import time
from pytools.tag import Tag


def _fused_loop_name_prefix_getter(tag: Tag) -> str:
    if isinstance(tag, NamedAxis):
        return f"i{tag.name}"
    else:
        raise NotImplementedError(type(tag))


class BatchedEinsumPytatoPyOpenCLArrayContext(
    BaseBatchedEinsumPytatoPyOpenCLArrayContext
        ):
    def __init__(
        self,
        queue, allocator=None,
        *,
        compile_trace_callback: Optional[Callable[[Any, str, Any], None]] = None,
        feinsum_db: Optional[str] = None,
        log_loopy_statistics: bool = False,
    ) -> None:
        import feinsum as fnsm

        super().__init__(
            queue, allocator,
            loop_fusion_axis_tag_t=NamedAxis,
            fallback_to_no_fusion=False,
            assume_all_indirection_maps_as_non_negative=True,
            compile_trace_callback=compile_trace_callback,
            feinsum_db=fnsm.DEFAULT_DB,
            log_loopy_statistics=log_loopy_statistics,
            fused_loop_name_prefix_getter=_fused_loop_name_prefix_getter
        )


def get_actx_t_priority(actx_t):
    # lower priority => gets executed first
    if issubclass(actx_t, PytatoJAXArrayContext):
        return 10
    else:
        return 1


def instantiate_actx_t(actx_t: Type[ArrayContext]) -> ArrayContext:
    import gc
    gc.collect()

    if issubclass(actx_t, (PyOpenCLArrayContext, PytatoPyOpenCLArrayContext)):
        import pyopencl as cl
        import pyopencl.tools as cl_tools

        ctx = cl.create_some_context()
        cq = cl.CommandQueue(ctx)
        allocator = cl_tools.MemoryPool(cl_tools.ImmediateAllocator(cq))
        return actx_t(cq, allocator)
    elif issubclass(actx_t, (EagerJAXArrayContext, PytatoJAXArrayContext)):
        import os
        if os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE") != "false":
            raise RuntimeError("environment variable 'XLA_PYTHON_CLIENT_PREALLOCATE'"
                               " is not set 'false'. This is required so that"
                               " backends other than JAX can allocate buffers on the"
                               " device.")

        from jax.config import config
        config.update("jax_enable_x64", True)
        return actx_t()
    else:
        raise NotImplementedError(actx_t)


def get_wallclock_time(f: Callable[[...], Any],
                       args: Tuple[Any, ...]) -> float:
    import gc

    # {{{ warmup rounds

    i_warmup = 0
    t_warmup = 0

    while i_warmup < 20 and t_warmup < 2:
        t_start = time()
        f(*args)
        t_end = time()
        t_warmup += (t_end - t_start)
        i_warmup += 1

    # }}}

    # {{{ actual timing rounds

    i_actual = 0
    t_actual = 0

    while i_actual < 100 and t_actual < 5:

        t_start = time()
        for _ in range(40):
            f(*args)
        t_end = time()

        gc.collect()

        t_actual += (t_end - t_start)
        i_actual += 40

    # }}}

    return t_actual / i_actual

# vim: fdm=marker
