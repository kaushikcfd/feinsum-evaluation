from arraycontext import (ArrayContext, PyOpenCLArrayContext,
                          PytatoPyOpenCLArrayContext,
                          EagerJAXArrayContext, PytatoJAXArrayContext)
from typing import Any, Callable, Mapping, Type, Tuple
from time import time


def get_actx_t_priority(actx_t):
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
        if os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] != "false":
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
                       args: Tuple[Any, ...],
                       kwargs: Mapping[str, Any]) -> float:

    # {{{ warmup rounds

    i_warmup = 0
    t_warmup = 0

    while i_warmup < 20 and t_warmup < 2:
        t_start = time()
        f(*args, **kwargs)
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
            f(*args, **kwargs)
        t_end = time()

        t_actual += (t_end - t_start)
        i_actual += 40

    # }}}

    return t_actual / i_actual

# vim: fdm=marker
