import timeit

import feinsum as f
from tabulate import tabulate


def get_time_to_canonicalize_in_msecs(expr: f.BatchedEinsum) -> float:
    return (
        timeit.timeit(lambda: f.canonicalize_einsum(expr), number=1000)
    )


def print_time_to_canonicalize_for_tccg_benchmark():
    from feinsum.utils import get_tccg_benchmark

    table = []
    for i in range(1, 49):
        einsum = get_tccg_benchmark(i)
        time_in_ms = get_time_to_canonicalize_in_msecs(einsum)
        print(f"Done with benchmark {i}.")
        table.append([i, f"{time_in_ms:.2f}"])

    print(
        tabulate(
            table,
            headers=["ibenchmark", "Time to canonicalize (in msecs)"],
            tablefmt="fancy",
        )
    )


if __name__ == "__main__":
    print_time_to_canonicalize_for_tccg_benchmark()
