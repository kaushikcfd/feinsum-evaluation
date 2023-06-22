import feinsum as f
import numpy as np
from typing import Tuple
from tabulate import tabulate


def _get_index_shape_str(i: int) -> Tuple[str, str]:
    if i == 1:
        return ("abc-bda-dc", "312 312 24 312")
    elif i == 2:
        return ("abc-dca-bd", "312 24 296 312")
    elif i == 3:
        return ("abcd-dbea-ec", "72 72 24 72 72")
    elif i == 4:
        return ("abcd-deca-be", "72 24 72 72 72")
    elif i == 5:
        return ("abcd-ebad-ce", "72 72 24 72 72")
    elif i == 6:
        return ("abcde-efbad-cf", "48 32 24 32 48 32")
    elif i == 7:
        return ("abcde-ecbfa-fd", "48 32 32 24 48 48")
    elif i == 8:
        return ("abcde-efcad-bf", "48 24 32 32 48 32")
    elif i == 9:
        return ("abcd-ea-ebcd", "72 72 72 72 72")
    elif i == 10:
        return ("abcd-eb-aecd", "72 72 72 72 72")
    elif i == 11:
        return ("abcd-ec-abed", "72 72 72 72 72")
    elif i == 12:
        return ("ab-ac-cb", "5136 5120 5136")
    elif i == 13:
        return ("ab-acd-dbc", "312 296 296 312")
    elif i == 14:
        return ("ab-cad-dcb", "312 296 312 312")
    elif i == 15:
        return ("abc-acd-db", "312 296 296 312")
    elif i == 16:
        return ("abc-ad-bdc", "312 312 296 296")
    elif i == 17:
        return ("abc-adc-bd", "312 312 296 296")
    elif i == 18:
        return ("abc-adc-db", "312 296 296 312")
    elif i == 19:
        return ("abc-adec-ebd", "72 72 72 72 72")
    elif i == 20:
        return ("abcd-aebf-dfce", "72 72 72 72 72 72")
    elif i == 21:
        return ("abcd-aebf-fdec", "72 72 72 72 72 72")
    elif i == 22:
        return ("abcd-aecf-bfde", "72 72 72 72 72 72")
    elif i == 23:
        return ("abcd-aecf-fbed", "72 72 72 72 72 72")
    elif i == 24:
        return ("abcd-aedf-bfce", "72 72 72 72 72 72")
    elif i == 25:
        return ("abcd-aedf-fbec", "72 72 72 72 72 72")
    elif i == 26:
        return ("abcd-aefb-fdce", "72 72 72 72 72 72")
    elif i == 27:
        return ("abcd-aefc-fbed", "72 72 72 72 72 72")
    elif i == 28:
        return ("abcd-eafb-fdec", "72 72 72 72 72 72")
    elif i == 29:
        return ("abcd-eafc-bfde", "72 72 72 72 72 72")
    elif i == 30:
        return ("abcd-eafd-fbec", "72 72 72 72 72 72")
    elif i == 31:
        return ("abcdef-dega-gfbc", "24 16 16 24 16 16 24")
    elif i == 32:
        return ("abcdef-degb-gfac", "24 16 16 24 16 16 24")
    elif i == 33:
        return ("abcdef-degc-gfab", "24 16 16 24 16 16 24")
    elif i == 34:
        return ("abcdef-dfga-gebc", "24 16 16 24 16 16 24")
    elif i == 35:
        return ("abcdef-dfgb-geac", "24 16 16 24 16 16 24")
    elif i == 36:
        return ("abcdef-dfgc-geab", "24 16 16 24 16 16 24")
    elif i == 37:
        return ("abcdef-efga-gdbc", "24 16 16 16 24 16 24")
    elif i == 38:
        return ("abcdef-efgb-gdac", "24 16 16 16 24 16 24")
    elif i == 39:
        return ("abcdef-efgc-gdab", "24 16 16 16 24 16 24")
    elif i == 40:
        return ("abcdef-gdab-efgc", "24 16 16 16 24 16 24")
    elif i == 41:
        return ("abcdef-gdac-efgb", "24 16 16 16 24 16 24")
    elif i == 42:
        return ("abcdef-gdbc-efga", "24 16 16 16 24 16 24")
    elif i == 43:
        return ("abcdef-geab-dfgc", "24 16 16 24 16 16 24")
    elif i == 44:
        return ("abcdef-geac-dfgb", "24 16 16 24 16 16 24")
    elif i == 45:
        return ("abcdef-gebc-dfga", "24 16 16 24 16 16 24")
    elif i == 46:
        return ("abcdef-gfab-degc", "24 16 16 24 16 16 24")
    elif i == 47:
        return ("abcdef-gfac-degb", "24 16 16 24 16 16 24")
    elif i == 48:
        return ("abcdef-gfbc-dega", "24 16 16 24 16 16 24")
    else:
        raise NotImplementedError(i)


def _parse_tccg_benchmark(subscript: str, shape: str) -> f.FusedEinsum:
    output, inA, inB = subscript.split("-")
    axis_lens = {chr(97+i): int(axis_len)
                 for i, axis_len in enumerate(shape.split(" "))}

    shapeA = [axis_lens[idx] for idx in inA]
    shapeB = [axis_lens[idx] for idx in inB]

    return f.einsum(f"{inA},{inB}->{output}",
                    f.array(shapeA, np.float64),
                    f.array(shapeB, np.float64))


def _get_time_to_canonicalize_in_secs(expr: f.FusedEinsum) -> float:
    import timeit
    ncount = 1000
    return timeit.timeit(lambda: f.canonicalize_einsum(expr), number=ncount)/ncount


def plot_time_to_canonicalize():
    table = []
    for i in range(1, 49):
        einsum = _parse_tccg_benchmark(*_get_index_shape_str(i))
        time_in_ms = _get_time_to_canonicalize_in_secs(einsum) * 1000
        table.append([r"\texttt{" + einsum.get_subscripts() + "}",
                      f"{time_in_ms:.2f}"])

    print(tabulate(table, headers=["einsum", "Time to canonicalize (in msecs)"],
                   tablefmt="latex_raw"))


if __name__ == "__main__":
    plot_time_to_canonicalize()
