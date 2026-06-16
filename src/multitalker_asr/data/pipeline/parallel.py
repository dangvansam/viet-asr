"""Thread-pool helpers for client-side fan-out across HTTP service calls.

The pipeline orchestrator is torch-free and network-bound: every stage calls
remote model services over HTTP, which releases the GIL while waiting. Running
those calls on a ThreadPoolExecutor saturates the (already batching) vLLM/service
backends without changing results — order is preserved by index.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def parallel_map(
    fn: Callable[[T], R],
    items: List[T],
    workers: int,
    ordered: bool = True,
) -> List[R]:
    """Map fn over items on a thread pool, returning results in input order.

    workers <= 1 or a single item runs inline (no pool). Exceptions propagate
    from the corresponding item, matching a serial loop's failure semantics.
    """
    n = len(items)
    if n == 0:
        return []
    effective = max(1, min(int(workers), n))
    if effective == 1:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=effective) as pool:
        return list(pool.map(fn, items))


def parallel_call(
    calls: List[Callable[[], R]],
    workers: int,
) -> List[R]:
    """Run zero-arg callables concurrently, results in submission order."""
    return parallel_map(lambda c: c(), calls, workers)
