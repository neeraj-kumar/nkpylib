"""Various thread-related utilities.

Note that I'm starting over in 2024, as the old threadutils.py (no underscore) was getting too messy
and most of it is outdated.
"""

from __future__ import annotations

import logging
import threading
import time

from queue import Queue, Empty
from typing import Any, Iterable

logger = logging.getLogger(__name__)

def chained_producer_consumers(functions: list[function],
                               sleep_interval: float=0.1,
                               get_timeout: float=0.5) -> Iterable[Any]:
    """
    Start a chain of producer-consumer operations.

    Args:
    - functions: A list of functions each with one output, where the first function takes no input,
      and each subsequent function takes an item from the previous function.
    - sleep_interval: The interval between producing items in the producer threads.
    - get_timeout: The timeout for getting items from the input queues.

    Yields:
    - Items produced by the last function in the chain.

    The producer-consumer chain is implemented using a series of threads and queues, where each
    thread consumes items from the previous thread's output queue, processes them with the function,
    and puts the results into the next thread's input queue. The final output queue's items are
    yielded by the main processing thread of this function.

    Once the first function exhausts its input, it puts a Sentinel into the output queue to signal
    the end. Each subsequent function stops when it receives a Sentinel from the input queue, and
    also puts a Sentinel into the output queue to keep signalling forward.

    If a function raises a StopIteration, we need to stop all upstream functions as well. We do this
    by setting a stop_idx to the index of the function that raised the StopIteration, and each
    function checks its own index against this stop_idx before processing an item.
    """
    lock = threading.Lock()
    stop_idx = -1
    sentinel = object()

    def producer_consumer_wrapper(func, in_q, out_q, idx):
        """Wrapper for each function in the chain.

        This consumes items from the input queue, processes them with the function,
        and puts the results into the output queue.

        If `in_q` is None, then the function is assumed to be the initial producer.

        If the function raises a StopIteration, we set the stop_idx to our index.
        If the stop_idx is set to a value at our index or higher, we finish.
        """
        nonlocal stop_idx
        logger.debug(f'producer_consumer_wrapper: {func} {in_q} {out_q} {idx} {stop_idx}')
        try:
            if in_q is None:
                for item in func():
                    out_q.put(item)
                    with lock:
                        if stop_idx >= idx:
                            break
                # put the sentinel to signal the end
                out_q.put(sentinel)
            else:
                while True:
                    try:
                        item = in_q.get(timeout=get_timeout)
                        in_q.task_done()
                        if item is sentinel:
                            out_q.put(sentinel)
                            break
                        result = func(item)
                        out_q.put(result)
                    except Empty:
                        continue
                    with lock:
                        if stop_idx >= idx:
                            break
        except StopIteration:
            with lock:
                stop_idx = max(stop_idx, idx)
            out_q.put(sentinel)
            logger.debug(f'stopping thread for {func} {in_q} {out_q} {idx} {stop_idx}')
        logger.debug(f'exiting producer_consumer_wrapper for {func} {in_q} {out_q} {idx} {stop_idx}')

    queues: list[Queue] = [Queue() for _ in range(len(functions))]

    # Start all but the last function in separate threads
    threads = []
    for i in range(len(functions)):
        kwargs = dict(func=functions[i], in_q=None, out_q=queues[i], idx=i)
        if i > 0:
            kwargs['in_q'] = queues[i-1]
        logger.debug(f'starting thread {i} with kwargs {kwargs}')
        thread = threading.Thread(target=producer_consumer_wrapper, kwargs=kwargs)
        thread.start()
        threads.append(thread)

    # Yield results from the last queue
    while threads or queues[-1].qsize() > 0:
        try:
            result = queues[-1].get(timeout=get_timeout)
            queues[-1].task_done()
            if result is sentinel:
                break
            yield result
            # check threads for completion
            threads = [t for t in threads if t.is_alive()]
            logger.debug(f'yielded result, threads left: {len(threads)}')
        except Empty:
            continue
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    # example usage
    logging.basicConfig(level=logging.DEBUG)
    def example_producer():
        for i in range(20):
            logger.debug(f"Starting item {i}")
            yield f"item {i}"

    def example_processor(item):
        logger.debug(f"Processing {item}")
        return f"processed {item}"

    def example_final_processor(item):
        if item == "processed item 10": raise StopIteration
        return item.upper()

    # Function chain
    functions = [example_producer, example_processor, example_final_processor]

    # Start the chained producer-consumer processing
    for final_result in chained_producer_consumers(functions, sleep_interval=0.1):
        logger.debug(f"Final output: {final_result}")
