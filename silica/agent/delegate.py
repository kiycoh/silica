"""Fan-out sub-agent delegation via ThreadPoolExecutor.

From SILICA.md §8.4:
  tasks = list of rendered payloads; run_one(task) -> raw output.
  Hard-stop: if len(tasks) > 10 raise, don't truncate.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable


def delegate(
    tasks: list[dict],
    run_one: Callable[[dict], Any],
    max_workers: int = 7,
) -> list[Any]:
    """Fan-out tasks to parallel sub-agent workers.

    Args:
        tasks: list of rendered payload dicts
        run_one: callable that processes a single task
        max_workers: max parallel workers (default 7, hard cap 10)

    Returns:
        list of results in task order

    Raises:
        RuntimeError: if len(tasks) > 10 (hard-stop, must repartition)
    """
    if len(tasks) > 10:
        raise RuntimeError(
            f"fan-out {len(tasks)} > max 10: ripartizionare il payload"
        )
    if not tasks:
        return []

    import logging
    import litellm.exceptions

    from silica.agent.llm import retry_transient

    logger = logging.getLogger(__name__)

    TRANSIENT_EXCEPTIONS = (
        litellm.exceptions.RateLimitError,
        litellm.exceptions.APIConnectionError,
        litellm.exceptions.ServiceUnavailableError,
        litellm.exceptions.InternalServerError,
        litellm.exceptions.Timeout,
        ConnectionError,
        TimeoutError,
    )

    def run_one_with_retry(task: dict) -> Any:
        try:
            # base_delay=0.5 keeps the historical 1/2/4/8s ladder (+jitter).
            return retry_transient(
                lambda: run_one(task), TRANSIENT_EXCEPTIONS,
                attempts=5, base_delay=0.5, jitter=0.5,
            )
        except TRANSIENT_EXCEPTIONS as e:
            return {"error": f"LLM transient error after 5 attempts: {e}"}
        except Exception as e:
            logger.error("Permanent LLM or execution error: %s", e)
            return {"error": f"LLM permanent error: {e}"}

    with ThreadPoolExecutor(max_workers=min(max_workers, len(tasks))) as ex:
        return list(ex.map(run_one_with_retry, tasks))
