from abc import ABC, abstractmethod
import asyncio
from collections.abc import Coroutine
from typing import Any


class JobRunner(ABC):
    @abstractmethod
    def submit(self, job: Coroutine[Any, Any, None]) -> asyncio.Task:
        raise NotImplementedError


class LocalJobRunner(JobRunner):
    def submit(self, job: Coroutine[Any, Any, None]) -> asyncio.Task:
        return asyncio.create_task(job)


job_runner = LocalJobRunner()