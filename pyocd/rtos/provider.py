# pyOCD debugger
# Copyright (c) 2016 Arm Limited
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
import logging
from typing import Generic, List, Literal, TypeVar, overload, Sequence

from pyocd.core.target import Target
from pyocd.debug.symbols import SymbolProvider

LOG = logging.getLogger(__name__)


class TargetThread(object):
    """@brief Base class representing a thread on the target."""

    def __init__(self):
        pass

    @property
    def unique_id(self):
        raise NotImplementedError()

    @property
    def name(self):
        raise NotImplementedError()

    @property
    def description(self):
        raise NotImplementedError()

    @property
    def is_current(self) -> bool:
        raise NotImplementedError()

    @property
    def context(self):
        raise NotImplementedError()


T = TypeVar("T", bound=TargetThread)


class ThreadProvider(Generic[T], ABC):
    """@brief Base class for RTOS support plugins."""

    def __init__(self, target: Target) -> None:
        self._target = target
        self._target_context = self._target.get_target_context()
        self._last_run_token = -1
        self._read_from_target = False

    @overload
    def _lookup_symbols(
        self,
        symbolList: Sequence[str],
        symbolProvider: SymbolProvider,
        allowPartial: Literal[False],
    ) -> dict[str, int] | None: ...

    @overload
    def _lookup_symbols(
        self,
        symbolList: Sequence[str],
        symbolProvider: SymbolProvider,
        allowPartial: Literal[True],
    ) -> dict[str, int]: ...

    def _lookup_symbols(
        self,
        symbolList: Sequence[str],
        symbolProvider: SymbolProvider,
        allowPartial: bool = False,
    ) -> dict[str, int] | None:
        syms: dict[str, int] = {}
        for name in symbolList:
            addr = symbolProvider.get_symbol_value(name)
            LOG.debug(
                "Value for symbol %s = %s",
                name,
                hex(addr) if addr is not None else "<none>",
            )
            if addr is not None:
                syms[name] = addr
            elif not allowPartial:
                return None
        return syms

    def init(self, symbolProvider: SymbolProvider) -> bool:
        """@retval True The provider was successfully initialzed.
        @retval False The provider could not be initialized successfully.
        """
        raise NotImplementedError()

    def _build_thread_list(self) -> None:
        raise NotImplementedError()

    def _is_thread_list_dirty(self) -> bool:
        token = self._target.run_token
        if token == self._last_run_token:
            # Target hasn't run since we last updated threads, so there is nothing to do.
            return False
        self._last_run_token = token
        return True

    def update_threads(self):
        if self._is_thread_list_dirty() and self._read_from_target:
            self._build_thread_list()

    @abstractmethod
    def get_threads(self) -> List[T]: ...

    @abstractmethod
    def get_thread(self, threadId: int) -> T | None: ...

    def invalidate(self) -> None:
        raise NotImplementedError()

    @property
    def read_from_target(self) -> bool:
        return self._read_from_target

    @read_from_target.setter
    def read_from_target(self, value: bool):
        if value != self._read_from_target:
            self.invalidate()
        self._read_from_target = value

    @property
    def is_enabled(self):
        raise NotImplementedError()

    @property
    def current_thread(self):
        raise NotImplementedError()

    def is_valid_thread_id(self, threadId: int) -> bool:
        raise NotImplementedError()

    def get_current_thread_id(self) -> int | None:
        """From GDB's point of view, where Handler Mode is a thread"""
        raise NotImplementedError()

    def get_actual_current_thread_id(self) -> int | None:
        """From OS's point of view, so the current OS thread even in Handler Mode"""
        raise NotImplementedError()
