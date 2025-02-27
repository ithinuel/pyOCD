# pyOCD debugger
# Copyright (c) 2016-2020 Arm Limited
# Copyright (c) 2022 Intel Corporation
# Copyright (c) 2022 Chris Reed
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
from __future__ import annotations

from enum import Enum
import logging
from typing import Any, List, Generator, Mapping, Optional, Sequence, TYPE_CHECKING


from .provider import TargetThread, ThreadProvider
from .common import read_c_string, HandlerModeThread
from pyocd.core import exceptions
from pyocd.core.target import Target
from pyocd.core.plugin import Plugin
from pyocd.debug.context import DebugContext
from pyocd.coresight.cortex_m_core_registers import index_for_reg
from pyocd.utility.mask import twos_complement

if TYPE_CHECKING:
    from pyocd.debug.symbols import SymbolProvider
    from pyocd.core.core_registers import (
        CoreRegisterNameOrNumberType,
    )
    from pyocd.utility.notification import Notification

# Create a logger for this module.
LOG = logging.getLogger(__name__)


class TargetList(object):
    def __init__(self, context: DebugContext, ptr: int, next_offset: int) -> None:
        self._context = context
        self._list = ptr
        self._list_node_next_offset = next_offset

    def __iter__(self) -> Generator[int, Any, None]:
        node = self._context.read32(self._list)

        while node != 0:
            try:
                yield node

                # Read next list node pointer.
                node = self._context.read32(node + self._list_node_next_offset)
            except exceptions.TransferError:  # noqa: PERF203
                LOG.warning(
                    "TransferError while reading list elements (list=0x%08x, node=0x%08x), terminating list",
                    self._list,
                    node,
                )
                node = 0


class ZephyrThreadContext(DebugContext):
    """@brief Thread context for Zephyr."""

    STACK_FRAME_OFFSETS: Mapping[int, int] = {
        0: 0,  # r0
        1: 4,  # r1
        2: 8,  # r2
        3: 12,  # r3
        12: 16,  # r12
        14: 20,  # lr
        15: 24,  # pc
        16: 28,  # xpsr
    }

    CALLEE_SAVED_OFFSETS: Mapping[int, int] = {
        4: -32,  # r4
        5: -28,  # r5
        6: -24,  # r6
        7: -20,  # r7
        8: -16,  # r8
        9: -12,  # r9
        10: -8,  # r10
        11: -4,  # r11
        13: 0,  # r13/sp
    }

    def __init__(
        self, parent: DebugContext, thread: "ZephyrThread | HandlerModeThread"
    ) -> None:
        super().__init__(parent)
        self._thread = thread
        self._has_fpu: bool = self.core.has_fpu

    def read_core_registers_raw(
        self, reg_list: Sequence[CoreRegisterNameOrNumberType]
    ) -> list[int]:
        reg_vals: List[int] = []

        is_current = self._thread.is_current
        in_exception = is_current and self._parent.read_core_register("ipsr") > 0

        # If this is the current thread and we're not in an exception, just read the live registers.
        if is_current and not in_exception:
            LOG.debug("Reading live registers")
            return self._parent.read_core_registers_raw(reg_list)

        # Because of above tests, from now on, inException implies isCurrent;
        # we are generating the thread view for the RTOS thread where the
        # exception occurred; the actual Handler Mode thread view is produced
        # by HandlerModeThread
        if in_exception:
            # Reasonable to assume PSP is still valid
            sp = int(self._parent.read_core_register("psp"))
        else:
            sp = self._thread.get_stack_pointer()
        exception_frame = 0x20

        reg_indices = [
            index_for_reg(reg) if isinstance(reg, str) else reg for reg in reg_list
        ]
        for reg in reg_indices:
            # If this is a stack pointer register, add an offset to account for the exception stack frame
            if reg == 13:
                val = sp + exception_frame
                LOG.debug("Reading register %d = 0x%x", reg, val)
                reg_vals.append(val)
                continue

            # If this is a callee-saved register, read it from the thread structure
            if isinstance(self._thread, ZephyrThread):
                callee_offset = self.CALLEE_SAVED_OFFSETS.get(reg, None)
                if callee_offset is not None:
                    try:
                        addr = (
                            self._thread.base
                            + self._thread.offsets["t_stack_ptr"]
                            + callee_offset
                        )
                        val = self._parent.read32(addr)
                        reg_vals.append(val)
                        LOG.debug(
                            "Reading callee-saved register %d at 0x%08x = 0x%x",
                            reg,
                            addr,
                            val,
                        )
                    except exceptions.TransferError:
                        reg_vals.append(0)
                    continue

            # If this is a exception stack frame register, read it from the stack
            stack_frame_offset = self.STACK_FRAME_OFFSETS.get(reg, None)
            if stack_frame_offset is not None:
                try:
                    addr = int(sp + stack_frame_offset)
                    val = self._parent.read32(addr)
                    reg_vals.append(val)
                    LOG.debug(
                        "Reading stack frame register %d at 0x%08x = 0x%x",
                        reg,
                        addr,
                        val,
                    )
                except exceptions.TransferError:
                    reg_vals.append(0)
                continue

            # If we get here, this is a register not in any of the dictionaries
            val = self._parent.read_core_register_raw(reg)
            LOG.debug("Reading live register %d = 0x%x", reg, val)
            reg_vals.append(val)
            continue

        return reg_vals


class ThreadState(Enum):
    READY = 0
    PENDING = 1 << 1
    PRESTART = 1 << 2
    DEAD = 1 << 3
    SUSPENDED = 1 << 4
    POLLING = 1 << 5
    RUNNING = 1 << 6


_STATE_NAMES = {
    ThreadState.READY: "Ready",
    ThreadState.PENDING: "Pending",
    ThreadState.PRESTART: "Prestart",
    ThreadState.DEAD: "Dead",
    ThreadState.SUSPENDED: "Suspended",
    ThreadState.POLLING: "Polling",
    ThreadState.RUNNING: "Running",
}


class ZephyrThread(TargetThread):
    """@brief A Zephyr task."""

    def __init__(
        self,
        target_context: DebugContext,
        provider: "ZephyrThreadProvider",
        base: int,
        offsets: Mapping[str, int],
    ) -> None:
        super().__init__()
        self._target_context = target_context
        self._provider = provider
        self.base = base
        self._thread_context = ZephyrThreadContext(self._target_context, self)
        self.offsets = offsets
        self._state: ThreadState = ThreadState.READY
        self._priority = 0
        self._name = "Unnamed"

        try:
            self.update_info()
        except exceptions.TransferError:
            LOG.debug("Transfer error while reading thread info")

    def get_stack_pointer(self) -> int:
        # Get stack pointer saved in thread struct.
        addr = self.base + self.offsets["t_stack_ptr"]
        try:
            return self._target_context.read32(addr)
        except exceptions.TransferError:
            LOG.debug(
                "Transfer error while reading thread's stack pointer @ 0x%08x", addr
            )
            return 0

    def update_info(self) -> None:
        try:
            self._priority = twos_complement(
                self._target_context.read8(self.base + self.offsets["t_prio"]),
                width=8,
            )
            self._state = ThreadState(
                self._target_context.read8(self.base + self.offsets["t_state"])
            )

            if self._provider.version > 0:
                addr = self.base + self.offsets["t_name"]
                self._name = read_c_string(self._target_context, addr)

        except exceptions.TransferError:
            LOG.debug("Transfer error while reading thread info")

    @property
    def state(self) -> ThreadState:
        return self._state

    @state.setter
    def state(self, value: ThreadState) -> None:
        self._state = value

    @property
    def priority(self) -> int:
        return self._priority

    @property
    def unique_id(self) -> int:
        return self.base

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "%s; Priority %d" % (
            _STATE_NAMES.get(self.state, "UNKNOWN"),
            self.priority,
        )

    @property
    def is_current(self) -> bool:
        return self._provider.get_actual_current_thread_id() == self.unique_id

    @property
    def context(self) -> ZephyrThreadContext:
        return self._thread_context

    def __str__(self) -> str:
        return "<ZephyrThread@0x%08x id=%x name=%s>" % (
            id(self),
            self.unique_id,
            self.name,
        )

    def __repr__(self) -> str:
        return str(self)


class ZephyrThreadProvider(ThreadProvider[ZephyrThread | HandlerModeThread]):
    """@brief Thread provider for Zephyr."""

    ## Required Zephyr symbols.
    ZEPHYR_SYMBOLS: Sequence[str] = [
        "_kernel",
        "_kernel_thread_info_offsets",
        "_kernel_thread_info_size_t_size",
    ]

    ZEPHYR_OFFSETS: Sequence[str] = [
        "version",
        "k_curr_thread",
        "k_threads",
        "t_entry",
        "t_next_thread",
        "t_state",
        "t_user_options",
        "t_prio",
        "t_stack_ptr",
        "t_name",
    ]

    def __init__(self, target: Target) -> None:
        super().__init__(target)
        self._symbols: Mapping[str, int] = {}
        self._offsets: Mapping[str, int] | None = None
        self._version: int = 0
        self._all_threads: int | None = None
        self._curr_thread: int | None = None
        self._threads: Mapping[int, ZephyrThread | HandlerModeThread] = {}

    def init(self, symbol_provider: SymbolProvider) -> bool:
        # Lookup required symbols.
        self._symbols = self._lookup_symbols(self.ZEPHYR_SYMBOLS, symbol_provider, True)
        if len(self._symbols) == 0:
            return False
        if len(self._symbols) != len(self.ZEPHYR_SYMBOLS):
            LOG.warning(
                "Zephyr kernel detected. Build your Zephyr application with `CONFIG_DEBUG_THREAD_INFO=y` to "
                "enable thread awareness."
            )
            return False

        self._update()
        self._target.session.subscribe(  # pyright: ignore
            self.event_handler, Target.Event.POST_FLASH_PROGRAM
        )
        self._target.session.subscribe(self.event_handler, Target.Event.POST_RESET)  # pyright: ignore

        return True

    def _get_offsets(self) -> Optional[dict[str, int]]:
        # Read the kernel and thread structure member offsets
        size: int = self._target_context.read8(
            self._symbols["_kernel_thread_info_size_t_size"]
        )
        LOG.debug("_kernel_thread_info_size_t_size = %d", size)
        if size != 4:
            LOG.error("Unsupported _kernel_thread_info_size_t_size")
            return None

        offsets: dict[str, int] = {}
        for index, name in enumerate(self.ZEPHYR_OFFSETS):
            offset = self._symbols["_kernel_thread_info_offsets"] + index * size
            offsets[name] = self._target_context.read32(offset)
            LOG.debug("%s = 0x%04x", name, offsets[name])

        return offsets

    def _update(self) -> None:
        self._offsets = self._get_offsets()

        if self._offsets is None:
            self._version = 0
            self._all_threads = None
            self._curr_thread = None
            LOG.debug("_offsets, _all_threads, and _curr_thread are invalid")
        else:
            self._version = self._offsets["version"]
            self._all_threads = self._symbols["_kernel"] + self._offsets["k_threads"]
            self._curr_thread = (
                self._symbols["_kernel"] + self._offsets["k_curr_thread"]
            )
            LOG.debug(
                "version = %d, _all_threads = 0x%08x, _curr_thread = 0x%08x",
                self._version,
                self._all_threads,
                self._curr_thread,
            )

    def invalidate(self) -> None:
        self._threads = {}

    def event_handler(self, notification: Notification) -> None:
        if notification.event == Target.Event.POST_RESET:
            LOG.debug("Invalidating threads list: %s", repr(notification))
            self.invalidate()

        elif notification.event == Target.Event.POST_FLASH_PROGRAM:  # pyright: ignore
            self._update()

    def _build_thread_list(self) -> None:
        if (
            self._offsets is None
            or self._curr_thread is None
            or self._all_threads is None
        ):
            raise exceptions.InternalError()

        all_threads = TargetList(
            self._target_context, self._all_threads, self._offsets["t_next_thread"]
        )
        new_threads: dict[int, ZephyrThread | HandlerModeThread] = {}

        current_thread: int = self._target_context.read32(self._curr_thread)
        LOG.debug("currentThread = 0x%08x", current_thread)

        for thread_base in all_threads:
            try:
                # Reuse existing thread objects.
                if thread_base in self._threads:
                    t = self._threads[thread_base]

                    # Ask the thread object to update its state and priority.
                    if isinstance(t, ZephyrThread):
                        t.update_info()
                else:
                    t = ZephyrThread(
                        self._target_context, self, thread_base, self._offsets
                    )

                # Set thread state.
                if thread_base == current_thread and isinstance(t, ZephyrThread):
                    t.state = ThreadState.RUNNING

                LOG.debug("Thread 0x%08x (%s)", thread_base, t.name)
                new_threads[t.unique_id] = t
            except exceptions.TransferError:  # noqa: PERF203
                LOG.debug("TransferError while examining thread 0x%08x", thread_base)

        # Create fake handler mode thread.
        if self._target_context.read_core_register("ipsr") > 0:
            LOG.debug("creating handler mode thread")
            t = HandlerModeThread(self._target_context, self)
            new_threads[t.unique_id] = t

        self._threads = new_threads

    def get_threads(self) -> List[ZephyrThread | HandlerModeThread]:
        if not self.is_enabled:
            return []
        self.update_threads()
        return list(self._threads.values())

    def get_thread(self, thread_id: int) -> ZephyrThread | HandlerModeThread | None:
        if not self.is_enabled:
            return None
        self.update_threads()
        return self._threads.get(thread_id, None)

    @property
    def is_enabled(self) -> bool:
        return self.get_is_running()

    @property
    def current_thread(self) -> ZephyrThread | HandlerModeThread | None:
        if not self.is_enabled:
            return None
        self.update_threads()
        thread_id = self.get_current_thread_id()
        if thread_id is None:
            return None
        return self._threads.get(thread_id, None)

    def is_valid_thread_id(self, thread_id: int) -> bool:
        if not self.is_enabled:
            return False
        self.update_threads()
        return thread_id in self._threads

    def get_current_thread_id(self) -> int | None:
        if not self.is_enabled:
            return None
        if self._target_context.read_core_register("ipsr") > 0:
            return HandlerModeThread.UNIQUE_ID
        return self.get_actual_current_thread_id()

    def get_actual_current_thread_id(self) -> int | None:
        if not self.is_enabled or self._curr_thread is None:
            return None
        return self._target_context.read32(self._curr_thread)

    def get_is_running(self) -> bool:
        return self._offsets is not None

    @property
    def version(self) -> int:
        return self._version


class ZephyrPlugin(Plugin):
    """@brief Plugin class for the Zephyr RTOS."""

    def load(self) -> type[ZephyrThreadProvider]:
        return ZephyrThreadProvider

    @property
    def name(self) -> str:
        return "zephyr"

    @property
    def description(self) -> str:
        return "Zephyr"
