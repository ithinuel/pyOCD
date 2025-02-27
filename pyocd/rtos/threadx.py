# pyOCD debugger
# Copyright (c) 2020-2021 Federico Zuccardi Merli
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

from typing import Any, Generator, Mapping, Sequence, TYPE_CHECKING

from .provider import TargetThread, ThreadProvider
from .common import read_c_string, HandlerModeThread, EXC_RETURN_EXT_FRAME_MASK
from pyocd.core import exceptions
from pyocd.core.target import Target
from pyocd.core.plugin import Plugin
from pyocd.debug.context import DebugContext
from pyocd.coresight.cortex_m_core_registers import index_for_reg
from pyocd.coresight.core_ids import CoreArchitecture

import logging

if TYPE_CHECKING:
    from pyocd.utility.notification import Notification
    from pyocd.debug.symbols import SymbolProvider
    from pyocd.core.core_registers import CoreRegisterNameOrNumberType

TX_THREAD_ID = 0x54485244  # 'THRD'
THREAD_ID_OFFSET = 0
THREAD_STACK_POINTER_OFFSET = 8
# All the following offset may be messed up if thread extensions are defined.
# They should be read somehow from the elf.
THREAD_NAME_OFFSET = 40
THREAD_PRIORITY_OFFSET = 44
THREAD_STATE_OFFSET = 48
THREAD_NEXT_OFFSET = 136

# Create a logger for this module.
LOG = logging.getLogger(__name__)


class TargetList(object):
    def __init__(self, context: DebugContext, ptr: int) -> None:
        self._context = context
        self._list = ptr

    def __iter__(self) -> Generator[int, Any, None]:
        next_node = 0
        head = self._context.read32(self._list)
        node = head
        is_valid = head != 0

        while is_valid and next_node != head:
            try:
                # Check if this is really a thread
                if self._context.read32(node) == TX_THREAD_ID:
                    # Yields the thread pointer.
                    yield node
                else:
                    # Something is wrong. Might depend on a thread extension
                    is_valid = False
                    LOG.warning(
                        "Wrong thread ID found. Memory corruption or unknown extensions"
                    )

                next_node = self._context.read32(node + THREAD_NEXT_OFFSET)
                node = next_node
            except exceptions.TransferError:  # noqa: PERF203
                LOG.warning(
                    "TransferError while reading list elements (list=0x%08x, node=0x%08x), terminating list",
                    self._list,
                    node,
                )
                is_valid = False


class ThreadXThreadContext(DebugContext):
    """@brief Thread context for ThreadX."""

    # SP/PSP are handled specially, so it is not in these dicts.

    # Offsets are relative to stored SP in a task switch block, for the
    # combined software + hardware stacked registers. In exception case,
    # software registers are not stacked, so appropriate amount must be
    # subtracted.
    NOFPU_REGISTER_OFFSETS: Mapping[int, int] = {
        # Software stacked
        -1: 0,  # lr (exception)
        4: 4,  # r4
        5: 8,  # r5
        6: 12,  # r6
        7: 16,  # r7
        8: 20,  # r8
        9: 24,  # r9
        10: 28,  # r10
        11: 32,  # r11
        # Hardware stacked
        0: 36,  # r0
        1: 40,  # r1
        2: 44,  # r2
        3: 48,  # r3
        12: 52,  # r12
        14: 56,  # lr (thread)
        15: 60,  # pc
        16: 64,  # xpsr
    }

    # Cortex-m0 port of threadx reverses r4-7 and r8-11
    NOFPU_REGISTER_OFFSETS_V6M: Mapping[int, int] = {
        # Software stacked
        4: 20,  # r4
        5: 24,  # r5
        6: 28,  # r6
        7: 32,  # r7
        8: 4,  # r8
        9: 8,  # r9
        10: 12,  # r10
        11: 16,  # r11
    }

    FPU_REGISTER_OFFSETS: Mapping[int, int] = {
        # Software stacked
        -1: 0,  # lr (exception)
        0x50: 4,  # s16
        0x51: 8,  # s17
        0x52: 12,  # s18
        0x53: 16,  # s19
        0x54: 20,  # s20
        0x55: 24,  # s21
        0x56: 28,  # s22
        0x57: 32,  # s23
        0x58: 36,  # s24
        0x59: 40,  # s25
        0x5A: 44,  # s26
        0x5B: 48,  # s27
        0x5C: 52,  # s28
        0x5D: 56,  # s29
        0x5E: 60,  # s30
        0x5F: 64,  # s31
        4: 68,  # r4
        5: 72,  # r5
        6: 76,  # r6
        7: 80,  # r7
        8: 84,  # r8
        9: 88,  # r9
        10: 92,  # r10
        11: 96,  # r11
        # Hardware stacked
        0: 100,  # r0
        1: 104,  # r1
        2: 108,  # r2
        3: 112,  # r3
        12: 116,  # r12
        14: 120,  # lr
        15: 124,  # pc
        16: 128,  # xpsr
        0x40: 132,  # s0
        0x41: 136,  # s1
        0x42: 140,  # s2
        0x43: 144,  # s3
        0x44: 148,  # s4
        0x45: 152,  # s5
        0x46: 156,  # s6
        0x47: 160,  # s7
        0x48: 164,  # s8
        0x49: 168,  # s9
        0x4A: 172,  # s10
        0x4B: 176,  # s11
        0x4C: 180,  # s12
        0x4D: 184,  # s13
        0x4E: 188,  # s14
        0x4F: 192,  # s15
        33: 196,  # fpscr
        # (reserved word: 200)
    }

    def __init__(self, parent: DebugContext, thread: "ThreadXThread") -> None:
        super().__init__(parent)
        self._thread = thread
        self._has_fpu = self.core.has_fpu
        if self.core.architecture != CoreArchitecture.ARMv6M:
            # Use the default offsets for this istance
            self._nofpu_register_offsets = dict(self.NOFPU_REGISTER_OFFSETS)
        else:
            # Use a copy with the Cortex-M0 specific offsets for this istance
            self._nofpu_register_offsets = dict(self.NOFPU_REGISTER_OFFSETS)
            self._nofpu_register_offsets.update(self.NOFPU_REGISTER_OFFSETS_V6M)

    def read_core_registers_raw(  # noqa: C901
        self, reg_list: Sequence[CoreRegisterNameOrNumberType]
    ) -> list[int]:
        reg_vals: list[int] = []

        is_current = self._thread.is_current
        in_exception = is_current and self._parent.read_core_register("ipsr") > 0

        # If this is the current thread and we're not in an exception, just read the live registers.
        if is_current and not in_exception:
            return self._parent.read_core_registers_raw(reg_list)

        # Because of above tests, from now on, inException implies isCurrent;
        # we are generating the thread view for the RTOS thread where the
        # exception occurred; the actual Handler Mode thread view is produced
        # by HandlerModeThread
        if in_exception:
            # Reasonable to assume PSP is still valid
            sp = self._parent.read_core_register_raw("psp")
        else:
            sp = self._thread.get_stack_pointer()

        # Determine which register offset table to use and the offsets past the saved state.
        hw_stacked = 0x20
        sw_stacked = 0x24
        table: dict[int, int] = self._nofpu_register_offsets
        if self._has_fpu:
            try:
                if in_exception and self.core.is_vector_catch():
                    # Vector catch has just occurred, take live LR
                    exception_lr = self._parent.read_core_register_raw("lr")
                else:
                    # Read stacked exception return LR.
                    offset = self.FPU_REGISTER_OFFSETS[-1]
                    exception_lr = self._parent.read32(sp + offset)

                # Check bit 4 of the exception LR to determine if FPU registers were stacked.
                if (exception_lr & EXC_RETURN_EXT_FRAME_MASK) == 0:
                    table = dict(self.FPU_REGISTER_OFFSETS)
                    hw_stacked = 0x68
                    sw_stacked = 0x64
            except exceptions.TransferError:
                LOG.debug("Transfer error while reading thread's saved LR")

        reg_indices = [
            index_for_reg(reg) if isinstance(reg, str) else reg for reg in reg_list
        ]
        for reg in reg_indices:
            # Must handle stack pointer specially.
            if reg == 13:
                if in_exception:
                    reg_vals.append(sp + hw_stacked)
                else:
                    reg_vals.append(sp + sw_stacked + hw_stacked)
                continue

            # Look up offset for this register on the stack.
            sp_offset = table.get(reg, None)
            if sp_offset is None:
                reg_vals.append(self._parent.read_core_register_raw(reg))
                continue
            if in_exception:
                sp_offset -= sw_stacked

            try:
                if sp_offset >= 0:
                    reg_vals.append(self._parent.read32(sp + sp_offset))
                else:
                    # Not available - try live one
                    reg_vals.append(self._parent.read_core_register_raw(reg))
            except exceptions.TransferError:
                reg_vals.append(0)

        return reg_vals


class ThreadXThread(TargetThread):
    """@brief A ThreadX task."""

    STATE_NAMES: Mapping[int, str] = {
        0: "Ready",
        1: "Completed",
        2: "Terminated",
        3: "Suspended",
        4: "Sleep",
        5: "Queue",
        6: "Semaphore",
        7: "EventFlag",
        8: "BlockMemory",
        9: "ByteMemory",
        10: "IoDriver",
        11: "File",
        12: "TcpIp",
        13: "Mutex",
        14: "PriorityChange",
        99: "Unknown",
    }

    READY = 0
    PRIORITYCHANGE = 14
    UNKNOWN = 99

    def __init__(
        self, target_context: DebugContext, provider: ThreadXThreadProvider, base: int
    ) -> None:
        super(ThreadXThread, self).__init__()
        self._target_context = target_context
        self._provider = provider
        self._base = base
        self._state = self._target_context.read32(self._base + THREAD_STATE_OFFSET)
        self._priority = self._target_context.read32(
            self._base + THREAD_PRIORITY_OFFSET
        )
        self._name = ""
        name_ptr = self._target_context.read32(self._base + THREAD_NAME_OFFSET)
        if name_ptr != 0:
            self._name = read_c_string(self._target_context, name_ptr)
        if len(self._name) == 0:
            self._name = "Unnamed"
        self._thread_context = ThreadXThreadContext(self._target_context, self)

    def get_stack_pointer(self) -> int:
        # Get stack pointer saved in thread struct.
        try:
            return self._target_context.read32(self._base + THREAD_STACK_POINTER_OFFSET)
        except exceptions.TransferError:
            LOG.debug(
                "Transfer error while reading thread's stack pointer @ 0x%08x",
                self._base + THREAD_STACK_POINTER_OFFSET,
            )
            return 0

    def update_info(self) -> None:
        try:
            self._priority = self._target_context.read32(
                self._base + THREAD_PRIORITY_OFFSET
            )
            self._state = self._target_context.read32(self._base + THREAD_STATE_OFFSET)
            if not self.READY <= self._state <= self.PRIORITYCHANGE:
                self._state = self.UNKNOWN
        except exceptions.TransferError:
            LOG.debug("Transfer error while reading thread info")

    @property
    def state(self) -> int:
        return self._state

    @state.setter
    def state(self, value: int) -> None:
        self._state = value

    @property
    def priority(self) -> int:
        return self._priority

    @property
    def unique_id(self) -> int:
        return self._base

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        # return "%s; Priority %d" % (self.STATE_NAMES[self.state], self.priority)
        return "%s; Priority %d" % (self.STATE_NAMES[self.state], self.priority)

    @property
    def is_current(self) -> bool:
        return self._provider.get_actual_current_thread_id() == self.unique_id

    @property
    def context(self) -> ThreadXThreadContext:
        return self._thread_context

    def __str__(self) -> str:
        return "<ThreadXThread@0x%08x id=%x name=%s>" % (
            id(self),
            self.unique_id,
            self.name,
        )

    def __repr__(self) -> str:
        return str(self)


class ThreadXThreadProvider(ThreadProvider[ThreadXThread | HandlerModeThread]):
    """@brief Thread provider for ThreadX.

    To successfully initialize, the following ThreadX symbols are needed:
        _tx_thread_created_ptr:     pointer to list of created processes
        _tx_thread_created_count:   count of created processes
        _tx_thread_current_ptr:     current thread
        _tx_thread_system_state:    ThreadX state: initializing, run, interrupt
    """

    # Scheduler not yet up
    TX_INITIALIZE_IN_PROGRESS = 0xF0F0F0F0

    def __init__(self, target: Target) -> None:
        super(ThreadXThreadProvider, self).__init__(target)
        self._created_ptr = 0
        self._created_cnt = 0
        self._current_ptr = 0
        self._system_state = 0
        self._threads: dict[int, ThreadXThread | HandlerModeThread] = {}

    def init(self, symbol_provider: SymbolProvider) -> bool:
        created_ptr = symbol_provider.get_symbol_value("_tx_thread_created_ptr")
        if created_ptr is None:
            return False
        LOG.debug("ThreadX: _tx_thread_created_ptr = 0x%08x", self._created_ptr)
        self._created_ptr = created_ptr

        created_cnt = symbol_provider.get_symbol_value("_tx_thread_created_count")
        if created_cnt is None:
            return False
        self._created_cnt = created_cnt
        LOG.debug("ThreadX: _tx_thread_created_cnt = 0x%08x", self._created_cnt)

        current_ptr = symbol_provider.get_symbol_value("_tx_thread_current_ptr")
        if current_ptr is None:
            return False
        LOG.debug("ThreadX: _tx_thread_current_ptr = 0x%08x", self._current_ptr)
        self._current_ptr = current_ptr

        system_state = symbol_provider.get_symbol_value("_tx_thread_system_state")
        if system_state is None:
            return False
        LOG.debug("ThreadX: _tx_thread_system_state = 0x%08x", self._current_ptr)
        self._system_state = system_state

        self._target.session.subscribe(  # pyright: ignore
            self.event_handler, Target.Event.POST_FLASH_PROGRAM
        )
        self._target.session.subscribe(self.event_handler, Target.Event.POST_RESET)  # pyright: ignore

        return True

    def invalidate(self) -> None:
        self._threads = {}

    def event_handler(self, notification: Notification) -> None:
        # Invalidate threads list if flash is reprogrammed.
        LOG.debug("ThreadX: invalidating threads list: %s", repr(notification))
        self.invalidate()

    def _build_thread_list(self) -> None:
        # Read the number of threads.
        thread_count = self._target_context.read32(self._created_cnt)

        # Build up list of all the threads
        all_threads = TargetList(self._target_context, self._created_ptr)
        new_threads: dict[int, ThreadXThread | HandlerModeThread] = {}
        for thread_base in all_threads:
            try:
                # Reuse existing thread objects if possible.
                if thread_base in self._threads:
                    t = self._threads[thread_base]

                    # Ask the thread object to update its state and priority.
                    if isinstance(t, ThreadXThread):
                        t.update_info()
                else:
                    t = ThreadXThread(self._target_context, self, thread_base)
                LOG.debug("Thread 0x%08x (%s)", thread_base, t.name)
                new_threads[t.unique_id] = t
            except exceptions.TransferError:  # noqa: PERF203
                LOG.debug("TransferError while examining thread 0x%08x", thread_base)

        # Is the number of threads correct?
        if len(new_threads) != thread_count:
            LOG.warning(
                "ThreadX: thread count mismatch, %d expected, %d found",
                thread_count,
                len(new_threads),
            )

        # Create fake handler mode thread.
        if self._target_context.read_core_register("ipsr") > 0:
            LOG.debug("ThreadX: creating handler mode thread")
            t = HandlerModeThread(self._target_context, self)
            new_threads[t.unique_id] = t

        self._threads = new_threads

    def get_threads(self) -> list[ThreadXThread | HandlerModeThread]:
        if not self.is_enabled:
            return []
        self.update_threads()
        return list(self._threads.values())

    def get_thread(self, thread_id: int) -> ThreadXThread | HandlerModeThread | None:
        if not self.is_enabled:
            return None
        self.update_threads()
        return self._threads.get(thread_id, None)

    @property
    def is_enabled(self) -> bool:
        # The _tx_thread_system_state global is used to determine whether
        # the kernel is running. Before the kernel starts, it'll contain
        # TX_INITIALIZE_IN_PROGRESS and possibly TX_INITIALIZE_IN_PROGRESS+1.
        # On cortex-m ports it should otherwise be 0.
        # As it's used in other ports to indicate the interrupt nesting level, it's
        # safer to compare it with TX_INITIALIZE_IN_PROGRESS.

        try:
            return (
                self._target_context.read32(self._system_state)
                < self.TX_INITIALIZE_IN_PROGRESS
            )
        except exceptions.TransferFaultError:
            LOG.warning(
                "ThreadX: read system state failed, target memory might not be initialized yet."
            )
            return False

    @property
    def current_thread(self) -> ThreadXThread | HandlerModeThread | None:
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
        if not self.is_enabled:
            return None
        return self._target_context.read32(self._current_ptr)


class ThreadXPlugin(Plugin):
    """@brief Plugin class for ThreadX."""

    def load(self) -> type[ThreadXThreadProvider]:
        return ThreadXThreadProvider

    @property
    def name(self) -> str:
        return "threadx"

    @property
    def description(self) -> str:
        return "ThreadX"
