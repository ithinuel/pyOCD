# pyOCD debugger
# Copyright (c) 2016-2020 Arm Limited
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
from typing import Any, Generator, List, Mapping, Sequence, TYPE_CHECKING
import logging

from .provider import TargetThread, ThreadProvider
from .common import read_c_string, HandlerModeThread, EXC_RETURN_EXT_FRAME_MASK
from pyocd.core import exceptions
from pyocd.core.target import Target
from pyocd.core.plugin import Plugin
from pyocd.debug.context import DebugContext
from pyocd.coresight.cortex_m_core_registers import index_for_reg
from pyocd.trace.events import TraceEvent, TraceITMEvent
from pyocd.trace.sink import TraceEventFilter

if TYPE_CHECKING:
    from pyocd.utility.notification import Notification
    from pyocd.debug.symbols import SymbolProvider
    from pyocd.core.core_registers import CoreRegisterNameOrNumberType

KERNEL_FLAGS_OFFSET = 0x1C
IS_RUNNING_MASK = 0x1

ALL_OBJECTS_THREADS_OFFSET = 0

THREAD_STACK_POINTER_OFFSET = 0
THREAD_EXTENDED_FRAME_OFFSET = 4
THREAD_NAME_OFFSET = 8
THREAD_STACK_BOTTOM_OFFSET = 12
THREAD_PRIORITY_OFFSET = 16
THREAD_STATE_OFFSET = 17
THREAD_CREATED_NODE_OFFSET = 36

LIST_NODE_NEXT_OFFSET = 0
LIST_NODE_OBJ_OFFSET = 8

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
                # Read the object from the node.
                obj = self._context.read32(node + LIST_NODE_OBJ_OFFSET)
                yield obj

                next_node = self._context.read32(node + LIST_NODE_NEXT_OFFSET)
                node = next_node
            except exceptions.TransferError:  # noqa: PERF203
                LOG.warning(
                    "TransferError while reading list elements (list=0x%08x, node=0x%08x), terminating list",
                    self._list,
                    node,
                )
                is_valid = False


class ArgonThreadContext(DebugContext):
    """@brief Thread context for Argon."""

    # SP is handled specially, so it is not in these dicts.

    CORE_REGISTER_OFFSETS: Mapping[int, int] = {
        # Software stacked
        4: 0,  # r4
        5: 4,  # r5
        6: 8,  # r6
        7: 12,  # r7
        8: 16,  # r8
        9: 20,  # r9
        10: 24,  # r10
        11: 28,  # r11
        # Hardware stacked
        0: 32,  # r0
        1: 36,  # r1
        2: 40,  # r2
        3: 44,  # r3
        12: 48,  # r12
        14: 52,  # lr
        15: 56,  # pc
        16: 60,  # xpsr
    }

    FPU_EXTENDED_REGISTER_OFFSETS: Mapping[int, int] = {
        # Software stacked
        4: 0,  # r4
        5: 4,  # r5
        6: 8,  # r6
        7: 12,  # r7
        8: 16,  # r8
        9: 20,  # r9
        10: 24,  # r10
        11: 28,  # r11
        0x50: 32,  # s16
        0x51: 36,  # s17
        0x52: 40,  # s18
        0x53: 44,  # s19
        0x54: 48,  # s20
        0x55: 52,  # s21
        0x56: 56,  # s22
        0x57: 60,  # s23
        0x58: 64,  # s24
        0x59: 68,  # s25
        0x5A: 72,  # s26
        0x5B: 76,  # s27
        0x5C: 80,  # s28
        0x5D: 84,  # s29
        0x5E: 88,  # s30
        0x5F: 92,  # s31
        # Hardware stacked
        0: 96,  # r0
        1: 100,  # r1
        2: 104,  # r2
        3: 108,  # r3
        12: 112,  # r12
        14: 116,  # lr
        15: 120,  # pc
        16: 124,  # xpsr
        0x40: 128,  # s0
        0x41: 132,  # s1
        0x42: 136,  # s2
        0x43: 140,  # s3
        0x44: 144,  # s4
        0x45: 148,  # s5
        0x46: 152,  # s6
        0x47: 156,  # s7
        0x48: 160,  # s8
        0x49: 164,  # s9
        0x4A: 168,  # s10
        0x4B: 172,  # s11
        0x4C: 176,  # s12
        0x4D: 180,  # s13
        0x4E: 184,  # s14
        0x4F: 188,  # s15
        33: 192,  # fpscr
        # (reserved word: 196)
    }

    def __init__(self, parent: DebugContext, thread: "ArgonThread") -> None:
        super().__init__(parent)
        self._thread = thread
        self._has_fpu = self.core.has_fpu

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
        sw_stacked = 0x20
        table = self.CORE_REGISTER_OFFSETS
        if self._has_fpu:
            if in_exception and self.core.is_vector_catch():
                # Vector catch has just occurred, take live LR
                exception_lr = self._parent.read_core_register("lr")

                # Check bit 4 of the exception LR to determine if FPU registers were stacked.
                has_extended_frame = (
                    int(exception_lr) & EXC_RETURN_EXT_FRAME_MASK
                ) == 0
            else:
                # Can't really rely on finding live LR after initial
                # vector catch, so retrieve LR stored by OS on last
                # thread switch.
                has_extended_frame = self._thread.has_extended_frame

            if has_extended_frame:
                table = self.FPU_EXTENDED_REGISTER_OFFSETS
                hw_stacked = 0x68
                sw_stacked = 0x60

        reg_indices = [
            index_for_reg(reg) if isinstance(reg, str) else reg for reg in reg_list
        ]
        for reg in reg_indices:
            # Must handle stack pointer specially.
            if reg == 13:
                sp_addr = sp + hw_stacked + (sw_stacked if not in_exception else 0)
                reg_vals.append(sp_addr)
                continue

            # Look up offset for this register on the stack.
            sp_offset = table.get(reg, None)
            if sp_offset is None:
                try:
                    reg_vals.append(self._parent.read_core_register_raw(reg))
                except exceptions.TransferError:
                    reg_vals.append(0)
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


class ArgonThread(TargetThread):
    """@brief Base class representing a thread on the target."""

    UNKNOWN = 0
    SUSPENDED = 1
    READY = 2
    RUNNING = 3
    BLOCKED = 4
    SLEEPING = 5
    DONE = 6

    STATE_NAMES: Mapping[int, str] = {
        UNKNOWN: "Unknown",
        SUSPENDED: "Suspended",
        READY: "Ready",
        RUNNING: "Running",
        BLOCKED: "Blocked",
        SLEEPING: "Sleeping",
        DONE: "Done",
    }

    def __init__(
        self, target_context: DebugContext, provider: "ArgonThreadProvider", base: int
    ) -> None:
        super(ArgonThread, self).__init__()
        self._target_context = target_context
        self._provider = provider
        self._base = base
        self._thread_context = ArgonThreadContext(self._target_context, self)
        self._has_fpu = self._thread_context.core.has_fpu
        self._priority = 0
        self._state = self.UNKNOWN
        self._name = "?"

        try:
            self.update_info()

            ptr = self._target_context.read32(self._base + THREAD_NAME_OFFSET)
            self._name = read_c_string(self._target_context, ptr)
            LOG.debug("Thread@%x name=%x '%s'", self._base, ptr, self._name)
        except exceptions.TransferError:
            LOG.debug("Transfer error while reading thread info")

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
            self._priority = self._target_context.read8(
                self._base + THREAD_PRIORITY_OFFSET
            )

            self._state = self._target_context.read8(self._base + THREAD_STATE_OFFSET)
            if self._state > self.DONE:
                self._state = self.UNKNOWN
        except exceptions.TransferError:
            LOG.debug("Transfer error while reading thread info")

    @property
    def state(self) -> int:
        return self._state

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
        return "%s; Priority %d" % (self.STATE_NAMES[self.state], self.priority)

    @property
    def is_current(self) -> bool:
        return self._provider.get_actual_current_thread_id() == self.unique_id

    @property
    def context(self) -> DebugContext:
        return self._thread_context

    @property
    def has_extended_frame(self) -> bool:
        if not self._has_fpu:
            return False
        try:
            flag = self._target_context.read8(self._base + THREAD_EXTENDED_FRAME_OFFSET)
            return flag != 0
        except exceptions.TransferError:
            LOG.debug(
                "Transfer error while reading thread's extended frame flag @ 0x%08x",
                self._base + THREAD_EXTENDED_FRAME_OFFSET,
            )
            return False

    def __str__(self) -> str:
        return "<ArgonThread@0x%08x id=%x name=%s>" % (
            id(self),
            self.unique_id,
            self.name,
        )

    def __repr__(self) -> str:
        return str(self)


class ArgonThreadProvider(ThreadProvider[ArgonThread | HandlerModeThread]):
    """@brief Base class for RTOS support plugins."""

    ThreadType = ArgonThread

    def __init__(self, target: Target) -> None:
        super().__init__(target)
        self.g_ar: int | None = None
        self.g_ar_objects: int | None = None
        self._all_threads: int | None = None
        self._threads: dict[int, ArgonThread | HandlerModeThread] = {}

    def init(self, symbol_provider: SymbolProvider) -> bool:
        self.g_ar = symbol_provider.get_symbol_value("g_ar")
        if self.g_ar is None:
            return False
        LOG.debug("Argon: g_ar = 0x%08x", self.g_ar)

        self.g_ar_objects = symbol_provider.get_symbol_value("g_ar_objects")
        if self.g_ar_objects is None:
            return False
        LOG.debug("Argon: g_ar_objects = 0x%08x", self.g_ar_objects)

        self._all_threads = self.g_ar_objects + ALL_OBJECTS_THREADS_OFFSET

        self._target.session.subscribe(  # pyright: ignore
            self.event_handler, Target.Event.POST_FLASH_PROGRAM
        )
        self._target.session.subscribe(self.event_handler, Target.Event.POST_RESET)  # pyright: ignore

        return True

    def invalidate(self) -> None:
        self._threads = {}

    def event_handler(self, notification: Notification) -> None:
        # Invalidate threads list if flash is reprogrammed.
        LOG.debug("Argon: invalidating threads list: %s", repr(notification))
        self.invalidate()

    def _build_thread_list(self) -> None:
        if self._all_threads is None:
            raise exceptions.InternalError()

        all_threads = TargetList(self._target_context, self._all_threads)
        new_threads: dict[int, ArgonThread | HandlerModeThread] = {}
        for thread_base in all_threads:
            try:
                # Reuse existing thread objects if possible.
                if thread_base in self._threads:
                    t = self._threads[thread_base]

                    # Ask the thread object to update its state and priority.
                    if isinstance(t, ArgonThread):
                        t.update_info()
                else:
                    t = ArgonThread(self._target_context, self, thread_base)
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

    def get_threads(self) -> List[ArgonThread | HandlerModeThread]:
        if not self.is_enabled:
            return []
        self.update_threads()
        return list(self._threads.values())

    def get_thread(self, thread_id: int) -> ArgonThread | HandlerModeThread | None:
        if not self.is_enabled:
            return None
        self.update_threads()
        return self._threads.get(thread_id, None)

    @property
    def is_enabled(self) -> bool:
        return self.get_is_running()

    @property
    def current_thread(self) -> ArgonThread | HandlerModeThread | None:
        if not self.is_enabled:
            return None
        self.update_threads()
        thread_id = self.get_current_thread_id()
        if thread_id is None:
            LOG.debug("No current thread found")
            return None

        thread = self._threads.get(thread_id, None)
        if thread is None:
            LOG.debug(
                "Current thread id=%x not found in self._threads = %s",
                thread_id,
                repr(self._threads),
            )
        return thread

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
        if not self.is_enabled or self.g_ar is None:
            return None
        return self._target_context.read32(self.g_ar)

    def get_is_running(self) -> bool:
        if self.g_ar is None:
            return False
        try:
            flags = self._target_context.read32(self.g_ar + KERNEL_FLAGS_OFFSET)
            return (flags & IS_RUNNING_MASK) != 0
        except exceptions.TransferFaultError:
            LOG.warning(
                "Argon: read kernel flags failed, target memory might not be initialized yet."
            )
            return False


class ArgonTraceEventID(Enum):
    # 2 value: 0=previous thread's new state, 1=new thread id
    kArTraceThreadSwitch = 1  # noqa: N815
    # 1 value
    kArTraceThreadCreated = 2  # noqa: N815
    # 1 value
    kArTraceThreadDeleted = 3  # noqa: N815


class ArgonTraceEvent(TraceEvent):
    """@brief Argon kernel trace event."""

    def __init__(
        self, event_id: int, thread_id: int, name: str, state: int, ts: int = 0
    ) -> None:
        super().__init__("argon", ts)
        self._event_id = event_id
        self._thread_id = thread_id
        self._thread_name = name
        self._prev_thread_state = state

    @property
    def event_id(self) -> int:
        return self._event_id

    @property
    def thread_id(self) -> int:
        return self._thread_id

    @property
    def thread_name(self) -> str:
        return self._thread_name

    @property
    def prev_thread_state(self) -> int:
        return self._prev_thread_state

    def __str__(self) -> str:
        if self.event_id == ArgonTraceEventID.kArTraceThreadSwitch:
            state_name = ArgonThread.STATE_NAMES.get(
                self.prev_thread_state, "<invalid state>"
            )
            desc = "New thread = {}; old thread state = {}".format(
                self.thread_name, state_name
            )
        elif self.event_id == ArgonTraceEventID.kArTraceThreadCreated:
            desc = "Created thread {}".format(self.thread_id)
        elif self.event_id == ArgonTraceEventID.kArTraceThreadDeleted:
            desc = "Deleted thread {}".format(self.thread_id)
        else:
            desc = "Unknown kernel event #{}".format(self.event_id)
        return "[{}] Argon: {}".format(self.timestamp, desc)


class ArgonTraceEventFilter(TraceEventFilter):
    """@brief Trace event filter to identify Argon kernel trace events sent via ITM.

    As Argon kernel trace events are identified, the ITM trace events are replaced with instances
    of ArgonTraceEvent.
    """

    def __init__(self, threads: Mapping[int, str]) -> None:
        super().__init__()
        self._threads = threads
        self._pending_event: TraceITMEvent | None = None

    def filter(self, event: TraceEvent) -> TraceEvent | None:
        if isinstance(event, TraceITMEvent):
            if event.port == 31:
                event_id = ArgonTraceEventID(event.data >> 24)
                if event_id in (
                    ArgonTraceEventID.kArTraceThreadSwitch,
                    ArgonTraceEventID.kArTraceThreadCreated,
                    ArgonTraceEventID.kArTraceThreadDeleted,
                ):
                    self._pending_event = event
                    # Swallow the event.
                    return None
            elif event.port == 30 and self._pending_event is not None:
                event_id = self._pending_event.data >> 24
                thread_id = event.data
                name = self._threads.get(thread_id, "<unknown thread>")
                state = self._pending_event.data & 0x00FFFFFF

                # Create the Argon event.
                event = ArgonTraceEvent(
                    event_id, thread_id, name, state, self._pending_event.timestamp
                )

                self._pending_event = None

        return event


class ArgonPlugin(Plugin):
    """@brief Plugin class for the Argon RTOS."""

    def load(self) -> type[ArgonThreadProvider]:
        return ArgonThreadProvider

    @property
    def name(self) -> str:
        return "argon"

    @property
    def description(self) -> str:
        return "Argon RTOS"
