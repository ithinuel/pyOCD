# pyOCD debugger
# Copyright (c) 2006-2015 Arm Limited
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

from . import board as board
from . import core as core
from . import debug as debug
from . import flash as flash
from . import gdbserver as gdbserver
from . import target as target
from . import utility as utility
from . import coresight as coresight
from . import trace as trace

from ._version import version

__version__ = version
