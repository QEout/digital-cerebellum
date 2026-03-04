"""
Pre-defined cerebellar microzones.

Each microzone follows the biological principle: genetically determined
structure, experience-driven specialization.  The microzone's existence
is pre-programmed; its internal weights adapt through learning.
"""

from digital_cerebellum.microzones.tool_call import ToolCallMicrozone
from digital_cerebellum.microzones.payment import PaymentMicrozone
from digital_cerebellum.microzones.shell_command import ShellCommandMicrozone
from digital_cerebellum.microzones.file_operation import FileOperationMicrozone
from digital_cerebellum.microzones.api_call import APICallMicrozone
from digital_cerebellum.microzones.response_prediction import ResponsePredictionMicrozone

ALL_MICROZONES = [
    ToolCallMicrozone,
    PaymentMicrozone,
    ShellCommandMicrozone,
    FileOperationMicrozone,
    APICallMicrozone,
    ResponsePredictionMicrozone,
]

__all__ = [
    "ToolCallMicrozone",
    "PaymentMicrozone",
    "ShellCommandMicrozone",
    "FileOperationMicrozone",
    "APICallMicrozone",
    "ResponsePredictionMicrozone",
    "ALL_MICROZONES",
]
