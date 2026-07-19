"""The one result type every ANPR engine speaks.

Lives in its own module so engines can come and go (cloud client in v1,
local ONNX stack in v2) without the worker or the callbacks caring where
a result came from."""

from collections import namedtuple

PlateResult = namedtuple(
    "PlateResult", ["plate_text", "confidence", "status", "region"]
)  # status: success | failed | no_plate | throttled | skipped
