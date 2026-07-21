"""Function signatures → JSON-schema parameter specs, plus validation.

The HTTP gateway (serve.py) uses this to give every served function a
FastMCP-style contract for free: declare a typed Python function and the
schema, request validation, and machine-readable docs are derived from it.
The same schemas are what an MCP tool adapter would hand to a model.

Deliberately shallow: primitives, containers, Optional/unions, Literal.
Unknown annotations degrade to "accept anything" — never to a crash.
"""

from __future__ import annotations

import inspect
import json
import types
import typing
from typing import Any, Callable

_PRIMITIVES: dict[Any, dict[str, Any]] = {
    int: {"type": "integer"},
    float: {"type": "number"},
    str: {"type": "string"},
    bool: {"type": "boolean"},
    type(None): {"type": "null"},
    Any: {},
}

_ANYTHING: dict[str, Any] = {}


def type_to_schema(tp: Any) -> dict[str, Any]:
    if tp in _PRIMITIVES:
        return dict(_PRIMITIVES[tp])
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin in (list, set, frozenset, tuple):
        schema: dict[str, Any] = {"type": "array"}
        item_args = [a for a in args if a is not Ellipsis]
        if len(set(item_args)) == 1:
            items = type_to_schema(item_args[0])
            if items:
                schema["items"] = items
        return schema
    if origin is dict:
        schema = {"type": "object"}
        if len(args) == 2:
            values = type_to_schema(args[1])
            if values:
                schema["additionalProperties"] = values
        return schema
    if origin in (typing.Union, types.UnionType):
        variants = [type_to_schema(a) for a in args]
        if any(v == _ANYTHING for v in variants):
            return {}
        return {"anyOf": variants}
    if origin is typing.Literal:
        return {"enum": list(args)}
    return {}  # unknown annotation: accept anything


def fn_schema(fn: Callable) -> dict[str, Any]:
    """JSON-schema object describing fn's parameters (called with kwargs)."""
    try:
        sig = inspect.signature(fn)
        hints = typing.get_type_hints(fn)
    except Exception:
        return {"type": "object", "additionalProperties": True}

    props: dict[str, Any] = {}
    required: list[str] = []
    open_extras = False
    for name, param in sig.parameters.items():
        if param.kind is param.VAR_KEYWORD:
            open_extras = True  # **kwargs — can't enumerate, allow extras
            continue
        if param.kind is param.VAR_POSITIONAL:
            continue  # *args can never be fed by a JSON object of named params
        schema = type_to_schema(hints.get(name, Any))
        if param.default is not param.empty:
            try:
                json.dumps(param.default)
                schema["default"] = param.default
            except (TypeError, ValueError):
                pass
        else:
            required.append(name)
        props[name] = schema

    out: dict[str, Any] = {
        "type": "object",
        "properties": props,
        "additionalProperties": open_extras,
    }
    if required:
        out["required"] = required
    return out


def validate(schema: dict[str, Any], payload: Any) -> tuple[dict[str, Any], list[str]]:
    """Check `payload` (a decoded JSON body) against a fn_schema.

    Returns (kwargs, errors). kwargs has values coerced where lossless
    (int → float for "number" params); errors are human-readable strings.
    """
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        return {}, ["request body must be a JSON object of named parameters"]

    props = schema.get("properties", {})
    clean: dict[str, Any] = {}
    errors: list[str] = []

    for name in schema.get("required", []):
        if name not in payload:
            errors.append(f"missing required parameter '{name}'")
    for name, value in payload.items():
        if name not in props:
            if schema.get("additionalProperties", False):
                clean[name] = value
            else:
                errors.append(f"unknown parameter '{name}'")
            continue
        coerced, err = _check(value, props[name], name)
        if err:
            errors.append(err)
        else:
            clean[name] = coerced
    return clean, errors


def _check(value: Any, schema: dict[str, Any], path: str) -> tuple[Any, str | None]:
    if not schema or (set(schema) <= {"default"}):
        return value, None
    if "enum" in schema:
        if value in schema["enum"]:
            return value, None
        return None, f"'{path}' must be one of {schema['enum']!r}"
    if "anyOf" in schema:
        for variant in schema["anyOf"]:
            coerced, err = _check(value, variant, path)
            if err is None:
                return coerced, None
        return None, f"'{path}' matches none of the allowed types"

    kind = schema.get("type")
    # bool is an int subclass in Python — never accept it for numeric params.
    if kind == "integer":
        if isinstance(value, int) and not isinstance(value, bool):
            return value, None
        return None, f"'{path}' must be an integer, got {type(value).__name__}"
    if kind == "number":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value), None
        return None, f"'{path}' must be a number, got {type(value).__name__}"
    if kind == "string":
        if isinstance(value, str):
            return value, None
        return None, f"'{path}' must be a string, got {type(value).__name__}"
    if kind == "boolean":
        if isinstance(value, bool):
            return value, None
        return None, f"'{path}' must be a boolean, got {type(value).__name__}"
    if kind == "null":
        if value is None:
            return value, None
        return None, f"'{path}' must be null"
    if kind == "array":
        if not isinstance(value, list):
            return None, f"'{path}' must be an array, got {type(value).__name__}"
        items = schema.get("items")
        if not items:
            return value, None
        out = []
        for i, element in enumerate(value):
            coerced, err = _check(element, items, f"{path}[{i}]")
            if err:
                return None, err
            out.append(coerced)
        return out, None
    if kind == "object":
        if not isinstance(value, dict):
            return None, f"'{path}' must be an object, got {type(value).__name__}"
        values_schema = schema.get("additionalProperties")
        if not isinstance(values_schema, dict) or not values_schema:
            return value, None
        out = {}
        for key, element in value.items():
            coerced, err = _check(element, values_schema, f"{path}.{key}")
            if err:
                return None, err
            out[key] = coerced
        return out, None
    return value, None
