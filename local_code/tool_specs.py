"""Canonical tool specifications and lightweight argument validation."""

import json
from copy import deepcopy


TOOL_ORDER = [
    "search_web",
    "fetch_url",
    "repo_map",
    "repo_overview",
    "list_files",
    "search_files",
    "read_file",
    "run_command",
    "write_file",
    "replace_lines",
    "replace_in_file",
    "insert_after",
    "final",
]

TOOL_SPECS = {
    "search_web": {
        "description": "Search the web for current information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search terms."},
                "max_results": {"type": "integer", "description": "Maximum number of results to return.", "minimum": 1},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    "fetch_url": {
        "description": "Fetch text from a specific URL.",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "URL to fetch."}},
            "required": ["url"],
            "additionalProperties": False,
        },
    },
    "repo_map": {
        "description": "Return a compact map of the repository.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    "repo_overview": {
        "description": "Return a higher-level overview of the repository structure.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    "list_files": {
        "description": "List files below a path in the working directory.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Optional directory path, defaults to current directory."}},
            "additionalProperties": False,
        },
    },
    "search_files": {
        "description": "Search repository files with a text or regex query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Text or regex to search for."},
                "path": {"type": "string", "description": "Optional path to search within."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    "read_file": {
        "description": "Read a file from the working directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to read."},
                "start": {"type": "integer", "description": "1-indexed starting line.", "minimum": 1},
                "end": {"type": "integer", "description": "1-indexed ending line.", "minimum": 1},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    "run_command": {
        "description": "Run a shell command in the working directory, subject to permission checks.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run."},
                "timeout": {"type": "integer", "description": "Timeout in seconds.", "minimum": 1},
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
    "write_file": {
        "description": "Write full content to a file in execute mode.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write."},
                "content": {"type": "string", "description": "Full file content."},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    },
    "replace_lines": {
        "description": "Replace an inclusive line range in a file in execute mode.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start": {"type": "integer", "minimum": 1},
                "end": {"type": "integer", "minimum": 1},
                "content": {"type": "string"},
            },
            "required": ["path", "start", "end", "content"],
            "additionalProperties": False,
        },
    },
    "replace_in_file": {
        "description": "Replace exact text in a file in execute mode.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old": {"type": "string"},
                "new": {"type": "string"},
                "count": {"type": "integer", "minimum": 1},
            },
            "required": ["path", "old", "new"],
            "additionalProperties": False,
        },
    },
    "insert_after": {
        "description": "Insert text after an exact anchor in a file in execute mode.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "anchor": {"type": "string"},
                "content": {"type": "string"},
                "occurrence": {"type": "integer", "minimum": 1},
            },
            "required": ["path", "anchor", "content"],
            "additionalProperties": False,
        },
    },
    "final": {
        "description": "Finish the backend run with a structured report.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "findings": {"type": "array", "items": {"type": "string"}},
                "files_read": {"type": "array", "items": {"type": "string"}},
                "files_changed": {"type": "array", "items": {"type": "string"}},
                "diff_summary": {"type": "string"},
                "commands_run": {"type": "array", "items": {"type": "string"}},
                "tests_run": {"type": "array", "items": {"type": "string"}},
                "risks": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "open_questions": {"type": "array", "items": {"type": "string"}},
                "suggested_deeper_scope": {"type": "array", "items": {"type": "string"}},
                "needs_clarification": {"type": "boolean"},
                "clarifying_question": {"type": "string"},
                "needs_approval": {"type": "boolean"},
                "plan": {"type": "array", "items": {"type": "string"}},
                "message": {"description": "Optional nested report object or JSON string."},
            },
            "additionalProperties": False,
        },
    },
}


class ToolValidationError(ValueError):
    """Raised when a tool call does not match the local schema."""


def allowed_tool_specs(allowed_names):
    """Return ordered canonical specs for an allowed-tool set."""
    allowed = set(allowed_names)
    return {name: TOOL_SPECS[name] for name in TOOL_ORDER if name in allowed}


def tool_prompt_lines(allowed_names):
    """Render compact fallback-JSON tool descriptions from canonical specs."""
    lines = []
    for name, spec in allowed_tool_specs(allowed_names).items():
        schema = spec["parameters"]
        example_args = {}
        for prop, prop_schema in schema.get("properties", {}).items():
            if prop == "message":
                continue
            typ = prop_schema.get("type")
            if typ == "string":
                example_args[prop] = _string_placeholder(prop)
            elif typ == "integer":
                example_args[prop] = _integer_placeholder(prop)
            elif typ == "boolean":
                example_args[prop] = False
            elif typ == "array":
                example_args[prop] = ["..."]
        lines.append(f"- {name}: {json.dumps(example_args, ensure_ascii=False)}")
    return "\n".join(lines)


def native_tool_definitions(allowed_names):
    """Return Ollama/OpenAI-style tool definitions for the allowed tools."""
    definitions = []
    for name, spec in allowed_tool_specs(allowed_names).items():
        definitions.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": spec["description"],
                    "parameters": deepcopy(spec["parameters"]),
                },
            }
        )
    return definitions


def validate_tool_call(tool, args, allowed_names):
    """Validate and normalize a tool call against allowed names and schemas."""
    if tool not in allowed_names:
        raise ToolValidationError(f"Tool {tool!r} is not available in the current mode.")
    if tool not in TOOL_SPECS:
        raise ToolValidationError(f"Tool {tool!r} is not defined by the local tool registry.")
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ToolValidationError(f"Tool {tool!r} args must be an object.")
    return _validate_object(tool, args, TOOL_SPECS[tool]["parameters"])


def _validate_object(tool, args, schema):
    normalized = dict(args)
    required = schema.get("required") or []
    for key in required:
        if key not in normalized:
            raise ToolValidationError(f"Tool {tool!r} missing required argument {key!r}.")
    properties = schema.get("properties") or {}
    if schema.get("additionalProperties") is False:
        extra = sorted(set(normalized) - set(properties))
        if extra:
            raise ToolValidationError(f"Tool {tool!r} got unknown argument(s): {', '.join(extra)}.")
    for key, value in list(normalized.items()):
        if key not in properties:
            continue
        normalized[key] = _validate_value(tool, key, value, properties[key])
    return normalized


def _validate_value(tool, key, value, schema):
    expected = schema.get("type")
    if expected is None:
        return value
    if expected == "string":
        if not isinstance(value, str):
            raise ToolValidationError(f"Tool {tool!r} argument {key!r} must be a string.")
        return value
    if expected == "integer":
        if isinstance(value, bool):
            raise ToolValidationError(f"Tool {tool!r} argument {key!r} must be an integer.")
        if isinstance(value, str) and value.strip().isdigit():
            value = int(value.strip())
        if not isinstance(value, int):
            raise ToolValidationError(f"Tool {tool!r} argument {key!r} must be an integer.")
        minimum = schema.get("minimum")
        if minimum is not None and value < minimum:
            raise ToolValidationError(f"Tool {tool!r} argument {key!r} must be >= {minimum}.")
        return value
    if expected == "boolean":
        if not isinstance(value, bool):
            raise ToolValidationError(f"Tool {tool!r} argument {key!r} must be a boolean.")
        return value
    if expected == "array":
        if not isinstance(value, list):
            raise ToolValidationError(f"Tool {tool!r} argument {key!r} must be an array.")
        item_schema = schema.get("items") or {}
        return [_validate_value(tool, f"{key}[]", item, item_schema) for item in value]
    if expected == "object":
        if not isinstance(value, dict):
            raise ToolValidationError(f"Tool {tool!r} argument {key!r} must be an object.")
        return value
    return value


def _string_placeholder(name):
    placeholders = {
        "query": "search terms",
        "url": "https://...",
        "path": "file path",
        "command": "shell command",
        "content": "replacement lines",
        "old": "exact old text",
        "new": "replacement text",
        "anchor": "exact anchor text",
        "summary": "...",
        "diff_summary": "...",
    }
    return placeholders.get(name, "...")


def _integer_placeholder(name):
    placeholders = {
        "max_results": 5,
        "start": 1,
        "end": 500,
        "timeout": 30,
        "count": 1,
        "occurrence": 1,
    }
    return placeholders.get(name, 1)
