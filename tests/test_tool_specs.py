import pytest

from local_code.tool_specs import native_tool_definitions, tool_prompt_lines, validate_tool_call, ToolValidationError


def test_tool_prompt_lines_are_derived_from_allowed_tools():
    rendered = tool_prompt_lines({"read_file", "final"})

    assert "read_file" in rendered
    assert '"path"' in rendered
    assert "write_file" not in rendered


def test_native_tool_definitions_use_function_schema():
    tools = native_tool_definitions({"read_file"})

    assert tools == [
        {
            "type": "function",
            "function": {
                "name": "read_file",
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
        }
    ]


def test_validate_tool_call_normalizes_integer_strings():
    args = validate_tool_call("read_file", {"path": "README.md", "start": "2"}, {"read_file"})

    assert args == {"path": "README.md", "start": 2}


def test_validate_tool_call_rejects_missing_required_args():
    with pytest.raises(ToolValidationError, match="missing required argument 'path'"):
        validate_tool_call("read_file", {"start": 1}, {"read_file"})


def test_validate_tool_call_rejects_unknown_args():
    with pytest.raises(ToolValidationError, match="unknown argument"):
        validate_tool_call("read_file", {"path": "README.md", "surprise": True}, {"read_file"})
