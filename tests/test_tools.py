import pytest
from pathlib import Path
from local_code.tools import resolve_path, replace_in_file, write_file, read_file, insert_after


class TestResolvePath:
    def test_relative_path_within_workdir(self, tmp_path):
        result = resolve_path(str(tmp_path), "src/main.py")
        assert result == tmp_path.resolve() / "src" / "main.py"

    def test_dot_path(self, tmp_path):
        result = resolve_path(str(tmp_path), ".")
        assert result == tmp_path.resolve()

    def test_blocks_parent_traversal(self, tmp_path):
        with pytest.raises(ValueError, match="escapes workdir"):
            resolve_path(str(tmp_path), "../../etc/passwd")

    def test_blocks_absolute_outside(self, tmp_path):
        with pytest.raises(ValueError, match="escapes workdir"):
            resolve_path(str(tmp_path), "/etc/passwd")

    def test_blocks_encoded_traversal(self, tmp_path):
        with pytest.raises(ValueError, match="escapes workdir"):
            resolve_path(str(tmp_path), "subdir/../../../etc/passwd")


class TestFileOps:
    def test_write_and_read(self, tmp_path):
        write_file(str(tmp_path), "hello.txt", "line1\nline2\n")
        result = read_file(str(tmp_path), "hello.txt", 1, 10)
        assert "line1" in result
        assert "line2" in result

    def test_write_outside_workdir_blocked(self, tmp_path):
        with pytest.raises(ValueError, match="escapes workdir"):
            write_file(str(tmp_path), "../../evil.txt", "bad")

    def test_replace_in_file(self, tmp_path):
        write_file(str(tmp_path), "f.py", "foo = 1\nfoo = 2\n")
        result = replace_in_file(str(tmp_path), "f.py", "foo", "bar", count=1)
        assert "Replaced" in result
        content = read_file(str(tmp_path), "f.py", 1, 10)
        assert "bar = 1" in content
        assert "foo = 2" in content

    def test_replace_missing_text(self, tmp_path):
        write_file(str(tmp_path), "f.py", "hello\n")
        result = replace_in_file(str(tmp_path), "f.py", "nothere", "x")
        assert "not found" in result.lower()

    def test_insert_after(self, tmp_path):
        write_file(str(tmp_path), "f.py", "def foo():\n    pass\n")
        result = insert_after(str(tmp_path), "f.py", "def foo():", "\n    # inserted")
        assert "Inserted" in result
        content = read_file(str(tmp_path), "f.py", 1, 10)
        assert "inserted" in content
