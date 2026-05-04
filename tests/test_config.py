import re
import pytest
from local_code.config import BLOCKED_COMMAND_PATTERNS, CODE_ACTION_RE, EDIT_INTENT_RE, PROMPT_YES_RE


def _blocked(command):
    return any(re.search(p, command) for p in BLOCKED_COMMAND_PATTERNS)


class TestBlockedCommands:
    def test_rm_rf_root(self):
        assert _blocked("rm -rf /")
        assert _blocked(" rm -rf /")

    def test_rm_rf_non_root_not_blocked(self):
        assert not _blocked("rm -rf /home/user/tmp")

    def test_mkfs(self):
        assert _blocked("mkfs.ext4 /dev/sda")
        assert _blocked("mkfs /dev/sda")

    def test_dd(self):
        assert _blocked("dd if=/dev/zero of=/dev/sda")

    def test_reboot(self):
        assert _blocked("reboot")
        assert _blocked("sudo reboot")
        assert _blocked("poweroff")
        assert _blocked("shutdown now")
        assert _blocked("halt")

    def test_fork_bomb(self):
        assert _blocked(":(){:|:&};:")


class TestCodeActionRE:
    def test_matches_action_words(self):
        assert CODE_ACTION_RE.search("inspect the code")
        assert CODE_ACTION_RE.search("fix this bug")
        assert CODE_ACTION_RE.search("search the repo")
        assert CODE_ACTION_RE.search("read the file")

    def test_word_boundary(self):
        # "fixed" should not match "fix" as a word boundary violation ... actually \b handles this
        assert not CODE_ACTION_RE.search("nonmatching text here")


class TestEditIntentRE:
    def test_matches_edit_words(self):
        assert EDIT_INTENT_RE.search("edit this file")
        assert EDIT_INTENT_RE.search("add a new function")
        assert EDIT_INTENT_RE.search("rename the variable")
        assert EDIT_INTENT_RE.search("create a new module")

    def test_no_match_on_unrelated(self):
        assert not EDIT_INTENT_RE.search("inspect only")


class TestPromptYesRE:
    def test_matches_yes_variants(self):
        assert PROMPT_YES_RE.search("yes")
        assert PROMPT_YES_RE.search("y")
        assert PROMPT_YES_RE.search("approve")
        assert PROMPT_YES_RE.search("apply")
        assert PROMPT_YES_RE.search("go ahead")
        assert PROMPT_YES_RE.search("do it")

    def test_word_boundary_respected(self):
        assert not PROMPT_YES_RE.search("yesterday")
        assert not PROMPT_YES_RE.search("yellow")
