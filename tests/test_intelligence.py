import json
from pathlib import Path

import pytest

from local_code.intelligence import (
    ComponentRecord,
    ConventionRecord,
    DecisionRecord,
    FactRecord,
    FileAssociationRecord,
    IntelligenceStore,
    IntelligenceValidationError,
    LifecycleStatusRecord,
    PrincipleRecord,
    ProjectIdentityRecord,
    RecordKind,
    RelationshipRecord,
    WorkflowRecord,
    stable_record_id,
)
from local_code.memory import ensure_memory_files, load_repo_memory


RECORD_TYPES = (
    ProjectIdentityRecord,
    PrincipleRecord,
    FactRecord,
    DecisionRecord,
    ComponentRecord,
    RelationshipRecord,
    LifecycleStatusRecord,
    ConventionRecord,
    WorkflowRecord,
    FileAssociationRecord,
)


def test_all_intelligence_categories_round_trip_with_common_fields(tmp_path):
    store = IntelligenceStore.load(tmp_path / ".rist")
    for index, record_type in enumerate(RECORD_TYPES):
        statement = f"record {index}"
        store.upsert(record_type(
            id=stable_record_id(record_type.KIND, statement),
            statement=statement,
            confidence=0.75,
            source="test",
            source_ref="tests/test_intelligence.py",
            supersedes=("old-id",),
            details={"index": index},
        ))
    store.sync_markdown_views()

    loaded = IntelligenceStore.load(tmp_path / ".rist")
    assert {record.kind for record in loaded.values()} == set(RecordKind)
    assert all(record.source == "test" for record in loaded.values())
    document = json.loads((tmp_path / ".rist" / "intelligence.json").read_text())
    assert document["schema_version"] == 1
    assert all(set(("id", "kind", "statement", "status", "confidence", "source", "source_ref", "created_at", "updated_at", "supersedes")) <= item.keys() for item in document["records"])


def test_markdown_is_editable_view_and_preserves_record_identity(tmp_path):
    base = tmp_path / ".rist"
    store = IntelligenceStore.load(base)
    record = FactRecord(id="fact:stable", statement="Original fact", source="scanner", source_ref="README.md")
    store.upsert(record)
    store.sync_markdown_views()

    project = base / "project.md"
    project.write_text(project.read_text().replace("Original fact", "Human-edited fact").replace('"confidence":1.0', '"confidence":0.9'), encoding="utf-8")
    loaded = IntelligenceStore.load(base)

    edited = loaded.records["fact:stable"]
    assert edited.statement == "Human-edited fact"
    assert edited.confidence == 0.9
    assert edited.source == "scanner"
    assert "Human-edited fact" in project.read_text()


def test_new_markdown_bullet_is_imported_with_stable_id(tmp_path):
    base = tmp_path / ".rist"
    IntelligenceStore.load(base)
    project = base / "project.md"
    project.write_text(project.read_text().replace("_No records._", "_No records._", 2).replace("## Facts\n\n_No records._", "## Facts\n\n- Python 3.11 is required"), encoding="utf-8")

    first = IntelligenceStore.load(base)
    record = next(record for record in first.values() if record.statement == "Python 3.11 is required")
    second = IntelligenceStore.load(base)
    assert record.id == next(item.id for item in second.values() if item.statement == record.statement)
    assert record.source == "markdown_edit"


def test_legacy_markdown_and_private_logs_migrate(tmp_path):
    base = tmp_path / ".rist"
    base.mkdir()
    (base / "project.md").write_text("# Project Notes\n\n- Stack: Python 3.11\n", encoding="utf-8")
    (base / "architecture.md").write_text("# Architecture\n\n- CLI delegates to agent\n", encoding="utf-8")
    (base / "decisions.md").write_text("# Decisions\n\n- Keep dependencies optional\n", encoding="utf-8")
    (base / "runs.jsonl").write_text('{"status":"ok"}\n', encoding="utf-8")
    (base / "chat_history.jsonl").write_text('{"role":"user","content":"private"}\n', encoding="utf-8")

    paths = ensure_memory_files(tmp_path)
    store = IntelligenceStore.load(base)

    assert {record.kind for record in store.values()} == {RecordKind.FACT, RecordKind.COMPONENT, RecordKind.DECISION}
    assert paths["runs"] == base / "private" / "runs.jsonl"
    assert paths["chat_history"] == base / "private" / "chat_history.jsonl"
    assert not (base / "runs.jsonl").exists()
    assert not (base / "chat_history.jsonl").exists()
    assert "recent runs" not in load_repo_memory(tmp_path)
    assert "private" not in load_repo_memory(tmp_path)


def test_version_zero_document_is_loaded_and_rewritten(tmp_path):
    base = tmp_path / ".rist"
    base.mkdir()
    (base / "intelligence.json").write_text(json.dumps({"version": 0, "records": [{"kind": "identity", "text": "Rist is a coding agent"}]}), encoding="utf-8")

    store = IntelligenceStore.load(base)

    assert next(iter(store.values())).kind == RecordKind.PROJECT_IDENTITY
    assert json.loads((base / "intelligence.json").read_text())["schema_version"] == 1


def test_invalid_persisted_schema_is_rejected(tmp_path):
    base = tmp_path / ".rist"
    base.mkdir()
    (base / "intelligence.json").write_text('{"schema_version": 1, "records": [{"id": "x"}]}', encoding="utf-8")

    with pytest.raises(IntelligenceValidationError, match="kind"):
        IntelligenceStore.load(base)
