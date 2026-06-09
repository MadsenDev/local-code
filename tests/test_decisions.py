import json

import pytest

from local_code.intelligence.decisions import DecisionService, DecisionStatus


def service(tmp_path):
    return DecisionService.load(tmp_path / ".rist" / "project")


def test_duplicate_decisions_are_rejected(tmp_path):
    decisions = service(tmp_path)
    original = decisions.add(title="Use PostgreSQL for persistence", rationale="Transactional storage", affected_components=["database"])
    with pytest.raises(ValueError, match=original.id):
        decisions.add(title="Use PostgreSQL for persistence", rationale="Duplicate", affected_components=["database"])


def test_contradictory_decisions_create_structured_conflict(tmp_path):
    decisions = service(tmp_path)
    accepted = decisions.add(title="Enable server-side rendering", rationale="Render pages on the server", affected_components=["web"])
    decisions.accept(accepted.id)
    candidate = {
        "title": "Disable server-side rendering",
        "rationale": "Do not render pages on the server",
        "affected_components": ["web"],
    }
    conflicts = decisions.conflicts_for("Change web rendering", [candidate])
    assert conflicts
    assert conflicts[0].decision_id == accepted.id
    assert conflicts[0].kind in {"deterministic", "semantic"}
    assert conflicts[0].to_dict()["explanation"]


def test_supersession_accepts_new_decision_and_links_old(tmp_path):
    decisions = service(tmp_path)
    old = decisions.add(title="Use REST", rationale="Simple API")
    decisions.accept(old.id)
    new = decisions.add(title="Use GraphQL", rationale="Typed client queries")
    superseded, replacement = decisions.supersede(old.id, new.id)
    assert superseded.status == DecisionStatus.SUPERSEDED
    assert superseded.superseding_decision == replacement.id
    assert replacement.status == DecisionStatus.ACCEPTED


def test_rejected_candidate_preserves_rejected_alternatives(tmp_path):
    decisions = service(tmp_path)
    [candidate] = decisions.extract_candidates({"decision_candidates": [{
        "title": "Choose SQLite",
        "rationale": "Local operation",
        "alternatives": ["PostgreSQL", "MySQL"],
        "consequences": ["Single writer"],
        "affected_components": ["storage"],
    }]})
    rejected = decisions.reject(candidate.id, "Does not support expected concurrency")
    assert rejected.status == DecisionStatus.REJECTED
    assert rejected.alternatives == ("PostgreSQL", "MySQL")
    assert candidate.id not in decisions.pending


def test_malformed_model_extraction_never_enters_review_queue(tmp_path):
    decisions = service(tmp_path)
    assert decisions.extract_candidates("not json") == []
    assert decisions.extract_candidates({"decision_candidates": "not a list"}) == []
    assert decisions.extract_candidates({"decision_candidates": [{"rationale": "missing title"}, 42]}) == []
    assert decisions.pending == {}
    assert decisions.decisions == {}


def test_user_edited_markdown_reconciles_by_stable_id(tmp_path):
    decisions = service(tmp_path)
    item = decisions.add(
        title="Use queues for background work",
        rationale="Keep requests responsive",
        alternatives=["Inline execution"],
        consequences=["Requires a worker"],
        affected_components=["jobs"],
        source_references=["docs/design.md"],
    )
    decisions.accept(item.id)
    markdown = decisions.markdown_path.read_text()
    markdown = markdown.replace("Use queues for background work", "Use durable queues for background work")
    markdown = markdown.replace("Keep requests responsive", "Keep API requests responsive")
    decisions.markdown_path.write_text(markdown)

    reloaded = DecisionService.load(decisions.base_path)
    edited = reloaded.decisions[item.id]
    assert edited.id == item.id
    assert edited.title == "Use durable queues for background work"
    assert edited.rationale == "Keep API requests responsive"
    assert json.loads(reloaded.path.read_text())["decisions"][0]["id"] == item.id


def test_candidates_require_review_and_can_be_edited_merged_and_accepted(tmp_path):
    decisions = service(tmp_path)
    candidates = decisions.extract_candidates({"decision_candidates": [
        {"title": "Use Redis", "rationale": "Fast cache", "affected_components": ["cache"]},
        {"title": "Add cache TTLs", "rationale": "Bound staleness", "affected_components": ["cache"]},
    ]}, source_run="run-1")
    assert not decisions.decisions
    decisions.edit_candidate(candidates[0].id, title="Use Redis cache")
    merged = decisions.merge_candidates([candidates[0].id, candidates[1].id], title="Use Redis with TTLs")
    accepted = decisions.accept(merged.id)
    assert accepted.status == DecisionStatus.ACCEPTED
    assert accepted.id == merged.id
    assert not decisions.pending


def test_contract_context_contains_relevant_decisions_and_conflict_contract(tmp_path):
    decisions = service(tmp_path)
    item = decisions.add(title="Do not use Redis cache", rationale="Avoid external cache operations", affected_components=["cache"])
    decisions.accept(item.id)
    context = decisions.contract_context("Use Redis cache for the cache component")
    assert context["relevant_decisions"][0]["id"] == item.id
    assert context["decision_conflicts"]
    assert context["requires_deviation_explanation"] is True
