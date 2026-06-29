from datetime import datetime, timedelta, timezone

from local_code.tui.workspace_memory import DecisionFilter, DecisionStatus, DecisionStore, DecisionType


def test_decision_store_sorts_newest_first_and_filters():
    store = DecisionStore()
    old = datetime(2026, 1, 1, tzinfo=timezone.utc)
    new = old + timedelta(minutes=5)
    store.add(type=DecisionType.PLAN, title="Refactor provider abstraction", reason="Reduce duplication", files=["provider.py"], timestamp=old)
    store.add(type=DecisionType.DECISION, title="Use managed llama runtime", confidence=0.9, status=DecisionStatus.ACCEPTED, files=["runtime.py"], timestamp=new)

    assert [item.title for item in store.all()] == ["Use managed llama runtime", "Refactor provider abstraction"]
    assert [item.title for item in store.filter(DecisionFilter(type=DecisionType.PLAN))] == ["Refactor provider abstraction"]
    assert [item.title for item in store.filter(DecisionFilter(file="runtime"))] == ["Use managed llama runtime"]
    assert [item.title for item in store.filter(DecisionFilter(min_confidence=0.8))] == ["Use managed llama runtime"]


def test_workspace_memory_search_and_dismiss():
    store = DecisionStore()
    record = store.add(type="assumption", title="Project uses FastAPI", summary="Detected app imports", confidence=0.75)

    assert store.filter(DecisionFilter(search="fastapi")) == [record]
    assert store.filter(DecisionFilter(search="django")) == []
    assert store.dismiss(record.id).status == DecisionStatus.DISMISSED
    assert store.filter(DecisionFilter(status=DecisionStatus.DISMISSED))[0].id == record.id


def test_activity_event_conversion_for_proposal_acceptance_and_rejection():
    store = DecisionStore()
    accepted = store.ingest_activity_event({"kind": "apply", "text": "Proposal accepted", "files": ["a.py"]})
    rejected = store.ingest_activity_event({"kind": "reject", "text": "Proposal rejected"})
    question = store.ingest_activity_event({"kind": "question", "title": "Waiting for user approval", "text": "Edit a.py"})

    assert accepted.type == DecisionType.DECISION
    assert accepted.status == DecisionStatus.ACCEPTED
    assert accepted.files == ("a.py",)
    assert rejected.type == DecisionType.REJECTED
    assert question.type == DecisionType.QUESTION
    assert question.status == DecisionStatus.PENDING
