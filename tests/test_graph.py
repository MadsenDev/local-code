import json
from pathlib import Path

from local_code.intelligence.graph import (
    DependencyGraph,
    EdgeKind,
    GraphQuery,
    JavaScriptParserAdapter,
    NodeKind,
    apply_semantic_enrichment,
)
from local_code.intelligence.indexer import index_repository
from tests.test_indexer import NOW, init_repo


def load_graph(repo: Path) -> DependencyGraph:
    payload = json.loads((repo / ".rist/project/dependency-graph.json").read_text(encoding="utf-8"))
    return DependencyGraph.from_dict(payload)


def find_node(graph: DependencyGraph, kind: NodeKind, qualified_name: str):
    return next(node for node in graph.nodes.values() if node.kind == kind and node.qualified_name == qualified_name)


def test_python_ast_graph_extracts_symbols_imports_commands_tests_and_evidence(tmp_path):
    repo = init_repo(tmp_path / "python-graph", {
        "pyproject.toml": '[project]\nname="sample"\ndependencies=["httpx>=1"]\n[project.scripts]\nsample="sample.cli:main"\n',
        "src/sample/providers.py": "class Provider:\n    pass\n\nclass LocalProvider(Provider):\n    def run(self):\n        return 1\n",
        "src/sample/cli.py": "from .providers import LocalProvider\n\ndef main():\n    return LocalProvider().run()\n\nif __name__ == '__main__':\n    main()\n",
        "tests/test_providers.py": "from sample.providers import LocalProvider\n\ndef test_run():\n    assert LocalProvider().run() == 1\n",
    })

    index_repository(repo, now=NOW)
    graph = load_graph(repo)

    assert find_node(graph, NodeKind.CLASS, "sample.providers.Provider")
    assert find_node(graph, NodeKind.FUNCTION, "sample.cli.main")
    assert find_node(graph, NodeKind.TEST, "tests.test_providers.test_run")
    provider_module = find_node(graph, NodeKind.MODULE, "sample.providers")
    test_module = find_node(graph, NodeKind.MODULE, "tests.test_providers")
    assert any(edge.kind == EdgeKind.IMPORTS and edge.source == test_module.id and edge.target == provider_module.id for edge in graph.edges.values())
    test_edge = next(edge for edge in graph.edges.values() if edge.kind == EdgeKind.TESTS and edge.source == test_module.id and edge.target == provider_module.id)
    assert test_edge.confidence == 1.0
    assert test_edge.evidence[0].location.path == "tests/test_providers.py"
    assert any(node.kind == NodeKind.COMMAND and node.name == "python-main" for node in graph.nodes.values())
    assert any(node.kind == NodeKind.COMMAND and node.name == "sample" for node in graph.nodes.values())
    assert any(node.kind == NodeKind.PACKAGE and node.name == "httpx" for node in graph.nodes.values())
    assert (repo / ".rist/project/architecture.md").read_text().startswith("# Architecture\n")


def test_javascript_typescript_extraction_uses_parser_adapter():
    parsed = JavaScriptParserAdapter().parse(
        "src/provider.ts",
        "// class Fake {}\nimport { Base } from './base';\nexport interface Provider {}\nexport class LocalProvider {}\nconst x = require('runtime');\n",
    )

    assert [(item.kind, item.name) for item in parsed.symbols] == [
        (NodeKind.INTERFACE, "Provider"),
        (NodeKind.CLASS, "LocalProvider"),
    ]
    assert [item.module for item in parsed.imports] == ["./base", "runtime"]


def test_enrichment_requires_citations_and_queries_components(tmp_path):
    repo = init_repo(tmp_path / "queries", {
        "src/providers.py": "class ProviderInterface:\n    pass\nclass CustomProvider(ProviderInterface):\n    pass\n",
        "tests/test_providers.py": "from providers import CustomProvider\ndef test_provider(): assert CustomProvider()\n",
    })
    index_repository(repo, now=NOW)
    graph = load_graph(repo)
    deterministic_edges = set(graph.edges)

    rejected = apply_semantic_enrichment(graph, [{"name": "Invented", "member_node_ids": ["missing"], "citations": []}])

    assert rejected
    assert deterministic_edges <= graph.edges.keys()
    query = GraphQuery(graph)
    placement = query.where_does_provider_belong("another provider")
    assert placement["component"]["name"] == "Provider Layer"
    component_id = placement["component"]["id"]
    assert any(node.qualified_name == "tests.test_providers" for node in query.tests_covering(component_id))
    assert any(node.name == "CustomProvider" for node in query.dependents_of("ProviderInterface"))


def test_incremental_index_removes_and_reports_stale_graph_records(tmp_path):
    repo = init_repo(tmp_path / "stale", {"src/old_provider.py": "class OldProvider:\n    pass\n"})
    index_repository(repo, now=NOW)
    before = load_graph(repo)
    old = find_node(before, NodeKind.CLASS, "old_provider.OldProvider")

    (repo / "src/old_provider.py").unlink()
    (repo / "src/runtime.py").write_text("def run():\n    return True\n", encoding="utf-8")
    report = index_repository(repo, now=NOW)
    after = load_graph(repo)

    assert old.id not in after.nodes
    assert old.id in report["stale_graph_nodes"]
    assert find_node(after, NodeKind.FUNCTION, "runtime.run")
    assert after.metadata["incremental"]["stale_node_ids"]
