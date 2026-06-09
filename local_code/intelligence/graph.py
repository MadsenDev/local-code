"""Typed dependency graph extraction and queries for repository intelligence.

The graph deliberately separates language parsing from graph assembly. Python is
parsed with :mod:`ast`; JavaScript and TypeScript go through a parser adapter so
a full parser (for example tree-sitter) can be supplied without changing graph
semantics.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import tomllib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

GRAPH_SCHEMA_VERSION = 1


class NodeKind(StrEnum):
    FILE = "file"
    MODULE = "module"
    PACKAGE = "package"
    CLASS = "class"
    INTERFACE = "interface"
    FUNCTION = "function"
    IMPORT = "import"
    COMMAND = "command"
    TEST = "test"
    COMPONENT = "component"


class EdgeKind(StrEnum):
    DECLARES = "declares"
    IMPORTS = "imports"
    DEPENDS_ON = "depends_on"
    ENTRY_POINT = "entry_point"
    TESTS = "tests"
    CONTAINS = "contains"
    IMPLEMENTS = "implements"


@dataclass(frozen=True, slots=True)
class SourceLocation:
    path: str
    line: int = 1
    column: int = 0
    end_line: int | None = None
    end_column: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "line": self.line,
            "column": self.column,
            "end_line": self.end_line,
            "end_column": self.end_column,
        }


@dataclass(frozen=True, slots=True)
class Evidence:
    kind: str
    source: str
    location: SourceLocation | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"kind": self.kind, "source": self.source}
        if self.location:
            value["location"] = self.location.to_dict()
        if self.detail:
            value["detail"] = self.detail
        return value


def _stable_id(prefix: str, *parts: str) -> str:
    identity = "\0".join(parts)
    slug = re.sub(r"[^a-z0-9]+", "-", parts[-1].lower()).strip("-")[:40] or prefix
    digest = hashlib.sha256(identity.encode()).hexdigest()[:12]
    return f"{prefix}:{slug}:{digest}"


@dataclass(frozen=True, slots=True)
class GraphNode:
    id: str
    kind: NodeKind
    name: str
    qualified_name: str
    language: str | None = None
    location: SourceLocation | None = None
    confidence: float = 1.0
    evidence: tuple[Evidence, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind.value,
            "name": self.name,
            "qualified_name": self.qualified_name,
            "language": self.language,
            "confidence": self.confidence,
            "evidence": [item.to_dict() for item in self.evidence],
            "attributes": dict(self.attributes),
        }
        if self.location:
            value["location"] = self.location.to_dict()
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GraphNode":
        return cls(
            id=str(value["id"]), kind=NodeKind(value["kind"]), name=str(value["name"]),
            qualified_name=str(value["qualified_name"]), language=value.get("language"),
            location=_location_from_dict(value.get("location")), confidence=float(value.get("confidence", 1.0)),
            evidence=tuple(_evidence_from_dict(item) for item in value.get("evidence", ())),
            attributes=dict(value.get("attributes", {})),
        )


@dataclass(frozen=True, slots=True)
class GraphEdge:
    id: str
    kind: EdgeKind
    source: str
    target: str
    confidence: float
    evidence: tuple[Evidence, ...]
    inferred: bool = False
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "kind": self.kind.value, "source": self.source, "target": self.target,
            "confidence": self.confidence, "evidence": [item.to_dict() for item in self.evidence],
            "inferred": self.inferred, "attributes": dict(self.attributes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GraphEdge":
        return cls(
            id=str(value["id"]), kind=EdgeKind(value["kind"]), source=str(value["source"]),
            target=str(value["target"]), confidence=float(value["confidence"]),
            evidence=tuple(_evidence_from_dict(item) for item in value.get("evidence", ())),
            inferred=bool(value.get("inferred", False)), attributes=dict(value.get("attributes", {})),
        )


def _location_from_dict(value: Any) -> SourceLocation | None:
    if not isinstance(value, Mapping):
        return None
    return SourceLocation(str(value["path"]), int(value.get("line", 1)), int(value.get("column", 0)), value.get("end_line"), value.get("end_column"))


def _evidence_from_dict(value: Mapping[str, Any]) -> Evidence:
    return Evidence(str(value["kind"]), str(value["source"]), _location_from_dict(value.get("location")), str(value.get("detail", "")))


@dataclass(slots=True)
class DependencyGraph:
    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: dict[str, GraphEdge] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: GraphNode) -> str:
        existing = self.nodes.get(node.id)
        if existing and existing != node:
            evidence = tuple(dict.fromkeys((*existing.evidence, *node.evidence)))
            attributes = {**existing.attributes, **node.attributes}
            self.nodes[node.id] = GraphNode(
                node.id, node.kind, node.name, node.qualified_name, node.language or existing.language,
                node.location or existing.location, max(node.confidence, existing.confidence), evidence, attributes,
            )
        else:
            self.nodes[node.id] = node
        return node.id

    def add_edge(self, kind: EdgeKind, source: str, target: str, *, confidence: float,
                 evidence: Sequence[Evidence], inferred: bool = False,
                 attributes: Mapping[str, Any] | None = None) -> str:
        edge_id = _stable_id("edge", kind.value, source, target)
        edge = GraphEdge(edge_id, kind, source, target, confidence, tuple(evidence), inferred, attributes or {})
        existing = self.edges.get(edge_id)
        if existing:
            edge = GraphEdge(
                edge_id, kind, source, target, max(existing.confidence, confidence),
                tuple(dict.fromkeys((*existing.evidence, *edge.evidence))), existing.inferred and inferred,
                {**existing.attributes, **edge.attributes},
            )
        self.edges[edge_id] = edge
        return edge_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": GRAPH_SCHEMA_VERSION,
            "metadata": self.metadata,
            "nodes": [self.nodes[key].to_dict() for key in sorted(self.nodes)],
            "edges": [self.edges[key].to_dict() for key in sorted(self.edges)],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DependencyGraph":
        nodes = [GraphNode.from_dict(item) for item in value.get("nodes", ())]
        edges = [GraphEdge.from_dict(item) for item in value.get("edges", ())]
        return cls({item.id: item for item in nodes}, {item.id: item for item in edges}, dict(value.get("metadata", {})))


@dataclass(frozen=True, slots=True)
class ParsedSymbol:
    kind: NodeKind
    name: str
    line: int
    column: int = 0
    end_line: int | None = None
    parent: str | None = None
    bases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ParsedImport:
    module: str
    line: int
    column: int = 0
    names: tuple[str, ...] = ()
    relative_level: int = 0


@dataclass(frozen=True, slots=True)
class ParsedUnit:
    symbols: tuple[ParsedSymbol, ...] = ()
    imports: tuple[ParsedImport, ...] = ()
    command_entry_points: tuple[tuple[str, int], ...] = ()


class ParserAdapter(Protocol):
    """Language parser boundary used by graph extraction."""

    languages: frozenset[str]

    def parse(self, path: str, source: str) -> ParsedUnit: ...


class PythonAstAdapter:
    languages = frozenset({"Python"})

    def parse(self, path: str, source: str) -> ParsedUnit:
        try:
            tree = ast.parse(source, filename=path)
        except (SyntaxError, ValueError):
            return ParsedUnit()
        symbols: list[ParsedSymbol] = []
        imports: list[ParsedImport] = []
        commands: list[tuple[str, int]] = []

        class Visitor(ast.NodeVisitor):
            parents: list[str] = []

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                bases = tuple(_python_name(base) for base in node.bases if _python_name(base))
                symbols.append(ParsedSymbol(NodeKind.CLASS, node.name, node.lineno, node.col_offset, getattr(node, "end_lineno", None), self.parents[-1] if self.parents else None, bases))
                self.parents.append(node.name)
                self.generic_visit(node)
                self.parents.pop()

            def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
                kind = NodeKind.TEST if node.name.startswith("test") else NodeKind.FUNCTION
                symbols.append(ParsedSymbol(kind, node.name, node.lineno, node.col_offset, getattr(node, "end_lineno", None), self.parents[-1] if self.parents else None))
                self.parents.append(node.name)
                self.generic_visit(node)
                self.parents.pop()

            visit_FunctionDef = _function
            visit_AsyncFunctionDef = _function

            def visit_Import(self, node: ast.Import) -> None:
                for alias in node.names:
                    imports.append(ParsedImport(alias.name, node.lineno, node.col_offset, (alias.asname or alias.name,)))

            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                imports.append(ParsedImport(node.module or "", node.lineno, node.col_offset, tuple(alias.name for alias in node.names), node.level))

            def visit_If(self, node: ast.If) -> None:
                if _is_python_main_guard(node.test):
                    commands.append(("python-main", node.lineno))
                self.generic_visit(node)

        Visitor().visit(tree)
        return ParsedUnit(tuple(symbols), tuple(imports), tuple(commands))


def _python_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _python_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _is_python_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.Compare) or len(node.ops) != 1 or not isinstance(node.ops[0], ast.Eq):
        return False
    values = (node.left, *node.comparators)
    return any(isinstance(item, ast.Name) and item.id == "__name__" for item in values) and any(isinstance(item, ast.Constant) and item.value == "__main__" for item in values)


class JavaScriptParserAdapter:
    """Dependency-free token parser adapter for JS/TS.

    It is intentionally replaceable. Unlike regex-only extraction it tokenizes
    comments, strings, identifiers and punctuation before recognizing imports
    and declarations. A tree-sitter adapter can implement the same protocol.
    """

    languages = frozenset({"JavaScript", "TypeScript"})

    def parse(self, path: str, source: str) -> ParsedUnit:
        tokens = _js_tokens(source)
        symbols: list[ParsedSymbol] = []
        imports: list[ParsedImport] = []
        index = 0
        while index < len(tokens):
            token, line, column = tokens[index]
            if token in {"class", "interface", "function"} and index + 1 < len(tokens):
                name = tokens[index + 1][0]
                if _identifier(name):
                    kind = {"class": NodeKind.CLASS, "interface": NodeKind.INTERFACE, "function": NodeKind.FUNCTION}[token]
                    symbols.append(ParsedSymbol(kind, name, line, column))
            elif token in {"const", "let", "var"} and index + 2 < len(tokens) and _identifier(tokens[index + 1][0]):
                cursor = index + 2
                while cursor < len(tokens) and tokens[cursor][0] not in {";", "\n"}:
                    if tokens[cursor][0] == "=>":
                        symbols.append(ParsedSymbol(NodeKind.FUNCTION, tokens[index + 1][0], line, column))
                        break
                    cursor += 1
            elif token == "import":
                module = ""
                cursor = index + 1
                while cursor < len(tokens) and tokens[cursor][0] not in {";", "\n"}:
                    if tokens[cursor][0] == "from" and cursor + 1 < len(tokens):
                        module = _strip_js_string(tokens[cursor + 1][0])
                        break
                    if tokens[cursor][0][0:1] in {'"', "'"}:
                        module = _strip_js_string(tokens[cursor][0])
                        break
                    cursor += 1
                if module:
                    imports.append(ParsedImport(module, line, column))
            elif token == "require" and index + 2 < len(tokens) and tokens[index + 1][0] == "(":
                module = _strip_js_string(tokens[index + 2][0])
                if module:
                    imports.append(ParsedImport(module, line, column))
            index += 1
        return ParsedUnit(tuple(symbols), tuple(imports))


def _identifier(value: str) -> bool:
    return bool(value) and (value[0].isalpha() or value[0] in "_$") and all(char.isalnum() or char in "_$" for char in value)


def _strip_js_string(value: str) -> str:
    return value[1:-1] if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'", "`"} else ""


def _js_tokens(source: str) -> list[tuple[str, int, int]]:
    tokens: list[tuple[str, int, int]] = []
    index = 0
    line = 1
    column = 0
    punctuation = set("{}()[];,.:")
    while index < len(source):
        char = source[index]
        if char == "\n":
            tokens.append(("\n", line, column)); line += 1; column = 0; index += 1; continue
        if char.isspace():
            column += 1; index += 1; continue
        if source.startswith("//", index):
            end = source.find("\n", index)
            if end < 0: break
            column += end - index; index = end; continue
        if source.startswith("/*", index):
            end = source.find("*/", index + 2)
            end = len(source) if end < 0 else end + 2
            segment = source[index:end]
            newline_count = segment.count("\n")
            if newline_count:
                line += newline_count; column = len(segment.rsplit("\n", 1)[-1])
            else: column += len(segment)
            index = end; continue
        if char in "'\"`":
            start, start_line, start_column, quote = index, line, column, char
            index += 1; column += 1
            while index < len(source):
                if source[index] == "\\": index += 2; column += 2; continue
                if source[index] == quote: index += 1; column += 1; break
                if source[index] == "\n": line += 1; column = 0; index += 1
                else: index += 1; column += 1
            tokens.append((source[start:index], start_line, start_column)); continue
        if char in punctuation:
            tokens.append((char, line, column)); index += 1; column += 1; continue
        start, start_column = index, column
        while index < len(source) and not source[index].isspace() and source[index] not in punctuation and source[index] not in "'\"`":
            if source.startswith("//", index) or source.startswith("/*", index): break
            index += 1; column += 1
        if index == start:
            index += 1; column += 1
        else:
            tokens.append((source[start:index], line, start_column))
    return tokens


DEFAULT_ADAPTERS: tuple[ParserAdapter, ...] = (PythonAstAdapter(), JavaScriptParserAdapter())


def _module_name(path: str) -> str:
    value = PurePosixPath(path)
    without_suffix = value.with_suffix("")
    parts = list(without_suffix.parts)
    if parts and parts[0] in {"src", "lib"}:
        parts.pop(0)
    if parts and parts[-1] in {"__init__", "index"}:
        parts.pop()
    return ".".join(parts) or without_suffix.name


def _node(kind: NodeKind, name: str, qualified: str, *, language: str | None = None,
          location: SourceLocation | None = None, confidence: float = 1.0,
          evidence: Sequence[Evidence] = (), attributes: Mapping[str, Any] | None = None) -> GraphNode:
    return GraphNode(_stable_id("node", kind.value, qualified), kind, name, qualified, language, location, confidence, tuple(evidence), attributes or {})


def _file_evidence(path: str, detail: str = "") -> Evidence:
    return Evidence("filesystem", path, SourceLocation(path), detail)


def build_dependency_graph(
    root: Path,
    files: Sequence[Mapping[str, Any]],
    manifests: Sequence[Mapping[str, Any]],
    commands: Mapping[str, str],
    *,
    previous: Mapping[str, Any] | None = None,
    adapters: Sequence[ParserAdapter] = DEFAULT_ADAPTERS,
    enrichment: Sequence[Mapping[str, Any]] | None = None,
) -> DependencyGraph:
    """Merge filesystem, manifest, and parser evidence into a current graph."""
    graph = DependencyGraph()
    module_ids: dict[str, str] = {}
    file_ids: dict[str, str] = {}
    adapters_by_language = {language: adapter for adapter in adapters for language in adapter.languages}

    for item in sorted(files, key=lambda value: str(value["path"])):
        path = str(item["path"])
        location = SourceLocation(path)
        evidence = (_file_evidence(path),)
        file_node = _node(NodeKind.FILE, PurePosixPath(path).name, path, language=item.get("language"), location=location, evidence=evidence, attributes={"test": bool(item.get("test")), "entrypoint": bool(item.get("entrypoint")), "sha256": item.get("sha256")})
        file_ids[path] = graph.add_node(file_node)
        language = item.get("language")
        adapter = adapters_by_language.get(language)
        if adapter is None:
            continue
        module = _module_name(path)
        module_node = _node(NodeKind.MODULE, module.rsplit(".", 1)[-1], module, language=language, location=location, evidence=(Evidence("static_analysis", adapter.__class__.__name__, location),))
        module_ids[module] = graph.add_node(module_node)
        graph.add_edge(EdgeKind.DECLARES, file_node.id, module_node.id, confidence=1.0, evidence=evidence)
        try:
            source = (root / path).read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        parsed = adapter.parse(path, source)
        symbol_ids: dict[str, str] = {}
        for symbol in parsed.symbols:
            qualified = f"{module}.{symbol.parent + '.' if symbol.parent else ''}{symbol.name}"
            symbol_location = SourceLocation(path, symbol.line, symbol.column, symbol.end_line)
            symbol_evidence = (Evidence("static_analysis", adapter.__class__.__name__, symbol_location),)
            symbol_node = _node(symbol.kind, symbol.name, qualified, language=language, location=symbol_location, evidence=symbol_evidence, attributes={"bases": list(symbol.bases)})
            symbol_ids[symbol.name] = graph.add_node(symbol_node)
            parent_id = symbol_ids.get(symbol.parent or "", module_node.id)
            graph.add_edge(EdgeKind.DECLARES, parent_id, symbol_node.id, confidence=1.0, evidence=symbol_evidence)
            for base in symbol.bases:
                target = _node(NodeKind.INTERFACE, base.rsplit(".", 1)[-1], base, language=language, confidence=0.75, evidence=symbol_evidence)
                graph.add_node(target)
                graph.add_edge(EdgeKind.IMPLEMENTS, symbol_node.id, target.id, confidence=0.75, evidence=symbol_evidence, inferred=True)
        for imported in parsed.imports:
            import_location = SourceLocation(path, imported.line, imported.column)
            resolved_name = _resolve_import(module, imported)
            import_node = _node(NodeKind.IMPORT, resolved_name, f"{module}:import:{resolved_name}:{imported.line}", language=language, location=import_location, evidence=(Evidence("static_analysis", adapter.__class__.__name__, import_location),), attributes={"names": list(imported.names)})
            graph.add_node(import_node)
            graph.add_edge(EdgeKind.DECLARES, module_node.id, import_node.id, confidence=1.0, evidence=import_node.evidence)
            target = _node(NodeKind.MODULE, resolved_name.rsplit(".", 1)[-1], resolved_name, language=language, confidence=0.9, evidence=import_node.evidence, attributes={"external": resolved_name not in module_ids})
            graph.add_node(target)
            graph.add_edge(EdgeKind.IMPORTS, module_node.id, target.id, confidence=0.95, evidence=import_node.evidence)
        if item.get("entrypoint"):
            _add_command(graph, path, path, file_node.id, "filesystem_entrypoint", 0.9)
        for command_name, line in parsed.command_entry_points:
            _add_command(graph, command_name, path, module_node.id, "static_analysis", 1.0, line)

    _resolve_internal_modules(graph, module_ids)
    _add_manifest_evidence(graph, root, manifests, commands, file_ids)
    _add_test_relationships(graph)
    apply_semantic_enrichment(graph, enrichment)

    old_nodes = {str(item["id"]) for item in (previous or {}).get("nodes", ()) if isinstance(item, Mapping) and "id" in item}
    old_edges = {str(item["id"]) for item in (previous or {}).get("edges", ()) if isinstance(item, Mapping) and "id" in item}
    graph.metadata["incremental"] = {
        "stale_node_ids": sorted(old_nodes - graph.nodes.keys()),
        "stale_edge_ids": sorted(old_edges - graph.edges.keys()),
        "new_node_ids": sorted(graph.nodes.keys() - old_nodes),
        "new_edge_ids": sorted(graph.edges.keys() - old_edges),
    }
    graph.metadata["counts"] = {"nodes": len(graph.nodes), "edges": len(graph.edges)}
    return graph


def _resolve_import(module: str, imported: ParsedImport) -> str:
    if imported.module.startswith("."):
        parent = module.split(".")[:-1]
        parts = imported.module.split("/")
        while parts and parts[0] in {".", ".."}:
            marker = parts.pop(0)
            if marker == ".." and parent:
                parent.pop()
        tail = [PurePosixPath(item).stem for item in parts if item]
        return ".".join((*parent, *tail))
    if not imported.relative_level:
        return imported.module
    parent = module.split(".")[:-1]
    keep = max(0, len(parent) - imported.relative_level + 1)
    return ".".join([*parent[:keep], *([imported.module] if imported.module else [])])


def _resolve_internal_modules(graph: DependencyGraph, module_ids: Mapping[str, str]) -> None:
    for edge_id, edge in list(graph.edges.items()):
        if edge.kind != EdgeKind.IMPORTS:
            continue
        target = graph.nodes.get(edge.target)
        internal_id = module_ids.get(target.qualified_name if target else "")
        if internal_id and internal_id != edge.target:
            del graph.edges[edge_id]
            graph.add_edge(EdgeKind.IMPORTS, edge.source, internal_id, confidence=edge.confidence, evidence=edge.evidence)
            if edge.target not in {item.source for item in graph.edges.values()} | {item.target for item in graph.edges.values()}:
                graph.nodes.pop(edge.target, None)


def _add_command(graph: DependencyGraph, name: str, path: str, target: str, evidence_kind: str, confidence: float, line: int = 1) -> None:
    location = SourceLocation(path, line or 1)
    evidence = (Evidence(evidence_kind, path, location),)
    command = _node(NodeKind.COMMAND, name, f"command:{name}:{path}", location=location, confidence=confidence, evidence=evidence)
    graph.add_node(command)
    graph.add_edge(EdgeKind.ENTRY_POINT, command.id, target, confidence=confidence, evidence=evidence, inferred=confidence < 1.0)


def _add_manifest_evidence(graph: DependencyGraph, root: Path, manifests: Sequence[Mapping[str, Any]], commands: Mapping[str, str], file_ids: Mapping[str, str]) -> None:
    packages: dict[str, str] = {}
    manifest_commands: set[tuple[str, str]] = set()
    for manifest in manifests:
        path = str(manifest["path"])
        name = str(manifest.get("name") or manifest.get("workspace") or PurePosixPath(path).parent)
        location = SourceLocation(path)
        evidence = (Evidence("manifest", path, location),)
        package = _node(NodeKind.PACKAGE, name, f"package:{name}:{manifest.get('workspace', '.')}", location=location, evidence=evidence, attributes={"manifest": path, "workspace": manifest.get("workspace", ".")})
        packages[name] = graph.add_node(package)
        if path in file_ids:
            graph.add_edge(EdgeKind.DECLARES, file_ids[path], package.id, confidence=1.0, evidence=evidence)
        for dependency in _manifest_dependencies(root / path):
            target = _node(NodeKind.PACKAGE, dependency, f"package:{dependency}", confidence=1.0, evidence=evidence, attributes={"external": dependency not in packages})
            graph.add_node(target)
            graph.add_edge(EdgeKind.DEPENDS_ON, package.id, target.id, confidence=1.0, evidence=evidence)
        declared_commands = {}
        if isinstance(manifest.get("scripts"), Mapping):
            declared_commands.update({str(key): str(value) for key, value in manifest["scripts"].items()})
        if isinstance(manifest.get("entry_points"), Mapping):
            declared_commands.update({str(key): str(value) for key, value in manifest["entry_points"].items()})
        for command_name, command_text in sorted(declared_commands.items()):
            _add_manifest_command(graph, path, command_name, command_text, package.id)
            manifest_commands.add((path, command_name))
    for name, command_text in sorted(commands.items()):
        manifest_path = next((str(item["path"]) for item in manifests if name in item.get("scripts", {})), str(manifests[0]["path"]) if manifests else "")
        if not manifest_path or (manifest_path, name) in manifest_commands:
            continue
        target = file_ids.get(manifest_path) or next((node.id for node in graph.nodes.values() if node.kind == NodeKind.PACKAGE and node.attributes.get("manifest") == manifest_path), None)
        if target:
            _add_manifest_command(graph, manifest_path, name, command_text, target)


def _add_manifest_command(graph: DependencyGraph, manifest_path: str, name: str, command_text: str, target: str) -> None:
    location = SourceLocation(manifest_path)
    evidence = (Evidence("manifest", manifest_path, location, command_text),)
    command = _node(NodeKind.COMMAND, name, f"manifest-command:{manifest_path}:{name}", location=location, evidence=evidence, attributes={"command": command_text})
    graph.add_node(command)
    graph.add_edge(EdgeKind.ENTRY_POINT, command.id, target, confidence=1.0, evidence=evidence)


def _manifest_dependencies(path: Path) -> list[str]:
    try:
        if path.name == "package.json":
            value = json.loads(path.read_text(encoding="utf-8"))
            names: set[str] = set()
            for key in ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies"):
                if isinstance(value.get(key), dict): names.update(str(item) for item in value[key])
            return sorted(names)
        if path.name == "pyproject.toml":
            value = tomllib.loads(path.read_text(encoding="utf-8"))
            dependencies = value.get("project", {}).get("dependencies", [])
            return sorted({_dependency_name(str(item)) for item in dependencies if _dependency_name(str(item))})
    except (OSError, ValueError, TypeError):
        pass
    return []


def _dependency_name(value: str) -> str:
    match = re.match(r"[A-Za-z0-9_.-]+", value.strip())
    return match.group(0) if match else ""


def _add_test_relationships(graph: DependencyGraph) -> None:
    modules = [node for node in graph.nodes.values() if node.kind == NodeKind.MODULE]
    sources = [node for node in modules if not _looks_like_test(node)]
    for test in [node for node in modules if _looks_like_test(node)]:
        imported_targets = {edge.target for edge in graph.edges.values() if edge.kind == EdgeKind.IMPORTS and edge.source == test.id}
        candidates = [node for node in sources if node.id in imported_targets]
        confidence = 1.0
        evidence_kind = "static_analysis"
        if not candidates:
            stem = _test_stem(test.name)
            candidates = [node for node in sources if node.name == stem or node.qualified_name.endswith(f".{stem}")]
            confidence = 0.7
            evidence_kind = "naming_convention"
        for source in candidates:
            location = test.location or SourceLocation(test.qualified_name)
            evidence = (Evidence(evidence_kind, location.path, location, f"{test.qualified_name} -> {source.qualified_name}"),)
            graph.add_edge(EdgeKind.TESTS, test.id, source.id, confidence=confidence, evidence=evidence, inferred=confidence < 1.0)


def _looks_like_test(node: GraphNode) -> bool:
    path = node.location.path.lower() if node.location else ""
    return node.name.startswith("test_") or node.name.endswith((".test", ".spec")) or any(part in {"test", "tests", "__tests__", "spec", "specs"} for part in PurePosixPath(path).parts)


def _test_stem(name: str) -> str:
    value = re.sub(r"^(test_|spec_)", "", name)
    return re.sub(r"[._-](test|spec)$", "", value)


COMPONENT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Provider Layer", ("provider", "openai", "ollama", "llama")),
    ("Runtime Layer", ("runtime", "routing", "agent", "tool", "cli")),
    ("Memory Layer", ("memory", "context", "intelligence", "store", "record")),
    ("UI", ("ui", "tui", "view", "frontend", "component")),
)


def apply_semantic_enrichment(graph: DependencyGraph, suggestions: Sequence[Mapping[str, Any]] | None = None) -> list[str]:
    """Add components while retaining all deterministic edges on any failure.

    Model suggestions are accepted only when every member cites an existing graph
    node ID and at least one valid node ID or repository-relative file evidence.
    """
    rejected: list[str] = []
    assignments: dict[str, set[str]] = defaultdict(set)
    for node in graph.nodes.values():
        if node.kind not in {NodeKind.FILE, NodeKind.MODULE, NodeKind.CLASS, NodeKind.INTERFACE}:
            continue
        searchable = f"{node.name} {node.qualified_name} {(node.location.path if node.location else '')}".lower()
        for component, markers in COMPONENT_RULES:
            if any(marker in searchable for marker in markers):
                assignments[component].add(node.id)
                break
    for suggestion in suggestions or ():
        try:
            name = str(suggestion["name"]).strip()
            members = {str(item) for item in suggestion["member_node_ids"]}
            citations = {str(item) for item in suggestion["citations"]}
            valid_files = {node.location.path for node in graph.nodes.values() if node.location}
            if not name or not members or not members <= graph.nodes.keys() or not citations or not all(citation in graph.nodes or citation in valid_files for citation in citations):
                raise ValueError("missing valid graph node IDs or file citations")
            assignments[name].update(members)
        except (KeyError, TypeError, ValueError) as exc:
            rejected.append(str(exc))
    for name, members in sorted(assignments.items()):
        citations = tuple(Evidence("semantic_enrichment", member, graph.nodes[member].location) for member in sorted(members))
        component = _node(NodeKind.COMPONENT, name, f"component:{name}", confidence=0.8, evidence=citations, attributes={"member_count": len(members)})
        graph.add_node(component)
        for member in sorted(members):
            graph.add_edge(EdgeKind.CONTAINS, component.id, member, confidence=0.8, evidence=(Evidence("semantic_enrichment", member, graph.nodes[member].location),), inferred=True)
    graph.metadata["enrichment"] = {"rejected": rejected, "components": sorted(assignments)}
    return rejected


def render_architecture(graph: DependencyGraph) -> str:
    """Render a concise, deterministic graph view."""
    components = sorted((node for node in graph.nodes.values() if node.kind == NodeKind.COMPONENT), key=lambda node: node.name)
    lines = ["# Architecture", "", "Generated from `.rist/project/dependency-graph.json`.", "", "## Components", ""]
    if not components:
        lines.append("_No components inferred._")
    for component in components:
        members = [graph.nodes[edge.target] for edge in graph.edges.values() if edge.kind == EdgeKind.CONTAINS and edge.source == component.id and edge.target in graph.nodes]
        examples = ", ".join(f"`{item.qualified_name}` ({item.id})" for item in sorted(members, key=lambda node: node.qualified_name)[:6])
        lines.append(f"- **{component.name}** `{component.id}` — {len(members)} nodes" + (f": {examples}" if examples else ""))
    lines.extend(("", "## Key relationships", ""))
    relationships = [edge for edge in graph.edges.values() if edge.kind in {EdgeKind.DEPENDS_ON, EdgeKind.IMPORTS, EdgeKind.TESTS}]
    for edge in sorted(relationships, key=lambda item: (item.kind.value, item.source, item.target))[:20]:
        source = graph.nodes.get(edge.source)
        target = graph.nodes.get(edge.target)
        if source and target:
            lines.append(f"- `{source.qualified_name}` **{edge.kind.value.replace('_', ' ')}** `{target.qualified_name}` ({edge.confidence:.2f})")
    if not relationships:
        lines.append("_No dependency relationships inferred._")
    return "\n".join(lines).rstrip() + "\n"


class GraphQuery:
    """Convenience queries over typed graph records."""

    def __init__(self, graph: DependencyGraph | Mapping[str, Any]):
        self.graph = graph if isinstance(graph, DependencyGraph) else DependencyGraph.from_dict(graph)

    def where_does_provider_belong(self, provider_name: str = "new provider") -> dict[str, Any]:
        components = [node for node in self.graph.nodes.values() if node.kind == NodeKind.COMPONENT and "provider" in node.name.lower()]
        examples = self._component_members(components[0].id) if components else [node for node in self.graph.nodes.values() if "provider" in node.qualified_name.lower()]
        return {"question": f"Where does {provider_name} belong?", "component": components[0].to_dict() if components else None, "examples": [node.to_dict() for node in examples[:10]]}

    def tests_covering(self, component_or_node_id: str) -> list[GraphNode]:
        targets = {component_or_node_id}
        if component_or_node_id in self.graph.nodes and self.graph.nodes[component_or_node_id].kind == NodeKind.COMPONENT:
            targets.update(node.id for node in self._component_members(component_or_node_id))
        test_ids = {edge.source for edge in self.graph.edges.values() if edge.kind == EdgeKind.TESTS and edge.target in targets}
        return sorted((self.graph.nodes[item] for item in test_ids if item in self.graph.nodes), key=lambda node: node.qualified_name)

    def dependents_of(self, interface_or_node_id: str) -> list[GraphNode]:
        target_ids = {interface_or_node_id}
        target_ids.update(node.id for node in self.graph.nodes.values() if node.qualified_name == interface_or_node_id or node.name == interface_or_node_id)
        source_ids = {edge.source for edge in self.graph.edges.values() if edge.target in target_ids and edge.kind in {EdgeKind.IMPORTS, EdgeKind.DEPENDS_ON, EdgeKind.IMPLEMENTS}}
        return sorted((self.graph.nodes[item] for item in source_ids if item in self.graph.nodes), key=lambda node: node.qualified_name)

    def _component_members(self, component_id: str) -> list[GraphNode]:
        ids = {edge.target for edge in self.graph.edges.values() if edge.kind == EdgeKind.CONTAINS and edge.source == component_id}
        return sorted((self.graph.nodes[item] for item in ids if item in self.graph.nodes), key=lambda node: node.qualified_name)
