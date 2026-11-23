"""Chunking strategies.

Four strategies, chosen so their trade-offs are easy to see in the eval:

* ``file``      one chunk per file. Simple, keeps full context, but big files blow the context budget
                and the embedding model only "sees" the first ~512 tokens of each file.
* ``fixed``     sliding windows of ~N tokens snapped to line boundaries, with overlap. Ignores code
                structure, so a window can start halfway through one method and end inside the next.
* ``function``  structure-aware with no dependencies: one chunk per function/method/property (Python via
                ``ast``, C-family languages via a brace scanner), markdown by heading. Small neighbours
                are merged, oversized members are split into windows.
* ``ast``       same idea, but members come from a real parser (tree-sitter), so it handles any language
                tree-sitter knows and isn't fooled by unusual formatting. Falls back to ``function``.

Every chunk records its start/end line (for citations) and a *parent* span (for small-to-big retrieval:
search with the small chunk, give the LLM the whole method or the neighbouring windows).
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from .models import Chunk, SourceFile, approx_tokens

BRACE_LANGUAGES = {
    "csharp", "java", "kotlin", "go", "javascript", "typescript", "tsx", "c", "cpp", "rust", "swift", "php", "scala",
}
CHAR_LITERAL_LANGUAGES = {"csharp", "java", "kotlin", "go", "c", "cpp", "rust", "swift", "scala"}
PREPROCESSOR_LANGUAGES = {"csharp", "c", "cpp"}


@dataclass
class Segment:
    start: int  # 1-based inclusive
    end: int
    kind: str
    symbol: str | None = None
    parent: tuple[int, int] | None = None


class Chunker:
    name = "base"

    def chunk(self, f: SourceFile) -> list[Chunk]:
        raise NotImplementedError

    def chunk_all(self, files: list[SourceFile]) -> list[Chunk]:
        out: list[Chunk] = []
        for f in files:
            out.extend(self.chunk(f))
        return out

    def describe(self) -> dict:
        return {"name": self.name, **{k: v for k, v in vars(self).items() if not k.startswith("_")}}


# --------------------------------------------------------------------------- helpers


def _trim(lines: list[str], s: int, e: int) -> tuple[int, int]:
    while s <= e and not lines[s - 1].strip():
        s += 1
    while e >= s and not lines[e - 1].strip():
        e -= 1
    return s, e


def _make_chunk(f: SourceFile, lines: list[str], seg: Segment) -> Chunk | None:
    # trim blank lines at both ends so citations point at real code
    s, e = _trim(lines, seg.start, seg.end)
    if s > e:
        return None
    ps = pe = None
    if seg.parent:
        ps, pe = _trim(lines, *seg.parent)
        ps, pe = min(ps, s), max(pe, e)
        if (ps, pe) == (s, e):
            ps = pe = None
    return Chunk(
        path=f.path,
        start_line=s,
        end_line=e,
        text="\n".join(lines[s - 1 : e]),
        kind=seg.kind,
        language=f.language,
        symbol=seg.symbol,
        parent_start=ps,
        parent_end=pe,
    )


def _windows(lines: list[str], start: int, end: int, max_tokens: int, overlap_tokens: int) -> list[tuple[int, int]]:
    """Split lines[start..end] (1-based, inclusive) into token-bounded windows with overlap."""
    spans: list[tuple[int, int]] = []
    i = start
    while i <= end:
        tokens, j = 0, i
        while j <= end and (tokens == 0 or tokens + approx_tokens(lines[j - 1]) <= max_tokens):
            tokens += approx_tokens(lines[j - 1]) if lines[j - 1].strip() else 0
            j += 1
        spans.append((i, j - 1))
        if j > end:
            break
        # step back so consecutive windows overlap by ~overlap_tokens
        back, k = 0, j - 1
        while k > i + 1 and back < overlap_tokens:
            back += approx_tokens(lines[k - 1])
            k -= 1
        i = max(k + 1, i + 1)
    return spans


def _seg_tokens(lines: list[str], seg: Segment) -> int:
    return sum(approx_tokens(l) for l in lines[seg.start - 1 : seg.end] if l.strip())


def _fill_gaps(members: list[Segment], n_lines: int, containers: list[Segment]) -> list[Segment]:
    """Return members plus 'gap' segments covering every line no member covers."""

    def container_at(line: int) -> str | None:
        best = None
        for c in containers:
            if c.start <= line <= c.end and (best is None or c.start >= best.start):
                best = c
        return best.symbol if best else None

    out: list[Segment] = []
    cursor = 1
    for m in sorted(members, key=lambda s: s.start):
        if m.start > cursor:
            out.append(Segment(cursor, m.start - 1, "gap", container_at(cursor)))
        out.append(m)
        cursor = max(cursor, m.end + 1)
    if cursor <= n_lines:
        out.append(Segment(cursor, n_lines, "gap", container_at(cursor)))
    return out


def _merge_and_split(
    lines: list[str], segs: list[Segment], min_tokens: int, max_tokens: int, overlap_tokens: int
) -> list[Segment]:
    # drop whitespace-only segments
    segs = [s for s in segs if any(l.strip() for l in lines[s.start - 1 : s.end])]

    # greedily merge small neighbours (e.g. runs of one-line properties) up to max_tokens
    merged: list[Segment] = []
    for s in segs:
        if merged:
            prev = merged[-1]
            pt, st = _seg_tokens(lines, prev), _seg_tokens(lines, s)
            if (pt < min_tokens or st < min_tokens) and pt + st <= max_tokens:
                names: list[str] = []
                for sym in (prev.symbol, s.symbol):
                    for n in (sym or "").split(", "):
                        if n and n not in names:
                            names.append(n)
                # drop a bare container name when a member of it is already listed
                names = [n for n in names if not any(o != n and o.startswith(n + ".") for o in names)]
                symbol = ", ".join(names[:5]) or None
                kind = prev.kind if prev.kind == s.kind else "group"
                merged[-1] = Segment(prev.start, s.end, kind, symbol)
                continue
        merged.append(s)

    # split anything still too big; each piece remembers the whole member as its parent
    out: list[Segment] = []
    for s in merged:
        if _seg_tokens(lines, s) <= max_tokens:
            out.append(s)
            continue
        for a, b in _windows(lines, s.start, s.end, max_tokens, overlap_tokens):
            out.append(Segment(a, b, s.kind, s.symbol, parent=(s.start, s.end)))
    return out


# --------------------------------------------------------------------------- strategies


class FileChunker(Chunker):
    name = "file"

    def chunk(self, f: SourceFile) -> list[Chunk]:
        lines = f.lines
        c = _make_chunk(f, lines, Segment(1, len(lines), "file", None))
        return [c] if c else []


class FixedChunker(Chunker):
    name = "fixed"

    def __init__(self, max_tokens: int = 400, overlap_tokens: int = 60):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, f: SourceFile) -> list[Chunk]:
        lines = f.lines
        spans = _windows(lines, 1, len(lines), self.max_tokens, self.overlap_tokens)
        chunks = []
        for i, (a, b) in enumerate(spans):
            # parent = this window plus its neighbours
            parent = (spans[max(0, i - 1)][0], spans[min(len(spans) - 1, i + 1)][1])
            c = _make_chunk(f, lines, Segment(a, b, "window", parent=parent))
            if c:
                chunks.append(c)
        return chunks


class FunctionChunker(Chunker):
    name = "function"

    def __init__(self, max_tokens: int = 800, min_tokens: int = 80, overlap_tokens: int = 60):
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_tokens = overlap_tokens

    def members(self, f: SourceFile, lines: list[str]) -> tuple[list[Segment], list[Segment]]:
        if f.language == "python":
            return python_members(f.text)
        if f.language in BRACE_LANGUAGES:
            return brace_members(lines, f.language)
        if f.language == "markdown":
            return markdown_sections(lines), []
        return [], []

    def chunk(self, f: SourceFile) -> list[Chunk]:
        lines = f.lines
        members, containers = self.members(f, lines)
        segs = _fill_gaps(members, len(lines), containers)
        for s in segs:
            if s.kind == "gap":
                s.symbol = _gap_symbol(lines, s)
        segs = _merge_and_split(lines, segs, self.min_tokens, self.max_tokens, self.overlap_tokens)
        return [c for s in segs if (c := _make_chunk(f, lines, s))]


class AstChunker(FunctionChunker):
    """Like ``function`` but members come from tree-sitter (pip install tree-sitter-language-pack)."""

    name = "ast"

    def members(self, f: SourceFile, lines: list[str]) -> tuple[list[Segment], list[Segment]]:
        found = treesitter_members(f.text, f.language, lines)
        return found if found is not None else super().members(f, lines)


CHUNKERS = {"file": FileChunker, "fixed": FixedChunker, "function": FunctionChunker, "ast": AstChunker}


def get_chunker(name: str, **kwargs) -> Chunker:
    try:
        return CHUNKERS[name](**kwargs)
    except KeyError:
        raise ValueError(f"Unknown chunker {name!r}; choose from {sorted(CHUNKERS)}") from None


def chunker_from_meta(desc: dict) -> Chunker:
    params = {k: v for k, v in desc.items() if k != "name"}
    return get_chunker(desc["name"], **params)


_DECL = re.compile(
    r"^\s*(?:(?:public|private|protected|internal|static|async|override|virtual|abstract|sealed|partial|"
    r"readonly|unsafe|extern|new|export|def|func|fn)\s+)+[\w<>\[\],.?\s]*?\b([A-Za-z_]\w*)\s*(?:<[^()]*>)?\s*\("
)


def _gap_symbol(lines: list[str], seg: Segment) -> str | None:
    """Name a gap by the declarations in it (expression-bodied members like `int X() => ...;`)."""
    names: list[str] = []
    for line in lines[seg.start - 1 : seg.end]:
        m = _DECL.match(line)
        if m and m.group(1) not in names and m.group(1) not in _CONTROL:
            names.append(m.group(1))
    owner = seg.symbol
    if not names:
        return owner
    return ", ".join(f"{owner}.{n}" if owner else n for n in names[:4])


def _is_comment_or_attr(line: str) -> bool:
    s = line.strip()
    return (s.startswith(("///", "//", "/*", "*", "[", "@", "#[")) and not s.startswith("*/")) or s.endswith("*/")


def _extend_leading_comments(members: list[Segment], lines: list[str]) -> None:
    for m in members:
        s = m.start
        while s > 1 and _is_comment_or_attr(lines[s - 2]):
            s -= 1
        m.start = s


# --------------------------------------------------------------------------- python (ast)


def python_members(source: str, split_class_over_lines: int = 80) -> tuple[list[Segment], list[Segment]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [], []
    members: list[Segment] = []
    containers: list[Segment] = []

    def start_of(node) -> int:
        decos = getattr(node, "decorator_list", [])
        return min([node.lineno] + [d.lineno for d in decos])

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            members.append(Segment(start_of(node), node.end_lineno, "function", node.name))
        elif isinstance(node, ast.ClassDef):
            start, end = start_of(node), node.end_lineno
            containers.append(Segment(start, end, "type", node.name))
            if end - start + 1 <= split_class_over_lines:
                members.append(Segment(start, end, "type", node.name))
                continue
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    members.append(Segment(start_of(sub), sub.end_lineno, "function", f"{node.name}.{sub.name}"))
    return members, containers


# --------------------------------------------------------------------------- tree-sitter

_TS_LANG = {"tsx": "tsx", "csharp": "csharp"}  # askcode language -> tree-sitter-language-pack name (default: same)

_TS_MEMBERS = {
    "method_declaration", "constructor_declaration", "destructor_declaration", "operator_declaration",
    "conversion_operator_declaration", "property_declaration", "indexer_declaration", "event_declaration",
    "function_definition", "function_declaration", "method_definition", "function_item", "method",
    "singleton_method", "compact_constructor_declaration", "enum_declaration", "delegate_declaration",
    "generator_function_declaration", "abstract_method_signature", "method_signature",
}
_TS_CONTAINERS = {
    "class_declaration", "struct_declaration", "interface_declaration", "record_declaration",
    "record_struct_declaration", "namespace_declaration", "file_scoped_namespace_declaration",
    "class_definition", "impl_item", "trait_item", "mod_item", "class", "module", "object_declaration",
    "class_specifier", "struct_specifier", "namespace_definition", "abstract_class_declaration",
    "trait_declaration", "enum_item",
}
_TS_TRANSPARENT = {
    "export_statement", "declaration_list", "class_body", "field_declaration_list", "body", "block",
    "template_declaration", "linkage_specification", "ERROR",
}


def _first_branch_only(text: str) -> str:
    """Blank out #else/#elif branches and directive lines (line numbers are preserved).

    Code like `#if X  void F() {  #else  void F()\\n{  #endif` has unbalanced braces in each branch
    taken together; compilers only ever see one branch, so the parser should too.
    """
    out, skip = [], []
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("#"):
            d = s[1:].strip().split(None, 1)[0] if len(s) > 1 else ""
            if d in ("if", "ifdef", "ifndef"):
                skip.append(False)
            elif d in ("elif", "else") and skip:
                skip[-1] = True
            elif d == "endif" and skip:
                skip.pop()
            if d in ("if", "ifdef", "ifndef", "elif", "else", "endif"):
                out.append("")
                continue
        out.append("" if any(skip) else line)
    return "\n".join(out)


def treesitter_members(text: str, language: str, lines: list[str]) -> tuple[list[Segment], list[Segment]] | None:
    """Members via tree-sitter. Returns None when tree-sitter or the grammar isn't available."""
    if language in ("markdown", "text"):
        return None
    try:
        from tree_sitter_language_pack import get_parser

        parser = get_parser(_TS_LANG.get(language, language))
    except Exception:
        return None
    if language in PREPROCESSOR_LANGUAGES:
        text = _first_branch_only(text)
    tree = parser.parse(text.encode("utf-8"))
    members: list[Segment] = []
    containers: list[Segment] = []

    def name_of(node) -> str | None:
        n = node.child_by_field_name("name")
        if n is None and node.type in ("lexical_declaration", "variable_declaration"):
            decl = next((c for c in node.named_children if c.type == "variable_declarator"), None)
            n = decl.child_by_field_name("name") if decl else None
        if n is None and node.type == "type_declaration":
            spec = next((c for c in node.named_children if c.type == "type_spec"), None)
            n = spec.child_by_field_name("name") if spec else None
        return n.text.decode("utf-8", "replace") if n is not None else None

    def qualify(owner: str | None, name: str | None) -> str | None:
        return f"{owner}.{name}" if owner and name else (name or owner)

    def is_function_var(node) -> bool:
        if node.type not in ("lexical_declaration", "variable_declaration"):
            return False
        for d in node.named_children:
            v = d.child_by_field_name("value") if d.type == "variable_declarator" else None
            if v is not None and v.type in ("arrow_function", "function", "function_expression", "class"):
                return True
        return False

    def walk(node, owner: str | None) -> None:
        for child in node.named_children:
            t = child.type
            span = (child.start_point[0] + 1, child.end_point[0] + 1)
            if t == "decorated_definition":  # python decorators wrap the def/class
                inner = child.child_by_field_name("definition")
                if inner is not None and inner.type == "function_definition":
                    members.append(Segment(*span, "function", qualify(owner, name_of(inner))))
                elif inner is not None:
                    _container(child, inner, owner)
            elif t in _TS_MEMBERS or is_function_var(child):
                kind = "property" if t in ("property_declaration", "indexer_declaration", "event_declaration") else "function"
                members.append(Segment(*span, kind, qualify(owner, name_of(child))))
            elif t in _TS_CONTAINERS or t == "type_declaration" and "struct" in child.text[:200].decode("utf-8", "replace"):
                _container(child, child, owner)
            elif t in _TS_TRANSPARENT:
                walk(child, owner)

    def _container(outer, node, owner):
        name = qualify(owner, name_of(node))
        containers.append(Segment(outer.start_point[0] + 1, outer.end_point[0] + 1, "type", name))
        body = node.child_by_field_name("body")
        walk(body if body is not None else node, name)

    walk(tree.root_node, None)
    if tree.root_node.has_error and not members:
        return None  # parser gave up; let the brace scanner try
    # file-scoped namespace (C#): declarations are siblings, not children
    for c in tree.root_node.named_children:
        if c.type == "file_scoped_namespace_declaration":
            containers.append(Segment(c.start_point[0] + 1, len(lines), "type", name_of(c)))
    _extend_leading_comments(members, lines)
    members.sort(key=lambda s: (s.start, -s.end))
    outer: list[Segment] = []
    for m in members:
        if outer and m.start <= outer[-1].end:
            continue
        outer.append(m)
    return outer, containers


# --------------------------------------------------------------------------- C-family (brace scanner)

_CONTAINER_RE = re.compile(r"\b(class|struct|interface|record|enum|namespace|trait|impl|object|module|union)\b\s*([A-Za-z_]\w*)?")
_CONTROL = {
    "if", "else", "for", "foreach", "while", "do", "switch", "try", "catch", "finally", "using", "lock", "fixed",
    "unchecked", "checked", "unsafe", "return", "new", "get", "set", "init", "add", "remove", "case", "default",
    "await", "yield", "throw", "when", "select", "defer", "match", "loop", "static", "synchronized", "sizeof",
    "typeof", "nameof",
}


def _classify(header: str) -> tuple[str, str | None]:
    """Classify the text before a '{' as type / function / property / block."""
    h = re.sub(r"^\s*(\[[^\]]*\]\s*)+", "", header).strip()  # leading attributes / annotations
    h = re.sub(r"^\s*(@\w+(\([^)]*\))?\s*)+", "", h).strip()
    if not h:
        return "block", None
    before_paren = h.split("(", 1)[0]
    first = re.match(r"[A-Za-z_$][\w$]*", h)
    first_word = first.group(0) if first else ""

    m = _CONTAINER_RE.search(before_paren)
    if m and "=" not in before_paren:
        name = m.group(2)
        if m.group(1) == "struct" or not name:
            t = re.search(r"\btype\s+([A-Za-z_]\w*)", h)  # go: type X struct {
            name = t.group(1) if t else name
        return "type", name or m.group(1)

    if first_word in _CONTROL and first_word not in ("static",):
        return "block", None

    if h.startswith("func"):  # go
        g = re.match(r"func\s*(?:\([^)]*\)\s*)?([A-Za-z_]\w*)", h)
        return ("function", g.group(1)) if g else ("block", None)

    if "=>" in h or re.search(r"(?<![=!<>])=(?![=>])", before_paren):
        js = re.match(r"(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=", h)
        if js:
            return "function", js.group(1)
        return "block", None

    if "(" in h:
        names = re.findall(r"([A-Za-z_$~][\w$]*)\s*(?:<[^()]*>)?\s*$", before_paren)
        name = names[0] if names else None
        if not name or name in _CONTROL:
            return "block", None
        return "function", name

    ids = re.findall(r"[A-Za-z_]\w*", h)
    return ("property", ids[-1]) if ids else ("block", None)


def brace_members(lines: list[str], language: str) -> tuple[list[Segment], list[Segment]]:
    """Find top-level members (methods, properties, nested types' members) by tracking braces.

    Not a real parser: it skips strings and comments and only follows the first branch of
    ``#if/#else`` blocks, which is enough to keep brace depth correct on typical code.
    """
    stack: list[dict] = []
    members: list[Segment] = []
    containers: list[Segment] = []
    header: list[str] = []
    header_start: int | None = None
    in_block_comment = False
    string_delim: str | None = None  # active multi-line string delimiter
    pp_skip: list[bool] = []  # per #if nesting: are we in a later (#else/#elif) branch?

    def reset_header():
        nonlocal header, header_start
        header, header_start = [], None

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if language in PREPROCESSOR_LANGUAGES and stripped.startswith("#") and string_delim is None and not in_block_comment:
            directive = stripped[1:].strip().split(None, 1)[0] if len(stripped) > 1 else ""
            if directive in ("if", "ifdef", "ifndef"):
                pp_skip.append(False)
            elif directive in ("elif", "else") and pp_skip:
                pp_skip[-1] = True
            elif directive == "endif" and pp_skip:
                pp_skip.pop()
            continue
        if any(pp_skip):
            continue

        j, n = 0, len(line)
        while j < n:
            c = line[j]
            if in_block_comment:
                end = line.find("*/", j)
                if end == -1:
                    j = n
                    break
                in_block_comment, j = False, end + 2
                continue
            if string_delim is not None:
                if string_delim == '"""':
                    end = line.find('"""', j)
                    if end == -1:
                        j = n
                        break
                    string_delim, j = None, end + 3
                elif string_delim == '@"':
                    if c == '"' and line[j + 1 : j + 2] == '"':
                        j += 2
                    elif c == '"':
                        string_delim, j = None, j + 1
                    else:
                        j += 1
                else:  # " ' `
                    if c == "\\":
                        j += 2
                    elif c == string_delim:
                        string_delim, j = None, j + 1
                    else:
                        j += 1
                continue

            two = line[j : j + 2]
            if two == "//":
                break
            if two == "/*":
                in_block_comment, j = True, j + 2
                continue
            if line[j : j + 3] == '"""':
                string_delim, j = '"""', j + 3
                continue
            if c == '"':
                prefix = line[max(0, j - 2) : j]
                string_delim = '@"' if "@" in prefix and language == "csharp" else '"'
                j += 1
                continue
            if c == "`" and language in ("javascript", "typescript", "tsx", "go"):
                string_delim, j = "`", j + 1
                continue
            if c == "'":
                if language in CHAR_LITERAL_LANGUAGES:
                    m = re.match(r"'(\\.[^']{0,8}|[^'\\])'", line[j:])
                    if m:
                        j += m.end()
                        continue
                    j += 1  # e.g. rust lifetime
                    continue
                string_delim, j = "'", j + 1
                continue

            if c == "{":
                kind, name = _classify("".join(header))
                stack.append({"kind": kind, "name": name, "start": header_start or i})
                reset_header()
            elif c == "}":
                if stack:
                    b = stack.pop()
                    if b["kind"] == "type":
                        containers.append(Segment(b["start"], i, "type", b["name"]))
                    elif b["kind"] in ("function", "property") and all(p["kind"] == "type" for p in stack):
                        owner = next((p["name"] for p in reversed(stack) if p["name"]), None)
                        sym = f"{owner}.{b['name']}" if owner and b["name"] else b["name"]
                        members.append(Segment(b["start"], i, b["kind"], sym))
                reset_header()
            elif c == ";":
                reset_header()
            else:
                if header_start is None and not c.isspace():
                    header_start = i
                header.append(c)
            j += 1
        if string_delim in ('"', "'"):
            string_delim = None  # ordinary strings don't span lines
        header.append(" ")

    _extend_leading_comments(members, lines)
    # drop members nested inside other members (can happen with unbalanced code)
    members.sort(key=lambda s: (s.start, -s.end))
    outer: list[Segment] = []
    for m in members:
        if outer and m.start <= outer[-1].end:
            continue
        outer.append(m)
    return outer, containers


# --------------------------------------------------------------------------- markdown


def markdown_sections(lines: list[str]) -> list[Segment]:
    heads: list[tuple[int, str]] = []
    in_fence = False
    for i, line in enumerate(lines, 1):
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
        elif not in_fence and (m := re.match(r"^#{1,6}\s+(.*)", line)):
            heads.append((i, m.group(1).strip()))
        elif (
            not in_fence and i > 1 and re.match(r"^(-{3,}|={3,})\s*$", line)
            and lines[i - 2].strip() and not lines[i - 2].lstrip().startswith(("|", "-", "*", "#"))
        ):  # setext heading: "Title\n-----"
            heads.append((i - 1, lines[i - 2].strip()))
    segs = []
    for idx, (start, title) in enumerate(heads):
        end = heads[idx + 1][0] - 1 if idx + 1 < len(heads) else len(lines)
        segs.append(Segment(start, end, "section", title))
    return segs
