#!/usr/bin/env python3
"""
One-off migration of scripts, tests and notebooks to the configuration with ``solver_config``.

    python tools/migrate_config.py rewrite PATH [PATH ...] [--dry-run]
    python tools/migrate_config.py check PATH [PATH ...]

``rewrite`` changes Python files and the code cells of notebooks in place:

* ``SimulationConfig(...)`` calls: the scheme keywords go into ``solver_config=FVSolverConfig(...)`` or
  ``FDSolverConfig(...)``, the passive-scalar keywords into ``passive_scalar_config=PassiveScalarConfig(...)``,
  the diffusion keywords into ``diffusion_config=DiffusionConfig(...)``. The solver is the one the old code ran:
  ``solver_mode`` defaults to ``FINITE_DIFFERENCE``; the finite-difference solver ran its low-storage integrator
  for ``RK4_LSRK`` and SSPRK(5,4) for every other ``time_integrator``; the finite-volume solver ran VL2 for
  ``time_integrator=VL2`` whatever ``split`` said, otherwise the unsplit scheme (``RK2_SSP``) or the split one
  (``MUSCL``). Keywords the old code ignored in that mode are dropped, each with a line in the report.
* ``dict(...)`` calls that hold ``solver_mode`` together with scheme keywords are rewritten like a call.
* ``X._replace(...)`` with moved keywords becomes a nested ``_replace`` on the container.
* ``BackendConfig(halo_exchange="ppermute")`` gets the integer constant.
* the renamed constants (``MUSCL`` -> ``MUSCL_HANCOCK``, ``RK4_SSP`` -> ``SSPRK54``, ``RK4_LSRK`` -> ``LSRK4``)
  are renamed in imports and uses, new names are added to the astronomix import of the scope, constants that
  are no longer used are removed from it.

Only the call or the import in question is rewritten; everything else keeps its formatting. A notebook is
written back through its JSON (code cells only) and only if the unchanged notebook survives that round trip
byte for byte. What cannot be resolved statically is reported as ``UNRESOLVED`` and left as it is: keywords
through ``**kwargs``, a mode or an integrator chosen at run time, a combination the old code rejected.
``--kwargs 'EXPRESSION=fd'`` tells the tool that the dictionary behind ``**EXPRESSION`` holds only keywords of
the finite-difference container (``fv``: of the finite-volume container, ``none``: no moved keyword).

``check`` lists every remaining old keyword or removed constant in a tree (code, comments, strings, markdown)
and exits with status 1 if there is one. ``docs/source/apidocs``, ``ROADMAP.md`` and this file are skipped.

Standard library only.
"""

import argparse
import ast
import io
import json
import os
import re
import sys
import tokenize

# ----------------------------------------------------------------------------------------------------------
# The mapping
# ----------------------------------------------------------------------------------------------------------

OLD_INTEGRATORS = ("RK2_SSP", "MUSCL", "RK4_SSP", "RK4_LSRK", "VL2")
RENAMED_CONSTANTS = {"MUSCL": "MUSCL_HANCOCK", "RK4_SSP": "SSPRK54", "RK4_LSRK": "LSRK4"}
REMOVED_CONSTANTS = ("UNSPLIT", "SPLIT", "MUSCL", "RK4_SSP", "RK4_LSRK")
#: constants an import may lose when the rewrite drops their last use
SCHEME_CONSTANTS = (
    "RK2_SSP", "MUSCL_HANCOCK", "VL2", "SSPRK54", "LSRK4", "MINMOD", "VAN_LEER", "SUPERBEE", "DOUBLE_MINMOD",
    "HLL", "HLLC", "LAX_FRIEDRICHS", "HLLD", "UNSPLIT", "SPLIT",
)  # fmt: skip
#: names the rewrite introduces (constants go to an import of constants, containers next to SimulationConfig)
NEW_CONSTANTS = ("DONOR_CELL", "PIECEWISE_LINEAR", "GLM", "CONSTRAINED_TRANSPORT", "PPERMUTE", "PAIR_ALL_TO_ALL")
NEW_CONTAINERS = ("FVSolverConfig", "FDSolverConfig", "PassiveScalarConfig", "DiffusionConfig")

FV_FIELDS = (
    "time_integrator", "reconstruction", "limiter", "riemann_solver", "divergence_control",
    "glm_extended_source", "first_order_flux_correction",
)  # fmt: skip
FD_FIELDS = (
    "time_integrator", "weno_z", "weno_admissible_face_state", "weno_positivity_preserving",
    "weno_ad_frozen_weights", "ct_upwind_dissipation", "ct_upwind_strength", "dual_energy",
)  # fmt: skip
#: old keywords that moved into a solver container under the same name
FV_KEYWORDS = ("limiter", "riemann_solver", "glm_extended_source", "first_order_flux_correction")
FD_KEYWORDS = FD_FIELDS[1:]
#: old keyword -> field of the new container
PASSIVE_SCALAR_KEYWORDS = {
    "num_passive_scalars": "num_scalars",
    "passive_scalar_bounds": "bounds",
    "passive_scalar_cfl": "cfl",
    "max_passive_scalar_substeps": "max_substeps",
}
DIFFUSION_KEYWORDS = {
    "diffusion": "viscosity",
    "viscosity_type": "viscosity_type",
    "resistivity": "resistivity",
    "thermal_conduction": "thermal_conduction",
    "conduction_density_weighted": "conduction_density_weighted",
    "conduction_order": "conduction_order",
}
#: old keywords that became module constants (keyword -> the default, the only value that can be dropped)
REMOVED_KEYWORDS = {"weno_epsilon": 1e-7, "dual_energy_eta": 1e-3}
#: old keywords that decide integrator, reconstruction and divergence control together
INTEGRATOR_KEYWORDS = ("time_integrator", "split", "first_order_fallback", "vl2_constrained_transport")
OLD_KEYWORDS = (
    set(INTEGRATOR_KEYWORDS) | set(FV_KEYWORDS) | set(FD_KEYWORDS) | set(PASSIVE_SCALAR_KEYWORDS)
    | set(DIFFUSION_KEYWORDS) | set(REMOVED_KEYWORDS)
)  # fmt: skip
#: old keywords that exist nowhere in the new configuration (the others are fields of a container now)
OLD_ONLY_KEYWORDS = (
    {"split", "first_order_fallback", "vl2_constrained_transport", "diffusion"}
    | set(PASSIVE_SCALAR_KEYWORDS) | set(REMOVED_KEYWORDS)
)  # fmt: skip
#: old keywords that are also common words or fields of other containers (not enough to recognise a dictionary)
AMBIGUOUS_KEYWORDS = {"split", "diffusion", "resistivity", "limiter"}
CONTAINER_FIELDS = ("solver_config", "passive_scalar_config", "diffusion_config")
HALO_EXCHANGE_CONSTANTS = {"ppermute": "PPERMUTE", "pair_all_to_all": "PAIR_ALL_TO_ALL"}
MAX_LINE = 88


# ----------------------------------------------------------------------------------------------------------
# Source text helpers
# ----------------------------------------------------------------------------------------------------------


class Source:
    """A source text with the conversion from ``ast`` positions (line, UTF-8 column) to offsets."""

    def __init__(self, text):
        self.text = text
        self.lines = text.splitlines(keepends=True)
        self.starts = [0]
        for line in self.lines:
            self.starts.append(self.starts[-1] + len(line))

    def offset(self, lineno, column):
        line = self.lines[lineno - 1] if lineno <= len(self.lines) else ""
        if not line.isascii():
            column = len(line.encode("utf-8")[:column].decode("utf-8"))
        return self.starts[lineno - 1] + column

    def span(self, node):
        return self.offset(node.lineno, node.col_offset), self.offset(node.end_lineno, node.end_col_offset)

    def segment(self, node):
        start, end = self.span(node)
        return self.text[start:end]

    def line_start(self, offset):
        return self.text.rfind("\n", 0, offset) + 1

    def indent(self, offset):
        """The leading white space of the line that holds ``offset``."""
        start = self.line_start(offset)
        return re.match(r"[ \t]*", self.text[start:]).group(0)

    def lineno(self, offset):
        return self.text.count("\n", 0, offset) + 1


def split_entries(inner):
    """Split the text between the parentheses of a call into its arguments.

    Returns ``(entries, tail_comments)``. An entry is a dictionary with the argument text (``text``, verbatim,
    without the comma), the comment lines above it (``leading``) and the comment behind it (``trailing``).
    """
    wrapped = "(" + inner + ")"
    starts = [0]
    for line in wrapped.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))

    def offset(position):
        return starts[position[0] - 1] + position[1]

    entries, current, last, pending, depth, new_line = [], None, None, [], 0, True

    def finish():
        nonlocal current, last
        if current is not None:
            current["text"] = wrapped[current["start"] : current["end"]]
            if current["inner_comment"] is not None:
                current["trailing"] = current["inner_comment"]
            entries.append(current)
            last, current = current, None

    for token in tokenize.generate_tokens(io.StringIO(wrapped).readline):
        if token.type in (tokenize.NEWLINE, tokenize.ENDMARKER, tokenize.INDENT, tokenize.DEDENT):
            continue
        if token.type == tokenize.NL:
            new_line = True
            continue
        if token.type == tokenize.COMMENT:
            if depth == 1 and current is None:
                if last is not None and not new_line and last["trailing"] is None:
                    last["trailing"] = token.string
                else:
                    pending.append(token.string)
            elif depth == 1:
                current["inner_comment"] = token.string
            continue
        if token.type == tokenize.OP and token.string in "([{":
            depth += 1
            if depth == 1:
                continue
        elif token.type == tokenize.OP and token.string in ")]}":
            depth -= 1
            if depth == 0:
                break
        elif token.type == tokenize.OP and token.string == "," and depth == 1:
            finish()
            new_line = False
            continue
        if current is None:
            current = dict(start=offset(token.start), end=offset(token.end), leading=pending, trailing=None)
            current["inner_comment"] = None
            pending = []
        else:
            current["end"] = offset(token.end)
            current["inner_comment"] = None
        new_line = False
    finish()
    for entry in entries:
        text = entry["text"]
        match = re.match(r"([A-Za-z_]\w*)\s*=(?!=)\s*", text)
        if text.startswith("**"):
            entry["kind"], entry["name"], entry["value"] = "star2", None, text[2:].strip()
        elif text.startswith("*"):
            entry["kind"], entry["name"], entry["value"] = "star", None, text[1:].strip()
        elif match:
            entry["kind"], entry["name"], entry["value"] = "keyword", match.group(1), text[match.end() :]
        else:
            entry["kind"], entry["name"], entry["value"] = "positional", None, text
    return entries, pending


def new_entry(name, value, source=None):
    """An argument ``name=value`` that takes over the comments of the entry ``source``."""
    entry = dict(kind="keyword", name=name, value=value, text=f"{name}={value}", leading=[], trailing=None)
    if source is not None:
        entry["leading"], entry["trailing"] = source["leading"], source["trailing"]
    return entry


def shifted(text, by="    "):
    """``text`` with its continuation lines indented by four more spaces."""
    lines = text.split("\n")
    return "\n".join([lines[0]] + [by + line if line.strip() else line for line in lines[1:]])


def render_inline(entries):
    return ", ".join(entry["text"] for entry in entries)


def can_inline(entries):
    return all("\n" not in e["text"] and not e["leading"] and not e["trailing"] for e in entries)


def render_container(constructor, entries, indent, prefix_length):
    """``constructor(...)`` with ``entries``: on one line if that fits, otherwise one keyword per line."""
    inline = f"{constructor}({render_inline(entries)})"
    if can_inline(entries) and prefix_length + len(inline) + 1 <= MAX_LINE:
        return inline
    return f"{constructor}(\n{render_lines(entries, indent + '    ', shift=True)}{indent})"


def render_lines(entries, indent, tail_comments=(), shift=False):
    out = []
    for entry in entries:
        for comment in entry["leading"]:
            out.append(f"{indent}{comment}\n")
        text = shifted(entry["text"]) if shift else entry["text"]
        trailing = f"  {entry['trailing']}" if entry["trailing"] else ""
        out.append(f"{indent}{text},{trailing}\n")
    for comment in tail_comments:
        out.append(f"{indent}{comment}\n")
    return "".join(out)


# ----------------------------------------------------------------------------------------------------------
# The rewrite of one call
# ----------------------------------------------------------------------------------------------------------


class Unresolved(Exception):
    """The call cannot be rewritten statically; it stays as it is."""


def constant_name(text):
    """The constant a value names (``VL2``, ``module.VL2``), or None."""
    match = re.fullmatch(r"(?:[A-Za-z_]\w*\.)*([A-Z][A-Z0-9_]*)", text.strip())
    return match.group(1) if match else None


def is_wrapped_safely(text):
    """Whether ``text`` can stand as the condition of a conditional expression without parentheses."""
    try:
        node = ast.parse(f"({text})", mode="eval").body
    except SyntaxError:
        return False
    return not isinstance(node, (ast.IfExp, ast.Lambda, ast.NamedExpr, ast.Yield, ast.YieldFrom, ast.Starred))


def condition(text):
    return text if is_wrapped_safely(text) else f"({text})"


def scheme_entries(keywords, mode, drops):
    """The entries of the solver container for the old keywords of one configuration.

    ``keywords`` maps the old keyword to its entry; ``mode`` is ``"FV"`` or ``"FD"``. Returns
    ``(constructor, entries)``; appends a line to ``drops`` for every keyword the old code ignored. Raises
    ``Unresolved`` for what needs a decision by hand.
    """
    for keyword, default in REMOVED_KEYWORDS.items():
        if keyword in keywords:
            try:
                same = float(ast.literal_eval(keywords[keyword]["value"])) == default
            except (ValueError, SyntaxError):
                same = False
            if not same:
                raise Unresolved(f"{keyword}={keywords[keyword]['value']} is not the default of the module constant")
            drops.append(f"{keyword}={keywords[keyword]['value']} dropped (the default; a module constant now)")

    integrator = keywords.get("time_integrator")
    split = keywords.get("split")
    fallback = keywords.get("first_order_fallback")
    transport = keywords.get("vl2_constrained_transport")
    fields = {}

    if mode == "FV":
        integrator_name = constant_name(integrator["value"]) if integrator else "RK2_SSP"
        if integrator_name not in OLD_INTEGRATORS:
            raise Unresolved(f"time_integrator={integrator['value']} is not one of the old integrator constants")
        split_name = constant_name(split["value"]) if split else "UNSPLIT"
        if split_name not in ("UNSPLIT", "SPLIT"):
            raise Unresolved(f"split={split['value']} is not UNSPLIT or SPLIT")
        fallback_value = fallback["value"].strip() if fallback else "False"
        if integrator_name == "VL2":
            new_integrator = "VL2"
            if split_name == "SPLIT":
                drops.append("split=SPLIT dropped (time_integrator=VL2 selected the VL2 scheme whatever split said)")
        elif split_name == "UNSPLIT":
            if integrator_name != "RK2_SSP":
                raise Unresolved(f"split=UNSPLIT with time_integrator={integrator_name} was rejected by the old code")
            new_integrator = "RK2_SSP"
            if fallback_value != "False":
                drops.append(
                    "first_order_fallback with the unsplit scheme: kept as reconstruction=DONOR_CELL "
                    "(the array form ignores it, the Pallas kernel honours it, as before)"
                )
        elif integrator_name == "MUSCL":
            new_integrator = "MUSCL_HANCOCK"
        elif fallback_value == "True":
            new_integrator = "MUSCL_HANCOCK"
            drops.append(
                f"time_integrator={integrator_name} with split=SPLIT and first_order_fallback=True ran the "
                "donor-cell split scheme: time_integrator=MUSCL_HANCOCK"
            )
        else:
            raise Unresolved(f"split=SPLIT with time_integrator={integrator_name} was rejected by the old code")
        if integrator is not None or split_name == "SPLIT":
            fields["time_integrator"] = new_entry("time_integrator", new_integrator, integrator or split)
        if fallback_value == "True":
            fields["reconstruction"] = new_entry("reconstruction", "DONOR_CELL", fallback)
        elif fallback_value != "False":
            value = f"DONOR_CELL if {condition(fallback_value)} else PIECEWISE_LINEAR"
            fields["reconstruction"] = new_entry("reconstruction", value, fallback)
        if transport is not None:
            transport_value = transport["value"].strip()
            if transport_value == "True":
                fields["divergence_control"] = new_entry("divergence_control", "CONSTRAINED_TRANSPORT", transport)
            elif transport_value != "False":
                value = f"CONSTRAINED_TRANSPORT if {condition(transport_value)} else GLM"
                fields["divergence_control"] = new_entry("divergence_control", value, transport)
        for keyword in FV_KEYWORDS:
            if keyword in keywords:
                fields[keyword] = keywords[keyword]
        for keyword in FD_KEYWORDS:
            if keyword in keywords:
                value = keywords[keyword]["value"].strip()
                if keyword == "ct_upwind_dissipation" and value != "False":
                    raise Unresolved("ct_upwind_dissipation in a finite-volume configuration was rejected by the old code")
                drops.append(f"{keyword}={value} dropped (ignored by the finite-volume solver)")
        return "FVSolverConfig", [fields[name] for name in FV_FIELDS if name in fields]

    if integrator is not None:
        value = integrator["value"].strip()
        name = constant_name(value)
        if name == "RK4_LSRK":
            fields["time_integrator"] = new_entry("time_integrator", "LSRK4", integrator)
        elif name == "RK4_SSP":
            fields["time_integrator"] = new_entry("time_integrator", "SSPRK54", integrator)
        elif name in OLD_INTEGRATORS:
            drops.append(f"time_integrator={name} dropped (the finite-difference solver ran SSPRK(5,4) for it)")
        else:
            fields["time_integrator"] = new_entry("time_integrator", conditional_integrator(value), integrator)
    if split is not None:
        drops.append(f"split={split['value'].strip()} dropped (ignored by the finite-difference solver)")
    if fallback is not None:
        drops.append(f"first_order_fallback={fallback['value'].strip()} dropped (ignored by the finite-difference solver)")
    if transport is not None:
        if transport["value"].strip() != "False":
            raise Unresolved("vl2_constrained_transport in a finite-difference configuration was rejected by the old code")
        drops.append("vl2_constrained_transport=False dropped (ignored by the finite-difference solver)")
    for keyword in FV_KEYWORDS:
        if keyword in keywords:
            drops.append(f"{keyword}={keywords[keyword]['value'].strip()} dropped (ignored by the finite-difference solver)")
    for keyword in FD_KEYWORDS:
        if keyword in keywords:
            fields[keyword] = keywords[keyword]
    return "FDSolverConfig", [fields[name] for name in FD_FIELDS if name in fields]


def conditional_integrator(value):
    """``A if condition else B`` with old integrator constants, for a finite-difference configuration."""
    try:
        node = ast.parse(f"({value})", mode="eval").body
    except SyntaxError:
        node = None
    if isinstance(node, ast.IfExp) and isinstance(node.body, ast.Name) and isinstance(node.orelse, ast.Name):
        if node.body.id in OLD_INTEGRATORS and node.orelse.id in OLD_INTEGRATORS:
            first = "LSRK4" if node.body.id == "RK4_LSRK" else "SSPRK54"
            second = "LSRK4" if node.orelse.id == "RK4_LSRK" else "SSPRK54"
            test = ast.get_source_segment(f"({value})", node.test)
            return f"{first} if {test} else {second}"
    raise Unresolved(f"time_integrator={value} is chosen at run time")


def moved_group(entries, mapping):
    """The entries of a passive-scalar or diffusion container, renamed to the fields of the container."""
    out = []
    for entry in entries:
        if entry["kind"] == "keyword" and entry["name"] in mapping:
            renamed = dict(entry)
            renamed["name"] = mapping[entry["name"]]
            renamed["text"] = f"{renamed['name']}={entry['value']}"
            out.append(renamed)
    return out


class Rewriter:
    """Rewrites one Python source text; collects the report lines."""

    def __init__(self, label, report, hints=None):
        self.label = label
        self.report = report
        #: what the dictionary of a ``**expression`` holds, by the text of the expression (``--kwargs``)
        self.hints = hints or {}

    def say(self, kind, lineno, message):
        self.report.append((kind, f"{self.label}:{lineno}", message))

    # -- targets ------------------------------------------------------------------------------------------

    def targets(self, source, tree):
        """The calls that may need a rewrite: ``(offset of the opening parenthesis, kind, node)``."""
        found = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            name = function.id if isinstance(function, ast.Name) else function.attr if isinstance(function, ast.Attribute) else None
            names = {keyword.arg for keyword in node.keywords}
            kind = None
            if name == "SimulationConfig":
                if names & OLD_KEYWORDS or None in names:
                    kind = "config"
            elif name == "dict" and isinstance(function, ast.Name):
                # a keyword with a string value belongs to a description of something else
                old = {
                    keyword.arg for keyword in node.keywords
                    if keyword.arg in OLD_KEYWORDS
                    and not (isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str))
                }  # fmt: skip
                if old - AMBIGUOUS_KEYWORDS or len(old) > 1 or (old and "solver_mode" in names):
                    kind = "dict"
            elif name == "_replace" and isinstance(function, ast.Attribute):
                if names & OLD_KEYWORDS and self.is_configuration(source.segment(function.value)):
                    kind = "replace"
            elif name == "BackendConfig":
                for keyword in node.keywords:
                    if keyword.arg == "halo_exchange" and isinstance(keyword.value, ast.Constant):
                        if isinstance(keyword.value.value, str):
                            kind = "halo"
            if kind is not None:
                end_of_function = source.offset(function.end_lineno, function.end_col_offset)
                found.append((source.text.index("(", end_of_function), kind, node))
        return found

    @staticmethod
    def is_configuration(receiver):
        """Whether ``receiver._replace`` is a replace on a ``SimulationConfig`` (not on a container or on
        parameters)."""
        last = receiver.strip().split(".")[-1]
        if last in CONTAINER_FIELDS or "param" in last.lower() or last.endswith(")"):
            return False
        return not re.search(r"registered|helper|solver|scheme|preset|ATHENAPK_VL2|FD_PAPER", last)

    # -- one source ---------------------------------------------------------------------------------------

    def rewrite(self, text):
        """The rewritten source text (the same object if nothing changed)."""
        try:
            ast.parse(text)
        except SyntaxError as error:
            self.say("UNRESOLVED", error.lineno or 0, f"syntax error, not rewritten: {error.msg}")
            return text
        original = text
        limit = len(text) + 1
        while True:
            source = Source(text)
            tree = ast.parse(text)
            candidates = [target for target in self.targets(source, tree) if target[0] < limit]
            if not candidates:
                break
            parenthesis, kind, node = max(candidates, key=lambda target: target[0])
            limit = parenthesis
            try:
                replacement = getattr(self, "rewrite_" + kind)(source, tree, node, parenthesis)
            except Unresolved as problem:
                self.say("UNRESOLVED", node.lineno, f"{kind}: {problem}")
                continue
            if replacement is not None:
                end = source.offset(node.end_lineno, node.end_col_offset)
                text = text[:parenthesis] + replacement + text[end:]
        text = self.rename_constants(text)
        text = self.fix_imports(text, original)
        return text

    # -- layout -------------------------------------------------------------------------------------------

    def arguments(self, source, node, parenthesis):
        end = source.offset(node.end_lineno, node.end_col_offset)
        return split_entries(source.text[parenthesis + 1 : end - 1])

    def render_call(self, source, node, parenthesis, entries, tail_comments):
        """The text ``(...)`` of a call with ``entries``, in the layout of the original call."""
        end = source.offset(node.end_lineno, node.end_col_offset)
        old = source.text[parenthesis:end]
        base = source.indent(parenthesis)
        if "\n" not in old and can_inline(entries) and not tail_comments:
            line_start = source.line_start(parenthesis)
            line_end = source.text.find("\n", end)
            line_end = len(source.text) if line_end < 0 else line_end
            new_length = (parenthesis - line_start) + len(render_inline(entries)) + 2 + (line_end - end)
            if new_length <= max(MAX_LINE, line_end - line_start):
                return f"({render_inline(entries)})"
        if "\n" in old:
            inner = old[1:-1]
            first = re.match(r"[ \t]*\n([ \t]*)\S", inner)
            indent = first.group(1) if first else base + "    "
            last_line = old[old.rfind("\n") + 1 : -1]
            closing = last_line if not last_line.strip() else base
        else:
            indent, closing = base + "    ", base
        return f"(\n{render_lines(entries, indent, tail_comments)}{closing})"

    def container_entry(self, field, constructor, entries, indent, source_entry=None):
        value = render_container(constructor, entries, indent, len(indent) + len(field) + 1)
        entry = new_entry(field, value)
        if can_inline(entries) and source_entry is not None and "\n" not in value:
            pass
        return entry

    def entry_indent(self, source, node, parenthesis):
        end = source.offset(node.end_lineno, node.end_col_offset)
        old = source.text[parenthesis:end]
        first = re.match(r"\([ \t]*\n([ \t]*)\S", old)
        return first.group(1) if first else source.indent(parenthesis) + "    "

    # -- SimulationConfig(...) and dict(...) -----------------------------------------------------------------

    def rewrite_config(self, source, tree, node, parenthesis, dictionary=False):
        entries, tail_comments = self.arguments(source, node, parenthesis)
        keywords = {entry["name"]: entry for entry in entries if entry["kind"] == "keyword"}
        if any(entry["kind"] in ("positional", "star") for entry in entries):
            raise Unresolved("positional arguments")
        mode_text = keywords["solver_mode"]["value"].strip() if "solver_mode" in keywords else None
        hinted = {}
        for entry in entries:
            if entry["kind"] != "star2":
                continue
            if entry["value"] in self.hints:
                hinted[id(entry)] = self.hints[entry["value"]]
                continue
            carried = resolve_dictionary(source, tree, entry["value"])
            if carried is None:
                raise Unresolved(f"keywords through **{entry['value']} (not a dictionary literal of this file)")
            moved = sorted(set(carried) & OLD_KEYWORDS)
            if moved:
                raise Unresolved(f"**{entry['value']} carries old keywords ({', '.join(moved)})")
            if "solver_mode" in carried:
                mode_text = carried["solver_mode"].strip()
        if not set(keywords) & OLD_KEYWORDS and not set(hinted.values()) - {"none"}:
            return None
        if dictionary and mode_text is None:
            raise Unresolved("a dictionary with old keywords and without solver_mode (the mode is set elsewhere)")
        mode = {None: "FD", "FINITE_DIFFERENCE": "FD", "FINITE_VOLUME": "FV"}.get(
            mode_text if mode_text is None else constant_name(mode_text) or mode_text
        )
        if mode is None:
            raise Unresolved(f"solver_mode={mode_text} is chosen at run time")

        drops = []
        constructor, solver = scheme_entries(keywords, mode, drops)
        for entry in entries:
            # a dictionary that holds only keywords of the solver container (told by --kwargs)
            if hinted.get(id(entry), "none") != "none":
                if hinted[id(entry)] != mode.lower():
                    raise Unresolved(f"**{entry['value']} holds {hinted[id(entry)]} keywords, the mode is {mode}")
                solver.append(entry)
        passive = moved_group(entries, PASSIVE_SCALAR_KEYWORDS)
        diffusion = moved_group(entries, DIFFUSION_KEYWORDS)
        indent = self.entry_indent(source, node, parenthesis)
        solver_names = set(INTEGRATOR_KEYWORDS) | set(FV_KEYWORDS) | set(FD_KEYWORDS) | set(REMOVED_KEYWORDS)
        out, placed = [], set()
        for entry in entries:
            name = entry["name"] if entry["kind"] == "keyword" else None
            if name in solver_names or hinted.get(id(entry), "none") != "none":
                group, constructed, field = "solver", (constructor, solver), "solver_config"
            elif name in PASSIVE_SCALAR_KEYWORDS:
                group, constructed, field = "passive", ("PassiveScalarConfig", passive), "passive_scalar_config"
            elif name in DIFFUSION_KEYWORDS:
                group, constructed, field = "diffusion", ("DiffusionConfig", diffusion), "diffusion_config"
            else:
                out.append(entry)
                continue
            if group not in placed and constructed[1]:
                placed.add(group)
                out.append(self.container_entry(field, constructed[0], constructed[1], indent))
        for message in drops:
            self.say("DROPPED", node.lineno, message)
        if dictionary:
            self.say(
                "NOTE", node.lineno,
                "dictionary rewritten as a complete scheme description: the call that takes it must not add "
                "scheme keywords",
            )  # fmt: skip
        return self.render_call(source, node, parenthesis, out, tail_comments)

    def rewrite_dict(self, source, tree, node, parenthesis):
        return self.rewrite_config(source, tree, node, parenthesis, dictionary=True)

    # -- X._replace(...) --------------------------------------------------------------------------------------

    def rewrite_replace(self, source, tree, node, parenthesis):
        entries, tail_comments = self.arguments(source, node, parenthesis)
        receiver = source.segment(node.func.value)
        keywords = {entry["name"]: entry for entry in entries if entry["kind"] == "keyword"}
        if any(entry["kind"] != "keyword" for entry in entries):
            raise Unresolved("_replace with ** or positional arguments")
        blocked = [name for name in ("solver_mode",) + INTEGRATOR_KEYWORDS + tuple(REMOVED_KEYWORDS) if name in keywords]
        if blocked:
            raise Unresolved(f"_replace of {', '.join(blocked)} (decides the scheme together with other keywords)")
        if "\n" in receiver or not re.fullmatch(r"[\w.]+", receiver):
            raise Unresolved(f"_replace on the expression {receiver.splitlines()[0]}")
        indent = self.entry_indent(source, node, parenthesis)
        solver = [entry for entry in entries if entry["name"] in FV_KEYWORDS or entry["name"] in FD_KEYWORDS]
        passive = moved_group(entries, PASSIVE_SCALAR_KEYWORDS)
        diffusion = moved_group(entries, DIFFUSION_KEYWORDS)
        out, placed = [], set()
        for entry in entries:
            name = entry["name"]
            if name in FV_KEYWORDS or name in FD_KEYWORDS:
                field, group = "solver_config", solver
            elif name in PASSIVE_SCALAR_KEYWORDS:
                field, group = "passive_scalar_config", passive
            elif name in DIFFUSION_KEYWORDS:
                field, group = "diffusion_config", diffusion
            else:
                out.append(entry)
                continue
            if field not in placed:
                placed.add(field)
                out.append(self.container_entry(field, f"{receiver}.{field}._replace", group, indent))
        if solver:
            self.say(
                "NOTE", node.lineno,
                f"{receiver}.solver_config._replace(...): {receiver} must carry a solver_config (not None)",
            )  # fmt: skip
        return self.render_call(source, node, parenthesis, out, tail_comments)

    # -- BackendConfig(halo_exchange="...") -------------------------------------------------------------------

    def rewrite_halo(self, source, tree, node, parenthesis):
        entries, tail_comments = self.arguments(source, node, parenthesis)
        out = []
        for entry in entries:
            if entry["kind"] == "keyword" and entry["name"] == "halo_exchange":
                try:
                    constant = HALO_EXCHANGE_CONSTANTS[ast.literal_eval(entry["value"])]
                except (ValueError, SyntaxError, KeyError):
                    raise Unresolved(f"halo_exchange={entry['value']} is not one of the known strings")
                entry = new_entry("halo_exchange", constant, entry)
            out.append(entry)
        return self.render_call(source, node, parenthesis, out, tail_comments)

    # -- constants and imports --------------------------------------------------------------------------------

    def rename_constants(self, text):
        """Rename the renamed constants in uses and imports; report the removed ones that are still used."""
        source = Source(text)
        edits = []
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Name) and node.id in RENAMED_CONSTANTS:
                start, end = source.span(node)
                edits.append((start, end, RENAMED_CONSTANTS[node.id]))
            elif isinstance(node, ast.Attribute) and node.attr in RENAMED_CONSTANTS:
                end = source.offset(node.end_lineno, node.end_col_offset)
                edits.append((end - len(node.attr), end, RENAMED_CONSTANTS[node.attr]))
            elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("astronomix"):
                statement_start, statement_end = source.span(node)
                statement = source.text[statement_start:statement_end]
                names_start = statement.index("import") + len("import")
                for alias in node.names:
                    if alias.name in RENAMED_CONSTANTS:
                        match = re.search(rf"\b{alias.name}\b", statement[names_start:])
                        start = statement_start + names_start + match.start()
                        edits.append((start, start + len(alias.name), RENAMED_CONSTANTS[alias.name]))
            if isinstance(node, (ast.Name, ast.Attribute)):
                name = node.id if isinstance(node, ast.Name) else node.attr
                if name in ("SPLIT", "UNSPLIT"):
                    self.say("UNRESOLVED", node.lineno, f"{name} is still used (the constant is gone)")
        for start, end, new in sorted(edits, reverse=True):
            text = text[:start] + new + text[end:]
        return text

    def fix_imports(self, text, original):
        """Add the names the rewrite introduced to the astronomix imports and remove the constants that lost
        their last use."""
        if text == original:
            return text
        for _ in range(40):
            source = Source(text)
            tree = ast.parse(text)
            edit = self.next_import_edit(source, tree, original)
            if edit is None:
                break
            start, end, new = edit
            text = text[:start] + new + text[end:]
        return text

    def next_import_edit(self, source, tree, original):
        imports = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("astronomix") and node.level == 0
        ]  # fmt: skip
        parents = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent

        def scope(node):
            node = parents.get(node)
            while node is not None and not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                node = parents.get(node)
            return node

        def scopes(node):
            chain, current = [], scope(node)
            while current is not None:
                chain.append(current)
                current = scope(current)
            return chain + [None]

        star = any(alias.name == "*" for node in imports for alias in node.names)
        bound = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    bound.setdefault(alias.asname or alias.name.split(".")[0], []).append(scope(node))
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                bound.setdefault(node.id, []).append(scope(node))
        used = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used.setdefault(node.id, []).append(node)

        # a name the rewrite introduced that no import of an enclosing scope provides
        for name in NEW_CONTAINERS + NEW_CONSTANTS + tuple(RENAMED_CONSTANTS.values()) + SCHEME_CONSTANTS:
            for use in used.get(name, []):
                chain = scopes(use)
                if star or any(where in chain for where in bound.get(name, [])):
                    continue
                target = self.import_for(name, imports, chain, scope)
                if target is None:
                    self.say("UNRESOLVED", use.lineno, f"{name} is not imported (no astronomix import to extend)")
                    return None
                return self.changed_import(source, target, add=name)
        # a scheme constant whose last use the rewrite removed
        before = {
            node.id for node in ast.walk(ast.parse(original))
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }  # fmt: skip
        for node in imports:
            for alias in node.names:
                name = RENAMED_CONSTANTS.get(alias.name, alias.name)
                old_name = {new: old for old, new in RENAMED_CONSTANTS.items()}.get(name, name)
                if alias.asname is None and name in SCHEME_CONSTANTS and name not in used:
                    if old_name in before or name in before or name in ("SPLIT", "UNSPLIT"):
                        return self.changed_import(source, node, remove=alias.name)
        return None

    @staticmethod
    def import_for(name, imports, chain, scope):
        """The import statement that should provide ``name``: in the innermost scope of the use that has an
        astronomix import; constants go to an import of constants, containers next to ``SimulationConfig``."""
        constant = name.isupper()
        for where in chain:
            local = [node for node in imports if scope(node) is where]
            if not local:
                continue

            def rank(node):
                names = [alias.name for alias in node.names]
                has_config = "SimulationConfig" in names
                has_constants = any(n.isupper() and n not in ("STATE_TYPE", "FIELD_TYPE") for n in names)
                if constant:
                    return (0 if has_constants and not has_config else 1 if has_constants else 2 if has_config else 3)
                return 0 if has_config else 1 if any(not n.isupper() and n[:1].isupper() for n in names) else 2

            return min(local, key=lambda node: (rank(node), node.lineno))
        return None

    def changed_import(self, source, node, add=None, remove=None):
        """The edit ``(start, end, text)`` that adds a name to or removes a name from an import statement."""
        start, end = source.span(node)
        old = source.text[start:end]
        names = [(alias.name, alias.asname) for alias in node.names]
        plain = [name for name, _ in names]
        if remove is not None:
            names = [(name, asname) for name, asname in names if name != remove]
            if not names:
                line_start = source.line_start(start)
                line_end = source.text.find("\n", end)
                line_end = len(source.text) if line_end < 0 else line_end + 1
                if not source.text[line_start:start].strip() and not source.text[end:line_end].strip():
                    return line_start, line_end, ""
                return start, end, "pass"
        one_per_line = "\n" in old and all(
            re.search(rf"^[ \t]*{re.escape(name)}(?:[ \t]+as[ \t]+\w+)?[ \t]*,?[ \t]*(?:#.*)?$", old, re.M) for name in plain
        )
        if one_per_line:
            lines = old.split("\n")

            def line_of(name):
                pattern = rf"^[ \t]*{re.escape(name)}(?:[ \t]+as[ \t]+\w+)?[ \t]*,?[ \t]*(?:#.*)?$"
                return next(index for index, line in enumerate(lines) if re.match(pattern, line))

            if remove is not None:
                del lines[line_of(remove)]
            else:
                indent = re.match(r"[ \t]*", lines[line_of(plain[0])]).group(0)
                later = [name for name in plain if name > add] if plain == sorted(plain) else []
                last = line_of(plain[-1])
                if not lines[last].split("#")[0].rstrip().endswith(","):
                    lines[last] = re.sub(r"^([^#]*\S)", r"\1,", lines[last], count=1)
                position = line_of(later[0]) if later else last + 1
                lines.insert(position, f"{indent}{add},")
            return start, end, "\n".join(lines)
        if add is not None:
            names.append((add, None))
            if plain == sorted(plain):
                names.sort(key=lambda item: item[0])
        listed = [f"{name} as {asname}" if asname else name for name, asname in names]
        head = f"from {node.module} import "
        base = source.indent(start)
        single = head + ", ".join(listed)
        if "\n" not in old and "(" not in old and len(base) + len(single) <= max(MAX_LINE, len(base) + len(old)):
            return start, end, single
        if "\n" not in old and len(base) + len(single) + 2 <= MAX_LINE:
            return start, end, f"{head}({', '.join(listed)})"
        body = "".join(f"{base}    {item},\n" for item in listed)
        return start, end, f"{head}(\n{body}{base})"


def resolve_dictionary(source, tree, expression):
    """The keys (with the text of their values) of the dictionary that ``**expression`` unpacks, if it is a
    literal of this file that nothing changes afterwards; None otherwise."""
    try:
        node = ast.parse(f"({expression})", mode="eval").body
    except SyntaxError:
        return None
    if isinstance(node, ast.Name):
        name, values = node.id, []
        for other in ast.walk(tree):
            if isinstance(other, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in other.targets):
                values.append(other.value)
            elif isinstance(other, ast.AnnAssign) and isinstance(other.target, ast.Name) and other.target.id == name:
                values.append(other.value)
            elif isinstance(other, (ast.AugAssign, ast.Subscript, ast.Attribute, ast.arg, ast.For, ast.NamedExpr)):
                # the dictionary is changed, bound a second time or it is a function argument
                if isinstance(other, ast.arg) and other.arg == name:
                    return None
                target = getattr(other, "value", None) if isinstance(other, (ast.Subscript, ast.Attribute)) else None
                if isinstance(target, ast.Name) and target.id == name:
                    if isinstance(other, ast.Subscript) and not isinstance(other.ctx, ast.Load):
                        return None
                    if isinstance(other, ast.Attribute) and other.attr in ("update", "pop", "setdefault", "clear"):
                        return None
                if isinstance(other, ast.AugAssign) and isinstance(other.target, ast.Name) and other.target.id == name:
                    return None
                if isinstance(other, (ast.For, ast.NamedExpr)) and isinstance(other.target, ast.Name):
                    if other.target.id == name:
                        return None
        if len(values) != 1 or values[0] is None:
            return None
        node, text = values[0], source
    else:
        text = None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict" and not node.args:
        if all(keyword.arg for keyword in node.keywords):
            return {
                keyword.arg: (text.segment(keyword.value) if text else ast.unparse(keyword.value))
                for keyword in node.keywords
            }
    if isinstance(node, ast.Dict) and all(isinstance(k, ast.Constant) and isinstance(k.value, str) for k in node.keys):
        return {k.value: (text.segment(v) if text else ast.unparse(v)) for k, v in zip(node.keys, node.values)}
    return None


# ----------------------------------------------------------------------------------------------------------
# Notebooks
# ----------------------------------------------------------------------------------------------------------

CELL_MARK = "# %%migrate-config-cell {}%%\n"
MAGIC_MARK = "# %%migrate-config-magic%% "


def notebook_code(notebook):
    """The code cells of a notebook as one Python source (magic lines commented out), and their indices."""
    parts, indices = [], []
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        lines = "".join(cell["source"]).split("\n")
        lines = [MAGIC_MARK + line if line.lstrip().startswith(("%", "!")) else line for line in lines]
        parts.append(CELL_MARK.format(index) + "\n".join(lines) + "\n")
        indices.append(index)
    return "".join(parts), indices


def notebook_cells(code):
    """Inverse of ``notebook_code``: cell index -> source text."""
    cells = {}
    for part in re.split(r"(?m)^(?=# %%migrate-config-cell \d+%%\n)", code):
        match = re.match(r"# %%migrate-config-cell (\d+)%%\n", part)
        if not match:
            continue
        lines = part[match.end() :][:-1].split("\n")
        lines = [line.replace(MAGIC_MARK, "", 1) if line.startswith(MAGIC_MARK) else line for line in lines]
        cells[int(match.group(1))] = "\n".join(lines)
    return cells


def dump_notebook(notebook):
    return json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"


def rewrite_notebook(path, report, dry_run, hints=None):
    with open(path, encoding="utf-8", newline="") as file:
        raw = file.read()
    notebook = json.loads(raw)
    code, _ = notebook_code(notebook)
    new_code = Rewriter(path, report, hints).rewrite(code)
    if new_code == code:
        return False
    if dump_notebook(notebook) != raw:
        report.append(("UNRESOLVED", f"{path}:0", "the notebook does not survive a JSON round trip byte for byte; rewrite by hand"))
        return False
    for index, text in notebook_cells(new_code).items():
        cell = notebook["cells"][index]
        if text != "".join(cell["source"]):
            lines = text.split("\n")
            cell["source"] = [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    if not dry_run:
        with open(path, "w", encoding="utf-8", newline="") as file:
            file.write(dump_notebook(notebook))
    return True


def rewrite_python(path, report, dry_run, hints=None):
    with open(path, encoding="utf-8", newline="") as file:
        text = file.read()
    new_text = Rewriter(path, report, hints).rewrite(text)
    if new_text == text:
        return False
    compile(new_text, path, "exec")
    if not dry_run:
        with open(path, "w", encoding="utf-8", newline="") as file:
            file.write(new_text)
    return True


# ----------------------------------------------------------------------------------------------------------
# Check mode
# ----------------------------------------------------------------------------------------------------------

#: identifiers that exist nowhere in the new package (searched in comments, strings, markdown)
TEXT_PATTERNS = [
    (re.compile(r"\b(UNSPLIT|RK4_SSP|RK4_LSRK)\b"), "removed constant {}"),
    (re.compile(r"(?<![\w.\"'-])(SPLIT|MUSCL)(?![\w-])(?=\s*[,)`\]]|\s*$)"), "removed constant {}"),
    (re.compile(r"=\s*(SPLIT|MUSCL)\b(?![-_ ]?[Hh]ancock)"), "removed constant {}"),
    (
        re.compile(
            r"\b(first_order_fallback|vl2_constrained_transport|passive_scalar_bounds|passive_scalar_cfl"
            r"|max_passive_scalar_substeps)\b"
        ),
        "old keyword {}",
    ),
    # the two removed fields: as a field only, the Pallas kernels keep a parameter of the name weno_epsilon
    (re.compile(r"\b(?:config|cfg|SimulationConfig|FDSolverConfig)\.(weno_epsilon|dual_energy_eta)\b"), "removed field {}"),
    (re.compile(r"\b(num_passive_scalars)\s*=(?!=)"), "old keyword {}"),
    (re.compile(r"\b(?:config|cfg|SimulationConfig)\.(num_passive_scalars)\b"), "old field {}"),
    (
        re.compile(
            r"\b(?:config|cfg|SimulationConfig)\.(time_integrator|split|limiter|riemann_solver|glm_extended_source"
            r"|first_order_flux_correction|weno_z|weno_admissible_face_state|weno_positivity_preserving"
            r"|weno_ad_frozen_weights|ct_upwind_dissipation|ct_upwind_strength|dual_energy|diffusion"
            r"|viscosity_type|resistivity|thermal_conduction|conduction_density_weighted|conduction_order)\b"
        ),
        "old field config.{}",
    ),
    (re.compile(r"\bhalo_exchange\s*=\s*([\"'](?:ppermute|pair_all_to_all)[\"'])"), "halo_exchange as a string {}"),
]
CALL_IN_TEXT = re.compile(r"\bSimulationConfig\(")
KEYWORD_IN_TEXT = re.compile(r"(?<![\w.])(" + "|".join(sorted(OLD_KEYWORDS)) + r")\s*=(?!=)")
SKIPPED = ("docs/source/apidocs", "ROADMAP.md", "tools/migrate_config.py", ".git/")


def check_text(label, text, findings, first_line=1):
    """Old names in free text: comments, strings, markdown."""
    for number, line in enumerate(text.split("\n"), first_line):
        for pattern, message in TEXT_PATTERNS:
            for match in pattern.finditer(line):
                findings.append((f"{label}:{number}", message.format(match.group(1))))
    for match in CALL_IN_TEXT.finditer(text):
        depth, position = 1, match.end()
        while position < len(text) and depth:
            depth += {"(": 1, ")": -1}.get(text[position], 0)
            position += 1
        body = text[match.end() : position - 1]
        level, top = 0, []
        for character in body:
            top.append(character if level == 0 else " ")
            level += {"(": 1, "[": 1, "{": 1, ")": -1, "]": -1, "}": -1}.get(character, 0)
        for keyword in KEYWORD_IN_TEXT.finditer("".join(top)):
            number = first_line + text.count("\n", 0, match.end() + keyword.start())
            findings.append((f"{label}:{number}", f"old keyword {keyword.group(1)} in a SimulationConfig(...) in text"))


def is_record(value):
    """Whether a dictionary value is a converted setting (``int(...)``, ``bool(...)``, ...) or a string."""
    if isinstance(value, ast.Constant):
        return isinstance(value.value, str)
    function = value.func if isinstance(value, ast.Call) else None
    return isinstance(function, ast.Name) and function.id in ("int", "bool", "float", "str", "list")


def helper_signature(function):
    """The named parameters of a function, and whether its ``**keywords`` reach a ``SimulationConfig``
    (directly, through ``_replace`` or through a dictionary that is updated with them)."""
    arguments = function.args
    explicit = {argument.arg for argument in arguments.posonlyargs + arguments.args + arguments.kwonlyargs}
    if arguments.kwarg is None:
        return explicit, False
    rest = arguments.kwarg.arg
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]

    def called(node):
        return node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)

    def starred(node):
        values = [keyword.value for keyword in node.keywords if keyword.arg is None]
        return {value.id for value in values if isinstance(value, ast.Name)}

    def first_name(node):
        return node.args[0].id if node.args and isinstance(node.args[0], ast.Name) else None

    configurations = [node for node in calls if called(node) in ("SimulationConfig", "_replace")]
    updated = {
        node.func.value.id
        for node in calls
        if called(node) == "update" and isinstance(node.func.value, ast.Name) and first_name(node) == rest
    }
    forwards = any(starred(node) & ({rest} | updated) for node in configurations)
    return explicit, forwards


def check_code(label, text, findings):
    """Old names in Python code (and in its comments and strings)."""
    try:
        tree = ast.parse(text)
    except SyntaxError as error:
        findings.append((f"{label}:{error.lineno}", f"syntax error: {error.msg}"))
        return
    source = Source(text)

    def found(node, message):
        findings.append((f"{label}:{node.lineno}", message))

    helpers = {
        node.name: helper_signature(node)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in REMOVED_CONSTANTS:
            found(node, f"removed constant {node.id}")
        elif isinstance(node, ast.Attribute):
            receiver = source.segment(node.value).strip()
            last = receiver.split(".")[-1]
            if node.attr in REMOVED_CONSTANTS and node.attr.isupper() and not isinstance(node.value, ast.Constant):
                found(node, f"removed constant {receiver}.{node.attr}")
            elif node.attr in OLD_ONLY_KEYWORDS - {"split", "diffusion", "num_passive_scalars"}:
                found(node, f"old field {receiver}.{node.attr}")
            elif node.attr in OLD_KEYWORDS - {"split"} and re.fullmatch(r"(\w*_)?(config|cfg)", last):
                if last not in CONTAINER_FIELDS and not re.search(r"solver|diffusion|scalar", last):
                    found(node, f"old field {receiver}.{node.attr}")
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("astronomix"):
            for alias in node.names:
                if alias.name in REMOVED_CONSTANTS:
                    found(node, f"import of the removed constant {alias.name}")
        elif isinstance(node, ast.Call):
            function = node.func
            name = function.id if isinstance(function, ast.Name) else function.attr if isinstance(function, ast.Attribute) else None
            names = {keyword.arg for keyword in node.keywords}
            if name == "SimulationConfig":
                for keyword in sorted(names & OLD_KEYWORDS):
                    found(node, f"old keyword {keyword} in SimulationConfig(...)")
            elif name == "dict" and isinstance(function, ast.Name):
                old = names & OLD_ONLY_KEYWORDS - AMBIGUOUS_KEYWORDS
                if "solver_mode" in names or names & {"mhd", "dimensionality", "num_cells", "box_size"}:
                    old |= names & OLD_KEYWORDS
                # a value converted with int(...), bool(...), ... is a record of a setting, not a keyword
                old -= {keyword.arg for keyword in node.keywords if is_record(keyword.value)}
                for keyword in sorted(old):
                    found(node, f"old keyword {keyword} in a dictionary of configuration keywords")
            elif name == "_replace" and isinstance(function, ast.Attribute):
                receiver = source.segment(function.value).strip()
                old = names & OLD_ONLY_KEYWORDS - AMBIGUOUS_KEYWORDS
                if Rewriter.is_configuration(receiver) and not re.search(r"\)$", receiver):
                    old |= names & OLD_KEYWORDS
                if re.search(r"registered_variables|registry", receiver.split(".")[-1]):
                    old = set()  # the variable registry has fields of the same names
                for keyword in sorted(old):
                    found(node, f"old keyword {keyword} in {receiver}._replace(...)")
            elif name is not None and name not in NEW_CONTAINERS:
                # a helper that hands its keywords on: old names reach the configuration through it
                explicit, forwards = helpers.get(name, (set(), False))
                old = names & OLD_ONLY_KEYWORDS - AMBIGUOUS_KEYWORDS
                if forwards:
                    old |= names & OLD_KEYWORDS
                for keyword in sorted(old - explicit):
                    found(node, f"old keyword {keyword} in a call of {name}(...)")
            for keyword in node.keywords:
                value = keyword.value
                if keyword.arg == "halo_exchange" and isinstance(value, ast.Constant) and isinstance(value.value, str):
                    found(node, f"halo_exchange as a string ({value.value!r})")
                if keyword.arg in ("time_integrator", "split", "limiter", "riemann_solver"):
                    if isinstance(value, ast.Constant) and isinstance(value.value, int):
                        if not isinstance(value.value, bool):
                            found(node, f"{keyword.arg}={value.value}: a number in place of a constant")
        elif isinstance(node, ast.Dict):
            keys = {key.value for key in node.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)}
            old = keys & OLD_ONLY_KEYWORDS - AMBIGUOUS_KEYWORDS
            if "solver_mode" in keys:
                old |= keys & OLD_KEYWORDS
            for keyword in sorted(old):
                found(node, f"old keyword {keyword!r} in a dictionary of configuration keywords")
    # comments and strings are free text
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError):
        tokens = []
    for token in tokens:
        if token.type == tokenize.COMMENT or (token.type == tokenize.STRING and len(token.string) > 30):
            check_text(label, token.string, findings, token.start[0])


def check_path(path, findings):
    if path.endswith(".ipynb"):
        with open(path, encoding="utf-8") as file:
            notebook = json.load(file)
        code, _ = notebook_code(notebook)
        found = []
        check_code(path, code.replace(MAGIC_MARK, "#"), found)
        cell_starts = [(m.start(), int(m.group(1))) for m in re.finditer(r"(?m)^# %%migrate-config-cell (\d+)%%\n", code)]
        line_cells = []
        for start, index in cell_starts:
            line_cells.append((code.count("\n", 0, start) + 1, index))
        for where, message in found:
            line = int(where.rsplit(":", 1)[1] or 0)
            cell = max((item for item in line_cells if item[0] <= line), default=(0, "?"))
            findings.append((f"{path}#cell{cell[1]}:{line - cell[0]}", message))
        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") != "code":
                check_text(f"{path}#cell{index}", "".join(cell["source"]), findings)
    elif path.endswith(".py"):
        with open(path, encoding="utf-8") as file:
            check_code(path, file.read(), findings)
    else:
        with open(path, encoding="utf-8", errors="replace") as file:
            check_text(path, file.read(), findings)


# ----------------------------------------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------------------------------------


def files_under(paths, suffixes):
    for path in paths:
        if os.path.isfile(path):
            yield path
            continue
        for directory, subdirectories, names in os.walk(path):
            subdirectories[:] = sorted(d for d in subdirectories if not d.startswith(".") and d != "__pycache__")
            for name in sorted(names):
                full = os.path.join(directory, name)
                if name.endswith(suffixes) and not any(skip in full.replace(os.sep, "/") for skip in SKIPPED):
                    yield full


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("rewrite", "check"))
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--dry-run", action="store_true", help="rewrite: report only, change no file")
    parser.add_argument(
        "--kwargs", action="append", default=[], metavar="EXPRESSION=fd|fv|none",
        help="rewrite: what the dictionary of a '**EXPRESSION' argument holds: only keywords of the "
        "finite-difference container (fd), of the finite-volume container (fv), or no moved keyword (none)",
    )  # fmt: skip
    arguments = parser.parse_args()
    hints = dict(item.rsplit("=", 1) for item in arguments.kwargs)
    if set(hints.values()) - {"fd", "fv", "none"}:
        parser.error("--kwargs takes EXPRESSION=fd, EXPRESSION=fv or EXPRESSION=none")

    if arguments.command == "rewrite":
        report, changed = [], []
        for path in files_under(arguments.paths, (".py", ".ipynb")):
            rewrite = rewrite_notebook if path.endswith(".ipynb") else rewrite_python
            if rewrite(path, report, arguments.dry_run, hints):
                changed.append(path)
        for kind, where, message in report:
            print(f"{kind:<10} {where}: {message}")
        unresolved = sum(kind == "UNRESOLVED" for kind, _, _ in report)
        print(f"{len(changed)} file(s) {'would be ' if arguments.dry_run else ''}rewritten, {unresolved} unresolved")
        return 0

    findings = []
    for path in files_under(arguments.paths, (".py", ".ipynb", ".md", ".rst", ".txt", ".toml", ".cfg", ".yml", ".yaml")):
        check_path(path, findings)
    for where, message in findings:
        print(f"{where}: {message}")
    print(f"{len(findings)} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
