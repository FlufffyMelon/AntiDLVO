import time
from contextlib import contextmanager
from typing import Dict, Optional, List


class _ProfileNode:
    """Internal node for hierarchical profiling."""

    def __init__(self, label: str, parent: Optional["_ProfileNode"] = None):
        self.label: str = label
        self.parent: Optional[_ProfileNode] = parent
        self.children: Dict[str, _ProfileNode] = {}
        self.time: float = 0.0
        self.count: int = 0

    def get_or_create_child(self, label: str) -> "_ProfileNode":
        node = self.children.get(label)
        if node is None:
            node = _ProfileNode(label, parent=self)
            self.children[label] = node
        return node


class Profiler:
    """
    Lightweight profiler for timing labeled code sections with hierarchical nesting.
    Use as: with profiler.measure("label"): ...
    Accumulates total time and counts per label and prints an indented hierarchy.
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        # Flat stats retained for backward compatibility and non-timed counters
        self._stats: Dict[str, Dict[str, float]] = {}
        # Hierarchical tree
        self._root = _ProfileNode("root", parent=None)
        self._stack: List[_ProfileNode] = [self._root]

    @contextmanager
    def measure(self, label: str):
        if not self.enabled:
            yield
            return
        # Enter hierarchy
        parent = self._stack[-1]
        node = parent.get_or_create_child(label)
        self._stack.append(node)
        start = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - start
            # Update hierarchical node
            node.time += dt
            node.count += 1
            # Update flat stats (kept for compatibility and increment/add usage)
            entry = self._stats.setdefault(label, {"time": 0.0, "count": 0.0})
            entry["time"] += dt
            entry["count"] += 1
            # Leave hierarchy
            self._stack.pop()

    def add(self, label: str, duration_s: float):
        if not self.enabled:
            return
        entry = self._stats.setdefault(label, {"time": 0.0, "count": 0.0})
        entry["time"] += duration_s
        entry["count"] += 1

    def increment(self, label: str, count: int = 1):
        if not self.enabled:
            return
        entry = self._stats.setdefault(label, {"time": 0.0, "count": 0.0})
        entry["count"] += count

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        return {k: dict(v) for k, v in self._stats.items()}

    def _append_tree_lines(
        self, node: _ProfileNode, total_runtime_s: float, lines: List[str], depth: int
    ) -> None:
        # Sort children by time descending
        children = sorted(node.children.values(), key=lambda n: n.time, reverse=True)
        for child in children:
            pct = (child.time / total_runtime_s * 100.0) if total_runtime_s > 0 else 0.0
            indent = " " * 4 * depth
            avg_time = (child.time / child.count) if child.count > 0 else 0.0
            # Report total time as avg_time * calls (equals accumulated child.time), along with calls and avg per call
            lines.append(
                f"{indent}{child.label}: total={child.time:.6f} s, calls={child.count}, avg={avg_time:.6f} s, {pct:.1f}%"
            )
            self._append_tree_lines(child, total_runtime_s, lines, depth + 1)

    def summary_text(self, total_moves: int, total_runtime_s: float) -> str:
        if not self.enabled:
            return "Profiling disabled."
        lines: List[str] = []
        lines.append("Performance summary:")
        lines.append(f"  Total runtime: {total_runtime_s:.3f} s")
        if total_moves > 0:
            lines.append(
                f"  Avg time per MC step: {total_runtime_s / total_moves:.6f} s"
            )
        lines.append("")
        lines.append(
            "Hierarchical breakdown (total time, calls, avg per call, % of total):"
        )
        # Print hierarchy starting from root's children
        self._append_tree_lines(self._root, total_runtime_s, lines, depth=0)

        # Optionally include flat-only counters (labels that never appeared in hierarchy)
        # These might come from increment() calls and have zero time.
        hierarchical_labels = set()

        def _collect_labels(n: _ProfileNode):
            for c in n.children.values():
                hierarchical_labels.add(c.label)
                _collect_labels(c)

        _collect_labels(self._root)
        extras = [
            (label, data)
            for label, data in self._stats.items()
            if label not in hierarchical_labels
        ]
        if extras:
            lines.append("")
            lines.append("Non-timed counters:")
            for label, data in sorted(extras, key=lambda kv: kv[0]):
                t = data.get("time", 0.0)
                c = int(data.get("count", 0))
                pct = (t / total_runtime_s * 100.0) if total_runtime_s > 0 else 0.0
                lines.append(f"    {label}: {t:.6f} s, {c} calls, {pct:.1f}%")

        return "\n".join(lines)
