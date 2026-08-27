"""Headless render check for a Streamlit page.

Runs a page module with a stub `streamlit` so every widget returns its default
and every chart is actually built *and serialised*. That last part is the point:
`st.altair_chart` here calls `chart.to_dict()`, which is where Altair raises the
spec errors (facet data placement, unknown field names) that otherwise only
surface in a browser.

Usage:
    python tools/render_check.py dashboard/pages/2_Live_Simulation.py
"""

from __future__ import annotations

import runpy
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class _Ctx:
    """Stands in for a column / expander / tab / spinner context manager."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _SessionState(dict):
    """Streamlit's session_state supports both item and attribute access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class _Stop(Exception):
    """Raised by the stub's st.stop(), mirroring Streamlit's own control flow."""


def _passthrough_decorator(*d_args, **d_kwargs):
    """Mimics @st.cache_data and @st.cache_data(show_spinner=False)."""
    if len(d_args) == 1 and not d_kwargs and callable(d_args[0]):
        return d_args[0]

    def wrap(fn):
        return fn

    return wrap


class _StubStreamlit(types.ModuleType):
    def __init__(self, name: str, charts: list):
        super().__init__(name)
        self._charts = charts

    # --- widgets that the page branches on -----------------------------
    def selectbox(self, label, options, index=0, **kw):
        options = list(options)
        return options[index or 0] if options else None

    def radio(self, label, options, index=0, **kw):
        return self.selectbox(label, options, index, **kw)

    def multiselect(self, label, options, default=None, **kw):
        if default is not None:
            return list(default)
        return list(options)[:2]

    def slider(self, label, min_value=0, max_value=1, value=None, step=None,
               *a, **kw):
        return min_value if value is None else value

    def select_slider(self, label, options, value=None, **kw):
        return value if value is not None else list(options)[0]

    def number_input(self, label, min_value=0, max_value=None, value=None, **kw):
        return min_value if value is None else value

    def text_input(self, label, value="", **kw):
        return value

    def checkbox(self, label, value=False, **kw):
        return value

    def toggle(self, label, value=False, **kw):
        return value

    def button(self, *a, **kw):
        # Return True so button-gated code paths (the week rollout) actually run.
        return True

    def form_submit_button(self, *a, **kw):
        return True

    # --- charts: serialise so Altair validates the spec ------------------
    def altair_chart(self, chart, **kw):
        chart.to_dict()  # raises on a bad spec
        self._charts.append(chart)

    def vega_lite_chart(self, spec, **kw):
        self._charts.append(spec)

    # --- layout ---------------------------------------------------------
    def _child(self):
        return _StubStreamlit("streamlit.child", self._charts)

    def columns(self, spec, **kw):
        n = spec if isinstance(spec, int) else len(spec)
        return [self._child() for _ in range(n)]

    def tabs(self, labels, **kw):
        return [self._child() for _ in labels]

    def expander(self, *a, **kw):
        return self._child()

    def container(self, *a, **kw):
        return self._child()

    def spinner(self, *a, **kw):
        return self._child()

    def form(self, *a, **kw):
        return self._child()

    def empty(self, *a, **kw):
        return self._child()

    def stop(self):
        raise _Stop()

    # `with st.sidebar:` is valid Streamlit, so the stub must support it too.
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    # --- caching --------------------------------------------------------
    cache_data = staticmethod(_passthrough_decorator)
    cache_resource = staticmethod(_passthrough_decorator)

    # --- everything else is a no-op that swallows its arguments ----------
    def __getattr__(self, name):
        if name == "session_state":
            state = _SessionState()
            self.__dict__["session_state"] = state
            return state
        if name == "sidebar":
            side = _StubStreamlit("streamlit.sidebar", self._charts)
            self.__dict__["sidebar"] = side
            return side
        if name.startswith("_"):
            raise AttributeError(name)

        def _noop(*a, **kw):
            return None

        return _noop


def main(page: str) -> int:
    charts: list = []
    stub = _StubStreamlit("streamlit", charts)
    sys.modules["streamlit"] = stub
    sys.path.insert(0, str(ROOT))

    path = (ROOT / page).resolve()
    print(f"rendering {path.relative_to(ROOT)} ...")
    try:
        runpy.run_path(str(path), run_name="__main__")
    except _Stop:
        print("  page called st.stop() -- a guard fired, not a crash")
    print(f"OK: {len(charts)} chart(s) built and serialised")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
