"""Research Console: HTML/SVG strings built with the standard library, shown through Colab/IPython's display bridge."""

import html
import json
import time
from typing import List, Optional

from . import code_hash

PALETTE = {"ink": "#1c2126", "muted": "#5c646c", "line": "#d8d4cb", "accent": "#22706a", "ochre": "#b5762a",
           "oxblood": "#8c2f2b", "soft": "#f6f4ef", "sage": "#eef2ef"}

EXAMPLE_REQUEST = {  # the brief's "multiply by two" example, shown when the console opens
    "schema_version": "nova.request/1.1", "request_id": "demo-1",
    "models": ["nova", "transformer", "recurrent", "ablated"],
    "settings": {"seed": 0, "deterministic": True, "device": "auto", "max_steps": 6,
                 "confidence_threshold": 0.5, "trace_level": "full"},
    "task": {"domain": "sequence",
             "support": [{"input": [2, 4, 6], "output": [4, 8, 12]}, {"input": [3, 5], "output": [6, 10]},
                         {"input": [1, 7], "output": [2, 14]}],
             "query": [{"input": [7, 1, 9, 12]}]},
}


def _scale(x: float, lo: float, hi: float) -> str:
    t = 0.0 if hi <= lo else max(0.0, min(1.0, (x - lo) / (hi - lo)))
    a, b = (245, 238, 228), (140, 47, 43)
    return "#%02x%02x%02x" % tuple(int(a[i] + t * (b[i] - a[i])) for i in range(3))


def svg_heatmap(R: List[List[Optional[float]]], cell: int = 30) -> str:
    k = len(R)
    vals = [v for i, row in enumerate(R) for j, v in enumerate(row) if i != j and v is not None]
    lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
    parts = [f'<svg width="{cell * k + 60}" height="{cell * k + 40}" font-family="sans-serif" font-size="11">']
    parts.append(f'<text x="{30 + cell * k / 2}" y="12" text-anchor="middle" fill="{PALETTE["muted"]}">demo j</text>')
    for i in range(k):
        parts.append(f'<text x="22" y="{36 + i * cell + cell / 2}" text-anchor="end" fill="{PALETTE["muted"]}">g{i + 1}</text>')
        for j in range(k):
            v = R[i][j]
            fill = "#e9e6e0" if i == j or v is None else _scale(v, lo, hi)
            title = "self" if i == j else f"{v:.3f}" if v is not None else "n/a"
            parts.append(f'<rect x="{30 + j * cell}" y="{22 + i * cell}" width="{cell - 2}" height="{cell - 2}" fill="{fill}">'
                         f'<title>guess {i + 1} on demo {j + 1}: {title}</title></rect>')
    parts.append("</svg>")
    return "".join(parts)


def svg_bars(values: List[float], label: str, colour: str) -> str:
    w = 28
    parts = [f'<svg width="{w * len(values) + 20}" height="90" font-family="sans-serif" font-size="11">',
             f'<text x="0" y="11" fill="{PALETTE["muted"]}">{html.escape(label)}</text>']
    for i, v in enumerate(values):
        h = 60 * max(0.0, min(1.0, v))
        parts.append(f'<rect x="{10 + i * w}" y="{78 - h}" width="{w - 6}" height="{h}" fill="{colour}"><title>{v:.3f}</title></rect>')
        parts.append(f'<text x="{10 + i * w + (w - 6) / 2}" y="89" text-anchor="middle" fill="{PALETTE["muted"]}">{i + 1}</text>')
    parts.append("</svg>")
    return "".join(parts)


def svg_line(values: List[float], halt: Optional[int] = None) -> str:
    if not values:
        return ""
    w, h, top = 220, 90, max(values) or 1.0
    xs = [20 + i * (w - 40) / max(1, len(values) - 1) for i in range(len(values))]
    ys = [h - 15 - (h - 30) * v / top for v in values]
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    parts = [f'<svg width="{w}" height="{h}" font-family="sans-serif" font-size="11">',
             f'<text x="0" y="11" fill="{PALETTE["muted"]}">fit error per step</text>',
             f'<polyline points="{pts}" fill="none" stroke="{PALETTE["accent"]}" stroke-width="2"/>']
    for i, (x, y) in enumerate(zip(xs, ys)):
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{PALETTE["accent"]}"><title>step {i}: {values[i]:.3f}</title></circle>')
    if halt is not None and halt < len(xs):
        parts.append(f'<line x1="{xs[halt]:.1f}" y1="14" x2="{xs[halt]:.1f}" y2="{h - 10}" stroke="{PALETTE["oxblood"]}" stroke-dasharray="3 3"/>')
    parts.append("</svg>")
    return "".join(parts)


def svg_learning_curves(curves: dict, budgets: List[float], title: str) -> str:
    """curves[model] = accuracy at each budget. Log-scaled x axis, one line per model."""
    import math
    colours = [PALETTE["accent"], PALETTE["ochre"], PALETTE["oxblood"], "#4a6a8a", "#7a6a9a", "#8a8a5a", PALETTE["muted"]]
    w, h, pad = 520, 260, 40
    lo, hi = math.log(min(budgets)), math.log(max(budgets)) if len(budgets) > 1 else math.log(min(budgets)) + 1
    X = lambda b: pad + (w - 2 * pad) * (math.log(b) - lo) / max(1e-9, hi - lo)
    Y = lambda a: h - pad - (h - 2 * pad) * a
    parts = [f'<svg width="{w + 150}" height="{h}" font-family="sans-serif" font-size="11">',
             f'<text x="{pad}" y="16" font-weight="600" fill="{PALETTE["ink"]}">{html.escape(title)}</text>',
             f'<line x1="{pad}" y1="{h - pad}" x2="{w - pad}" y2="{h - pad}" stroke="{PALETTE["line"]}"/>',
             f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h - pad}" stroke="{PALETTE["line"]}"/>']
    for a in (0, 0.5, 1):
        parts.append(f'<text x="{pad - 6}" y="{Y(a) + 4}" text-anchor="end" fill="{PALETTE["muted"]}">{a:.1f}</text>')
    for b in budgets:
        parts.append(f'<text x="{X(b)}" y="{h - pad + 14}" text-anchor="middle" fill="{PALETTE["muted"]}">{int(b)}</text>')
    for i, (name, accs) in enumerate(sorted(curves.items())):
        col = colours[i % len(colours)]
        pts = " ".join(f"{X(b):.1f},{Y(a):.1f}" for b, a in zip(budgets, accs))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"/>')
        parts.append(f'<text x="{w - pad + 10}" y="{pad + 14 * i}" fill="{col}">{html.escape(name)}</text>')
    parts.append(f'<text x="{w / 2}" y="{h - 6}" text-anchor="middle" fill="{PALETTE["muted"]}">optimizer updates (log scale)</text></svg>')
    return "".join(parts)


def html_table(rows: List[dict], title: str = "") -> str:
    if not rows:
        return f"<p>{html.escape(title)}: no rows</p>"
    cols = list(rows[0])
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else html.escape(str(v))
    head = "".join(f"<th style='text-align:left;padding:3px 8px;border-bottom:1px solid {PALETTE['accent']}'>{html.escape(c)}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td style='padding:3px 8px;border-bottom:1px solid {PALETTE['line']}'>{fmt(r.get(c, ''))}</td>" for c in cols) + "</tr>" for r in rows)
    return f"<div class='nova'><b>{html.escape(title)}</b><table style='border-collapse:collapse'>{head}{body}</table></div>"


CSS = f"""<style>
.nova{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;color:{PALETTE['ink']};font-size:13px}}
.nova .card{{border:1px solid {PALETTE['line']};border-radius:6px;padding:10px 12px;margin:8px 0;background:#fff}}
.nova .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px}}
.nova h4{{margin:0 0 6px;color:{PALETTE['accent']}}} .nova .muted{{color:{PALETTE['muted']}}}
.nova .err{{border-color:{PALETTE['oxblood']};background:#fbf1f0}} .nova .badge{{display:inline-block;padding:1px 7px;border-radius:9px;font-size:11px;color:#fff;background:{PALETTE['oxblood']}}}
.nova .bar{{height:8px;background:{PALETTE['soft']};border-radius:4px;position:relative}} .nova .bar span{{position:absolute;left:0;top:0;bottom:0;background:{PALETTE['accent']};border-radius:4px}}
.nova code,.nova textarea{{font-family:Menlo,monospace;font-size:12px}} .nova textarea{{width:100%;height:220px}}
.nova button{{margin-right:6px;padding:4px 10px}} .nova .stats span{{margin-right:12px}}
</style>"""


def render_errors(errors: List[dict]) -> str:
    rows = "".join(f"<li><code>{html.escape(e.get('code', ''))}</code> {html.escape(str(e.get('message', '')))}"
                   f"{' <span class=muted>(' + html.escape(str(e['field'])) + ')</span>' if e.get('field') else ''}</li>" for e in errors)
    return f'<div class="card err"><h4>Errors</h4><ul>{rows}</ul></div>'


def render_response(response: dict, threshold: float = 0.5) -> str:
    """Pure function: the same response always gives the same HTML, so rendering can be unit-tested without a browser."""
    if response.get("status") != "ok":
        return f'<div class="nova">{CSS}{render_errors(response.get("errors", []))}</div>'
    cards = []
    for name, r in response["results"].items():
        conf_rows = "".join(
            f'<div>{html.escape(json.dumps(p))[:200]} <span class="muted">conf {c:.2f}</span>'
            f'{" <span class=badge>abstain</span>" if a else ""}<div class="bar"><span style="width:{100 * c:.0f}%"></span></div></div>'
            for p, c, a in zip(r["predictions"], r["confidence"], r["abstained"]))
        comp = r["compute"]
        ck = r.get("checkpoint", {})
        stats = (f'<div class="stats muted"><span>latency {comp["latency_ms"]} ms</span><span>peak mem {comp["peak_mem_bytes"]} B</span>'
                 f'<span>steps {comp["refine_steps"]}</span><span>executor calls {comp["executor_calls"]}</span>'
                 f'<span>params {ck.get("params", "?")}</span><span>ckpt {html.escape(str(ck.get("id", "?")))[:40]}</span></div>')
        trace = ""
        tr = r.get("trace")
        if tr:
            trace += svg_line(tr.get("rho", []), tr.get("halt_step"))
            trace += svg_bars(tr.get("w") or [], "trust w", PALETTE["accent"])
            trace += svg_bars(tr.get("v") or [], "reliability v", PALETTE["ochre"])
            if tr.get("R"):
                trace += svg_heatmap(tr["R"])
        cards.append(f'<div class="card"><h4>{html.escape(name)}</h4>{conf_rows}{stats}<div>{trace}</div></div>')
    return f'<div class="nova">{CSS}<div class="grid">{"".join(cards)}</div></div>'


class Console:
    def __init__(self, backend):
        self.backend = backend
        self.last = {"request": None, "response": None}

    def handle(self, action: str, request_text: str) -> dict:
        try:
            request = json.loads(request_text)
        except json.JSONDecodeError as err:
            resp = {"status": "error", "errors": [{"code": "E_SCHEMA", "message": f"invalid JSON: {err}", "field": None}]}
            return {"html": render_response(resp), "ok": False}
        if action == "validate":
            resp = self.backend.validate(request)
            ok = resp["status"] == "ok"
            msg = '<div class="nova card">Request is valid. You can run it.</div>' if ok else render_response(resp)
            return {"html": msg, "ok": ok}
        resp = self.backend.compare(request) if action == "run" else self.backend.trace(request)
        self.last = {"request": request, "response": resp}
        return {"html": render_response(resp, request.get("settings", {}).get("confidence_threshold", 0.5)), "ok": resp["status"] == "ok"}

    def run(self, request_json: str):  # callable cell (works outside Colab too)
        from IPython.display import HTML, display
        display(HTML(self.handle("run", request_json)["html"]))

    def record(self) -> dict:
        return {"schema": "nova.runrecord/1", "time": time.time(), "code_hash": code_hash(),
                "health": self.backend.health(), **self.last}

    def export(self, path: str) -> str:
        with open(path, "w") as fh:
            json.dump(self.record(), fh, indent=2, default=str)
        return path

    def html(self, request: Optional[dict] = None) -> str:
        text = html.escape(json.dumps(request or EXAMPLE_REQUEST, indent=2))
        # the page below is HTML plus a small JavaScript block that calls Python through google.colab.kernel.invokeFunction
        return f"""<div class="nova">{CSS}
<div class="card"><h4>NOVA Research Console</h4>
<div class="muted">Edit the request (schema <code>nova.request/1.1</code>): <code>support</code> = demonstrations with input and output,
<code>query</code> = questions with input only. Validate first; Run stays disabled until the request is valid.</div>
<textarea id="nova-req">{text}</textarea>
<div>seed <input id="nova-seed" type="number" value="0" style="width:60px"> max steps <input id="nova-steps" type="number" min="0" max="8" value="6" style="width:50px">
threshold <input id="nova-thr" type="number" step="0.05" min="0" max="1" value="0.5" style="width:60px">
device <select id="nova-dev"><option>auto</option><option>cpu</option><option>cuda</option></select>
trace <select id="nova-trace"><option>full</option><option>summary</option><option>none</option></select></div>
<div style="margin-top:6px"><button id="nova-val">Validate</button><button id="nova-run" disabled>Run (compare)</button>
<button id="nova-exp" disabled>Export JSON record</button></div></div>
<div id="nova-out"></div></div>
<script>
(function(){{
  const req=document.getElementById('nova-req'), out=document.getElementById('nova-out');
  const run=document.getElementById('nova-run'), exp=document.getElementById('nova-exp');
  let last=null;
  function withControls(){{ let r; try{{ r=JSON.parse(req.value); }}catch(e){{ return req.value; }}
    r.settings=Object.assign(r.settings||{{}},{{seed:+document.getElementById('nova-seed').value,
      max_steps:+document.getElementById('nova-steps').value, confidence_threshold:+document.getElementById('nova-thr').value,
      device:document.getElementById('nova-dev').value, trace_level:document.getElementById('nova-trace').value}});
    return JSON.stringify(r); }}
  async function call(action){{
    if(!(window.google&&google.colab&&google.colab.kernel)){{ out.innerHTML='<div class="card err">Not in Colab: use the callable cell <code>console.run(request_json)</code>.</div>'; return null; }}
    const res=await google.colab.kernel.invokeFunction('nova.console',[action,withControls()],{{}});
    const data=res.data['application/json']; out.innerHTML=data.html; return data; }}
  req.addEventListener('input',()=>{{ run.disabled=true; }});
  document.getElementById('nova-val').onclick=async()=>{{ const d=await call('validate'); run.disabled=!(d&&d.ok); }};
  run.onclick=async()=>{{ const d=await call('run'); exp.disabled=!d; last=d; }};
  exp.onclick=async()=>{{ const res=await google.colab.kernel.invokeFunction('nova.export',[],{{}});
    const blob=new Blob([JSON.stringify(res.data['application/json'],null,2)],{{type:'application/json'}});
    const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='nova_run_record.json'; a.click(); }};
}})();
</script>"""


def launch(backend, request: Optional[dict] = None) -> Console:
    """Register Colab callbacks (if available) and display the console. Returns the Console for callable-cell use."""
    from IPython.display import HTML, JSON, display
    console = Console(backend)
    try:
        from google.colab import output
        output.register_callback("nova.console", lambda action, text: JSON(console.handle(action, text)))
        output.register_callback("nova.export", lambda: JSON(json.loads(json.dumps(console.record(), default=str))))
    except ImportError:  # plain Jupyter or tests: buttons explain the callable cell instead
        pass
    display(HTML(console.html(request)))
    return console
