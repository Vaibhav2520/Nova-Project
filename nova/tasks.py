"""The puzzle factory (Procedural Generalization Suite): rules, puzzle generation, fingerprints and batching."""

import dataclasses
import hashlib
import heapq
import itertools
import json
import random
from collections import deque
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

GENERATOR_VERSION = "pgs/1"  # version of the puzzle factory; part of every fingerprint

PAD, BOS, EOS, SEP, ROW, EDGE, LBR, RBR, SRC, DST, DEMO, QRY = range(12)
VALUE_OFFSET = 16  # value tokens start at 16 (12..15 are kept free for later)
NUM_VALUES = 80
VOCAB_SIZE = VALUE_OFFSET + NUM_VALUES
NUM_SYMBOLS = 32
SPECIAL_NAMES = ["PAD", "BOS", "EOS", "SEP", "ROW", "EDGE", "LBR", "RBR", "SRC", "DST", "DEMO", "QRY"]


def vt(value: int) -> int:
    """Turn a value (0..79) into its token number."""
    if not 0 <= value < NUM_VALUES:
        raise ValueError(f"value {value} cannot be written as a token")
    return VALUE_OFFSET + value


def token_name(token: int) -> str:
    """Readable name of a token, used in traces and the console."""
    if token < len(SPECIAL_NAMES):
        return SPECIAL_NAMES[token]
    return f"RSV{token}" if token < VALUE_OFFSET else str(token - VALUE_OFFSET)


@dataclasses.dataclass(frozen=True)
class Primitive:
    name: str
    family: str
    params: Tuple[tuple, ...]
    fn: Callable
    terminal: bool = False  # True if the rule must be the last step (its output is not a normal input)
    identity: Tuple[tuple, ...] = ()  # parameter settings that would do nothing (banned in compositions)


Program = Tuple[Tuple[str, tuple], ...]

def _perm_table(i: int) -> List[int]:
    """Permutation table i over the 32 symbols (never the identity)."""
    rng = random.Random(9000 + i)
    while True:
        table = rng.sample(range(NUM_SYMBOLS), NUM_SYMBOLS)
        if table != list(range(NUM_SYMBOLS)):
            return table


def seq_copy(x, *_):
    return list(x)


def seq_rotate(x, k):
    return list(x[k % len(x):]) + list(x[:k % len(x)]) if x else []


def seq_dedup(x):
    seen, out = set(), []
    for s in x:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def seq_bracket(x):  # bracket rule (our declared reading of the brief's "bracket")
    out = []
    for i, s in enumerate(x):
        if i == 0 or x[i - 1] != s:
            if i > 0:
                out.append(-RBR)  # negative numbers mark special tokens inside rule outputs
            out.append(-LBR)
        out.append(s)
    if x:
        out.append(-RBR)
    return out


def seq_map_perm(x, i):
    table = _perm_table(i)
    return [table[s] for s in x]


def seq_map_affine(x, a, b):  # affine map a*x+b mod 32 (includes "multiply by two")
    return [(a * s + b) % NUM_SYMBOLS for s in x]


def seq_reduce(x, op):
    if op == "count":
        return [len(x)]
    if op == "sum":
        return [sum(x) % NUM_SYMBOLS]
    if op == "max":
        return [max(x)] if x else [0]
    return [min(x)] if x else [0]


def seq_sort(x, key):
    keyfn = (lambda s: s) if key == "value" else (lambda s: s % 4)
    return sorted(x, key=keyfn)


def seq_mirror(x):
    return list(x) + list(reversed(x))


SEQUENCE_RULES = {
    "copy": Primitive("copy", "copy", ((),), seq_copy),
    "rotate": Primitive("rotate", "rotate", ((1,), (2,), (3,)), seq_rotate),
    "dedup": Primitive("dedup", "deduplicate", ((),), seq_dedup),
    "bracket": Primitive("bracket", "bracket", ((),), seq_bracket, terminal=True),
    "map_perm": Primitive("map_perm", "map", tuple((i,) for i in range(4)), seq_map_perm),
    "map_affine": Primitive("map_affine", "map", tuple((a, b) for a in (1, 2, 3, 4, 5, 7) for b in (0, 1, 3)),
                            seq_map_affine, identity=((1, 0),)),
    "reduce": Primitive("reduce", "reduce", (("count",), ("sum",), ("max",), ("min",)), seq_reduce, terminal=True),
    "sort": Primitive("sort", "stable_sort", (("value",), ("mod4",)), seq_sort),
    "mirror": Primitive("mirror", "mirror", ((),), seq_mirror),
}


def _blank(h, w):
    return [[0] * w for _ in range(h)]


def grid_objects(g):
    """Return objects as (colour, list of cells). An object is a 4-connected group of one colour."""
    h, w = len(g), len(g[0])
    seen = [[False] * w for _ in range(h)]
    objs = []
    for r in range(h):
        for c in range(w):
            if g[r][c] and not seen[r][c]:
                colour, cells, todo = g[r][c], [], [(r, c)]
                seen[r][c] = True
                while todo:
                    cr, cc = todo.pop()
                    cells.append((cr, cc))
                    for nr, nc in ((cr + 1, cc), (cr - 1, cc), (cr, cc + 1), (cr, cc - 1)):
                        if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and g[nr][nc] == colour:
                            seen[nr][nc] = True
                            todo.append((nr, nc))
                objs.append((colour, sorted(cells)))
    return objs


def grid_translate(g, dx, dy):
    h, w = len(g), len(g[0])
    out = _blank(h, w)
    for r in range(h):
        for c in range(w):
            if g[r][c] and 0 <= r + dy < h and 0 <= c + dx < w:
                out[r + dy][c + dx] = g[r][c]
    return out


def grid_slide(g, direction):
    h, w = len(g), len(g[0])
    dr, dc = {"right": (0, 1), "left": (0, -1), "up": (-1, 0), "down": (1, 0)}[direction]
    out = [row[:] for row in g]
    objs = grid_objects(g)
    lead = lambda o: max(r * dr + c * dc for r, c in o[1])
    for colour, cells in sorted(objs, key=lead, reverse=True):
        for r, c in cells:
            out[r][c] = 0
        while all(0 <= r + dr < h and 0 <= c + dc < w and out[r + dr][c + dc] == 0 for r, c in cells):
            cells = [(r + dr, c + dc) for r, c in cells]
        for r, c in cells:
            out[r][c] = colour
    return out


def _colour_table(i):
    rng = random.Random(7000 + i)
    while True:
        table = [0] + rng.sample(range(1, 10), 9)
        if table != list(range(10)):
            return table


def grid_recolour_map(g, i):
    table = _colour_table(i)
    return [[table[x] for x in row] for row in g]


def grid_recolour_largest(g, colour):
    objs = grid_objects(g)
    out = [row[:] for row in g]
    if objs:
        _, cells = max(objs, key=lambda o: len(o[1]))  # largest one (first in scan order wins ties)
        for r, c in cells:
            out[r][c] = colour
    return out


def grid_flip_h(g):
    return [row[::-1] for row in g]


def grid_flip_v(g):
    return [row[:] for row in g[::-1]]


def grid_rot90(g):
    h, w = len(g), len(g[0])
    return [[g[h - 1 - c][r] for c in range(h)] for r in range(w)]


def grid_fill_enclosed(g, colour):
    h, w = len(g), len(g[0])
    outside = [[False] * w for _ in range(h)]
    todo = [(r, c) for r in range(h) for c in range(w) if (r in (0, h - 1) or c in (0, w - 1)) and g[r][c] == 0]
    for r, c in todo:
        outside[r][c] = True
    while todo:
        r, c = todo.pop()
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= nr < h and 0 <= nc < w and not outside[nr][nc] and g[nr][nc] == 0:
                outside[nr][nc] = True
                todo.append((nr, nc))
    return [[colour if g[r][c] == 0 and not outside[r][c] else g[r][c] for c in range(w)] for r in range(h)]


def grid_mark_border(g, colour):
    h, w = len(g), len(g[0])
    out = [row[:] for row in g]
    for _, cells in grid_objects(g):
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells):
            for r, c in cells:
                out[r][c] = colour
    return out


def grid_count_bar(g):
    n = len(grid_objects(g))
    return [[1] * n] if n else [[0]]


def grid_gravity(g):
    h, w = len(g), len(g[0])
    out = _blank(h, w)
    for c in range(w):
        column = [g[r][c] for r in range(h) if g[r][c]]
        for i, colour in enumerate(column):
            out[h - len(column) + i][c] = colour
    return out


GRID_RULES = {
    "translate": Primitive("translate", "motion", ((1, 0), (-1, 0), (0, 1), (0, -1)), grid_translate),
    "slide": Primitive("slide", "interaction", (("right",), ("left",), ("up",)), grid_slide),
    "recolour_map": Primitive("recolour_map", "recolour", tuple((i,) for i in range(6)), grid_recolour_map),
    "recolour_largest": Primitive("recolour_largest", "recolour", ((1,), (2,), (3,)), grid_recolour_largest),
    "flip_h": Primitive("flip_h", "symmetry", ((),), grid_flip_h),
    "flip_v": Primitive("flip_v", "symmetry", ((),), grid_flip_v),
    "rot90": Primitive("rot90", "symmetry", ((),), grid_rot90),
    "fill_enclosed": Primitive("fill_enclosed", "enclosure", ((4,), (5,)), grid_fill_enclosed),
    "mark_border": Primitive("mark_border", "connectivity", ((6,), (7,)), grid_mark_border),
    "count_bar": Primitive("count_bar", "counting", ((),), grid_count_bar, terminal=True),
    "gravity": Primitive("gravity", "gravity", ((),), grid_gravity),
}


@dataclasses.dataclass
class Graph:
    n: int
    edges: List[Tuple[int, int]]
    colours: List[int]
    src: int
    dst: int


def _adjacency(g: Graph, undirected=False):
    adj = [[] for _ in range(g.n)]
    for u, v in g.edges:
        adj[u].append(v)
        if undirected:
            adj[v].append(u)
    return [sorted(set(a)) for a in adj]


def graph_reverse(g: Graph):
    return Graph(g.n, [(v, u) for u, v in g.edges], list(g.colours), g.src, g.dst)


def graph_drop_colour(g: Graph, colour):
    keep = [(u, v) for u, v in g.edges if g.colours[u] != colour and g.colours[v] != colour]
    return Graph(g.n, keep, list(g.colours), g.src, g.dst)


def graph_reach(g: Graph):
    adj, seen, todo = _adjacency(g), {g.src}, deque([g.src])
    while todo:
        for v in adj[todo.popleft()]:
            if v not in seen:
                seen.add(v)
                todo.append(v)
    return sorted(seen)


def graph_path_avoid(g: Graph, colour):  # query: shortest path that avoids a colour (-1 = no constraint)
    adj, parent, todo = _adjacency(g), {g.src: None}, deque([g.src])
    while todo:
        u = todo.popleft()
        if u == g.dst:
            path = []
            while u is not None:
                path.append(u)
                u = parent[u]
            return path[::-1]
        for v in adj[u]:
            blocked = colour >= 0 and g.colours[v] == colour and v != g.dst
            if v not in parent and not blocked:
                parent[v] = u
                todo.append(v)
    return []


def _components(g: Graph):
    adj, label = _adjacency(g, undirected=True), [-1] * g.n
    for start in range(g.n):
        if label[start] < 0:
            label[start], todo = start, [start]
            while todo:
                for v in adj[todo.pop()]:
                    if label[v] < 0:
                        label[v] = start
                        todo.append(v)
    return label


def graph_comp_label(g: Graph):
    return _components(g)


def graph_largest_comp(g: Graph):
    label = _components(g)
    best = max(sorted(set(label)), key=lambda l: label.count(l))  # largest; smallest label wins ties
    return [i for i in range(g.n) if label[i] == best]


def graph_topo(g: Graph):  # query: stable topological order (smallest ready node first)
    adj, indeg = _adjacency(g), [0] * g.n
    for u in range(g.n):
        for v in adj[u]:
            indeg[v] += 1
    ready = [u for u in range(g.n) if indeg[u] == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        u = heapq.heappop(ready)
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(ready, v)
    return order + sorted(set(range(g.n)) - set(order))  # nodes on cycles (if any) go last, sorted


def graph_cycle_members(g: Graph):
    adj, members = _adjacency(g), []
    for u in range(g.n):
        seen, todo = set(adj[u]), list(adj[u])
        while todo and u not in seen:
            for v in adj[todo.pop()]:
                if v not in seen:
                    seen.add(v)
                    todo.append(v)
        if u in seen:
            members.append(u)
    return members


GRAPH_RULES = {
    "reverse": Primitive("reverse", "component_transform", ((),), graph_reverse),
    "drop_colour": Primitive("drop_colour", "component_transform", ((1,), (2,)), graph_drop_colour),
    "reach": Primitive("reach", "reachability", ((),), graph_reach, terminal=True),
    "path_avoid": Primitive("path_avoid", "shortest_path", ((-1,), (1,), (2,)), graph_path_avoid, terminal=True),
    "comp_label": Primitive("comp_label", "components", ((),), graph_comp_label, terminal=True),
    "largest_comp": Primitive("largest_comp", "components", ((),), graph_largest_comp, terminal=True),
    "topo": Primitive("topo", "topological", ((),), graph_topo, terminal=True),
    "cycle_members": Primitive("cycle_members", "cycles", ((),), graph_cycle_members, terminal=True),
}

RULES = {"sequence": SEQUENCE_RULES, "grid": GRID_RULES, "graph": GRAPH_RULES}
SEALED_FAMILIES = {"sequence": "mirror", "grid": "gravity", "graph": "cycles"}


def enumerate_programs(domain: str, max_depth: int) -> List[Program]:
    """Every program up to max_depth steps. Terminal rules only at the end; no identity steps and no rule twice in a row."""
    rules = RULES[domain]
    steps = [(name, p) for name, r in sorted(rules.items()) for p in r.params]
    programs = []
    for depth in range(1, max_depth + 1):
        for chain in itertools.product(steps, repeat=depth):
            if _valid_chain(domain, chain):
                programs.append(tuple(chain))
    return programs


def _valid_chain(domain: str, chain) -> bool:
    rules = RULES[domain]
    for i, (name, params) in enumerate(chain):
        last = i == len(chain) - 1
        if rules[name].terminal and not last:
            return False
        # identity steps would duplicate shorter programs
        if len(chain) > 1 and (name == "copy" or params in rules[name].identity):
            return False
        if i > 0 and chain[i - 1][0] == name:
            return False
    if domain == "graph" and not rules[chain[-1][0]].terminal:  # graph programs must end with a query
        return False
    return True


def program_families(domain: str, program: Program) -> Tuple[str, ...]:
    return tuple(RULES[domain][name].family for name, _ in program)


def describe(program: Program) -> str:
    return " -> ".join(name + (str(list(p)) if p else "") for name, p in program)


def run_program(domain: str, program: Program, value):
    """Apply each step of the program in order and return the result."""
    for name, params in program:
        value = RULES[domain][name].fn(value, *params)
    return value


def program_signature(domain: str, program: Program, regime: dict) -> str:
    """sha256 of every field that defines how instances are generated. Any changed field changes the signature."""
    spec = {"generator": GENERATOR_VERSION, "domain": domain, "program": [[n, list(p)] for n, p in program], "regime": regime}
    return "sha256:" + hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


REGIMES = {
    "sequence": {"train": {"len": [4, 16]}, "large": {"len": [32, 64]}},
    "grid": {"train": {"side": [4, 7], "objects": [1, 3], "distractor": False},  # small grids (at most 58 tokens)
             # bigger grids with noise pixels (at most 158 tokens)
             "large": {"side": [9, 12], "objects": [4, 6], "distractor": True}},
    "graph": {"train": {"nodes": [6, 12], "model": "er"}, "large": {"nodes": [14, 24], "model": "pa"}},
}


def sample_sequence(rng: random.Random, regime: dict):
    n = rng.randint(*regime["len"])
    alphabet = rng.sample(range(NUM_SYMBOLS), rng.randint(3, 12))  # small alphabet, so repeats and runs are common
    return [rng.choice(alphabet) for _ in range(n)]


def sample_grid(rng: random.Random, regime: dict):
    h, w = rng.randint(*regime["side"]), rng.randint(*regime["side"])
    g = _blank(h, w)
    def free(cells):
        return all(0 <= r < h and 0 <= c < w for r, c in cells) and all(
            g[r + dr][c + dc] == 0 for r, c in cells for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if 0 <= r + dr < h and 0 <= c + dc < w)
    for _ in range(rng.randint(*regime["objects"])):
        for _attempt in range(40):
            oh, ow = rng.randint(1, 3), rng.randint(1, 3)
            r0, c0 = rng.randint(0, h - 1), rng.randint(0, w - 1)
            hollow = oh == 3 and ow == 3 and rng.random() < 0.5
            cells = [(r0 + i, c0 + j) for i in range(oh) for j in range(ow) if not (hollow and i == 1 and j == 1)]
            if free(cells):
                colour = rng.randint(1, 9)
                for r, c in cells:
                    g[r][c] = colour
                break
    if regime.get("distractor"):  # held-out distractor process: isolated noise pixels
        for _ in range(rng.randint(2, 5)):
            r, c = rng.randint(0, h - 1), rng.randint(0, w - 1)
            if free([(r, c)]):
                g[r][c] = 9
    return g


def sample_graph(rng: random.Random, regime: dict, dag: bool):
    n = rng.randint(*regime["nodes"])
    order = list(range(n))
    rng.shuffle(order)
    rank = {u: i for i, u in enumerate(order)}
    edges = set()
    if regime["model"] == "er":  # Erdos-Renyi model: every pair gets an edge with probability p
        p = rng.uniform(2.0, 3.0) / (2 * (n - 1)) * (2 if dag else 1)
        for u in range(n):
            for v in range(n):
                if u != v and rng.random() < p and (not dag or rank[u] < rank[v]):
                    edges.add((u, v))
    else:  # preferential attachment: new nodes prefer well-connected nodes
        degree = [0] * n
        for i in range(1, n):
            for _ in range(rng.randint(1, 2)):
                weights = [degree[j] + 1 for j in range(i)]
                j = rng.choices(range(i), weights=weights)[0]
                a, b = order[i], order[j]
                u, v = (b, a) if (dag or rng.random() < 0.5) else (a, b)
                if dag and rank[u] > rank[v]:
                    u, v = v, u
                edges.add((u, v))
                degree[a] += 1
                degree[b] += 1
    edge_list = sorted(edges)
    rng.shuffle(edge_list)  # then shuffle, so the input order carries no hint
    src, dst = rng.sample(range(n), 2)
    return Graph(n, edge_list, [rng.randint(0, 3) for _ in range(n)], src, dst)


def sample_input(domain: str, rng: random.Random, regime: dict, program: Program):
    if domain == "sequence":
        return sample_sequence(rng, regime)
    if domain == "grid":
        return sample_grid(rng, regime)
    return sample_graph(rng, regime, dag=program[-1][0] == "topo")


def encode_input(domain: str, value) -> List[int]:
    if domain == "sequence":
        return [vt(s) for s in value]
    if domain == "grid":
        return encode_grid(value)
    g = value
    toks = [vt(g.n)] + [vt(c) for c in g.colours]
    for u, v in g.edges:
        toks += [vt(u), vt(v), EDGE]
    return toks + [SRC, vt(g.src), DST, vt(g.dst)]


def encode_grid(g) -> List[int]:
    toks = [vt(len(g)), vt(len(g[0]))]
    for row in g:
        toks += [vt(x) for x in row] + [ROW]
    return toks


def encode_output(domain: str, value) -> List[int]:
    if domain == "grid":
        return encode_grid(value) + [EOS]
    return [(-s if s < 0 else vt(s)) for s in value] + [EOS]  # negative numbers are special tokens (brackets)


@dataclasses.dataclass
class Task:
    domain: str
    program: Program  # the hidden rule (never given to a model)
    signature: str
    supports: List[Tuple[List[int], List[int]]]
    queries: List[Tuple[List[int], List[int]]]
    depth: int = 1

    def instance_hash(self) -> str:
        body = json.dumps([self.signature, self.supports, self.queries], separators=(",", ":"))
        return "sha256:" + hashlib.sha256(body.encode()).hexdigest()


def seeded_rng(*parts) -> random.Random:
    """A random generator seeded from sha256 of the given parts, so puzzle i of split s is always identical."""
    seed = int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:16], 16)
    return random.Random(seed)


def make_task(domain: str, program: Program, regime_name: str, rng: random.Random,
              k_min: int, k_max: int, queries: int, max_len: int = 1000) -> Task:
    """Sample K demonstrations and Q questions from one program. Retries inputs that give over-long outputs."""
    regime = REGIMES[domain][regime_name]
    k = rng.randint(k_min, k_max)
    pairs = []
    tries = 0
    while len(pairs) < k + queries:
        tries += 1
        if tries > 200:
            raise RuntimeError(f"could not sample inputs for {describe(program)}")
        value = sample_input(domain, rng, regime, program)
        x = encode_input(domain, value)
        y = encode_output(domain, run_program(domain, program, value))
        if len(x) <= max_len and len(y) <= max_len:
            pairs.append((x, y))
    signature = program_signature(domain, program, {"name": regime_name, **regime})
    return Task(domain, program, signature, pairs[:k], pairs[k:], depth=len(program))


STREAM_SEGMENTS = (  # preregistered order of rule families in the continual stream
    ("sequence", "stable_sort", "abrupt"),
    ("grid", "symmetry", "abrupt"),
    ("graph", "reachability", "gradual"),
    ("sequence", "map", "abrupt"),
    ("sequence", "stable_sort", "abrupt"),
)


def stream_choice(position: int, seg_len: int, ramp: int, rng: random.Random) -> Tuple[str, str]:
    """Return (domain, family) for a position in the stream. Gradual segments blend in over `ramp` tasks."""
    seg = min(position // seg_len, len(STREAM_SEGMENTS) - 1)
    domain, family, kind = STREAM_SEGMENTS[seg]
    offset = position - seg * seg_len
    if kind == "gradual" and seg > 0 and offset < ramp and rng.random() > offset / ramp:
        domain, family, _ = STREAM_SEGMENTS[seg - 1]
    return domain, family


@dataclasses.dataclass
class TaskBatch:  # padded tensors for a batch of tasks (never contains query answers)
    support_inputs: torch.Tensor
    support_input_mask: torch.Tensor
    support_outputs: torch.Tensor
    support_output_mask: torch.Tensor
    support_mask: torch.Tensor
    query_inputs: torch.Tensor
    query_input_mask: torch.Tensor
    query_mask: torch.Tensor
    meta: dict  # domain, signatures, schema version: never passed to a model

    def to(self, device) -> "TaskBatch":
        moved = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        return TaskBatch(**{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in moved.items()})

    def answer_cap(self, ceiling: int) -> int:
        """Demos and questions of a task come from the same size range, so 2x the longest demo answer (+4) is always enough."""
        longest = int(self.support_output_mask.sum(-1).max()) if self.support_output_mask.numel() else 0
        return max(4, min(ceiling, 2 * longest + 4))

    @property
    def size(self) -> int:
        return self.support_inputs.shape[0]


@dataclasses.dataclass
class TargetBatch:
    query_outputs: torch.Tensor
    query_output_mask: torch.Tensor

    def to(self, device) -> "TargetBatch":
        return TargetBatch(self.query_outputs.to(device), self.query_output_mask.to(device))


def _pad(seqs: Sequence[Sequence[int]], length: int) -> List[List[int]]:
    return [list(s) + [PAD] * (length - len(s)) for s in seqs]


def collate(tasks: Sequence[Task], device="cpu") -> Tuple[TaskBatch, TargetBatch]:
    """Pad a list of tasks into one TaskBatch (model input) and one TargetBatch (answers)."""
    B = len(tasks)
    K = max(1, max(len(t.supports) for t in tasks))
    Q = max(1, max(len(t.queries) for t in tasks))
    L_in = max(1, max(len(x) for t in tasks for x, _ in t.supports + t.queries))
    L_out = max(1, max(len(y) for t in tasks for _, y in t.supports + t.queries))
    s_in = torch.full((B, K, L_in), PAD, dtype=torch.long)
    s_out = torch.full((B, K, L_out), PAD, dtype=torch.long)
    q_in = torch.full((B, Q, L_in), PAD, dtype=torch.long)
    q_out = torch.full((B, Q, L_out), PAD, dtype=torch.long)
    s_mask = torch.zeros(B, K, dtype=torch.bool)
    q_mask = torch.zeros(B, Q, dtype=torch.bool)
    for b, t in enumerate(tasks):
        for k, (x, y) in enumerate(t.supports):
            s_in[b, k, :len(x)] = torch.tensor(x)
            s_out[b, k, :len(y)] = torch.tensor(y)
            s_mask[b, k] = True
        for q, (x, y) in enumerate(t.queries):
            q_in[b, q, :len(x)] = torch.tensor(x)
            q_out[b, q, :len(y)] = torch.tensor(y)
            q_mask[b, q] = True
    # for humans and the ledger; never given to a model
    meta = {"schema": "nova.taskbatch/1", "domains": [t.domain for t in tasks],
            "signatures": [t.signature for t in tasks], "depths": [t.depth for t in tasks]}
    batch = TaskBatch(s_in, s_in != PAD, s_out, s_out != PAD, s_mask, q_in, q_in != PAD, q_mask, meta)
    return batch.to(device), TargetBatch(q_out, q_out != PAD).to(device)


def decode_grid(tokens: Sequence[int]):
    vals = [t for t in tokens if t not in (EOS, PAD)]
    h, w = vals[0] - VALUE_OFFSET, vals[1] - VALUE_OFFSET
    cells = [t - VALUE_OFFSET for t in vals[2:] if t != ROW]
    return [cells[r * w:(r + 1) * w] for r in range(h)]


def decode_graph(tokens: Sequence[int]) -> Graph:
    n = tokens[0] - VALUE_OFFSET
    colours = [t - VALUE_OFFSET for t in tokens[1:1 + n]]
    rest = list(tokens[1 + n:])
    cut = rest.index(SRC)
    edge_toks = rest[:cut]
    edges = [(edge_toks[i] - VALUE_OFFSET, edge_toks[i + 1] - VALUE_OFFSET) for i in range(0, len(edge_toks), 3)]
    return Graph(n, edges, colours, rest[cut + 1] - VALUE_OFFSET, rest[cut + 3] - VALUE_OFFSET)


def decode_value(domain: str, tokens: Sequence[int]):
    toks = [t for t in tokens if t != PAD]
    toks = toks[:toks.index(EOS)] if EOS in toks else toks
    if domain == "grid" and len(toks) >= 2 and all(t >= VALUE_OFFSET or t == ROW for t in toks):
        try:
            return decode_grid(toks)
        except Exception:
            pass
    return [t - VALUE_OFFSET if t >= VALUE_OFFSET else token_name(t) for t in toks]
