"""Crossing resolution: skeleton graph, ports, and embedding-guided joint matching.

Three linkers share the same graph extraction so the comparison in the post-processing
table changes one thing at a time:

``greedy``       the angle-only rule of the reference collagen pipeline: pick a port,
                 take the straightest partner within +/-30 degrees, repeat.  Order
                 dependent, and an early pick can strand a later fiber.
``angle_joint``  all ports of one crossing solved together by exact minimum-cost
                 matching with an unmatched option, on an angle+appearance cost.
``cfx``          ours: the same exact solver on a cost that also reads the learned
                 instance embedding and the predicted overlap multiplicity, and that
                 refuses to commit when the best and second-best solutions are close.

The embedding is what makes ``cfx`` different in kind: two ports belonging to the same
physical fiber were trained to have nearly identical embeddings, so continuation is
decided by learned appearance and not only by geometry.  That supervision was free -
the instance identity came from the label-first generator.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy import ndimage as ndi

from .topology import branch_count, match_ports, ordered_component, prune_spurs, zhang_suen

METHODS = ("greedy", "angle_joint", "cfx")

# Cost weights and the rejection margin are hyperparameters.  They are selected on the
# synthetic VALIDATION split by scripts/tools/tune_linker.py, which writes configs/linker.json;
# the values below are only the fallback when that file is absent.
COSTS = dict(angle=1.0, embedding=1.2, intensity=0.25, support=0.20, overlap=0.35,
             moments=0.10)
MARGIN_THRESHOLD = 0.12
ANGLE_LIMIT = 40.0


UNMATCHED_COST = 0.9


@lru_cache(maxsize=4)
def load_params(path="configs/linker.json"):
    """Return the tuned linker parameters, or the fallback defaults."""
    from pathlib import Path

    from .common import load_json

    p = Path(path)
    data = load_json(p) if p.exists() else {}
    costs = dict(COSTS)
    costs.update(data.get("costs", {}))
    return dict(costs=costs,
                margin_threshold=float(data.get("margin_threshold", MARGIN_THRESHOLD)),
                angle_limit=float(data.get("angle_limit", ANGLE_LIMIT)),
                unmatched=float(data.get("unmatched", UNMATCHED_COST)),
                source=str(p) if p.exists() else "built-in defaults")


def _port_embedding(embed, path, start=5, stop=22):
    """Mean unit embedding sampled on the segment, away from the crossing itself.

    Right at a junction two fibers overlap, so the embedding there is a mixture and
    carries little identity.  Sampling a window further along the arm reads the
    embedding where only one fiber is present.
    """
    if embed is None:
        return None
    pts = path[start:stop] if len(path) > start + 2 else path
    if not len(pts):
        pts = path
    v = embed[:, pts[:, 0], pts[:, 1]].mean(axis=1)
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v


def _bridge(binary, overlap2, pa, pb):
    q = np.rint(np.linspace(pa, pb, 24)).astype(int)
    q[:, 0] = np.clip(q[:, 0], 0, binary.shape[0] - 1)
    q[:, 1] = np.clip(q[:, 1], 0, binary.shape[1] - 1)
    support = float(np.mean(binary[q[:, 0], q[:, 1]] > 0))
    multi = float(np.mean(overlap2[q[:, 0], q[:, 1]])) if overlap2 is not None else 0.0
    return support, multi


def extract_graph(binary, radius=4, min_length=8, joint_prob=None):
    """Thin, find crossing regions, cut them out and order the remaining segments."""
    sk = prune_spurs(zhang_suen(binary), 6)
    candidates = branch_count(sk) >= 3
    expanded = ndi.binary_dilation(candidates, iterations=radius)
    if joint_prob is not None and candidates.any():
        # An acute overlap produces two nearby Y vertices; the learned crossing peak
        # merges them into a single crossing region.
        peaks = (joint_prob == ndi.maximum_filter(joint_prob, size=13)) & (joint_prob > 0.5)
        distance = ndi.distance_transform_edt(~candidates)
        for cy, cx in np.argwhere(peaks & (distance <= 10)):
            y0, y1 = max(0, cy - 10), min(sk.shape[0], cy + 11)
            x0, x1 = max(0, cx - 10), min(sk.shape[1], cx + 11)
            yy, xx = np.ogrid[y0:y1, x0:x1]
            expanded[y0:y1, x0:x1] |= ((yy - cy) ** 2 + (xx - cx) ** 2 <= 100)
    joint_labels, nj = ndi.label(expanded, structure=np.ones((3, 3)))
    cut = sk.copy()
    cut[expanded] = 0
    seg_labels, _ = ndi.label(cut, structure=np.ones((3, 3)))
    sizes = np.bincount(seg_labels.ravel())
    segments, rejected = [], 0
    for i, box in enumerate(ndi.find_objects(seg_labels), 1):
        if box is None or sizes[i] < min_length:
            continue
        coords = np.argwhere(seg_labels[box] == i) + np.array([box[0].start, box[1].start])
        path, closed = ordered_component(coords)
        if path is None:
            rejected += 1
            continue
        segments.append(dict(id=len(segments), points_yx=path, closed=closed))
    centres = {}
    for j, box in enumerate(ndi.find_objects(joint_labels), 1):
        if box is None:
            continue
        pts = np.argwhere((joint_labels[box] == j) & candidates[box]) + np.array(
            [box[0].start, box[1].start])
        if len(pts):
            centres[j] = pts.mean(axis=0)
    return sk, segments, joint_labels, centres, rejected


def collect_ports(segments, joint_labels, centres, gray, embed, tangent_span=12):
    ports = {j: [] for j in centres}
    for seg in segments:
        if seg["closed"]:
            continue
        for end in (0, 1):
            path = seg["points_yx"] if end == 0 else seg["points_yx"][::-1]
            y, x = path[0]
            near = joint_labels[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
            labels = [int(v) for v in np.unique(near[near > 0]) if int(v) in centres]
            if not labels:
                continue
            j = min(labels, key=lambda q: float(np.linalg.norm(path[0] - centres[q])))
            k = min(tangent_span, len(path) - 1)
            v = path[k].astype(float) - path[0]
            v /= max(np.linalg.norm(v), 1e-6)
            intensity = (float(np.mean(gray[path[:k + 1, 0], path[:k + 1, 1]])) / 255.0
                         if gray is not None else 0.0)
            ports[j].append(dict(segment=seg["id"], end=end, point_yx=path[0].tolist(),
                                 direction_yx=v.tolist(), intensity=intensity,
                                 instance_pixels=path[5:22].tolist() if len(path) > 7
                                 else path.tolist(),
                                 embedding=_port_embedding(embed, path)))
    return ports


def pair_costs(ports, centre, binary, overlap2, moments, method, angle_limit, costs=None):
    weights = costs or COSTS
    n = len(ports)
    cost = np.full((n, n), np.inf)
    detail = {}
    limit = 30.0 if method == "greedy" else angle_limit
    for a in range(n):
        for b in range(a + 1, n):
            if ports[a]["segment"] == ports[b]["segment"]:
                continue
            va = np.array(ports[a]["direction_yx"])
            vb = np.array(ports[b]["direction_yx"])
            deviation = abs(180 - math.degrees(math.acos(float(np.clip(va @ vb, -1, 1)))))
            if deviation > limit:
                continue
            terms = dict(angle=deviation / limit)
            if method != "greedy":
                terms["intensity"] = abs(ports[a]["intensity"] - ports[b]["intensity"])
                support, multi = _bridge(binary, overlap2,
                                         np.array(ports[a]["point_yx"]),
                                         np.array(ports[b]["point_yx"]))
                terms["support"] = 1.0 - support
                if moments is not None:
                    cy, cx = np.rint(centre).astype(int)
                    cy = int(np.clip(cy, 0, moments.shape[1] - 1))
                    cx = int(np.clip(cx, 0, moments.shape[2] - 1))
                    theta = math.atan2(va[0], va[1])
                    descriptor = np.array([math.cos(2 * theta), math.sin(2 * theta),
                                           math.cos(4 * theta), math.sin(4 * theta)])
                    terms["moments"] = float(np.mean((moments[:, cy, cx] - descriptor) ** 2))
            if method == "cfx":
                ea, eb = ports[a]["embedding"], ports[b]["embedding"]
                if ea is not None and eb is not None:
                    terms["embedding"] = float(1.0 - np.dot(ea, eb)) / 2.0
                # A real overlap shows two fibers stacked; a spurious bridge does not.
                terms["overlap"] = 1.0 - multi if overlap2 is not None else 0.0
            value = sum(weights.get(k, 0.0) * v for k, v in terms.items())
            cost[a, b] = cost[b, a] = value
            detail[(a, b)] = {k: round(float(v), 4) for k, v in terms.items()}
    return cost, detail


def solve(cost, method, rng, unmatched=0.9, margin_threshold=0.12):
    n = len(cost)
    if method == "greedy":
        available = [int(v) for v in rng.permutation(n)]
        pairs = []
        while available:
            a = available.pop()
            options = [b for b in available if np.isfinite(cost[a, b])]
            if options:
                b = min(options, key=lambda b: cost[a, b])
                available.remove(b)
                pairs.append([a, b])
        matched = {p for pair in pairs for p in pair}
        return dict(pairs=pairs, unmatched=[i for i in range(n) if i not in matched],
                    cost=float(sum(cost[a, b] for a, b in pairs)), margin=None,
                    ambiguous=False, reason="greedy_baseline")
    solution = match_ports(cost, unmatched, margin_threshold)
    if method == "angle_joint":
        solution = dict(solution, ambiguous=False, reason="joint_no_rejection")
    return solution


def analyze(binary, gray=None, prediction=None, method="cfx", radius=4, min_length=8,
            angle_limit=None, seed=42, margin_threshold=None, costs=None):
    """Full crossing analysis for one image.

    `prediction` is the activated bundle from `models.activate` squeezed to numpy, or
    None for the training-free baseline.  Cost weights and the rejection margin default
    to the values selected on the validation split.
    """
    if method not in METHODS:
        raise ValueError(f"unknown linking method {method!r}")
    tuned = load_params()
    costs = costs or tuned["costs"]
    margin_threshold = tuned["margin_threshold"] if margin_threshold is None else margin_threshold
    angle_limit = tuned["angle_limit"] if angle_limit is None else angle_limit
    unmatched = float(costs.get("unmatched", tuned["unmatched"]))
    pred = prediction or {}
    joint_prob = pred.get("joint")
    moments = pred.get("moments")
    embed = pred.get("embed")
    overlap = pred.get("overlap")
    overlap2 = overlap[2] if overlap is not None else None

    sk, segments, joint_labels, centres, rejected = extract_graph(
        binary, radius, min_length, joint_prob)
    ports = collect_ports(segments, joint_labels, centres, gray, embed)
    rng = np.random.default_rng(seed)
    links, joints = {}, []
    for j, centre in centres.items():
        plist = ports[j]
        cost, detail = pair_costs(plist, centre, binary, overlap2, moments, method,
                                  angle_limit, costs)
        solution = solve(cost, method, rng, unmatched=unmatched,
                         margin_threshold=margin_threshold)
        accepted = [] if solution["ambiguous"] else solution["pairs"]
        for a, b in accepted:
            ka = (plist[a]["segment"], plist[a]["end"])
            kb = (plist[b]["segment"], plist[b]["end"])
            links[ka] = (kb, j)
            links[kb] = (ka, j)
        cy, cx = np.rint(centre).astype(int)
        score = (float(joint_prob[int(np.clip(cy, 0, joint_prob.shape[0] - 1)),
                                  int(np.clip(cx, 0, joint_prob.shape[1] - 1))])
                 if joint_prob is not None else None)
        joints.append(dict(
            id=int(j), xy=[float(centre[1]), float(centre[0])], degree=len(plist),
            ports=[{k: v for k, v in p.items() if k != "embedding"} for p in plist],
            solution={k: v for k, v in solution.items() if k != "pairs"} | {"pairs": solution["pairs"]},
            accepted_pairs=accepted, pair_terms={f"{a}-{b}": t for (a, b), t in detail.items()},
            network_score=score,
            type="crossing_candidate" if len(plist) >= 4 else "branch_or_occlusion_candidate"))

    fibers, visited = [], set()
    order = sorted(range(len(segments)),
                   key=lambda i: ((i, 0) in links and (i, 1) in links, i))
    for sid in order:
        if sid in visited:
            continue
        entry = 0 if (sid, 0) not in links else 1 if (sid, 1) not in links else 0
        current, path, sids = sid, [], []
        closed = segments[sid]["closed"]
        while current not in visited:
            visited.add(current)
            sids.append(current)
            points = segments[current]["points_yx"]
            if entry == 1:
                points = points[::-1]
            if path:
                # Explicit interpolated bridge across the crossing: inference, not pixels.
                bridge = np.linspace(path[-1], points[0],
                                     max(2, int(np.linalg.norm(np.array(path[-1]) - points[0])) + 1))
                path.extend(bridge[1:-1].tolist())
            path.extend(points.tolist())
            nxt = links.get((current, 1 - entry))
            if nxt is None:
                break
            (current, entry), _ = nxt
            if current in visited:
                closed = True
                break
        arr = np.array(path, dtype=float)
        length = float(np.linalg.norm(np.diff(arr, axis=0), axis=1).sum()) if len(arr) > 1 else 0.0
        if closed and len(arr) > 1:
            length += float(np.linalg.norm(arr[-1] - arr[0]))
        fibers.append(dict(id=len(fibers), segment_ids=sids, points_xy=arr[:, ::-1].tolist(),
                           length_pixels=length, closed=closed))

    return sk, dict(
        joints=joints, fibers=fibers, segment_count=len(segments),
        rejected_branched_components=rejected,
        crossing_candidates=sum(j["degree"] >= 4 for j in joints),
        branch_candidates=sum(j["degree"] == 3 for j in joints),
        unresolved_joints=sum(bool(j["solution"]["ambiguous"]) for j in joints),
        method=method, seed=seed,
        coordinate_system="analysis pixels, x right y down; projected 2D geometry only")


def heatmap_points(heat, threshold=0.5, radius=6):
    maxima = (heat == ndi.maximum_filter(heat, size=2 * radius + 1)) & (heat >= threshold)
    labels, n = ndi.label(maxima)
    points = []
    for i in range(1, n + 1):
        sel = labels == i
        coords = np.argwhere(sel)
        best = coords[int(np.argmax(heat[sel]))]
        points.append([float(best[1]), float(best[0])])
    return points
