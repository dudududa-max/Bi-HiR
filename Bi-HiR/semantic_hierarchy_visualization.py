import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import networkx as nx
import matplotlib.pyplot as plt


def build_graph_from_json(obj: Dict[str, Any]) -> nx.DiGraph:
    G = nx.DiGraph()

    def add_node_recursive(node: Dict[str, Any]):
        nid = node["id"]
        children = node.get("children", []) or []
        ntype = node.get("type", "leaf" if len(children) == 0 else "internal")
        nlabel = node.get("label", str(nid))
        G.add_node(nid, label=nlabel, type=ntype)
        for child in children:
            cid = child["id"]
            cchildren = child.get("children", []) or []
            ctype = child.get("type", "leaf" if len(cchildren) == 0 else "internal")
            clabel = child.get("label", str(cid))
            G.add_node(cid, label=clabel, type=ctype)
            G.add_edge(nid, cid)
            add_node_recursive(child)

    if isinstance(obj, dict) and "id" in obj:
        add_node_recursive(obj)
    elif isinstance(obj, dict) and "roots" in obj and isinstance(obj["roots"], list):
        for r in obj["roots"]:
            add_node_recursive(r)
    elif isinstance(obj, list):
        for r in obj:
            add_node_recursive(r)
    else:
        raise ValueError("Unsupported JSON top-level structure: requires {id,...} or {roots:[...]} or [ ... ]")
    return G


def ensure_single_root(G: nx.DiGraph, virtual_root_id: Optional[str] = "__ROOT__"):
    roots = [n for n in G.nodes if G.in_degree(n) == 0]
    if len(roots) == 1:
        return roots[0]
    rid = virtual_root_id
    if rid in G:
        i = 1
        while f"{rid}_{i}" in G:
            i += 1
        rid = f"{rid}_{i}"
    G.add_node(rid, label="ROOT", type="internal")
    if len(roots) == 0:
        # If no root exists, connect the virtual root to all nodes
        for n in G.nodes:
            if n != rid:
                G.add_edge(rid, n)
    else:
        # If multiple roots exist, connect them to the virtual root
        for r in roots:
            G.add_edge(rid, r)
    return rid


def layout_tree(G: nx.DiGraph, root, x_spacing: float = 6.0, y_step: float = 2.0):
    leaves_cache = {}
    _count_leaves(G, root, leaves_cache)
    pos = {}
    _assign_pos(G, root, 0.0, x_spacing, 0.0, -abs(y_step), pos, leaves_cache)
    return pos

def _count_leaves(G, node, cache: Dict[Any, int]) -> int:
    if node in cache:
        return cache[node]
    children = list(G.successors(node))
    cache[node] = 1 if not children else sum(_count_leaves(G, c, cache) for c in children)
    return cache[node]

def _assign_pos(G, node, x_left, x_spacing, y, y_step, pos, leaves_cache) -> Tuple[float, float]:
    children = list(G.successors(node))
    if not children:
        cx = x_left + x_spacing / 2.0
        pos[node] = (cx, y)
        return x_left + x_spacing, cx
    child_centers, cur_x = [], x_left
    for c in sorted(children, key=lambda k: str(k)):  # Ensure stable order
        cur_x, cc = _assign_pos(G, c, cur_x, x_spacing, y + y_step, y_step, pos, leaves_cache)
        child_centers.append(cc)
    pos[node] = (sum(child_centers) / len(child_centers), y)
    return cur_x, pos[node][0]


def plot_tree(G: nx.DiGraph, root, save_path: str,
              figsize=(72, 36), x_spacing: float = 6.0, y_step: float = 2.0, rotate_labels: bool = True):
    connected = nx.descendants(G, root) | {root}
    H = G.subgraph(connected).copy()

    pos = layout_tree(H, root, x_spacing=x_spacing, y_step=y_step)
    plt.figure(figsize=figsize)

    nx.draw_networkx_edges(H, pos, arrows=True, arrowstyle="-|>", alpha=0.4, edge_color="#555")

    leaf_nodes = [n for n, d in H.nodes(data=True) if d.get("type") == "leaf"]
    inner_nodes = [n for n in H if n not in leaf_nodes]

    nx.draw_networkx_nodes(H, pos, nodelist=inner_nodes, node_shape="s", node_color="#AED7F4",
                           edgecolors="#222", linewidths=0.8)
    nx.draw_networkx_nodes(H, pos, nodelist=leaf_nodes, node_shape="o", node_color="#C2F4AE",
                           edgecolors="#222", linewidths=0.8)

    labels = nx.draw_networkx_labels(
        H, pos,
        labels={n: d.get("label", str(n)) for n, d in H.nodes(data=True)},
        font_size=16, horizontalalignment="left", verticalalignment="bottom",
    )
    if rotate_labels:
        for t in labels.values():
            t.set_rotation(270)

    plt.axis("off")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=330, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved: {save_path}")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.resolve()
    
    JSON_PATH = BASE_DIR / "JSON" / "CUB_200.json"
    SAVE_PATH = BASE_DIR / "semantic-hierarchy" / "CUB_200_tree.png"

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    G = build_graph_from_json(data)
    root = ensure_single_root(G)                     # Automatically detect/create root
    plot_tree(G, root, SAVE_PATH)                    # Generate and save
