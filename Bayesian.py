# Bayesian.py
import re
import numpy as np
import pandas as pd
from collections import OrderedDict
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# -----------------------
# Helper: parse CPT string -> list of floats
# -----------------------
def parse_cpt_values(cpt_string):
    """Parse nilai probabilitas dari string CPT (numbers separated or inside parentheses)."""
    if cpt_string is None:
        return []
    vals = [float(x) for x in re.findall(r'[0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?', str(cpt_string))]
    return vals

# -----------------------
# Learn CPTs from data (MLE with optional Laplace smoothing)
# Input:
#   df: pandas DataFrame containing columns for all nodes (columns names = node names)
#   node_list: list of node names (order independent)
#   states: dict or common list of states (if dict: states[node] -> list)
#   parents_map: dict {node: [parents]} if not provided, user supplies
# Output:
#   cpts_map: dict { (node, tuple(parents)) : list_of_probabilities_flattened }
#   where flattened ordering: iterate over parent cartesian product (lexicographic) and for each parent combination list P(node=state) in order states[node]
def learn_cpts_from_data(df, node_list, states, parents_map, laplace=1.0):
    """
    Estimate CPTs with counts from df.
    Returns ordered dict of CPTs.
    """
    from itertools import product
    cpts = OrderedDict()
    # Ensure states mapping
    node_states = {}
    if isinstance(states, dict):
        node_states = states
    else:
        # same states for all nodes
        for n in node_list:
            node_states[n] = list(states)

    for node in node_list:
        parents = parents_map.get(node, []) or []
        # If no parents: just frequency of node
        if len(parents) == 0:
            counts = df[node].value_counts().reindex(node_states[node], fill_value=0).astype(float)
            counts += laplace
            probs = (counts / counts.sum()).tolist()
            # store flattened: here length = #states
            cpts[(node, tuple(parents))] = probs
        else:
            # compute counts for each parent combination
            parent_states_lists = [node_states[p] for p in parents]
            probs_flat = []
            # iterate lexicographically over parent combinations
            for comb in product(*parent_states_lists):
                # filter rows matching parent combination
                mask = np.ones(len(df), dtype=bool)
                for p, val in zip(parents, comb):
                    mask &= (df[p] == val)
                subset = df[mask]
                counts = subset[node].value_counts().reindex(node_states[node], fill_value=0).astype(float)
                counts += laplace
                probs = (counts / counts.sum()).tolist()
                probs_flat.extend(probs)  # append child-state probs for this parent combo
            cpts[(node, tuple(parents))] = probs_flat
    return cpts, node_states

# or
def load_cpts_from_csv(path):
    df_cpts = pd.read_csv(path)
    cpts_loaded = {}

    for _, row in df_cpts.iterrows():
        node = row['node']

        # FIX untuk parent kosong ("" → NaN)
        if pd.isna(row['parents']):
            parents = []
        else:
            parents = str(row['parents']).strip().split()

        # parse CPT array
        probs = parse_cpt_values(row['data'])

        # simpan
        cpts_loaded[(node, tuple(parents))] = probs

    return cpts_loaded

# -----------------------
# Save CPTs to CSV (same format as original: node, parents(space separated), data(string of space-separated floats))
# -----------------------
def save_cpts_to_csv(cpts_map, filepath):
    
    rows = []
    for (node, parents), probs in cpts_map.items():
        parents_str = " ".join(parents)
        data_str = " ".join(str(float(x)) for x in probs)
        rows.append({'node': node, 'parents': parents_str, 'data': data_str})
    import pandas as pd
    pd.DataFrame(rows)[['node','parents','data']].to_csv(filepath, index=False)

# -----------------------
# Get parents helper
# -----------------------
def get_parents_map_from_cpts(cpts_map):
    parents_map = {}
    for (node, parents) in cpts_map.keys():
        parents_map.setdefault(node, list(parents))
    return parents_map

# -----------------------
# Get probability P(node=state | parent_evidence)
# cpts_map expected values are lists (floats) and ordering is:
# for each parent combination (lexicographic), list of probs for node.states in order
# -----------------------
def get_probability(node, state, parent_evidence, node_states, cpts_map):
    """
    parent_evidence: dict {parent: state}
    node_states: dict {node: [state_list]}
    """
    key = None
    # find entry for node (any parents)
    for (n, parents) in cpts_map.keys():
        if n == node:
            key = (n, parents)
            break
    if key is None:
        return 0.0
    parents = list(key[1])
    probs = cpts_map[key]
    # index into probs
    s_index = node_states[node].index(state)

    if len(parents) == 0:
        # probs is list of length S
        return probs[s_index]
    # build parent indices lexicographically
    parent_indices = []
    for p in parents:
        p_state = parent_evidence.get(p)
        if p_state is None:
            # parent state missing -> cannot evaluate this conditional
            return 0.0
        parent_indices.append(node_states[p].index(p_state))
    # compute index: for lexicographic product parents, with fastest-changing dimension = node state
    # we stored for each parent-combination the list [P(node=state1),P(node=state2),...]
    # so index = (parent_comb_index) * S + s_index
    # parent_comb_index computed as base-mixed radix
    base = [len(node_states[p]) for p in parents]
    comb_index = 0
    for idx, b in zip(parent_indices, base):
        comb_index = comb_index * b + idx
    index = comb_index * len(node_states[node]) + s_index
    if index < 0 or index >= len(probs):
        return 0.0
    return probs[index]

# -----------------------
# Calculate joint probability for full assignment (assignment is dict node->state)
# -----------------------
def calculate_joint_probability(assignment, node_states, cpts_map):
    prob = 1.0
    # iterate nodes in any order (product rule uses P(Xi | parents(Xi)))
    for (node, parents) in cpts_map.keys():
        parents = list(parents)
        # build parent evidence
        parent_evidence = {p: assignment[p] for p in parents}
        node_state = assignment[node]
        p = get_probability(node, node_state, parent_evidence, node_states, cpts_map)
        prob *= p
    return prob

# -----------------------
# Infer posterior P(query_node | evidence) by enumeration (exact marginalization)
# unobserved variables are all nodes not in evidence and not query_node
# -----------------------
def infer_posterior(query_node, evidence, node_states, cpts_map):
    all_nodes = list({n for (n,_) in cpts_map.keys()})
    states_q = node_states[query_node]
    result = {s: 0.0 for s in states_q}
    unobserved = [n for n in all_nodes if n not in evidence and n != query_node]

    from itertools import product
    # iterate over all states of unobserved variables
    states_lists = [node_states[n] for n in unobserved]
    for combo in product(*states_lists) if states_lists else [()]:
        assign = dict(zip(unobserved, combo))
        for s in states_q:
            full_assign = {**evidence, **assign, query_node: s}
            result[s] += calculate_joint_probability(full_assign, node_states, cpts_map)
    # normalize
    tot = sum(result.values())
    if tot > 0:
        for k in result:
            result[k] /= tot
    return result

# -----------------------
# Visualization: network + annotate nodes with CPT snippet (first few probs or full)
# -----------------------
def visualize_bayesian_network(cpts_map, node_states, show_cpt_full=False, figsize=(14,10)):
    G = nx.DiGraph()
    nodes = sorted({n for (n,_) in cpts_map.keys()})
    for n in nodes:
        G.add_node(n)
    for (child, parents) in cpts_map.keys():
        for p in parents:
            G.add_edge(p, child)

    # level layout
    def get_node_levels(graph):
        levels = {}
        roots = [n for n in graph.nodes() if graph.in_degree(n) == 0]
        queue = [(r, 0) for r in roots]
        visited = set(roots)
        while queue:
            node, lvl = queue.pop(0)
            levels[node] = lvl
            for ch in graph.successors(node):
                if ch not in visited:
                    visited.add(ch)
                    queue.append((ch, lvl+1))
        # nodes not reached (cycles?) -> place at max+1
        for n in graph.nodes():
            if n not in levels:
                levels[n] = max(levels.values())+1
        return levels

    levels = get_node_levels(G)
    level_nodes = {}
    for n, l in levels.items():
        level_nodes.setdefault(l, []).append(n)
    pos = {}
    for lvl, nodes_at in level_nodes.items():
        num = len(nodes_at)
        for i, node in enumerate(nodes_at):
            x = (i - num/2) * 2.5
            y = -lvl * 3
            pos[node] = (x, y)

    plt.figure(figsize=figsize)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20, width=1.8)
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, font_size=10)

    # annotate CPT (as small text near node)
    for node in nodes:
        # find its cpt
        for (n, parents), probs in cpts_map.items():
            if n == node:
                if show_cpt_full:
                    txt = "parents: {}\n{}".format(" ".join(parents) if parents else "-", " ".join([f"{p:.3f}" for p in probs]))
                else:
                    # show up to first 9 numbers
                    txt = "p(len={})".format(len(probs))
                plt.text(pos[node][0], pos[node][1]-0.6, txt,
                         fontsize=8, ha='center', va='top', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
                break

    legend_elements = [Patch(facecolor='lightblue', label='Nodes')]
    plt.legend(handles=legend_elements, loc='upper left')
    plt.axis('off')
    plt.title("Bayesian Network (nodes & CPT summary)")
    plt.tight_layout()
    plt.show()

# -----------------------
# Evaluation helpers:
#  - log_likelihood of dataset under model
#  - node_prediction_accuracy: predict node by argmax P(node|parents) on each row and compute accuracy
#  - BIC: -2*loglik + k*ln(N) where k = number of free parameters
# -----------------------
def log_likelihood(df, node_states, cpts_map):
    ll = 0.0
    for idx, row in df.iterrows():
        # compute joint prob for this complete row
        assign = row.to_dict()
        p = calculate_joint_probability(assign, node_states, cpts_map)
        if p > 0:
            ll += np.log(p)
        else:
            # very small penalty for zero-prob entries
            ll += -1e6
    return ll

def node_prediction_accuracy(df, node_states, cpts_map):
    accs = {}
    for (node, parents) in cpts_map.keys():
        parents = list(parents)
        if len(parents) == 0:
            # predict using marginal
            probs = cpts_map[(node, tuple(parents))]
            pred = node_states[node][int(np.argmax(probs))]
            accs[node] = (df[node] == pred).mean()
        else:
            correct = 0
            for idx, row in df.iterrows():
                parent_evidence = {p: row[p] for p in parents}
                # compute P(node=state|parents)
                probs = []
                for s in node_states[node]:
                    probs.append(get_probability(node, s, parent_evidence, node_states, cpts_map))
                if sum(probs) == 0:
                    pred = None
                else:
                    pred = node_states[node][int(np.argmax(probs))]
                if pred == row[node]:
                    correct += 1
            accs[node] = correct / len(df)
    return accs

def number_of_parameters(cpts_map, node_states):
    k = 0
    for (node, parents), probs in cpts_map.items():
        s = len(node_states[node])
        # free parameters per parent combination = s-1
        parent_card = 1
        for p in parents:
            parent_card *= len(node_states[p])
        k += parent_card * (s - 1)
    return k

def bic(df, node_states, cpts_map):
    ll = log_likelihood(df, node_states, cpts_map)
    k = number_of_parameters(cpts_map, node_states)
    n = len(df)
    return -2*ll + k * np.log(n)


def visualize_cpt_table(node, parents, cpt_dict):
    print(f"CPT for node: {node}")
    print(f"Parents: {parents}")

    # Jika CPT adalah dict satu level
    if isinstance(cpt_dict, dict):
        df = pd.DataFrame.from_dict(cpt_dict, orient='index', columns=['Probability'])
        display(df)
        return

    # Jika CPT adalah list of dicts (umum di BN)
    if isinstance(cpt_dict, list) and isinstance(cpt_dict[0], dict):
        df = pd.DataFrame(cpt_dict)
        display(df)
        return

    # Default: fallback ke bentuk lama
    cpt_values = np.array(cpt_dict)
    num_states = cpt_values.shape[-1]
    rows = len(cpt_values) // num_states
    table = cpt_values.reshape(rows, num_states)

    fig, ax = plt.subplots(figsize=(4, 0.6 * rows))
    ax.axis('off')
    ax.table(cellText=table, loc='center')
    plt.show()
    
def visualize_bn_with_cpts(G, cpts, pos=None):
    plt.figure(figsize=(16, 10))
    
    if pos is None:
        pos = nx.spring_layout(G, seed=42, k=0.9)

    # Draw Graph
    nx.draw_networkx_nodes(G, pos, node_color="#a9d6e5", node_size=1800)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", width=2)

    plt.title("Bayesian Network with CPT tables", fontsize=14)
    plt.axis('off')
    plt.show()

    # Draw CPT tables beside each node
    for (node, parents), cpt_values in cpts.items():
        print(f"=== CPT for {node} ===")
        visualize_cpt_table(node, list(parents), cpt_values)