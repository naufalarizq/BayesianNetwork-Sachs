# ============================================================
# IMPLEMENTASI BAYESIAN NETWORK FROM SCRATCH
# ============================================================

import re                          # Digunakan untuk parsing string CPT dari file teks/CSV
import numpy as np                 # Operasi numerik dan perhitungan probabilitas
import pandas as pd                # Manipulasi dataset observasi
from collections import OrderedDict  # Menjaga urutan CPT agar konsisten
import networkx as nx              # Representasi graf DAG Bayesian Network
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from IPython.display import display


# ============================================================
# HELPER: PARSING CPT
# ============================================================

def parse_cpt_values(cpt_string):
    """
    Fungsi ini mengubah CPT yang awalnya berbentuk string
    (misalnya "(0.2 0.3 0.5)") menjadi list float.

    Ini diperlukan karena inferensi Bayesian membutuhkan
    nilai probabilitas numerik, bukan representasi teks.
    """
    if cpt_string is None:
        return []

    # Regex digunakan agar fleksibel terhadap format CPT
    vals = [
        float(x) for x in re.findall(
            r'[0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?',
            str(cpt_string)
        )
    ]
    return vals


# ============================================================
# PARAMETER LEARNING (MLE + LAPLACE SMOOTHING)
# ============================================================

def learn_cpts_from_data(df, node_list, states, parents_map, laplace=1.0):
    """
    Fungsi ini mengimplementasikan pembelajaran parameter
    Bayesian Network dari data observasi.

    Secara teori:
    P(X | Parents(X)) = N(X, Parents(X)) / N(Parents(X))

    Laplace smoothing digunakan untuk mencegah probabilitas nol.
    """

    from itertools import product

    # CPT disimpan dalam OrderedDict agar urutan konsisten
    cpts = OrderedDict()

    # Menentukan state diskrit untuk setiap node
    # Ini penting karena CPT bergantung pada urutan state
    node_states = {}
    if isinstance(states, dict):
        node_states = states
    else:
        for n in node_list:
            node_states[n] = list(states)

    # Iterasi setiap node dalam Bayesian Network
    for node in node_list:

        # Ambil parent node sesuai struktur DAG
        parents = parents_map.get(node, []) or []

        # ====================================================
        # KASUS 1: NODE TANPA PARENT (ROOT NODE)
        # ====================================================
        # P(X) dihitung langsung dari frekuensi data
        if len(parents) == 0:

            # Hitung frekuensi kemunculan setiap state
            counts = (
                df[node]
                .value_counts()
                .reindex(node_states[node], fill_value=0)
                .astype(float)
            )

            # Laplace smoothing untuk menghindari probabilitas nol
            counts += laplace

            # Normalisasi menjadi distribusi probabilitas
            probs = (counts / counts.sum()).tolist()

            # Simpan sebagai CPT node tanpa parent
            cpts[(node, tuple(parents))] = probs

        # ====================================================
        # KASUS 2: NODE DENGAN PARENT
        # ====================================================
        # P(X | Y1, Y2, ...)
        else:
            parent_states_lists = [node_states[p] for p in parents]

            # CPT disimpan dalam bentuk flattened list
            probs_flat = []

            # Enumerasi semua kombinasi state parent (kartesian)
            for comb in product(*parent_states_lists):

                # Filter data yang sesuai kombinasi parent
                mask = np.ones(len(df), dtype=bool)
                for p, val in zip(parents, comb):
                    mask &= (df[p] == val)

                subset = df[mask]

                # Hitung distribusi child node
                counts = (
                    subset[node]
                    .value_counts()
                    .reindex(node_states[node], fill_value=0)
                    .astype(float)
                )

                # Laplace smoothing
                counts += laplace

                # Normalisasi
                probs = (counts / counts.sum()).tolist()

                # Disimpan berurutan untuk tiap kombinasi parent
                probs_flat.extend(probs)

            cpts[(node, tuple(parents))] = probs_flat

    return cpts, node_states


# ============================================================
# MENGAMBIL PROBABILITAS KONDISIONAL
# ============================================================

def get_probability(node, state, parent_evidence, node_states, cpts_map):
    """
    Mengambil nilai:
    P(node = state | parent_evidence)

    Fungsi ini adalah operasi fundamental dalam inferensi Bayesian Network.
    """

    # Cari CPT yang sesuai dengan node
    key = None
    for (n, parents) in cpts_map.keys():
        if n == node:
            key = (n, parents)
            break

    if key is None:
        return 0.0

    parents = list(key[1])
    probs = cpts_map[key]

    # Indeks state node
    s_index = node_states[node].index(state)

    # Jika node tidak memiliki parent
    if len(parents) == 0:
        return probs[s_index]

    # Hitung indeks kombinasi parent (mixed radix)
    parent_indices = []
    for p in parents:
        if p not in parent_evidence:
            return 0.0
        parent_indices.append(node_states[p].index(parent_evidence[p]))

    base = [len(node_states[p]) for p in parents]
    comb_index = 0
    for idx, b in zip(parent_indices, base):
        comb_index = comb_index * b + idx

    # Indeks akhir dalam CPT flattened
    index = comb_index * len(node_states[node]) + s_index

    return probs[index] if 0 <= index < len(probs) else 0.0


# ============================================================
# JOINT PROBABILITY
# ============================================================

def calculate_joint_probability(assignment, node_states, cpts_map):
    """
    Menghitung probabilitas gabungan:
    P(X1, X2, ..., Xn)

    Menggunakan chain rule Bayesian Network:
    ∏ P(Xi | Parents(Xi))
    """
    prob = 1.0

    for (node, parents) in cpts_map.keys():
        parents = list(parents)
        parent_evidence = {p: assignment[p] for p in parents}
        node_state = assignment[node]

        # Kalikan probabilitas lokal
        prob *= get_probability(
            node, node_state, parent_evidence,
            node_states, cpts_map
        )

    return prob


# ============================================================
# INFERENSI POSTERIOR (EXACT ENUMERATION)
# ============================================================

def infer_posterior(query_node, evidence, node_states, cpts_map):
    """
    Menghitung probabilitas posterior:
    P(query_node | evidence)

    Menggunakan metode exact inference dengan enumerasi
    seluruh variabel tersembunyi.
    """

    all_nodes = list({n for (n, _) in cpts_map.keys()})
    states_q = node_states[query_node]

    # Inisialisasi hasil posterior
    result = {s: 0.0 for s in states_q}

    # Variabel tersembunyi (tidak di-query dan tidak di-evidence)
    unobserved = [
        n for n in all_nodes
        if n not in evidence and n != query_node
    ]

    from itertools import product
    states_lists = [node_states[n] for n in unobserved]

    # Enumerasi semua kemungkinan variabel tersembunyi
    for combo in product(*states_lists) if states_lists else [()]:
        assign = dict(zip(unobserved, combo))

        for s in states_q:
            full_assign = {**evidence, **assign, query_node: s}

            # Hitung joint probability
            result[s] += calculate_joint_probability(
                full_assign, node_states, cpts_map
            )

    # Normalisasi agar menjadi distribusi probabilitas valid
    total = sum(result.values())
    if total > 0:
        for k in result:
            result[k] /= total

    return result


# ============================================================
# VISUALISASI BAYESIAN NETWORK
# Menampilkan struktur DAG dan ringkasan CPT
# ============================================================

def visualize_bayesian_network(cpts_map, node_states, show_cpt_full=False, figsize=(14,10)):
    # Membuat graf berarah (Directed Acyclic Graph)
    # Graf ini merepresentasikan struktur kausal Bayesian Network
    G = nx.DiGraph()

    # Mengambil semua node unik dari CPT
    nodes = sorted({n for (n,_) in cpts_map.keys()})

    # Menambahkan node ke graf
    for n in nodes:
        G.add_node(n)

    # Menambahkan edge dari parent ke child sesuai struktur BN
    for (child, parents) in cpts_map.keys():
        for p in parents:
            G.add_edge(p, child)

    # ========================================================
    # MENENTUKAN LEVEL NODE (ROOT → LEAF)
    # ========================================================
    # Tujuan: memastikan layout mengikuti arah kausal
    def get_node_levels(graph):
        levels = {}

        # Root node adalah node tanpa incoming edge
        roots = [n for n in graph.nodes() if graph.in_degree(n) == 0]

        # BFS untuk menentukan kedalaman node
        queue = [(r, 0) for r in roots]
        visited = set(roots)

        while queue:
            node, lvl = queue.pop(0)
            levels[node] = lvl

            # Anak node ditempatkan di level berikutnya
            for ch in graph.successors(node):
                if ch not in visited:
                    visited.add(ch)
                    queue.append((ch, lvl+1))

        # Jika ada node yang tidak terjangkau (keamanan tambahan)
        for n in graph.nodes():
            if n not in levels:
                levels[n] = max(levels.values()) + 1

        return levels

    # Menghitung level setiap node
    levels = get_node_levels(G)

    # Mengelompokkan node berdasarkan level
    level_nodes = {}
    for n, l in levels.items():
        level_nodes.setdefault(l, []).append(n)

    # Menentukan posisi (x, y) node agar tersusun rapi
    pos = {}
    for lvl, nodes_at in level_nodes.items():
        num = len(nodes_at)
        for i, node in enumerate(nodes_at):
            x = (i - num/2) * 2.5   # Spasi horizontal
            y = -lvl * 3            # Spasi vertikal antar level
            pos[node] = (x, y)

    # ========================================================
    # MENGGAMBAR GRAF BAYESIAN NETWORK
    # ========================================================
    plt.figure(figsize=figsize)

    # Menggambar edge (hubungan kausal)
    nx.draw_networkx_edges(
        G, pos, arrows=True,
        arrowsize=20, width=1.8
    )

    # Menggambar node
    nx.draw_networkx_nodes(
        G, pos,
        node_size=2000,
        node_color='lightblue'
    )

    # Menampilkan label node
    nx.draw_networkx_labels(G, pos, font_size=10)

    # ========================================================
    # ANOTASI CPT PADA NODE
    # ========================================================
    # Tujuan: memberikan informasi probabilistik secara ringkas
    for node in nodes:
        for (n, parents), probs in cpts_map.items():
            if n == node:

                # Jika ingin menampilkan CPT lengkap
                if show_cpt_full:
                    txt = (
                        "parents: {}\n{}"
                        .format(
                            " ".join(parents) if parents else "-",
                            " ".join([f"{p:.3f}" for p in probs])
                        )
                    )
                else:
                    # Default: hanya tampilkan ukuran CPT
                    txt = "p(len={})".format(len(probs))

                # Tampilkan teks di bawah node
                plt.text(
                    pos[node][0],
                    pos[node][1] - 0.6,
                    txt,
                    fontsize=8,
                    ha='center',
                    va='top',
                    bbox=dict(
                        facecolor='white',
                        alpha=0.7,
                        boxstyle='round'
                    )
                )
                break

    # Legenda sederhana
    legend_elements = [
        Patch(facecolor='lightblue', label='Nodes')
    ]
    plt.legend(handles=legend_elements, loc='upper left')

    plt.axis('off')
    plt.title("Bayesian Network (nodes & CPT summary)")
    plt.tight_layout()
    plt.show()


# ============================================================
# EVALUASI MODEL BAYESIAN NETWORK
# ============================================================

def log_likelihood(df, node_states, cpts_map):
    """
    Menghitung log-likelihood dataset terhadap model Bayesian Network.

    Secara teori:
    log P(Data | Model) = Σ log P(x_i | CPT)
    """
    ll = 0.0

    for _, row in df.iterrows():
        # Mengubah satu baris data menjadi assignment lengkap
        assign = row.to_dict()

        # Hitung joint probability untuk satu observasi
        p = calculate_joint_probability(assign, node_states, cpts_map)

        # Jika probabilitas valid, tambahkan log-nya
        if p > 0:
            ll += np.log(p)
        else:
            # Penalti besar jika probabilitas nol
            # (menandakan model tidak menjelaskan data)
            ll += -1e6

    return ll


def node_prediction_accuracy(df, node_states, cpts_map):
    """
    Mengukur akurasi prediksi tiap node berdasarkan CPT.

    Prediksi dilakukan dengan:
    argmax P(node | parents)

    Ini bukan supervised learning,
    melainkan evaluasi konsistensi probabilistik.
    """
    accs = {}

    for (node, parents) in cpts_map.keys():
        parents = list(parents)

        # Jika node tanpa parent → gunakan distribusi marginal
        if len(parents) == 0:
            probs = cpts_map[(node, tuple(parents))]
            pred = node_states[node][int(np.argmax(probs))]
            accs[node] = (df[node] == pred).mean()

        else:
            correct = 0

            for _, row in df.iterrows():
                # Ambil evidence parent
                parent_evidence = {p: row[p] for p in parents}

                # Hitung probabilitas untuk setiap state node
                probs = [
                    get_probability(
                        node, s, parent_evidence,
                        node_states, cpts_map
                    )
                    for s in node_states[node]
                ]

                # Prediksi state dengan probabilitas maksimum
                if sum(probs) > 0:
                    pred = node_states[node][int(np.argmax(probs))]
                else:
                    pred = None

                if pred == row[node]:
                    correct += 1

            accs[node] = correct / len(df)

    return accs


def number_of_parameters(cpts_map, node_states):
    """
    Menghitung jumlah parameter bebas dalam Bayesian Network.

    Secara teori:
    Untuk setiap CPT:
    (jumlah kombinasi parent) × (jumlah state - 1)
    """
    k = 0

    for (node, parents), _ in cpts_map.items():
        s = len(node_states[node])

        # Menghitung kardinalitas parent
        parent_card = 1
        for p in parents:
            parent_card *= len(node_states[p])

        # Parameter bebas
        k += parent_card * (s - 1)

    return k


def bic(df, node_states, cpts_map):
    """
    Bayesian Information Criterion (BIC)

    BIC = -2 log L + k log n

    Digunakan untuk menilai trade-off
    antara kecocokan data dan kompleksitas model.
    """
    ll = log_likelihood(df, node_states, cpts_map)
    k = number_of_parameters(cpts_map, node_states)
    n = len(df)

    return -2 * ll + k * np.log(n)


# ============================================================
# VISUALISASI CPT DALAM BENTUK TABEL
# ============================================================

def visualize_cpt_table(node, parents, cpt_dict):
    """
    Menampilkan CPT node dalam bentuk tabel
    untuk interpretasi manusia.
    """
    print(f"CPT for node: {node}")
    print(f"Parents: {parents}")

    # Jika CPT berbentuk dictionary
    if isinstance(cpt_dict, dict):
        df = pd.DataFrame.from_dict(
            cpt_dict,
            orient='index',
            columns=['Probability']
        )
        display(df)
        return

    # Jika CPT berupa list of dict
    if isinstance(cpt_dict, list) and isinstance(cpt_dict[0], dict):
        df = pd.DataFrame(cpt_dict)
        display(df)
        return

    # Jika CPT berupa flattened list
    cpt_values = np.array(cpt_dict)
    num_states = cpt_values.shape[-1]
    rows = len(cpt_values) // num_states
    table = cpt_values.reshape(rows, num_states)

    fig, ax = plt.subplots(figsize=(4, 0.6 * rows))
    ax.axis('off')
    ax.table(cellText=table, loc='center')
    plt.show()


def visualize_bn_with_cpts(G, cpts, pos=None):
    """
    Menampilkan graf Bayesian Network
    dan CPT setiap node secara terpisah.
    """
    plt.figure(figsize=(16, 10))

    # Jika posisi node belum ditentukan, gunakan spring layout
    if pos is None:
        pos = nx.spring_layout(G, seed=42, k=0.9)

    # Menggambar node, edge, dan label
    nx.draw_networkx_nodes(G, pos, node_color="#a9d6e5", node_size=1800)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", width=2)

    plt.title("Bayesian Network with CPT tables", fontsize=14)
    plt.axis('off')
    plt.show()

    # Menampilkan CPT setiap node
    for (node, parents), cpt_values in cpts.items():
        print(f"=== CPT for {node} ===")
        visualize_cpt_table(node, list(parents), cpt_values)
