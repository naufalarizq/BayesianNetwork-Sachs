import numpy as np
import re
import csv

# ============================================================================
# BAGIAN 1: PARSING FILE CSV UNTUK MEMBACA JARINGAN BAYESIAN
# ============================================================================
# Tujuan: Membaca file CSV yang berisi struktur jaringan Bayesian
# termasuk node-node, state-nya, dan Conditional Probability Table (CPT)

nodes = {}  # Dictionary untuk menyimpan node dan state-nya: {node_name: [states]}
cpts = {}   # Dictionary untuk menyimpan CPT data: {(node, parents_tuple): cpt_values_string}

# Definisikan states untuk setiap node (sama untuk semua node dalam Sachs)
states = ["LOW", "AVG", "HIGH"]

# ====== LANGKAH 1: BACA FILE CSV ======
# File CSV berisi kolom: node, parents, data
# - node: nama node
# - parents: parent nodes dipisahkan dengan spasi (kosong jika tidak ada parent)
# - data: probabilitas values dipisahkan dengan spasi

with open("data\sachs.bif\sachs_cpts.csv", 'r') as csvfile:
    reader = csv.DictReader(csvfile)  # Baca CSV dengan header
    for row in reader:
        node_name = row['node'].strip()  # Ambil nama node
        parents_str = row['parents'].strip()  # Ambil string parent
        cpt_data = row['data'].strip()  # Ambil data probabilitas
        
        # Jika parents kosong, artinya node ini tidak punya parent
        parent_vars = parents_str.split() if parents_str else []
        
        # Simpan node dan state-nya
        nodes[node_name] = states
        print(f"Node '{node_name}' dengan states: {states}")
        
        # Simpan CPT dengan key berupa tuple (node, parent_tuple)
        cpts[(node_name, tuple(parent_vars))] = cpt_data
        print(f"CPT untuk {node_name} diberikan {parent_vars if parent_vars else 'tanpa parent'}")

print("\nStruktur jaringan berhasil dibaca dari CSV.")

# ============================================================================
# BAGIAN 2: FUNGSI UNTUK MENGAKSES NILAI PROBABILITAS DARI CPT
# ============================================================================

# Fungsi 1: parse_cpt_values
# Mengekstrak semua angka (probabilitas) dari string CPT yang berisi parentheses bersarang
# Contoh input: "(((0.6721176592 0.3277794919 0.0001028489)...))"
# Output: [0.6721176592, 0.3277794919, 0.0001028489, ...]
def parse_cpt_values(cpt_string):
    """Parse nilai probabilitas dari string CPT, handle parentheses bersarang."""
    # Regex ini mencari semua angka termasuk yang menggunakan scientific notation
    # [0-9]+\.?[0-9]* = integer atau decimal
    # (?:[eE][+-]?[0-9]+)? = opsional scientific notation seperti 1e-05
    values = [float(x) for x in re.findall(r'[0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?', cpt_string)]
    return values

# Fungsi 2: get_probability
# Mengambil nilai probabilitas spesifik dari CPT berdasarkan node, state, dan parent state
# Contoh: get_probability('Erk', 'HIGH', {'Mek': 'HIGH', 'PKA': 'AVG'}, ...)
# Ini akan mengembalikan P(Erk=HIGH | Mek=HIGH, PKA=AVG)
def get_probability(node, state, parent_evidence, node_states, cpts):
    """Ambil probabilitas dari CPT yang sudah diparse secara manual."""
    # Cari CPT entry yang match dengan node yang dicari
    for (cpt_node, cpt_parents), cpt_data in cpts.items():
        if cpt_node == node:
            parent_vars = list(cpt_parents)  # List dari parent node names
            values = parse_cpt_values(cpt_data)  # Ekstrak semua probability values
            
            # ====== MENGHITUNG INDEX DALAM ARRAY CPT YANG DI-FLATTEN ======
            # CPT disimpan sebagai array 1D yang di-flatten
            # Kita perlu menghitung index berdasarkan state kombinasi
            # 
            # Contoh: P(Erk | Mek, PKA) dengan masing2 punya 3 state (LOW, AVG, HIGH)
            # Total entries: 3 * 3 * 3 = 27
            # Index dihitung dengan: state_index + parent1_index * 3 + parent2_index * 9
            
            state_index = node_states[node].index(state)  # Index state dari node
            multiplier = len(node_states[node])  # Berapa banyak state untuk node ini
            index = state_index  # Mulai dari state_index
            
            # Proses setiap parent node dari kiri ke kanan
            for parent in parent_vars:
                parent_state = parent_evidence.get(parent)  # Ambil state dari parent
                if parent_state is None:  # Jika parent state tidak ada, return 0
                    return 0.0
                parent_index = node_states[parent].index(parent_state)  # Index state parent
                index += parent_index * multiplier  # Tambah ke index dengan multiplier
                multiplier *= len(node_states[parent])  # Update multiplier untuk parent selanjutnya
            
            # Ambil value dari array pada index yang sudah dihitung
            return values[index] if index < len(values) else 0.0
    
    return 0.0

# ============================================================================
# BAGIAN 3: FUNGSI UNTUK MENGHITUNG JOINT PROBABILITY
# ============================================================================

all_node_names = list(nodes.keys())  # List semua node dalam jaringan
print(f"\nSemua node dalam jaringan: {all_node_names}")

# Fungsi: calculate_joint_probability
# Menghitung P(X1=x1, X2=x2, ..., Xn=xn) untuk assignment lengkap semua node
# Menggunakan chain rule: P(X1,...,Xn) = P(X1)*P(X2|X1)*P(X3|X1,X2)*...*P(Xn|X1,...,Xn-1)
def calculate_joint_probability(assignment, node_states, cpts):
    """Hitung joint probability untuk assignment lengkap."""
    prob = 1.0  # Mulai dengan probabilitas 1
    for node in all_node_names:
        state = assignment[node]  # Ambil state untuk node ini
        
        # Cari parent nodes untuk node saat ini
        # Melihat ke dalam cpts untuk menemukan relasi parent-child
        parents = [n for n in all_node_names if (node, tuple([n])) in cpts or any(n in p for _, p in cpts.keys() if _ == node)]
        
        # Ambil evidence (state) dari semua parent nodes
        parent_evidence = {p: assignment[p] for p in parents if p in assignment}
        
        # Hitung P(node=state | parents) dan kalikan dengan prob sebelumnya
        # Ini mengimplementasikan chain rule
        prob *= get_probability(node, state, parent_evidence, node_states, cpts)
    
    return prob

# ============================================================================
# BAGIAN 4: INFERENCE - MENGHITUNG POSTERIOR PROBABILITY
# ============================================================================

# Fungsi: infer_posterior
# Menghitung posterior probability: P(Query | Evidence)
# Menggunakan marginalization dan normalization
def infer_posterior(query_node, evidence, node_states, cpts):
    """Hitung posterior probability dari query_node diberikan evidence."""
    all_states = node_states[query_node]  # Semua possible states untuk query node
    posteriors = {state: 0.0 for state in all_states}  # Dictionary untuk menyimpan hasil
    
    # Identifikasi variabel yang tidak diamati (tidak di evidence dan bukan query)
    unobserved = [n for n in all_node_names if n not in evidence and n != query_node]
    
    # Untuk setiap state dari query node, hitung joint probability
    for query_state in all_states:
        total = 0.0
        
        # Fungsi nested: iterate_assignments
        # Recursively generate semua kombinasi state untuk unobserved variables
        # Ini melakukan marginalization (sum out) variabel yang tidak diobservasi
        def iterate_assignments(unobs_list, current_assignment):
            if not unobs_list:  # Base case: semua unobserved variables sudah di-assign
                # Buat complete assignment dengan: evidence + query_state + unobserved assignments
                assignment = {**evidence, query_node: query_state, **current_assignment}
                # Hitung joint probability untuk assignment lengkap ini
                return calculate_joint_probability(assignment, node_states, cpts)
            
            node = unobs_list[0]  # Ambil unobserved variable pertama
            total = 0.0
            # Iterate melalui semua possible states untuk variable ini
            for state in node_states[node]:
                # Recursive call untuk variable berikutnya
                total += iterate_assignments(unobs_list[1:], {**current_assignment, node: state})
            return total
        
        # Jalankan marginalization: sum out semua unobserved variables
        total = iterate_assignments(unobserved, {})
        posteriors[query_state] = total
    
    # ====== NORMALISASI ======
    # P(Query | Evidence) = P(Query, Evidence) / P(Evidence)
    # P(Evidence) = sum dari semua posteriors (sebelum normalisasi)
    total_prob = sum(posteriors.values())
    if total_prob > 0:
        # Normalisasi: bagi setiap posterior dengan total
        posteriors = {s: p / total_prob for s, p in posteriors.items()}
    
    return posteriors

# ============================================================================
# BAGIAN 5: MENJALANKAN INFERENCE DENGAN EVIDENCE TERTENTU
# ============================================================================

# Tentukan evidence (observasi): apa yang kita ketahui
# Contoh: PKC=HIGH, PKA=AVG, Raf=HIGH
evidence = {
    'PKC': 'HIGH',
    'PKA': 'AVG',
    'Raf': 'HIGH'
}

# QUERY 1: Apa probabilitas Erk diberikan evidence di atas?
query_node = 'Erk'
posterior_probabilities = infer_posterior(query_node, evidence, nodes, cpts)

print(f"\nProbabilitas Posterior untuk {query_node} Diberikan Evidence:")
print(f"Evidence: {evidence}\n")
for state, prob in posterior_probabilities.items():
    print(f"P({query_node}={state} | Evidence) = {prob:.6f}")

most_likely_state = max(posterior_probabilities, key=posterior_probabilities.get)
print(f"\nState paling mungkin untuk {query_node} adalah: {most_likely_state} (Probabilitas: {posterior_probabilities[most_likely_state]:.6f})")

# QUERY 2: Apa probabilitas PIP2 diberikan evidence?
query_node = 'PIP2'
posterior_probabilities = infer_posterior(query_node, evidence, nodes, cpts)

print(f"\n{'='*60}")
print(f"\nProbabilitas Posterior untuk {query_node} Diberikan Evidence:")
print(f"Evidence: {evidence}\n")
for state, prob in posterior_probabilities.items():
    print(f"P({query_node}={state} | Evidence) = {prob:.6f}")

most_likely_state = max(posterior_probabilities, key=posterior_probabilities.get)
print(f"\nState paling mungkin untuk {query_node} adalah: {most_likely_state} (Probabilitas: {posterior_probabilities[most_likely_state]:.6f})")

# QUERY 3: Apa probabilitas Akt diberikan evidence?
query_node = 'Akt'
posterior_probabilities = infer_posterior(query_node, evidence, nodes, cpts)

print(f"\n{'='*60}")
print(f"\nProbabilitas Posterior untuk {query_node} Diberikan Evidence:")
print(f"Evidence: {evidence}\n")
for state, prob in posterior_probabilities.items():
    print(f"P({query_node}={state} | Evidence) = {prob:.6f}")

most_likely_state = max(posterior_probabilities, key=posterior_probabilities.get)
print(f"\nState paling mungkin untuk {query_node} adalah: {most_likely_state} (Probabilitas: {posterior_probabilities[most_likely_state]:.6f})")

print(f"\n{'='*60}")
print("\nInference selesai.")