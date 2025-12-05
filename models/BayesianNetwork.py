class BayesianNetworkLearning:
    def __init__(self, structure):
        self.structure = structure
        self.cpts = {} # Menyimpan tabel probabilitas
    def train(self, df_train):
        for node, parents in self.structure.items():
            if not parents:
                # Jika tidak punya parent, hitung peluang kemunculan biasa (Prior)
                self.cpts[node] = df_train[node].value_counts(normalize=True).to_dict()
            else:
                # Jika punya parent, hitung peluang bersyarat
                self.cpts[node] = {}
                try:
                    # Group data berdasarkan nilai-nilai parent
                    groups = df_train.groupby(parents)
                    for parent_vals, subset in groups:
                        # Pastikan format key konsisten (tuple)
                        if len(parents) == 1: parent_vals = (parent_vals,)
                        
                        # Hitung probabilitas node target pada kondisi parent tersebut
                        self.cpts[node][parent_vals] = subset[node].value_counts(normalize=True).to_dict()
                except Exception as e:
                    print(f"Info: Node {node} mungkin memiliki kombinasi parent yang jarang muncul.")

    def predict_proba(self, target, evidence_dict):
        """Mengambil probabilitas dari tabel CPT yang sudah dilatih"""
        parents = self.structure.get(target, [])
        # Kasus 1: Root Node
        if not parents:
            return self.cpts.get(target, {})
        # Kasus 2: Child Node
        # Buat key pencarian berdasarkan evidence
        key = tuple(evidence_dict.get(p) for p in parents)
        if key in self.cpts[target]:
            return self.cpts[target][key]