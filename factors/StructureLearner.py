class StructureLearner:
    def __init__(self, threshold=0.1):
        self.threshold = threshold
        self.edges = []
        self.structure_dict = {}
    def fit(self, df_num):
        # Gunakan korelasi Spearman karena data kita ordinal (0,1,2)
        corr_matrix = df_num.corr(method='spearman').abs()
        columns = df_num.columns.tolist()
        self.structure_dict = {col: [] for col in columns}
        print(f"threshold korelasi > {self.threshold}...")
        # Iterasi mencari hubungan
        # CONSTRAINT: j < i (Kolom kiri mempengaruhi kolom kanan)
        # Ini adalah cara sederhana mencegah Cycle (Looping) tanpa algoritma rumit
        for i in range(len(columns)):
            for j in range(i):
                child = columns[i]
                parent = columns[j]
                # Jika korelasi kuat, anggap ada hubungan (Parent -> Child)
                if corr_matrix.iloc[i, j] > self.threshold:
                    self.structure_dict[child].append(parent)
                    self.edges.append((parent, child))
        print(f"{len(self.edges)} hubungan (arcs).")
        return self.structure_dict, self.edges