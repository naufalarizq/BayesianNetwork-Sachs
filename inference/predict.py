class VariableElimination:
    def __init__(self, model, df_train):
        """
        model: Objek BayesianNetworkLearning yang sudah dilatih
        df_train: Dataframe training asli (X_train_raw) untuk menghitung probabilitas
        """
        self.model = model
        self.df = df_train
    def query(self, variables, evidence=None):
        """
        Meniru fungsi pgmpy.inference.VariableElimination.query
        """
        target = variables[0]
        subset = self.df.copy()
        # 1. Filter Data Berdasarkan Evidence
        if evidence:
            for key, val in evidence.items():
                # Pastikan evidence valid
                if key in subset.columns:
                    subset = subset[subset[key] == val]
                else:
                    print(f"Warning: Variabel evidence '{key}' tidak ditemukan.")
        # 2. Hitung Probabilitas Target
        if len(subset) == 0:
            print("Warning: Kombinasi evidence ini tidak pernah muncul di data training.")
            return None
        # Value counts normalize=True memberikan probabilitas
        result = subset[target].value_counts(normalize=True).sort_index()
        # Format tampilan agar mirip pgmpy
        print(f"+{'-'*20}+{'-'*15}+")
        print(f"| {target:<18} | {'phi(' + target + ')':<13} |")
        print(f"+{'-'*20}+{'-'*15}+")
        for state, prob in result.items():
            # Jika state berupa angka, coba mapping balik jika memungkinkan (opsional)
            print(f"| {str(state):<18} | {prob:.4f}        |")
        print(f"+{'-'*20}+{'-'*15}+")
        return result