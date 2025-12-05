from pgmpy.readwrite import BIFReader
from pgmpy.sampling import BayesianModelSampling


def sample(path: str, size, output_path):
    reader = BIFReader(path)
    model = reader.get_model()
    s = BayesianModelSampling(model)
    data = s.forward_sample(size=size)
    data.to_csv(output_path)


if __name__ == '__main__':
    sample(r"data\sachs.bif\sachs.bif", 5000, r"data\sachs.bif\sachs.csv")