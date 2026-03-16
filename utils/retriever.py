import faiss
import numpy as np

class VectorRetriever:
    def __init__(self, embeddings):
        self.embeddings = np.array(embeddings).astype("float32")
        dimension = self.embeddings.shape[1]

        # create FAISS index
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)

    def retrieve(self, query_embedding, k=3):
        query_embedding = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_embedding, k)

        return indices[0]