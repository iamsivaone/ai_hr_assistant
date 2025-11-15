"""Vector store wrapper — supports FAISS locally, and a stub for Pinecone (configurable)."""

from typing import List, Tuple
import os
from pathlib import Path
import numpy as np

from sentence_transformers import SentenceTransformer
import faiss

EMB_MODEL = "all-MiniLM-L6-v2"


class FaissStore:
    def __init__(self, index_path: str = "./data/indexes/faiss_index"):
        """
        Initialize a FaissStore instance.

        Args:
            index_path (str, optional): Path to the FAISS index file. Defaults to "./data/indexes/faiss_index".

        Initializes a FaissStore instance, loading the FAISS index from disk if it exists.
        """
        self.index_path = Path(index_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.model = SentenceTransformer(EMB_MODEL)
        self.index = None
        self.ids = []

        if self.index_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                self.ids = list(
                    np.load(str(self.index_path) + ".ids.npy", allow_pickle=True)
                )
            except Exception:
                self.index = None
                self.ids = []

    def _emb(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into a NumPy array of embeddings.

        Args:
            texts (List[str]): List of texts to encode.

        Returns:
            np.ndarray: NumPy array of shape (len(texts), d) where d is the dimensionality of the embeddings.
        """
        return self.model.encode(texts, convert_to_numpy=True)

    def add(self, texts: List[str], ids: List[str]):
        """
        Add a list of texts with their corresponding IDs to the FAISS index.

        Args:
            texts (List[str]): List of texts to add to the index.
            ids (List[str]): List of IDs corresponding with the texts.

        Raises:
            ValueError: If the number of texts and ids do not match.

        Saves the updated index and IDs to disk after adding the new texts.
        """
        if len(texts) != len(ids):
            raise ValueError("Number of texts and ids must match")

        vecs = self._emb(texts)
        d = vecs.shape[1]

        if self.index is None:
            self.index = faiss.IndexFlatL2(d)

        self.index.add(vecs)
        self.ids.extend(ids)

        # Save index and IDs
        faiss.write_index(self.index, str(self.index_path))
        np.save(str(self.index_path) + ".ids.npy", np.array(self.ids))

    def query(self, text: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Query the FAISS index for the k nearest neighbors to the given text.

        Args:
            text (str): Text to query the index with.
            k (int, optional): Number of nearest neighbors to return. Defaults to 5.

        Returns:
            List[Tuple[str, float]]: List of tuples containing the ID and distance of each of the k nearest neighbors.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        vec = self._emb([text])
        D, I = self.index.search(vec, k)

        results = []
        for dist, idx in zip(D[0], I[0]):
            if 0 <= idx < len(self.ids):
                results.append((self.ids[idx], float(dist)))
        return results
