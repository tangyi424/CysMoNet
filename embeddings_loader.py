# File: embeddings_loader.py
import numpy as np
from typing import Dict, Optional


def load_npz_embeddings(path: str) -> Dict[str, np.ndarray]:
    """Load embeddings saved as numpy .npz with keys as sequences.
    Returns a dict: sequence_str -> 1D numpy array.
    """
    data = np.load(path, allow_pickle=True)
    # if saved as object arrays or mapping
    try:
        # npz where each key is a sequence
        seq_keys = [k for k in data.files]
        out = {k: np.asarray(data[k]) for k in seq_keys}
        return out
    except Exception:
        # fallback: single item that's a dict
        arr = list(data.items())[0][1]
        if isinstance(arr, np.ndarray) and arr.dtype == object:
            # assume it's an array of (seq, emb) pairs
            return {k: np.asarray(v) for k, v in arr}
        raise


class EmbeddingStore:
    """Simple embedding store wrapper. Expects dict-like mapping seq->embedding
    """
    def __init__(self, mapping: Dict[str, np.ndarray]):
        self.mapping = mapping

    def get(self, seq: str) -> Optional[np.ndarray]:
        return self.mapping.get(seq)