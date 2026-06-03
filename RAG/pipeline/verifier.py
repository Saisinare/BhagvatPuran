# RAG/pipeline/verifier.py
from sentence_transformers import CrossEncoder
import numpy as np

# small NLI checkpoint (or reuse a cross-encoder)
NLI_MODEL = "cross-encoder/nli-roberta-base"  # example; replace if unavailable


class Verifier:
    def __init__(self, model_name=NLI_MODEL):
        try:
            self.model = CrossEncoder(model_name)
        except Exception:
            # fallback to reranker model if NLI model not available
            from sentence_transformers import CrossEncoder as CE
            self.model = CE("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def verify_claim(self, claim: str, passages: list):
        """
        For a claim (string) compute best entailment score against passages.
        Returns normalized score [0,1] (higher => more supported) and supporting passage index.
        """
        pairs = [[claim, p] for p in passages]
        scores = self.model.predict(pairs)

        # scores can be shape (n,) for reranker models or (n,3) for NLI (contradiction, neutral, entailment)
        scores = np.asarray(scores)
        if scores.ndim == 2 and scores.shape[1] >= 3:
            # Use entailment logit/probability (last column) as support score
            entail_scores = scores[:, -1]
        else:
            # Fallback: use 1D scores directly
            entail_scores = scores
        
        # Normalize scores to [0, 1] using sigmoid for interpretability
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        normalized_scores = sigmoid(entail_scores)
        
        best_idx = int(np.argmax(normalized_scores))
        best_score = float(normalized_scores[best_idx])
        
        return best_score, best_idx
