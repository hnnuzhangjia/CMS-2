"""Semantic entropy and confidence-guided contrastive decoding."""

import math
from typing import Dict, Tuple
from openai import OpenAI

from .clustering import _joint_clustering

def _score_clusters(professional_dist, amateur_dist):
    """Equations (5)-(6), with a numerically stable softmax."""
    if sum(professional_dist.values()) == 0:
        raise ValueError("At least one professional sample is required")
    alpha = _semantic_entropy(professional_dist)
    logits = {
        rep: (1.0 + alpha) * count - alpha * amateur_dist.get(rep, 0)
        for rep, count in professional_dist.items()
    }
    peak = max(logits.values())
    weights = {rep: math.exp(value - peak) for rep, value in logits.items()}
    total = sum(weights.values())
    scores = {rep: weight / total for rep, weight in weights.items()}
    return max(logits, key=logits.get), scores


def scd_decode(
    client: OpenAI,
    model: str,
    question: str,
    professional_dist: Dict[str, int],
    amateur_dist: Dict[str, int],
) -> Tuple[str, Dict[str, float]]:
    """Jointly cluster weighted representatives and return softmax SCD scores."""
    expert, amateur = _joint_clustering(
        client, model, question, professional_dist.items(), amateur_dist.items()
    )
    return _score_clusters(expert, amateur)
