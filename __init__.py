"""
SCD (Semantic Contrastive Decoding) — standalone package.

Usage:
    from scd import SCD, scd_decode, semantic_clustering, semantic_entropy

    # High-level API:
    scd = SCD(openai_api_key="sk-...")
    result = scd.decode(question="...", evidence_docs=[...], llm_func=my_llm)

    # Low-level API:
    best, scores = scd_decode(client, model, question, prof_dist, amat_dist)
"""

from .scd import SCD, scd_decode

__all__ = ["SCD", "scd_decode"]
