"""Bidirectional semantic equivalence and joint answer clustering."""

from typing import Dict, List
from openai import OpenAI

from .nli import _call_gpt4o_mini_nli


def _equivalent(client, model, question, answer1, answer2):
    """Semantic equivalence requires entailment in both directions."""
    if answer1 == answer2:
        return True
    return (
        _call_gpt4o_mini_nli(client, model, question, answer1, answer2) == "Entailment"
        and _call_gpt4o_mini_nli(client, model, question, answer2, answer1) == "Entailment"
    )


def _joint_clustering(client, model, question, professional, amateur):
    """Cluster ordered (answer, count) entries against first representatives."""
    expert_counts = {}
    amateur_counts = {}
    for entries, target in ((professional, expert_counts), (amateur, amateur_counts)):
        for answer, count in entries:
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError("Cluster frequencies must be non-negative integers")
            if count == 0:
                continue
            representative = next(
                (rep for rep in expert_counts
                 if _equivalent(client, model, question, rep, answer)), None
            )
            if representative is None:
                representative = answer
                expert_counts[representative] = 0
                amateur_counts[representative] = 0
            target[representative] += count
    return expert_counts, amateur_counts


