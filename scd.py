"""
SCD (Semantic Contrastive Decoding) - Standalone Implementation using GPT-4o-mini.

Core idea:
  1. For each evidence document, generate N answers from the LLM in "professional" mode.
  2. Cluster semantically equivalent answers using GPT-4o-mini NLI → answer distribution.
  3. Compute semantic entropy of the distribution; filter low-entropy evidence (≤ threshold).
  4. Generate professional (dict1) and amateur (dict2) answer distributions with filtered evidence.
  5. Apply contrastive decoding:
       score(answer) = (1 + entropy) × professional_freq - entropy × amateur_aligned_freq
     Pick the answer with the highest absolute score.

Reference:
  "Discerning and Resolving Knowledge Conflicts through Adaptive Decoding
   with Contextual Information-Entropy Constraint" (Wang et al., 2024)
"""

import math
from typing import Dict, List, Optional, Tuple, Callable

from openai import OpenAI


# ─── GPT-4o-mini NLI ─────────────────────────────────────────────────────────

_NLI_SYSTEM_PROMPT = """You are a semantic entailment judge. Given a question and two possible answers,
determine whether Answer 1 semantically entails Answer 2.

Entailment: Answer 1 logically implies Answer 2 (they have the same meaning or Answer 1 is more specific).
Contradiction: Answer 1 and Answer 2 are semantically opposite or incompatible.
Neutral: The answers are about the same topic but neither entails nor contradicts.

Respond with exactly one word: Entailment, Contradiction, or Neutral. No explanation."""


def _build_nli_prompt(question: str, answer1: str, answer2: str) -> str:
    return (
        f"Question: {question}\n"
        f"Answer 1: {answer1}\n"
        f"Answer 2: {answer2}\n\n"
        f"Is Answer 1 entailed by, contradictory to, or neutral to Answer 2?"
    )


def _call_gpt4o_mini_nli(
    client: OpenAI,
    model: str,
    question: str,
    answer1: str,
    answer2: str,
) -> str:
    """Call GPT-4o-mini for semantic entailment judgment. Returns 'Entailment', 'Contradiction', or 'Neutral'."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _NLI_SYSTEM_PROMPT},
                {"role": "user", "content": _build_nli_prompt(question, answer1, answer2)},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        result = response.choices[0].message.content.strip()
        # Normalise
        result_lower = result.lower()
        if "entail" in result_lower:
            return "Entailment"
        elif "contradict" in result_lower:
            return "Contradiction"
        elif "neutral" in result_lower:
            return "Neutral"
        return "Neutral"
    except Exception:
        return "Neutral"


# ─── Semantic Clustering ─────────────────────────────────────────────────────

def _semantic_clustering(
    client: OpenAI,
    model: str,
    question: str,
    answers: List[str],
) -> Dict[str, int]:
    """
    Cluster a list of generated answers by semantic equivalence using GPT-4o-mini NLI.

    For each answer after the first, check if it is entailed by any existing cluster
    representative (answer1 = existing, answer2 = new). If so, merge into that cluster;
    otherwise, create a new cluster.

    Args:
        client: OpenAI client instance.
        model: GPT-4o-mini model name (e.g., "gpt-4o-mini").
        question: The question text.
        answers: List of generated answer strings.

    Returns:
        dict mapping cluster-representative answer → frequency (count).
    """
    n = len(answers)
    if n == 0:
        return {}
    if n == 1:
        return {answers[0]: 1}

    parent = list(range(n))  # Union-Find parent pointers

    for i in range(n):
        for j in range(i + 1, n):
            label = _call_gpt4o_mini_nli(
                client, model, question, answers[i], answers[j],
            )
            if label == "Entailment":
                parent[j] = i  # answers[j] is entailed by answers[i] → merge j into i

    # Resolve clusters
    distribution: Dict[str, int] = {}
    for i in range(n):
        root = i
        while parent[root] != root:
            root = parent[root]
        rep = answers[root]
        distribution[rep] = distribution.get(rep, 0) + 1

    return distribution


# ─── Semantic Entropy ────────────────────────────────────────────────────────

def _semantic_entropy(distribution: Dict[str, int]) -> float:
    """
    Compute Shannon entropy (base 2) of an answer-frequency distribution.

    Args:
        distribution: {answer: frequency}

    Returns:
        Entropy value in bits. 0 if all answers are identical.
    """
    total = sum(distribution.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in distribution.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


# ─── SCD Core ────────────────────────────────────────────────────────────────

def scd_decode(
    client: OpenAI,
    model: str,
    question: str,
    professional_dist: Dict[str, int],
    amateur_dist: Dict[str, int],
) -> Tuple[str, Dict[str, float]]:
    """
    Semantic Contrastive Decoding: contrast professional vs amateur answer distributions.

    Algorithm:
      1. Compute semantic entropy 'a' of the professional distribution.
      2. Align amateur distribution (dict3):
         For each professional answer k1, find the first amateur answer k2 that is
         entailed by k1. Map dict3[k1] = amateur_freq(k2). If no entailment is found,
         amateur entries are mapped to themselves.
      3. For each answer key in the union of dict1 and dict3:
           score(key) = (1 + a) × professional_freq(key) - a × aligned_amateur_freq(key)
      4. Return the answer with the highest absolute score.

    Intuition:
      If both professional and amateur models produce the same answer, that answer is
      likely from parametric knowledge (not from evidence). Subtracting the amateur
      signal downweights such answers, amplifying evidence-driven answers.

    Args:
        client: OpenAI client for NLI calls.
        model: GPT-4o-mini model name.
        question: The question text.
        professional_dist: Professional-mode answer distribution {answer: freq}.
        amateur_dist: Amateur-mode answer distribution {answer: freq}.

    Returns:
        (best_answer, all_scores) where best_answer is the answer with highest |score|
        and all_scores maps every answer to its SCD score.
    """
    a = _semantic_entropy(professional_dist)

    keys1 = list(professional_dist.keys())
    keys2 = list(amateur_dist.keys())

    # Align amateur answers to professional answers via entailment
    dict3: Dict[str, int] = {}
    for k1 in keys1:
        matched = False
        for k2 in keys2:
            label = _call_gpt4o_mini_nli(client, model, question, k1, k2)
            if label == "Entailment":
                dict3[k1] = amateur_dist[k2]
                matched = True
                break
            dict3[k2] = amateur_dist[k2]
        if not matched:
            # Ensure all amateur keys exist in dict3
            for k2 in keys2:
                dict3[k2] = amateur_dist[k2]

    # Compute SCD scores
    all_keys = set(professional_dist) | set(dict3)
    result: Dict[str, float] = {}
    for key in all_keys:
        val1 = professional_dist.get(key, 0)
        val2 = dict3.get(key, 0)
        result[key] = (1.0 + a) * val1 - a * val2

    best_answer = max(result, key=lambda k: abs(result[k]))
    return best_answer, result


# ─── High-Level API ──────────────────────────────────────────────────────────

class SCD:
    """
    Semantic Contrastive Decoding with GPT-4o-mini for semantic equivalence judgment.

    Usage:
        def my_llm(prompt: str, temperature: float) -> str:
            # Call your LLM (OpenAI, vLLM, local model, etc.)
            ...

        scd = SCD(
            openai_api_key="sk-...",
            openai_base_url="https://api.openai.com/v1",
            nli_model="gpt-4o-mini",
        )

        result = scd.decode(
            question="What is the capital of France?",
            evidence_docs=["Paris is the capital of France.", "Lyon is a city."],
            llm_func=my_llm,
            num_samples=5,
            entropy_threshold=1.5,
        )
        print(result["answer"])  # "Paris"
    """

    _DEFAULT_NLI_MODEL = "gpt-4o-mini"
    _PROFESSIONAL_PROMPT = (
        "You are a professional question-answering model that can determine "
        "whether the provided evidence is correct and give the correct answer "
        "to the question. This evidence includes memory that is related to "
        "the answer, irrelevant or incorrect.\n"
        "Evidence:\n{evidence}\n"
        "Question: {question}\n"
        "Answer:"
    )
    _AMATEUR_PROMPT = (
        "You are an amateur in question answering. "
        "Please briefly answer the following question incorrectly.\n"
        "Evidence:\n{evidence}\n"
        "Question: {question}\n"
        "Answer:"
    )
    _NO_EVIDENCE_PROMPT = (
        "You are a professional question-answering model.\n"
        "You must Answer the following question in a single brief but complete sentence.\n"
        "Question: {question}\n"
        "Answer:"
    )

    def __init__(
        self,
        openai_api_key: str,
        openai_base_url: Optional[str] = None,
        nli_model: str = _DEFAULT_NLI_MODEL,
    ):
        """
        Args:
            openai_api_key: OpenAI API key for GPT-4o-mini NLI calls.
            openai_base_url: Custom base URL (for proxies / Azure).
            nli_model: Model name for NLI (default: "gpt-4o-mini").
        """
        if openai_base_url:
            self._client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_base_url,
            )
        else:
            self._client = OpenAI(api_key=openai_api_key)
        self._nli_model = nli_model

    # ── Public API ───────────────────────────────────────────────────────

    def decode(
        self,
        question: str,
        evidence_docs: List[str],
        llm_func: Callable[[str, float], str],
        num_samples: int = 5,
        entropy_threshold: float = 1.5,
        professional_temperature: float = 0.7,
        amateur_temperature: float = 0.9,
    ) -> Dict:
        """
        Run SCD decoding for a single question with evidence documents.

        Args:
            question: The question text.
            evidence_docs: List of evidence document strings.
            llm_func: Callable(prompt, temperature) → generated_text.
                      This is the main generation model (e.g., Llama, Mistral).
            num_samples: Number of samples to generate per evidence source (default: 5).
            entropy_threshold: Maximum semantic entropy for an evidence source to be
                               considered reliable (default: 1.5).
            professional_temperature: Temperature for professional-mode generation.
            amateur_temperature: Temperature for amateur-mode generation.

        Returns:
            dict with keys:
                - "answer": The best decoded answer (str).
                - "score": SCD score of the best answer (float).
                - "all_scores": {answer: score} for all candidates.
                - "professional_dist": Professional answer distribution.
                - "amateur_dist": Amateur answer distribution.
                - "filtered_evidence": Indices of evidence sources that passed the entropy filter.
                - "entropy_per_evidence": {index: entropy} for each evidence source.
        """
        # ── Step 1: Score each evidence source by semantic entropy ──
        entropy_per_evidence: Dict[int, float] = {}

        for idx, doc in enumerate(evidence_docs):
            prompt = self._PROFESSIONAL_PROMPT.format(
                evidence=doc, question=question,
            )
            answers = [llm_func(prompt, professional_temperature) for _ in range(num_samples)]
            dist = _semantic_clustering(
                self._client, self._nli_model, question, answers,
            )
            entropy_per_evidence[idx] = _semantic_entropy(dist)

        # Filter: keep only evidence with entropy ≤ threshold
        filtered_indices = [
            i for i, se in entropy_per_evidence.items() if se <= entropy_threshold
        ]

        # ── Step 2: No reliable evidence → fallback to no-context generation ──
        if not filtered_indices:
            prompt = self._NO_EVIDENCE_PROMPT.format(question=question)
            answer = llm_func(prompt, professional_temperature)
            return {
                "answer": answer,
                "score": 0.0,
                "all_scores": {answer: 0.0},
                "professional_dist": {},
                "amateur_dist": {},
                "filtered_evidence": [],
                "entropy_per_evidence": entropy_per_evidence,
            }

        # ── Step 3: Build evidence string from filtered sources ──
        combined_evidence = "\n".join(
            f"{i + 1}. {evidence_docs[i]}" for i in filtered_indices
        )

        # ── Step 4: Professional-mode distribution ──
        prof_prompt = self._PROFESSIONAL_PROMPT.format(
            evidence=combined_evidence, question=question,
        )
        prof_answers = [
            llm_func(prof_prompt, professional_temperature)
            for _ in range(num_samples)
        ]
        professional_dist = _semantic_clustering(
            self._client, self._nli_model, question, prof_answers,
        )

        # ── Step 5: Amateur-mode distribution ──
        amat_prompt = self._AMATEUR_PROMPT.format(
            evidence=combined_evidence, question=question,
        )
        amat_answers = [
            llm_func(amat_prompt, amateur_temperature)
            for _ in range(num_samples)
        ]
        amateur_dist = _semantic_clustering(
            self._client, self._nli_model, question, amat_answers,
        )

        # ── Step 6: SCD contrastive decoding ──
        best_answer, all_scores = scd_decode(
            self._client,
            self._nli_model,
            question,
            professional_dist,
            amateur_dist,
        )

        return {
            "answer": best_answer,
            "score": all_scores[best_answer],
            "all_scores": all_scores,
            "professional_dist": professional_dist,
            "amateur_dist": amateur_dist,
            "filtered_evidence": filtered_indices,
            "entropy_per_evidence": entropy_per_evidence,
        }
