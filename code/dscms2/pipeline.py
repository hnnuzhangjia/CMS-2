"""Two-stage evidence filtering and semantic decoding pipeline."""

from typing import Callable, Dict, List, Optional
from openai import OpenAI

from .clustering import _joint_clustering, _semantic_clustering
from .decoding import _score_clusters, _semantic_entropy


class SCD:
    """Semantic Contrastive Decoding with GPT-4o-mini for semantic equivalence judgment."""

    _DEFAULT_NLI_MODEL = "gpt-4o-mini"
    _PROFESSIONAL_PROMPT = (
        "You are a professional question-answering model that can determine "
        "whether the provided evidence is correct and give the correct answer "
        "to the question. This evidence includes memory that is related to "
        "the answer, irrelevant or incorrect.\n"
        "Evidence:\n{evidence}\n"
        "You must answer the following question in a single brief but complete sentence.\n"
        "Question: {question}\n"
        "Answer:"
    )
    _AMATEUR_PROMPT = (
        "You are an amateur in question answering. "
        "You must answer the following question in a single brief but complete sentence.\n"
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
        if openai_base_url:
            self._client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_base_url,
            )
        else:
            self._client = OpenAI(api_key=openai_api_key)
        self._nli_model = nli_model


    def decode(
        self,
        question: str,
        evidence_docs: List[str],
        llm_func: Callable[[str, float], str],
        num_samples: int = 10,
        entropy_threshold: float = 1.0,
        professional_temperature: float = 0.9,
        amateur_temperature: float = 0.9,
    ) -> Dict:
        """Run SCD decoding for a single question with evidence documents."""
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or num_samples < 1:
            raise ValueError("num_samples must be a positive integer")

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

        filtered_indices = [
            i for i, se in entropy_per_evidence.items() if se <= entropy_threshold
        ]

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

        combined_evidence = "\n".join(
            f"{i + 1}. {evidence_docs[i]}" for i in filtered_indices
        )

        prof_prompt = self._PROFESSIONAL_PROMPT.format(
            evidence=combined_evidence, question=question,
        )
        prof_answers = [
            llm_func(prof_prompt, professional_temperature)
            for _ in range(num_samples)
        ]

        amat_prompt = self._AMATEUR_PROMPT.format(question=question)
        amat_answers = [
            llm_func(amat_prompt, amateur_temperature)
            for _ in range(num_samples)
        ]
        professional_dist, amateur_dist = _joint_clustering(
            self._client, self._nli_model, question,
            ((answer, 1) for answer in prof_answers),
            ((answer, 1) for answer in amat_answers),
        )
        best_answer, all_scores = _score_clusters(professional_dist, amateur_dist)

        return {
            "answer": best_answer,
            "score": all_scores[best_answer],
            "all_scores": all_scores,
            "professional_dist": professional_dist,
            "amateur_dist": amateur_dist,
            "filtered_evidence": filtered_indices,
            "entropy_per_evidence": entropy_per_evidence,
        }
