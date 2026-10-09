"""Semantic entailment judgments through an OpenAI-compatible API."""

from openai import OpenAI


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
        f"Does Answer 1 entail Answer 2? Respond with Entailment, Contradiction, or Neutral."
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
