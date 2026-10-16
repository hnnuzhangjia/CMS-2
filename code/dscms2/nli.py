from openai import OpenAI


_NLI_PROMPT = (
    "We are evaluating answers to the question: {question}. Here are two possible answers:\n"
    "Possible Answer 1: {text1}\n"
    "Possible Answer 2: {text2}\n"
    "Does Possible Answer 1 semantically entail Possible Answer 2? "
    "Your response must be entailment, contradiction, or neutral.\n"
    "Answer:"
)


def _build_nli_prompt(question: str, answer1: str, answer2: str) -> str:
    return _NLI_PROMPT.format(
        question=question,
        text1=answer1,
        text2=answer2,
    )


def _call_gpt4o_mini_nli(
    client: OpenAI,
    model: str,
    question: str,
    answer1: str,
    answer2: str,
) -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": _build_nli_prompt(question, answer1, answer2)},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        result = (response.choices[0].message.content or "").strip().rstrip(".").casefold()
        return {
            "entailment": "Entailment",
            "contradiction": "Contradiction",
            "neutral": "Neutral",
        }.get(result, "Neutral")
    except Exception:
        return "Neutral"

'''Clarification on Appendix A.3: The semantic equivalence prompt compares two individual answers, rather than an answer against a list. Equivalence requires entailment in both directions, as described in Appendix B and implemented in the code.'''

