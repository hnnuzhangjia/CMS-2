# DS-CMS²: Dual-Stage Confidence Measurement in Semantic Space

Official implementation of **DS-CMS²**, a training-free framework for mitigating knowledge conflicts in retrieval-augmented generation (RAG).

> **[Mitigating Knowledge Conflicts of Retrieval-Augmented Generation through Dual-Stage Confidence Measurement in Semantic Space](./Mitigating_Knowledge_Conflicts_of_Retrieval-Augmented_Generation_through_Dual-Stage_Confidence_Measurement_in_Semantic_Space.pdf)**  
> Jia Zhang, Zhiheng Zhang, Zeao Ji, Tengfei Ma, and Daojian Zeng. COLM 2026.

## Overview

RAG can be harmed by two distinct forms of conflict:

- **Inter-context conflict:** retrieved documents disagree with one another or contain irrelevant/misleading evidence.
- **Context-memory conflict:** retrieved evidence disagrees with the language model's parametric knowledge.

DS-CMS² addresses both without additional training by measuring confidence in **semantic space** at two stages:

1. **Pre-inference knowledge filtering.** For each evidence document, the model samples multiple answers, clusters semantically equivalent answers, and computes semantic entropy. Documents whose entropy exceeds a threshold are filtered out; if none remain, the method falls back to closed-book generation.
2. **Post-inference semantic contrastive decoding (SCD).** The model generates answer distributions in professional (context-aware) and amateur modes, aligns their semantic classes, and uses the expert semantic entropy as a dynamic contrast strength. This emphasizes answers supported by reliable context while avoiding fixed-strength overcorrection.

## Installation

~~~bash
git clone https://github.com/hnnuzhangjia/CMS-2.git
cd CMS-2
pip install -r requirements.txt
~~~

Requirements: Python 3.8+ and an OpenAI-compatible API endpoint for semantic entailment judgments (GPT-4o-mini by default).

## Quick start

~~~python
from scd import SCD
from openai import OpenAI

# Your generator can use OpenAI, vLLM, or any compatible backend.
generator = OpenAI()

def llm_func(prompt: str, temperature: float) -> str:
    response = generator.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()

scd = SCD(
    openai_api_key="YOUR_API_KEY",
    nli_model="gpt-4o-mini",
)

result = scd.decode(
    question="Who was the first US-born winner of golf's British Open?",
    evidence_docs=[
        "Walter Charles Hagen became the first American-born golfer to win the British Open in 1922.",
        "The first US-born winner of the British Open was Bobby Jones, who won in 1930.",
    ],
    llm_func=llm_func,
    num_samples=5,
    entropy_threshold=1.5,
)

print(result["answer"])
print(result["filtered_evidence"])
print(result["entropy_per_evidence"])
~~~

## Output

`SCD.decode()` returns a dictionary containing:

- `answer`: selected final answer;
- `score` and `all_scores`: semantic contrastive-decoding scores;
- `professional_dist` and `amateur_dist`: semantic answer-cluster distributions;
- `filtered_evidence`: indices of evidence documents retained by pre-inference filtering;
- `entropy_per_evidence`: semantic entropy for every evidence document.

## Implementation notes

- `scd.py` provides the high-level `SCD` interface and the lower-level `scd_decode` function.
- Semantic clustering currently uses GPT-4o-mini as an NLI judge. The paper also evaluates a local DeBERTa-v3 NLI alternative, so this component can be replaced for fully local deployment.
- The implementation estimates semantic entropy from sampled answer-cluster frequencies.

## Citation

~~~bibtex
@inproceedings{zhang2026dscms2,
  title={Mitigating Knowledge Conflicts of Retrieval-Augmented Generation through Dual-Stage Confidence Measurement in Semantic Space},
  author={Zhang, Jia and Zhang, Zhiheng and Ji, Zeao and Ma, Tengfei and Zeng, Daojian},
  booktitle={Conference on Language Modeling (COLM)},
  year={2026}
}
~~~

## License

No license file is currently included. Please contact the authors before using this code beyond the repository's intended research purposes.

