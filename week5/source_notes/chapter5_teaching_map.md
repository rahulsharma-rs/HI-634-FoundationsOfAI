# Chapter 5 teaching map

| Chapter concepts | Notebook treatment | Observable learner evidence |
|---|---|---|
| Text source, purpose, and unit | Define one synthetic note as the document unit and one current documented need as the binary reference | Learner can separate notes, people, mentions, and labels |
| Cleaning, sentences, and tokens | Preserve raw text, normalize whitespace, inspect sentence and token offsets | Learner identifies which transformations change character positions or meaning |
| Vocabulary and vectorization | Build one-hot, Bag-of-Words, n-gram, and TF-IDF representations | Learner reads vocabulary-aligned matrices and states what each representation loses |
| Similarity and embeddings | Demonstrate a negation failure in TF-IDF cosine similarity and build compact co-occurrence/SVD vectors | Learner distinguishes representation similarity from factual agreement |
| Rules and extraction | Apply regex, EntityRuler patterns, educational normalization, and a deliberately brittle assertion rule | Learner preserves spans and separates entity detection from context |
| Classification | Compare a most-frequent baseline with a leakage-safe TF-IDF/logistic pipeline | Learner explains why the vectorizer must be fitted inside the training pipeline |
| Evaluation | Compute confusion counts, precision, recall, F1, exact-span metrics, and precision at k | Learner connects errors to reviewer workload and missed evidence |
| Retrieval and summarization | Rank synthetic notes and expose an extractive summary's omission | Learner defines relevance and checks source completeness |
| Privacy, fairness, and security | Demonstrate incomplete regex redaction and complete an NLP Pipeline Card | Learner identifies residual PHI risk, language gaps, ownership, monitoring, and rollback |

## Pedagogical sequence

`purpose → source and unit → raw text → tokens → vocabulary → sparse vectors → dense representation → rules/models → task-specific evaluation → human review and safeguards`

The notebook includes recall, application, analysis, and design prompts. All numerical results are teaching-corpus results and are explicitly separated from claims of clinical validity.
