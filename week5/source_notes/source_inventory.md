# Week 5 source inventory

The user supplied the following Chapter 5 files in the course workspace on 22 September 2026. They contain the same chapter in two formats and are not copied into this public teaching repository.

| Source | Format | Coverage and reliability | Use in this module |
|---|---|---|---|
| `HI634_Chapter_5_Natural_Language_Processing_in_Health_Services.docx` | Microsoft Word | Primary course chapter; 17,437 words after local text extraction | Main source for concepts, examples, terminology, learning outcomes, cautions, and references |
| `HI634_Chapter_5_Natural_Language_Processing_in_Health_Services.pdf` | PDF, 48 pages | Rendered companion to the DOCX; metadata shows creation on 22 September 2026 | Cross-check of chapter identity, length, and publication-style structure |
| `~$634_Chapter_5_Natural_Language_Processing_in_Health_Services.docx` | Office lock file | Temporary editor artifact; not course content | Excluded |

## Extracted chapter structure

The lesson draws especially from these chapter sections:

- Sections 2–3: language sources, units of analysis, terminology, tasks, and intended use.
- Section 4: cleaning, segmentation, tokenization, normalization, counts, one-hot vectors, Bag-of-Words, n-grams, TF-IDF, cosine similarity, static embeddings, and contextual embeddings.
- Section 5: regular expressions, dictionaries, entity spans, concept normalization, assertion, experiencer, temporality, sections, and hybrid extraction.
- Sections 6–7: leakage-safe text classification, baselines, precision, recall, F1, span evaluation, error review, and reviewer burden.
- Sections 8–9: retrieval, extractive summarization, and the progressive synthetic social-needs case.
- Sections 10–12: PHI, privacy, secure processing, language access, documentation bias, workflow integration, and the NLP Pipeline Card.
- Sections 13–17: future directions, key points, review questions, and hands-on exercises.

## Data and implementation decisions

- The UCI Drug Reviews (Druglib.com) corpus is discussed in the chapter, but it is not redistributed here. The chapter notes both a CC BY 4.0 repository label and older donor restrictions that require clarification.
- Every text record committed for Week 5 is explicitly synthetic and generated locally. It contains no patient data or protected health information.
- The chapter demonstrates Word2Vec with Gensim. Gensim 4.4.0 does not currently build under the repository's Python 3.14 environment, so the notebook demonstrates the same static dense-vector ideas with a deterministic co-occurrence matrix and truncated SVD. It does not claim that SVD is Word2Vec.
- No pretrained model or network call is required. spaCy uses a blank English tokenizer and rule components; NLTK supplies the Porter stemmer.

## Citation candidates retained from the chapter

- UCI Machine Learning Repository, Drug Reviews (Druglib.com), ID 461, DOI `10.24432/C55G6J`.
- Mikolov et al. (2013), word representations, DOI `10.48550/arXiv.1301.3781`.
- Devlin et al. (2019), BERT, DOI `10.18653/v1/N19-1423`.
- Chapman et al. (2001), NegEx, DOI `10.1006/jbin.2001.1029`.
- Harkema et al. (2009), ConText, DOI `10.1016/j.jbi.2009.05.002`.
- HHS guidance on HIPAA de-identification, minimum necessary, and the Security Rule.
- NIST AI Risk Management Framework 1.0, DOI `10.6028/NIST.AI.100-1`.
