"""Build the Week 5 NLP classroom notebook from version-controlled cell sources."""

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


OUTPUT = Path(__file__).resolve().parents[1] / "notebooks" / "week5_natural_language_processing_health_services.ipynb"


def md(cell_id: str, source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip(), id=cell_id)


def code(cell_id: str, source: str):
    return nbf.v4.new_code_cell(dedent(source).strip(), id=cell_id)


cells = [
    md(
        "w5-title",
        """
        # Week 5 Natural Language Processing in Health Services

        ## From documented language to accountable reviewer support

        This guided lab follows Chapter 5's central sequence:

        `raw text → sentences and tokens → vocabulary → numerical representation → rules or models → task-specific evaluation → human interpretation`

        The code makes intermediate representations visible and deliberately exposes failures involving negation, experiencer, historical context, language coverage, retrieval, and omission.

        > **Educational boundary:** Every note and label used here is synthetic. Nothing in this notebook is clinically validated. Do not use its outputs to diagnose, determine eligibility, contact a patient, deny a service, or replace authorized review.

        - **Document unit:** one synthetic care-management note
        - **Reference-positive meaning:** an explicitly documented current social need of the patient
        - **Reproducibility seed:** 42
        """,
    ),
    md(
        "w5-objectives",
        """
        ## Learning objectives

        By the end of the lesson, learners will be able to:

        1. **Trace** raw text through cleaning, sentence segmentation, tokenization, vocabulary construction, and numerical vectors.
        2. **Compare** one-hot, Bag-of-Words, n-gram, TF-IDF, and dense static representations by what they retain and discard.
        3. **Implement** transparent regex and dictionary extraction while preserving source spans and contextual uncertainty.
        4. **Build** a leakage-safe classical text-classification pipeline and compare it with a simple baseline.
        5. **Evaluate** note labels, entity spans, retrieval results, and extractive summaries with task-appropriate evidence.
        6. **Critique** an NLP workflow for privacy, documentation bias, language access, security, human review, and rollback.

        ### Lesson pathway

        Work from representation mechanics to a progressive social-needs case, then finish with an NLP Pipeline Card. Predict each output before running its cell and explain any result that appears counterintuitive.
        """,
    ),
    md(
        "w5-purpose",
        """
        # 1 Define the purpose and the unit before choosing a method

        The proposed workflow is intentionally limited: locate candidate evidence for an authorized care-management reviewer, preserve the source wording and context, and allow the reviewer to confirm, reject, clarify, or escalate it.

        A **note**, a **mention**, and a **person** are not interchangeable units. Ten matching notes can describe one person or copied history. A detected term is not automatically affirmed, current, or about the patient.

        **Think–pair–share:** For “find transportation barriers,” specify the document unit, positive definition, reviewer, permitted action, abstention path, and unacceptable false-positive and false-negative outcomes.
        """,
    ),
    code(
        "w5-setup",
        """
        from collections import Counter
        from importlib.metadata import version
        from pathlib import Path
        import hashlib
        import html
        import platform
        import re

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        import sklearn
        import spacy

        from IPython.display import display
        from nltk.stem import PorterStemmer
        from spacy.lang.en.stop_words import STOP_WORDS
        from spacy.lookups import Lookups
        from sklearn.decomposition import TruncatedSVD
        from sklearn.dummy import DummyClassifier
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            precision_recall_fscore_support,
            precision_score,
            recall_score,
        )
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import normalize

        RANDOM_STATE = 42
        np.random.seed(RANDOM_STATE)
        pd.set_option("display.max_colwidth", 120)
        sns.set_theme(style="whitegrid", context="notebook")


        def find_repo_root() -> Path:
            current = Path.cwd().resolve()
            for candidate in (current, *current.parents):
                if (candidate / "week5" / "data").is_dir():
                    return candidate
            raise FileNotFoundError("Could not find the repository's week5/data directory.")


        REPO_ROOT = find_repo_root()
        DATA_DIR = REPO_ROOT / "week5" / "data"

        assert platform.python_version_tuple()[:2] == ("3", "14")
        print(f"Python: {platform.python_version()}")
        for package in ["numpy", "pandas", "scikit-learn", "nltk", "spacy"]:
            print(f"{package}: {version(package)}")
        print(f"Data folder: {DATA_DIR}")
        """,
    ),
    md(
        "w5-data-provenance",
        """
        # 2 Data provenance and documentary states

        Three local CSV files support the lesson:

        - 12 synthetic care-management notes with authored reference labels and distinct documentary states;
        - 32 separately worded synthetic training examples; and
        - 10 stress cases covering negation, uncertainty, history, experiencer, templates, and Spanish text.

        The public UCI Drug Reviews corpus discussed in Chapter 5 is **not redistributed or downloaded** here because its listed access terms require clarification. The required notebook is fully offline and makes no performance claim about that corpus.

        A reference value of 0 does not mean “the person has no need.” It can mean stable, negated, uncertain, historical, another experiencer, unassessed, or unrelated. Preserve those differences.
        """,
    ),
    code(
        "w5-load-data",
        """
        EXPECTED_HASHES = {
            "synthetic_care_management_notes.csv": "f17edcadd01231334e84306d64439bca4f0d553e9b49fd629cad7a2a453d9a0c",
            "synthetic_social_needs_training.csv": "73454d7c90a04aeb6ecd186f85dab7bd31b49f4af6aea8834dd039d571b1939b",
            "synthetic_language_challenge.csv": "9d0696afaca45b05445af0e4cb664ea77179b5db010b85d3d4e7c432d2636d8f",
        }


        def sha256_file(path: Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest()


        for filename, expected in EXPECTED_HASHES.items():
            observed = sha256_file(DATA_DIR / filename)
            assert observed == expected, f"Unexpected data snapshot: {filename}"

        notes = pd.read_csv(DATA_DIR / "synthetic_care_management_notes.csv")
        training = pd.read_csv(DATA_DIR / "synthetic_social_needs_training.csv")
        challenge = pd.read_csv(DATA_DIR / "synthetic_language_challenge.csv")

        assert notes.shape == (12, 5)
        assert training.shape == (32, 4)
        assert challenge.shape == (10, 5)
        assert not notes["text"].duplicated().any()
        assert notes["text"].str.strip().ne("").all()

        display(notes)
        display(notes.groupby("documentation_status", sort=False).size().rename("notes").to_frame())
        """,
    ),
    md(
        "w5-cleaning",
        """
        # 3 Preserve raw text before conservative cleaning

        Cleaning should repair an identified technical issue. It should not indiscriminately erase punctuation, negation, sections, doses, or offsets. Keep the original string beside any transformed version so evidence can be traced back to what was documented.
        """,
    ),
    code(
        "w5-clean-code",
        """
        def clean_text(text: str) -> str:
            decoded = html.unescape(str(text))
            return re.sub(r"\\s+", " ", decoded).strip()


        raw_text = "  No chest pain.\\nPt cannot afford medication.  "
        cleaned_text = clean_text(raw_text)
        notes["clean_text"] = notes["text"].map(clean_text)

        assert raw_text != cleaned_text
        assert "No chest pain" in raw_text and "No chest pain" in cleaned_text
        print("Raw representation:", repr(raw_text))
        print("Clean representation:", repr(cleaned_text))
        """,
    ),
    md(
        "w5-tokenization",
        """
        # 4 Sentence segmentation and tokenization

        A computer begins with characters. A tokenizer chooses units; a sentencizer chooses boundaries. Abbreviations, contractions, dates, headings, doses, and punctuation can make those choices consequential. We use a blank spaCy English pipeline, so no pretrained model or external download is involved.
        """,
    ),
    code(
        "w5-token-code",
        """
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")

        sample = "No chest pain. Pt can't attend at 9:30."
        doc = nlp(sample)
        sentence_rows = [
            {"sentence": sentence.text, "start": sentence.start_char, "end": sentence.end_char}
            for sentence in doc.sents
        ]
        token_rows = [
            {"token": token.text, "start": token.idx, "is_punct": token.is_punct, "is_alpha": token.is_alpha}
            for token in doc
        ]

        assert all(
            sample[row["start"]:row["end"]] == row["sentence"]
            for row in sentence_rows
        )
        display(pd.DataFrame(sentence_rows))
        display(pd.DataFrame(token_rows))
        """,
    ),
    md(
        "w5-normalization",
        """
        # 5 Normalization without destroying meaning

        Lowercasing, stop-word removal, stemming, and lemmatization are choices—not a mandatory checklist. A generic stop-word list can remove “no” and make a denial resemble an affirmation. A stem is not necessarily a valid word, and a tiny lookup table is not a clinical lemmatizer.
        """,
    ),
    code(
        "w5-normalization-code",
        """
        stop_sample = "No chest pain and no transportation problems."
        words = [token.lower_ for token in nlp(stop_sample) if token.is_alpha]
        removed_all = [word for word in words if word not in STOP_WORDS]
        retain = {"no", "not", "never", "without", "cannot", "denies"}
        custom_stop_words = set(STOP_WORDS) - retain
        removed_selectively = [word for word in words if word not in custom_stop_words]

        stemmer = PorterStemmer()
        lemma_nlp = spacy.blank("en")
        lemmatizer = lemma_nlp.add_pipe("lemmatizer", config={"mode": "lookup"})
        lookups = Lookups()
        lookups.add_table(
            "lemma_lookup",
            {"patients": "patient", "medications": "medication", "was": "be", "running": "run"},
        )
        lemmatizer.initialize(lookups=lookups)
        morphology = pd.DataFrame(
            [
                {"surface": token.text, "stem": stemmer.stem(token.text), "supplied_lemma": token.lemma_}
                for token in lemma_nlp("patients medications was running")
            ]
        )

        assert "no" not in removed_all and "no" in removed_selectively
        print("Original tokens:", words)
        print("Generic stop-word removal:", removed_all)
        print("Negation-preserving example:", removed_selectively)
        display(morphology)
        """,
    ),
    md(
        "w5-vectorization",
        """
        # 6 From vocabulary positions to sparse lexical vectors

        A vocabulary maps recognized terms to stable column positions. Those positions are addresses, not ordered clinical values. One-hot encoding marks one term; Bag-of-Words counts terms across documents; n-grams retain short local sequences; TF-IDF reweights counts using corpus distribution.
        """,
    ),
    code(
        "w5-counts-onehot",
        """
        word_counts = Counter()
        for parsed in nlp.pipe(notes["clean_text"]):
            word_counts.update(token.lower_ for token in parsed if token.is_alpha)

        vocabulary = {"transportation": 0, "clinic": 1, "pain": 2, "medication": 3}
        one_hot = np.eye(len(vocabulary), dtype=int)
        one_hot_table = pd.DataFrame(one_hot, index=vocabulary, columns=vocabulary)

        assert one_hot_table.loc["medication"].tolist() == [0, 0, 0, 1]
        assert "housing" not in vocabulary
        print("Most frequent visible tokens:", word_counts.most_common(10))
        display(one_hot_table)
        """,
    ),
    code(
        "w5-bow",
        """
        tiny_corpus = [
            "patient needs transportation",
            "patient needs medication",
            "patient reports transportation problems",
        ]
        count_vectorizer = CountVectorizer()
        count_matrix = count_vectorizer.fit_transform(tiny_corpus)
        bow_table = pd.DataFrame(
            count_matrix.toarray(), columns=count_vectorizer.get_feature_names_out()
        )

        assert count_matrix.shape == (3, 6)
        assert count_matrix.nnz == 10
        print("Sparse matrix shape:", count_matrix.shape)
        print("Stored nonzero values:", count_matrix.nnz)
        display(bow_table)
        """,
    ),
    code(
        "w5-ngrams",
        """
        phrase_corpus = ["chest pain", "no chest pain", "patient denies chest pain"]
        ngram_rows = []
        vocabularies = {}
        for maximum_n in [1, 2, 3]:
            vectorizer = CountVectorizer(ngram_range=(1, maximum_n))
            matrix = vectorizer.fit_transform(phrase_corpus)
            vocabulary_terms = vectorizer.get_feature_names_out().tolist()
            vocabularies[maximum_n] = vocabulary_terms
            ngram_rows.append(
                {"maximum_n": maximum_n, "features": len(vocabulary_terms), "vocabulary": vocabulary_terms}
            )

        assert "no chest pain" not in vocabularies[2]
        assert "no chest pain" in vocabularies[3]
        display(pd.DataFrame(ngram_rows))
        """,
    ),
    code(
        "w5-tfidf",
        """
        raw_tfidf = TfidfVectorizer(norm=None)
        weighted = raw_tfidf.fit_transform(tiny_corpus)
        unit_tfidf = TfidfVectorizer()
        unit_matrix = unit_tfidf.fit_transform(tiny_corpus)

        idf_table = pd.DataFrame(
            {"term": raw_tfidf.get_feature_names_out(), "idf": raw_tfidf.idf_}
        )
        normalized_table = pd.DataFrame(
            unit_matrix.toarray(), columns=unit_tfidf.get_feature_names_out()
        )

        row_norms = np.sqrt((unit_matrix.multiply(unit_matrix)).sum(axis=1)).A1
        assert np.allclose(row_norms, 1.0)
        display(idf_table.round(3))
        display(normalized_table.round(3))
        print("Row L2 norms:", row_norms.round(3))
        """,
    ),
    md(
        "w5-similarity",
        """
        # 7 Similarity is not agreement

        Cosine similarity compares directions in one shared vector space. It does not prove entailment, factual agreement, or an affirmed need. Lexical overlap can make opposite statements look more similar than paraphrases.
        """,
    ),
    code(
        "w5-cosine",
        """
        comparison_texts = [
            "Patient cannot afford medication.",
            "Medication cost is a financial burden.",
            "Patient can afford medication.",
        ]
        similarity_vectorizer = TfidfVectorizer()
        similarity_vectors = similarity_vectorizer.fit_transform(comparison_texts)
        similarity = cosine_similarity(similarity_vectors)
        similarity_table = pd.DataFrame(
            similarity,
            index=["cannot_afford", "financial_burden", "can_afford"],
            columns=["cannot_afford", "financial_burden", "can_afford"],
        )

        assert similarity[0, 2] > similarity[0, 1]
        display(similarity_table.round(3))
        print("The opposite-polarity sentence ranks as more similar because it shares more words.")
        """,
    ),
    md(
        "w5-embeddings",
        """
        # 8 Static dense representations and their limits

        Word2Vec learns compact static word vectors from nearby-word prediction. Chapter 5 demonstrates that algorithm with Gensim. Gensim 4.4.0 does not build under this repository's Python 3.14 environment, so the next cell uses a transparent alternative: word-window co-occurrence, positive pointwise mutual information, and truncated SVD.

        This is **not Word2Vec**. It does demonstrate the relevant representation concepts: dense learned coordinates, context-based association, one static vector per word, document averaging, out-of-vocabulary coverage, and the lack of built-in negation or causal meaning. Contextual transformer embeddings are a later bridge and still require task-specific validation.
        """,
    ),
    code(
        "w5-dense-embedding",
        """
        embedding_texts = pd.concat([notes["text"], training["text"]], ignore_index=True).tolist()
        token_sequences = [
            [token.lower_ for token in parsed if token.is_alpha]
            for parsed in nlp.pipe(embedding_texts)
        ]
        embedding_counts = Counter(token for sequence in token_sequences for token in sequence)
        embedding_terms = sorted(term for term, count in embedding_counts.items() if count >= 2)
        term_to_index = {term: index for index, term in enumerate(embedding_terms)}

        cooccurrence = np.zeros((len(embedding_terms), len(embedding_terms)), dtype=float)
        window = 2
        for sequence in token_sequences:
            known = [term for term in sequence if term in term_to_index]
            for center_index, center in enumerate(known):
                left = max(0, center_index - window)
                right = min(len(known), center_index + window + 1)
                for context_index in range(left, right):
                    if context_index != center_index:
                        cooccurrence[term_to_index[center], term_to_index[known[context_index]]] += 1

        total = cooccurrence.sum()
        expected = np.outer(cooccurrence.sum(axis=1), cooccurrence.sum(axis=0)) / total
        ppmi = np.zeros_like(cooccurrence)
        observed = cooccurrence > 0
        ppmi[observed] = np.maximum(np.log(cooccurrence[observed] / expected[observed]), 0)

        dimensions = min(8, len(embedding_terms) - 1)
        svd = TruncatedSVD(n_components=dimensions, random_state=RANDOM_STATE)
        word_embeddings = normalize(svd.fit_transform(ppmi))


        def nearest_terms(term: str, count: int = 5) -> pd.DataFrame:
            index = term_to_index[term]
            scores = word_embeddings @ word_embeddings[index]
            order = np.argsort(-scores, kind="stable")
            rows = [
                {"term": embedding_terms[i], "cosine_similarity": scores[i]}
                for i in order
                if i != index
            ][:count]
            return pd.DataFrame(rows)


        def document_embedding(text: str):
            terms = [token.lower_ for token in nlp(text) if token.is_alpha]
            indices = [term_to_index[term] for term in terms if term in term_to_index]
            coverage = len(indices) / max(len(terms), 1)
            if not indices:
                return None, coverage
            return word_embeddings[indices].mean(axis=0), coverage


        assert word_embeddings.shape == (len(embedding_terms), dimensions)
        assert document_embedding("xyzabc qwerty")[0] is None
        print("Dense embedding matrix:", word_embeddings.shape)
        display(nearest_terms("transportation"))
        vector, coverage = document_embedding("Patient cannot afford medication.")
        print("Document vector dimensions:", None if vector is None else vector.shape)
        print("Known-token coverage:", round(coverage, 3))
        """,
    ),
    md(
        "w5-extraction",
        """
        # 9 Rules, dictionaries, entities, and evidence spans

        A regex identifies character patterns. A dictionary enumerates recognized expressions. Entity recognition locates a labeled span. Normalization proposes a concept mapping. Assertion, experiencer, and time are separate questions. Keeping the original span and offsets makes an output inspectable.
        """,
    ),
    code(
        "w5-regex",
        """
        fragment = "Appointment 2026-09-10. Metformin 500 mg listed."
        dates = re.findall(r"\\b\\d{4}-\\d{2}-\\d{2}\\b", fragment)
        dose_pattern = r"\\b([A-Za-z]+)\\s+(\\d+(?:\\.\\d+)?)\\s*(mg|mcg)\\b"
        dose_matches = [
            {"groups": match.groups(), "start": match.start(), "end": match.end()}
            for match in re.finditer(dose_pattern, fragment, flags=re.I)
        ]

        assert dates == ["2026-09-10"]
        assert dose_matches[0]["groups"] == ("Metformin", "500", "mg")
        print("Date-like strings:", dates)
        display(pd.DataFrame(dose_matches))
        print("Pattern matches do not validate the date, medication, dose, or clinical intent.")
        """,
    ),
    code(
        "w5-entities",
        """
        entity_nlp = spacy.blank("en")
        entity_nlp.add_pipe("sentencizer")
        ruler = entity_nlp.add_pipe("entity_ruler", config={"phrase_matcher_attr": "LOWER"})
        ruler.add_patterns(
            [
                {"label": "MEDICATION", "pattern": "metformin"},
                {"label": "SYMPTOM", "pattern": "chest pain"},
                {"label": "SYMPTOM", "pattern": "shortness of breath"},
                {"label": "SYMPTOM", "pattern": "SOB"},
                {"label": "SYMPTOM", "pattern": "dyspnea"},
                {"label": "TRANSPORTATION", "pattern": "transportation"},
                {"label": "TRANSPORTATION", "pattern": "ride"},
                {"label": "HOUSING", "pattern": "housing"},
                {"label": "HOUSING", "pattern": "rent"},
                {"label": "FOOD_INSECURITY", "pattern": "food insecurity"},
                {"label": "FOOD_INSECURITY", "pattern": "food"},
                {"label": "MEDICATION_COST", "pattern": "afford prescribed medication"},
            ]
        )

        entity_text = "No chest pain. Needs a ride. Metformin is listed."
        entities = pd.DataFrame(
            [
                {
                    "surface": entity.text,
                    "label": entity.label_,
                    "start": entity.start_char,
                    "end": entity.end_char,
                }
                for entity in entity_nlp(entity_text).ents
            ]
        )
        assert len(entities) == 3
        display(entities)
        """,
    ),
    code(
        "w5-concept-normalization",
        """
        concept_dictionary = {
            "sob": ("TEACHING_DYSPNEA", "shortness of breath"),
            "dyspnea": ("TEACHING_DYSPNEA", "shortness of breath"),
            "shortness of breath": ("TEACHING_DYSPNEA", "shortness of breath"),
        }
        normalization_text = "SOB documented; dyspnea discussed."
        normalization_rows = []
        for entity in entity_nlp(normalization_text).ents:
            normalization_rows.append(
                {
                    "surface": entity.text,
                    "mapping": concept_dictionary.get(entity.text.casefold(), "Unmapped: review needed"),
                }
            )

        assert len(normalization_rows) == 2
        display(pd.DataFrame(normalization_rows))
        print("TEACHING_DYSPNEA is an invented lesson identifier, not an official medical code.")
        """,
    ),
    md(
        "w5-context",
        """
        # 10 Negation, uncertainty, experiencer, and time

        Context cues have scope. “No fever but chest pain persists” should not negate chest pain. “Mother cannot pay rent” concerns another experiencer. “Previously homeless; now stable” is historical and resolved. The next function is intentionally incomplete so its failure is visible; it is not NegEx, ConText, or a validated clinical component.
        """,
    ),
    code(
        "w5-assertion",
        """
        def toy_assertion(sentence: str, term: str) -> str:
            lowered = sentence.lower()
            match = re.search(r"\\b" + re.escape(term.lower()) + r"\\b", lowered)
            if match is None:
                return "not mentioned"
            before, after = lowered[: match.start()], lowered[match.end() :]
            if re.search(r"\\bcannot be excluded\\b", after):
                return "uncertain"
            if re.search(r"\\b(possible|suspected)\\b", before):
                return "uncertain"
            if re.search(r"\\b(no|denies|without)\\b", before):
                return "negated"
            return "affirmed"


        context_examples = [
            ("Patient has chest pain.", "chest pain"),
            ("Patient denies chest pain.", "chest pain"),
            ("No evidence of pneumonia.", "pneumonia"),
            ("Pneumonia cannot be excluded.", "pneumonia"),
            ("No fever but chest pain persists.", "chest pain"),
        ]
        assertion_results = pd.DataFrame(
            [
                {"sentence": sentence, "term": term, "toy_status": toy_assertion(sentence, term)}
                for sentence, term in context_examples
            ]
        )

        assert assertion_results.iloc[-1]["toy_status"] == "negated"
        display(assertion_results)
        print("The final row is a deliberate scope error: chest pain is affirmed, not negated.")
        """,
    ),
    code(
        "w5-evidence-table",
        """
        evidence_rows = []
        social_labels = {"TRANSPORTATION", "HOUSING", "FOOD_INSECURITY", "MEDICATION_COST"}
        for row in notes.itertuples(index=False):
            parsed = entity_nlp(row.text)
            for entity in parsed.ents:
                if entity.label_ in social_labels:
                    evidence_rows.append(
                        {
                            "note_id": row.note_id,
                            "surface": entity.text,
                            "entity_type": entity.label_,
                            "start": entity.start_char,
                            "end": entity.end_char,
                            "authored_status": row.documentation_status,
                            "source_text": row.text,
                        }
                    )
        evidence = pd.DataFrame(evidence_rows)

        assert all(
            source[start:end] == surface
            for source, start, end, surface in evidence[["source_text", "start", "end", "surface"]].itertuples(index=False)
        )
        display(evidence)
        """,
    ),
    md(
        "w5-classification",
        """
        # 11 Leakage-safe classical text classification

        We compare a most-frequent-label baseline with TF-IDF plus logistic regression. The vectorizer is inside a `Pipeline`, so vocabulary and IDF weights are learned from training text only. The label is authored for this synthetic lesson; it is not a diagnosis, eligibility rule, or validated outcome.
        """,
    ),
    code(
        "w5-classifier",
        """
        X_train, X_test, y_train, y_test = train_test_split(
            training["text"],
            training["label"],
            test_size=0.25,
            stratify=training["label"],
            random_state=RANDOM_STATE,
        )

        baseline = DummyClassifier(strategy="most_frequent")
        baseline.fit(np.zeros((len(X_train), 1)), y_train)
        baseline_predictions = baseline.predict(np.zeros((len(X_test), 1)))

        text_model = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
                ("classifier", LogisticRegression(max_iter=1_000, random_state=RANDOM_STATE)),
            ]
        )
        text_model.fit(X_train, y_train)
        model_predictions = text_model.predict(X_test)

        classification_rows = []
        for name, predicted in [
            ("Most-frequent baseline", baseline_predictions),
            ("TF-IDF logistic regression", model_predictions),
        ]:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, predicted, average="binary", zero_division=0
            )
            classification_rows.append(
                {
                    "model": name,
                    "accuracy": accuracy_score(y_test, predicted),
                    "precision": precision,
                    "recall": recall,
                    "F1": f1,
                }
            )

        display(pd.DataFrame(classification_rows).round(3))
        print(classification_report(y_test, model_predictions, zero_division=0))
        display(
            pd.DataFrame(
                {"text": X_test, "reference": y_test, "prediction": model_predictions}
            ).sort_index()
        )
        """,
    ),
    code(
        "w5-language-challenge",
        """
        final_text_model = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
                ("classifier", LogisticRegression(max_iter=1_000, random_state=RANDOM_STATE)),
            ]
        ).fit(training["text"], training["label"])

        challenge_results = challenge.copy()
        challenge_results["model_prediction"] = final_text_model.predict(challenge["text"])
        challenge_results["correct"] = (
            challenge_results["model_prediction"]
            == challenge_results["reference_current_patient_need"]
        )
        display(challenge_results)
        print("English vocabulary coverage must not turn unsupported Spanish text into evidence of no need.")
        """,
    ),
    md(
        "w5-evaluation",
        """
        # 12 Task-specific evaluation and reviewer consequences

        Classification metrics, exact-span metrics, retrieval relevance, and summary review answer different questions. Precision measures how many flags are reference-positive; recall measures how many reference-positive items were found. Neither metric alone describes reviewer capacity, downstream benefit, or patient experience.
        """,
    ),
    code(
        "w5-keyword-evaluation",
        """
        reference = notes["reference_current_patient_need"].to_numpy()
        keyword_pattern = r"\\b(?:food|transportation|housing|rent|afford|ride)\\b"
        keyword_predictions = (
            notes["text"].str.contains(keyword_pattern, case=False, regex=True).astype(int).to_numpy()
        )
        matrix = confusion_matrix(reference, keyword_predictions, labels=[0, 1])
        keyword_metrics = pd.Series(
            {
                "precision": precision_score(reference, keyword_predictions),
                "recall": recall_score(reference, keyword_predictions),
                "F1": f1_score(reference, keyword_predictions),
                "flagged_notes": int(keyword_predictions.sum()),
                "total_notes": len(notes),
            }
        )

        assert matrix.tolist() == [[2, 6], [0, 4]]
        display(pd.DataFrame(matrix, index=["reference_0", "reference_1"], columns=["predicted_0", "predicted_1"]))
        display(keyword_metrics.round(3).to_frame("value"))
        """,
    ),
    code(
        "w5-span-evaluation",
        """
        span_text = "Needs a ride. No chest pain."


        def span_tuple(phrase: str, label: str):
            start = span_text.index(phrase)
            return ("note_A", start, start + len(phrase), label)


        gold_spans = {
            span_tuple("ride", "TRANSPORTATION"),
            span_tuple("chest pain", "SYMPTOM"),
        }
        predicted_spans = {
            ("note_A", entity.start_char, entity.end_char, entity.label_)
            for entity in entity_nlp(span_text).ents
        }
        true_positive_spans = len(gold_spans & predicted_spans)
        span_precision = true_positive_spans / max(len(predicted_spans), 1)
        span_recall = true_positive_spans / max(len(gold_spans), 1)

        assert span_precision == 1.0 and span_recall == 1.0
        print("Gold spans:", gold_spans)
        print("Predicted spans:", predicted_spans)
        print("Exact span precision and recall:", span_precision, span_recall)
        print("Perfect boundaries do not evaluate the negation of chest pain.")
        """,
    ),
    md(
        "w5-retrieval",
        """
        # 13 Retrieval, similarity, and extractive summarization

        Retrieval ranks documents for a query; relevance must be defined for that query. Extractive summarization selects existing sentences; it can still mislead by omitting the main problem, qualifier, or action.
        """,
    ),
    code(
        "w5-retrieval-code",
        """
        retriever = TfidfVectorizer(ngram_range=(1, 2))
        note_vectors = retriever.fit_transform(notes["text"])
        query = "transportation problem getting to appointment"
        query_vector = retriever.transform([query])
        assert query_vector.nnz > 0

        retrieval_scores = cosine_similarity(query_vector, note_vectors).ravel()
        top_indices = np.argsort(-retrieval_scores, kind="stable")[:3]
        retrieved = notes.iloc[top_indices][
            ["note_id", "text", "reference_current_patient_need", "documentation_status", "need_type"]
        ].copy()
        retrieved["similarity"] = retrieval_scores[top_indices]
        retrieved["narrow_relevance"] = (retrieved["need_type"] == "transportation") & (
            retrieved["reference_current_patient_need"] == 1
        )

        assert retrieved.iloc[0]["note_id"] == "N005"
        display(retrieved)
        print("Precision at 3 under affirmed-current-transportation relevance:", retrieved["narrow_relevance"].mean())
        """,
    ),
    code(
        "w5-summary-code",
        """
        summary_source = (
            "Patient cannot afford medication. "
            "Medication cost was discussed with care management. "
            "The patient reports no transportation problems. "
            "A financial assistance review is planned."
        )
        summary_sentences = [sentence.text for sentence in nlp(summary_source).sents]
        summary_vectorizer = TfidfVectorizer()
        summary_matrix = summary_vectorizer.fit_transform(summary_sentences)
        sentence_scores = np.asarray(summary_matrix.sum(axis=1)).ravel()
        selected = np.argsort(-sentence_scores, kind="stable")[:2]
        extractive_summary = " ".join(summary_sentences[index] for index in sorted(selected))

        display(pd.DataFrame({"sentence": summary_sentences, "score": sentence_scores}).round(3))
        print("Extractive summary:", extractive_summary)
        print("Contains the affordability problem?", "cannot afford" in extractive_summary.lower())
        print("Contains the planned action?", "assistance review" in extractive_summary.lower())
        """,
    ),
    md(
        "w5-privacy",
        """
        # 14 Privacy, documentation bias, language access, and secure processing

        Free text can expose direct and indirect identifiers in raw files, notebook outputs, logs, caches, vectorizer vocabularies, error examples, and saved models. Regex substitution is not HIPAA Safe Harbor or Expert Determination. Numerical vectors are not automatically anonymous.

        NLP also analyzes what was documented—not a complete account of reality. Missing text can mean an unasked question. An English-only model must not become a gatekeeper for assistance. Preserve unknown states, support correction, and provide language-concordant review or an approved interpretation pathway.
        """,
    ),
    code(
        "w5-redaction",
        """
        synthetic_private_text = (
            "Jordan Example, MRN 123456, attended on 2026-09-10. "
            "Phone 202-555-0147. The only harp repairer in Tiny Village."
        )
        redacted = re.sub(r"\\bMRN\\s+\\d+\\b", "MRN [REDACTED]", synthetic_private_text)
        redacted = re.sub(r"\\b\\d{4}-\\d{2}-\\d{2}\\b", "[DATE]", redacted)
        redacted = re.sub(r"\\b\\d{3}-\\d{3}-\\d{4}\\b", "[PHONE]", redacted)

        assert "Jordan Example" in redacted and "Tiny Village" in redacted
        print(redacted)
        print("The invented name, occupation, and location remain: this is NOT de-identified.")
        """,
    ),
    md(
        "w5-pipeline-card",
        """
        # 15 NLP Pipeline Card and release-readiness review

        | Field | Required design decision or evidence |
        |---|---|
        | Problem | Relevant social-information evidence may be hard to locate in approved notes |
        | Intended use | Prioritize evidence for an authorized reviewer; no autonomous eligibility or clinical decision |
        | Source and unit | Approved note source; candidate mention within a note; patient aggregation specified separately |
        | Reference standard | Written current-patient-need policy including assertion, time, experiencer, uncertainty, and exclusions |
        | Representation | Preserved source text plus versioned dictionary and/or training-fitted TF-IDF |
        | Output | Source span, category, context status, provenance, model/rule version, and review state |
        | Evaluation | Precision, recall, exact spans, contextual accuracy, support, coverage, workload, and external validity |
        | Human authority | Named role confirms, rejects, corrects, clarifies, or escalates |
        | Responsible AI | Documentation bias, stigmatizing labels, language access, accessibility, preferences, and recourse |
        | Secure AI | Approved processing boundary, access control, protected logs, governed exports, retention, and rollback |
        | Monitoring | Coverage, errors, drift, workload, corrections, incidents, and suspension criteria |

        **Release-readiness scenario:** A manager asks to connect this notebook to real notes because the phrase rule scored perfectly on the 12 constructed examples. Write an *abstain*, *restrict*, *modify*, or *advance only to another evaluation stage* recommendation. Include at least two Evaluation, two Responsible AI, and two Secure AI reasons.
        """,
    ),
    md(
        "w5-review",
        """
        # 16 Review and applied exercises

        ## Review questions

        1. Why is a vocabulary position an address rather than an ordered clinical measurement?
        2. What does TF-IDF retain, and why did the opposite-polarity sentence receive high cosine similarity?
        3. Why can perfect entity boundaries coexist with incorrect clinical interpretation?
        4. Why must the TF-IDF vectorizer be fitted inside the training pipeline?
        5. Under what relevance definition should “No transportation problems” appear in retrieval results?
        6. Why can a verbatim extractive summary still be unsafe or inadequate?
        7. Which artifacts beyond raw notes may expose sensitive language?
        8. How should unsupported Spanish text and “not documented” cases be routed without treating them as negative findings?

        ## Applied exercises

        - **Representation lab:** Add a contraction, abbreviation, dose, date, and section heading. Record unexpected tokens, offsets, and any lost information.
        - **Retrieval lab:** Change the query and score the top three results using two explicitly different relevance rubrics.
        - **Challenge-set lab:** Add one ambiguous, historical, family-related, negated, multilingual, and empty-template example. Reserve them from rule development.
        - **Workflow design:** Complete the Pipeline Card for a nonconfidential use case and specify ownership, correction, escalation, monitoring, and rollback.

        ### Evaluation criteria

        A strong submission preserves provenance; separates mention, context, and action; uses a leakage-safe split; reports task-specific metrics and support; analyzes at least two concrete errors; and states privacy, language, review, and deployment limits.
        """,
    ),
    code(
        "w5-reproducibility",
        """
        execution_record = {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scikit-learn": version("scikit-learn"),
            "nltk": version("nltk"),
            "spacy": version("spacy"),
            "random_seed": RANDOM_STATE,
            **{filename: sha256_file(DATA_DIR / filename) for filename in EXPECTED_HASHES},
        }
        display(pd.Series(execution_record, name="value").to_frame())
        """,
    ),
    md(
        "w5-takeaways",
        """
        # Key takeaways

        - NLP analyzes representations of documented language, not an objective or complete account of a person's circumstances.
        - Cleaning, tokenization, vocabulary, n-grams, and numerical encoding determine which distinctions remain available.
        - Sparse lexical vectors and dense learned vectors are different representations; similarity is not factual agreement.
        - Entity spans, normalized concepts, assertion, experiencer, time, and action are separate outputs requiring separate evidence.
        - Classification, extraction, retrieval, and summarization need different reference standards and evaluation methods.
        - Synthetic success does not establish clinical validity, transportability, fairness, privacy, or workflow benefit.
        - Privacy, language access, source provenance, human authority, monitoring, rollback, and retirement belong inside the NLP method.

        ## References and provenance

        - HI 634 Chapter 5, *Natural Language Processing in Health Services*.
        - UCI Machine Learning Repository, Drug Reviews (Druglib.com), ID 461, DOI [10.24432/C55G6J](https://doi.org/10.24432/C55G6J). Discussed but not redistributed here.
        - Mikolov et al., word representations, DOI [10.48550/arXiv.1301.3781](https://doi.org/10.48550/arXiv.1301.3781).
        - Devlin et al., BERT, DOI [10.18653/v1/N19-1423](https://doi.org/10.18653/v1/N19-1423).
        - Chapman et al., NegEx, DOI [10.1006/jbin.2001.1029](https://doi.org/10.1006/jbin.2001.1029).
        - Harkema et al., ConText, DOI [10.1016/j.jbi.2009.05.002](https://doi.org/10.1016/j.jbi.2009.05.002).
        - Synthetic CSV files were generated locally by `week5/scripts/generate_synthetic_data.py`; they contain no real patient information.
        """,
    ),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.14"},
    },
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(f"Wrote {OUTPUT}")
