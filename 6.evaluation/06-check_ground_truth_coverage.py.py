import json
from pathlib import Path



# BioAI Ground Truth Coverage Audit


print("=" * 70)
print("BioAI Ground Truth Coverage Audit")
print("=" * 70)


# ------------------------------------------------------------
# 1. Define project paths
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

CORPUS_PATH = (
    PROJECT_DIR
    / "5.retrieval"
    / "pubmed_corpus_v2.json"
)

QUESTIONS_PATH = (
    BASE_DIR
    / "01-retrieval_questions.json"
)


# ------------------------------------------------------------
# 2. Check files
# ------------------------------------------------------------

print("\nChecking files...")

if not CORPUS_PATH.exists():
    raise FileNotFoundError(
        f"Corpus not found:\n{CORPUS_PATH}"
    )

if not QUESTIONS_PATH.exists():
    raise FileNotFoundError(
        f"Evaluation questions not found:\n{QUESTIONS_PATH}"
    )

print(f"Corpus:")
print(CORPUS_PATH)

print(f"\nEvaluation questions:")
print(QUESTIONS_PATH)


# ------------------------------------------------------------
# 3. Load corpus
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("Loading corpus...")
print("=" * 70)

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    corpus = json.load(f)

print(f"Papers in corpus: {len(corpus)}")


# ------------------------------------------------------------
# 4. Build corpus PMID set
# ------------------------------------------------------------

corpus_pmids = set()

for paper in corpus:
    pmid = str(paper.get("pmid", "")).strip()

    if pmid:
        corpus_pmids.add(pmid)

print(f"Unique PMIDs in corpus: {len(corpus_pmids)}")


# ------------------------------------------------------------
# 5. Load evaluation questions
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("Loading evaluation questions...")
print("=" * 70)

with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
    questions = json.load(f)

print(f"Evaluation questions: {len(questions)}")


# ------------------------------------------------------------
# 6. Audit every question
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("Ground Truth Coverage")
print("=" * 70)


total_relevant = 0
total_found = 0
total_missing = 0


all_missing = set()


for i, item in enumerate(questions, start=1):

    question = item["question"]

    relevant_pmids = [
        str(pmid).strip()
        for pmid in item["relevant_pmids"]
    ]

    found = [
        pmid
        for pmid in relevant_pmids
        if pmid in corpus_pmids
    ]

    missing = [
        pmid
        for pmid in relevant_pmids
        if pmid not in corpus_pmids
    ]

    total_relevant += len(relevant_pmids)
    total_found += len(found)
    total_missing += len(missing)

    all_missing.update(missing)

    print("\n" + "-" * 70)
    print(f"Question {i}")
    print("-" * 70)

    print(f"Query:")
    print(question)

    print("\nRelevant PMIDs:")
    print(relevant_pmids)

    print("\nFound in Corpus V2:")
    print(found)

    print("\nMissing from Corpus V2:")
    print(missing)

    coverage = (
        len(found) / len(relevant_pmids)
        if relevant_pmids
        else 0
    )

    print(f"\nCoverage: {coverage:.3f}")


# ------------------------------------------------------------
# 7. Overall coverage
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("Overall Ground Truth Coverage")
print("=" * 70)

print(f"Total relevant PMIDs : {total_relevant}")
print(f"Found in corpus      : {total_found}")
print(f"Missing from corpus  : {total_missing}")

overall_coverage = (
    total_found / total_relevant
    if total_relevant
    else 0
)

print(f"Coverage             : {overall_coverage:.3f}")


# ------------------------------------------------------------
# 8. Missing PMIDs
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("Missing Ground Truth PMIDs")
print("=" * 70)

if all_missing:

    for pmid in sorted(all_missing):
        print(pmid)

else:

    print("None")


# ------------------------------------------------------------
# 9. Final interpretation
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("Interpretation")
print("=" * 70)

if overall_coverage == 1.0:

    print(
        "All ground-truth PMIDs are present in Corpus V2."
    )

    print(
        "Retrieval performance can therefore be evaluated "
        "directly against the current benchmark."
    )

elif overall_coverage > 0:

    print(
        "Some ground-truth PMIDs are present, "
        "but some are missing from Corpus V2."
    )

    print(
        "Do NOT interpret low Recall@5 as purely a retrieval "
        "ranking problem."
    )

    print(
        "First reconcile the evaluation benchmark and corpus."
    )

else:

    print(
        "None of the ground-truth PMIDs are present in Corpus V2."
    )

    print(
        "The current retrieval evaluation is not valid for "
        "this corpus."
    )


print("\n" + "=" * 70)
print("Audit completed.")
print("=" * 70)