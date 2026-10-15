import json
from datetime import datetime

from config import get_embed_model, get_collection, TOP_K


# ============================================================
# 1. Evaluation Dataset
# ============================================================

EVALUATION_DATASET = [
    {
        "id": "fever_temperature",
        "category": "Fever",
        "query": "What is the best way to measure my baby's temperature at home?",
        "ground_truth_evidence": [
            "measure body temperature with an electronic thermometer in the axilla",
            "electronic thermometer in the axilla",
            "chemical dot thermometer in the axilla",
            "infra-red tympanic thermometer",
        ],
    },

    {
        "id": "fever_sponging",
        "category": "Fever",
        "query": "Is it okay to use cold water sponging to cool down my child?",
        "ground_truth_evidence": [
            "Tepid sponging is not recommended for the treatment of fever"
        ],
    },

    {
        "id": "asthma_flare",
        "category": "Asthma and Breathing",
        "query": "How can I recognize if my child's asthma is flaring up?",
        "ground_truth_evidence": [
            "progressive increase in wheeze",
            "shortness of breath",
            "chest tightness",
            "cough",
            "acute asthma",
            "flare-ups",
            "exacerbations",
        ],
    },

    {
        "id": "diarrhea_dehydration",
        "category": "Diarrhoea and Vomiting",
        "query": "How do I know if my child is becoming dehydrated?",
        "ground_truth_evidence": [
            "Altered responsiveness",
            "Sunken eyes",
            "Dry mucous membranes",
            "Tachycardia",
            "Tachypnoea",
            "Reduced skin turgor",
            "Decreased urine output",
        ],
    },

    {
        "id": "gastroenteritis_prevention",
        "category": "General Care and Prevention",
        "query": "What is the most effective way to prevent the spread of stomach bugs in my family?",
        "ground_truth_evidence": [
            "washing hands with soap",
            "most important factor in preventing the spread of gastroenteritis",
            "hands should be washed after going to the toilet",
            "before preparing, serving or eating food",
        ],
    },
]


# ============================================================
# 2. Normalize text
# ============================================================

def normalize(text: str) -> str:
    """
    Normalize text to make matching more robust.
    """
    return " ".join(text.lower().split())


# ============================================================
# 3. Find Ground Truth Chunks
# ============================================================

def find_ground_truth_chunks(collection, evidence_list):
    """
    Find chunks in Chroma that contain the manually selected
    ground-truth evidence.

    These chunks are considered relevant for evaluation.
    """

    # Get all documents from the collection
    results = collection.get(
        include=["documents", "metadatas"]
    )

    ids = results["ids"]
    docs = results["documents"]
    metas = results["metadatas"]

    relevant_ids = set()
    matched_chunks = []

    normalized_evidence = [
        normalize(evidence)
        for evidence in evidence_list
    ]

    for chunk_id, doc, meta in zip(ids, docs, metas):

        normalized_doc = normalize(doc)

        matched = False
        matched_evidence = []

        for evidence in normalized_evidence:

            if evidence in normalized_doc:
                matched = True
                matched_evidence.append(evidence)

        if matched:
            relevant_ids.add(chunk_id)

            matched_chunks.append({
                "id": chunk_id,
                "metadata": meta,
                "matched_evidence": matched_evidence
            })

    return relevant_ids, matched_chunks


# ============================================================
# 4. Semantic Retrieval
# ============================================================

def retrieve(query: str, k: int):

    model = get_embed_model()

    query_embedding = model.encode(
        [query],
        prompt_name="query",
        convert_to_numpy=True
    )[0]

    collection = get_collection()

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k,
        include=[
            "documents",
            "metadatas",
            "distances"
        ],
    )

    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    semantic_scores = [
        1.0 - distance
        for distance in distances
    ]

    ranked = sorted(
        zip(
            ids,
            docs,
            metas,
            distances,
            semantic_scores
        ),
        key=lambda x: x[-1],
        reverse=True
    )

    return ranked


# ============================================================
# 5. Precision@K
# ============================================================

def precision_at_k(retrieved_ids, relevant_ids, k):

    top_k = retrieved_ids[:k]

    if not top_k:
        return 0.0

    relevant_retrieved = sum(
        1 for chunk_id in top_k
        if chunk_id in relevant_ids
    )

    return relevant_retrieved / len(top_k)


# ============================================================
# 6. Evaluate
# ============================================================

def evaluate():

    collection = get_collection()

    print("=" * 70)
    print("SEMANTIC RETRIEVAL EVALUATION")
    print("=" * 70)

    print(f"Number of evaluation queries: {len(EVALUATION_DATASET)}")

    all_results = []

    precision_5_scores = []
    precision_10_scores = []

    for item in EVALUATION_DATASET:

        query = item["query"]

        print("\n" + "-" * 70)
        print(f"Category: {item['category']}")
        print(f"Query: {query}")

        # ----------------------------------------------------
        # Find manually labeled relevant chunks
        # ----------------------------------------------------

        relevant_ids, matched_chunks = find_ground_truth_chunks(
            collection,
            item["ground_truth_evidence"]
        )

        print(
            f"Ground-truth relevant chunks found: "
            f"{len(relevant_ids)}"
        )

        if not relevant_ids:
            print(
                "WARNING: No ground-truth chunk was found. "
                "Check the evidence text."
            )

        # ----------------------------------------------------
        # Retrieve top 10
        # ----------------------------------------------------

        retrieved = retrieve(query, k=10)

        retrieved_ids = [
            row[0]
            for row in retrieved
        ]

        # ----------------------------------------------------
        # Precision@5
        # ----------------------------------------------------

        p5 = precision_at_k(
            retrieved_ids,
            relevant_ids,
            5
        )

        # ----------------------------------------------------
        # Precision@10
        # ----------------------------------------------------

        p10 = precision_at_k(
            retrieved_ids,
            relevant_ids,
            10
        )

        precision_5_scores.append(p5)
        precision_10_scores.append(p10)

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print(f"Precision@5 : {p5:.2f}")
        print(f"Precision@10: {p10:.2f}")

        print("\nTop retrieved chunks:")

        for rank, row in enumerate(retrieved, start=1):

            chunk_id, doc, meta, distance, semantic = row

            is_relevant = chunk_id in relevant_ids

            mark = "✓" if is_relevant else "✗"

            print(
                f"{rank:2d}. {mark} "
                f"semantic={semantic:.4f} "
                f"id={chunk_id}"
            )

        # ----------------------------------------------------
        # Save individual result
        # ----------------------------------------------------

        all_results.append({
            "id": item["id"],
            "category": item["category"],
            "query": query,
            "num_ground_truth_chunks": len(relevant_ids),
            "ground_truth_chunk_ids": list(relevant_ids),
            "matched_ground_truth_chunks": matched_chunks,
            "precision_at_5": p5,
            "precision_at_10": p10,
            "retrieved_chunks": [
                {
                    "rank": rank,
                    "chunk_id": row[0],
                    "semantic_score": row[4],
                    "relevant": row[0] in relevant_ids,
                    "source": row[2].get("source"),
                    "page": row[2].get("page"),
                }
                for rank, row in enumerate(retrieved, start=1)
            ],
        })

    # ========================================================
    # Average Results
    # ========================================================

    avg_p5 = (
        sum(precision_5_scores)
        / len(precision_5_scores)
    )

    avg_p10 = (
        sum(precision_10_scores)
        / len(precision_10_scores)
    )

    final_results = {
        "timestamp": datetime.now().isoformat(),
        "evaluation_type": "Semantic Retrieval",
        "num_queries": len(EVALUATION_DATASET),
        "avg_precision_at_5": avg_p5,
        "avg_precision_at_10": avg_p10,
        "individual_results": all_results,
    }

    # ========================================================
    # Save JSON
    # ========================================================

    output_file = "retrieval_evaluation_results.json"

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            final_results,
            f,
            indent=2,
            ensure_ascii=False
        )

    # ========================================================
    # Final Summary
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(f"Average Precision@5 : {avg_p5:.2%}")
    print(f"Average Precision@10: {avg_p10:.2%}")

    print(
        f"\nResults saved to: {output_file}"
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    evaluate()