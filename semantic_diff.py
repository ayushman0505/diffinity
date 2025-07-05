# utils/semantic_diff.py

from sentence_transformers import SentenceTransformer, util
import torch

model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embeddings(text_blocks):
    """
    Generate sentence embeddings for a list of text blocks.
    """
    return model.encode(text_blocks, convert_to_tensor=True)


def compute_semantic_diff(doc1_blocks, doc2_blocks, threshold=0.8):
    """
    Compares two lists of text blocks using cosine similarity.
    Returns a list of differences as tuples:
    ("ADDED" or "REMOVED", text, similarity_score)
    """
    doc1_embeddings = get_embeddings(doc1_blocks)
    doc2_embeddings = get_embeddings(doc2_blocks)

    diffs = []

    for i, emb1 in enumerate(doc1_embeddings):
        scores = util.cos_sim(emb1, doc2_embeddings)[0]
        max_score = torch.max(scores).item()
        if max_score < threshold:
            diffs.append(("REMOVED", doc1_blocks[i], max_score))

    for i, emb2 in enumerate(doc2_embeddings):
        scores = util.cos_sim(emb2, doc1_embeddings)[0]
        max_score = torch.max(scores).item()
        if max_score < threshold:
            diffs.append(("ADDED", doc2_blocks[i], max_score))

    return sorted(diffs, key=lambda x: x[2])

