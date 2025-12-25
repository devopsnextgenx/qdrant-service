#!/usr/bin/env python3
"""CLI to index all data in data/ into Qdrant"""
import argparse
from app.ingest import iter_all_documents, iter_captions, iter_stories
from app.embeddings import embedder
from app.qdrant_service import qdrant


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["all", "captions", "stories"], default="all")
    args = parser.parse_args()

    docs = []
    if args.type in ("all", "captions"):
        docs.extend(list(iter_captions()))
    if args.type in ("all", "stories"):
        docs.extend(list(iter_stories()))

    if not docs:
        print("No documents found to index")
        return

    texts = [d["text"] for d in docs]
    vecs = embedder.encode(texts)

    captions_points = []
    stories_points = []
    for d, v in zip(docs, vecs):
        p = {"id": d["id"], "vector": v.tolist() if hasattr(v, 'tolist') else v, "payload": d["payload"]}
        if d["payload"]["type"] == "caption":
            captions_points.append(p)
        else:
            stories_points.append(p)

    if captions_points:
        qdrant.ensure_collection("captions", embedder.dim)
        qdrant.upsert_points("captions", captions_points)
    if stories_points:
        qdrant.ensure_collection("stories", embedder.dim)
        qdrant.upsert_points("stories", stories_points)

    print(f"Indexed {len(docs)} documents")


if __name__ == "__main__":
    main()
