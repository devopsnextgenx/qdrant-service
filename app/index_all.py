#!/usr/bin/env python3
"""CLI to index all data in data/ into Qdrant"""
import argparse
import asyncio
import logging
from .ingest import iter_captions, iter_stories
from .embeddings import embedder
from .qdrant_service import qdrant
from .chunking import TextChunker
from .config import CONFIG
from .tracker import QdrantProcessTracker

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load config values
CAPTIONS_COLLECTION = CONFIG.get("qdrant", {}).get("collections", {}).get("captions", "captions")
STORIES_COLLECTION = CONFIG.get("qdrant", {}).get("collections", {}).get("stories", "stories")
INDEX_BATCH = CONFIG.get("indexing", {}).get("batch_size", 64)
CHUNK_SIZE = CONFIG.get("chunking", {}).get("chunk_size", 512)
CHUNK_OVERLAP = CONFIG.get("chunking", {}).get("overlap", 128)


async def _embed_texts(texts):
    """Embed texts using thread pool for CPU-bound work"""
    return await asyncio.to_thread(embedder.encode, texts)


async def index_documents(doc_type: str = "all"):
    """Index documents with chunking and batching"""
    logger.info("Indexing %s", doc_type)
    
    # Initialize chunker
    chunker = TextChunker(chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    logger.info(f"Initialized chunker with size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    
    # Initialize tracker
    tracker = QdrantProcessTracker()
    skip_captions = tracker.get_processed_files("captions")
    skip_stories = tracker.get_processed_files("stories")
    logger.info(f"Loaded tracker: {len(skip_captions)} captions, {len(skip_stories)} stories already processed")

    # Gather raw documents
    raw_docs = []
    if doc_type in ("all", "captions"):
        raw_docs.extend(list(iter_captions(skip_files=skip_captions)))
    if doc_type in ("all", "stories"):
        raw_docs.extend(list(iter_stories(skip_files=skip_stories)))

    if not raw_docs:
        logger.warning("No new documents found to index")
        print("No new documents found to index")
        return

    logger.info("Processing %s raw documents", len(raw_docs))
    
    # Chunk all documents
    all_chunks = []
    for doc in raw_docs:
        doc_id = doc['id']
        text = doc['text']
        
        # Extract metadata from original doc
        metadata = {
            'type': doc['payload'].get('type'),
            'source': doc['payload'].get('source'),
        }
        
        # Chunk the document
        chunks = chunker.chunk_text(text, doc_id, metadata)
        all_chunks.extend(chunks)
        
        logger.debug(f"Document {doc_id} split into {len(chunks)} chunks")
    
    logger.info(f"Created {len(all_chunks)} chunks from {len(raw_docs)} documents")
    
    # Track which files are being processed in this run
    newly_processed_files = {"captions": set(), "stories": set()}
    for doc in raw_docs:
        source = doc['payload'].get('source')
        dtype = doc['payload'].get('type')
        if dtype == "caption":
            newly_processed_files["captions"].add(source)
        else:
            newly_processed_files["stories"].add(source)

    # Process chunks in batches
    total_indexed = 0
    for i in range(0, len(all_chunks), INDEX_BATCH):
        batch = all_chunks[i : i + INDEX_BATCH]
        batch_num = i // INDEX_BATCH + 1
        logger.info("Indexing batch %s (%s chunks)", batch_num, len(batch))
        
        texts = [chunk["text"] for chunk in batch]
        vecs = await _embed_texts(texts)
        
        captions_points = []
        stories_points = []
        
        for chunk, vec in zip(batch, vecs):
            point = {
                "id": chunk["id"],
                "vector": vec.tolist() if hasattr(vec, "tolist") else vec,
                "payload": chunk["payload"]
            }
            
            if chunk["payload"]["type"] == "caption":
                captions_points.append(point)
            else:
                stories_points.append(point)

        # Upsert to collections
        if captions_points:
            qdrant.ensure_collection(CAPTIONS_COLLECTION, embedder.dim)
            qdrant.upsert_points(CAPTIONS_COLLECTION, captions_points)
        if stories_points:
            qdrant.ensure_collection(STORIES_COLLECTION, embedder.dim)
            qdrant.upsert_points(STORIES_COLLECTION, stories_points)

        total_indexed += len(batch)
        logger.info("Indexed batch %s - total indexed: %s", batch_num, total_indexed)

    # Save progress after all batches are done
    for source in newly_processed_files["captions"]:
        tracker.mark_as_processed(source, "captions")
    for source in newly_processed_files["stories"]:
        tracker.mark_as_processed(source, "stories")
    tracker.save()
    logger.info("Updated tracker with %d new files", len(newly_processed_files["captions"]) + len(newly_processed_files["stories"]))

    print(f"\n{'='*60}")
    print(f"Indexing complete!")
    print(f"{'='*60}")
    print(f"Raw documents:  {len(raw_docs)}")
    print(f"Chunks created: {len(all_chunks)}")
    print(f"Total indexed:  {total_indexed}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Index documents into Qdrant with chunking support"
    )
    parser.add_argument(
        "--type",
        choices=["all", "captions", "stories"],
        default="all",
        help="Type of documents to index (default: all)"
    )
    args = parser.parse_args()

    # Run async indexing
    asyncio.run(index_documents(args.type))


if __name__ == "__main__":
    main()