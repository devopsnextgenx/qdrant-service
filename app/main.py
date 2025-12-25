# ============================================
# MODIFIED: app/main.py
# ============================================
import asyncio
import os
import logging
from fastapi import FastAPI, HTTPException, Query
from typing import Optional

from .config import CONFIG
from .embeddings import embedder
from .qdrant_service import qdrant
from .ingest import iter_all_documents, iter_captions, iter_stories
from .chunking import TextChunker  # NEW IMPORT

logger = logging.getLogger(__name__)
app = FastAPI(title="Qdrant Index & Search")

CAPTIONS_COLLECTION = CONFIG.get("qdrant", {}).get("collections", {}).get("captions", "captions")
STORIES_COLLECTION = CONFIG.get("qdrant", {}).get("collections", {}).get("stories", "stories")

INDEX_BATCH = CONFIG.get("indexing", {}).get("batch_size", 64)
SEARCH_TOP_K = CONFIG.get("search", {}).get("top_k", 10)
SEARCH_SCORE_THRESHOLD = CONFIG.get("search", {}).get("score_threshold", 0.0)

# NEW: Initialize chunker with config values
CHUNK_SIZE = CONFIG.get("chunking", {}).get("chunk_size", 512)
CHUNK_OVERLAP = CONFIG.get("chunking", {}).get("overlap", 128)
chunker = TextChunker(chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

logger.info(f"Initialized chunker with size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")


@app.get("/health")
async def health():
    """Basic service health plus backend checks (Qdrant and embeddings)"""
    services = {}
    # Qdrant
    try:
        services['qdrant'] = qdrant.health()
    except Exception as e:
        services['qdrant'] = {'ok': False, 'error': str(e)}

    # Embeddings backend (if it has health method)
    emb_health = None
    if hasattr(embedder, 'health'):
        try:
            emb_health = embedder.health()
        except Exception as e:
            emb_health = {'ok': False, 'error': str(e)}
    else:
        emb_health = {'ok': True, 'note': 'no health probe available for this embedder'}

    services['embeddings'] = emb_health

    status = 'ok' if services.get('qdrant', {}).get('ok') and services.get('embeddings', {}).get('ok') else 'degraded'
    return {'status': status, 'services': services}


async def _embed_texts(texts):
    # encoding is CPU-bound; call in thread pool
    return await asyncio.to_thread(embedder.encode, texts)


@app.post("/index")
@app.get("/index")
async def index(type: Optional[str] = Query("all", description="captions|stories|all")):
    """Index data into Qdrant. Use ?type=captions|stories|all"""
    logger.info("Indexing %s", type)
    if type not in ("all", "captions", "stories"):
        raise HTTPException(status_code=400, detail="Invalid type")

    # Prepare iterators
    raw_docs = []
    if type in ("all", "captions"):
        raw_docs.extend(list(iter_captions()))
    if type in ("all", "stories"):
        raw_docs.extend(list(iter_stories()))

    if not raw_docs:
        return {"indexed": 0}

    logger.info("Processing %s raw documents", len(raw_docs))
    
    # NEW: Chunk all documents before processing
    all_chunks = []
    for doc in raw_docs:
        doc_id = doc['id']
        text = doc['text']
        
        # Extract metadata from original doc
        metadata = {
            'type': doc['payload'].get('type'),
            'source': doc['payload'].get('source'),
            # Add any other metadata fields you need
        }
        
        # Chunk the document
        chunks = chunker.chunk_text(text, doc_id, metadata)
        all_chunks.extend(chunks)
        
        logger.debug(f"Document {doc_id} split into {len(chunks)} chunks")
    
    logger.info(f"Created {len(all_chunks)} chunks from {len(raw_docs)} documents")
    
    # Process chunks in batches
    total_indexed = 0
    for i in range(0, len(all_chunks), INDEX_BATCH):
        batch = all_chunks[i : i + INDEX_BATCH]
        logger.info("Indexing batch %s (%s chunks)", i // INDEX_BATCH + 1, len(batch))
        
        texts = [chunk["text"] for chunk in batch]
        vecs = await _embed_texts(texts)
        
        captions_points = []
        stories_points = []
        logger.debug(f"Embedding batch {i // INDEX_BATCH + 1} with {len(texts)} texts")
        print(f"Embedding batch {i // INDEX_BATCH + 1} with {len(texts)} texts")
        
        for chunk, vec in zip(batch, vecs):
            point = {
                "id": chunk["id"],  # Now a valid UUID string
                "vector": vec.tolist() if hasattr(vec, "tolist") else vec, 
                "payload": chunk["payload"]
            }
            
            if chunk["payload"]["type"] == "caption":
                captions_points.append(point)
            else:
                stories_points.append(point)

        if captions_points:
            qdrant.ensure_collection(CAPTIONS_COLLECTION, embedder.dim)
            qdrant.upsert_points(CAPTIONS_COLLECTION, captions_points)
        if stories_points:
            qdrant.ensure_collection(STORIES_COLLECTION, embedder.dim)
            qdrant.upsert_points(STORIES_COLLECTION, stories_points)

        total_indexed += len(batch)
        logger.info("Indexed batch %s - total indexed: %s", i // INDEX_BATCH + 1, total_indexed)

    return {
        "indexed": total_indexed, 
        "raw_documents": len(raw_docs),
        "chunks_created": len(all_chunks)
    }


@app.get("/search")
async def search(
    q: str = Query(...), 
    type: Optional[str] = Query("captions", description="captions|stories"), 
    limit: Optional[int] = None
):
    if type not in ("captions", "stories"):
        raise HTTPException(status_code=400, detail="Invalid type")

    if limit is None:
        limit = SEARCH_TOP_K

    vecs = await _embed_texts([q])
    vec = vecs[0]
    coll = CAPTIONS_COLLECTION if type == "captions" else STORIES_COLLECTION
    
    results = qdrant.hybrid_search(
        coll, 
        vector=vec.tolist() if hasattr(vec, "tolist") else vec,
        text_query=q,
        limit=limit
    )
    
    # Apply score threshold filter
    filtered = [r for r in results if r.get("score", 0) >= SEARCH_SCORE_THRESHOLD]

    return {
        "results": filtered, 
        "meta": {
            "limit": limit, 
            "score_threshold": SEARCH_SCORE_THRESHOLD,
            "chunking_enabled": True,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP
        }
    }
