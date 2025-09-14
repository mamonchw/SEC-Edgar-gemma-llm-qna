"""Wrapper helpers for Chroma DB operations.

Provides lightweight functions so main logic stays clean.
"""
from __future__ import annotations

import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict
import uuid


def get_chroma_client(persist_directory: str):
    client = chromadb.PersistentClient(path=persist_directory)
    return client


def get_or_create_collection(client, name: str):
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def upsert_single(client, collection_name: str, embedding: List[float], metadata: Dict):
    coll = get_or_create_collection(client, collection_name)
    doc_id = metadata.get("id") or f"{metadata['ticker']}-{metadata['year']}-{metadata['chunk_index']}-{uuid.uuid4().hex[:8]}"
    coll.upsert(ids=[doc_id], embeddings=[embedding], metadatas=[metadata], documents=[metadata.get("text", "")])


def similarity_search(client, collection_name: str, query_embedding: List[float], k: int = 5):
    coll = get_or_create_collection(client, collection_name)
    res = coll.query(query_embeddings=[query_embedding], n_results=k, include=["documents", "metadatas", "distances"])
    # Standardize output
    results = []
    for i in range(len(res["ids"][0])):
        results.append({
            "id": res["ids"][0][i],
            "distance": res["distances"][0][i],
            "metadata": res["metadatas"][0][i],
            "document": res["documents"][0][i],
        })
    return results


def list_collections(client) -> List[str]:
    try:
        cols = client.list_collections()
    except Exception:
        return []
    names = []
    for c in cols:
        # chroma returns objects with name attribute
        name = getattr(c, "name", None)
        if name:
            names.append(name)
    return sorted(names)


__all__ = [
    "get_chroma_client",
    "upsert_single",
    "similarity_search",
    "get_or_create_collection",
    "list_collections",
]
