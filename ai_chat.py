from __future__ import annotations

import os
import json
import textwrap
from typing import List, Dict, Deque, Tuple
from collections import deque
import requests
from sentence_transformers import SentenceTransformer
from vector_store import get_chroma_client, similarity_search, get_or_create_collection, list_collections
import numpy as np


CLOUDFLARE_ENDPOINT = (
    "https://api.cloudflare.com/client/v4/accounts/" \
    "ad4b2a07c9f893b15b2364bc2630dabe/ai/run/@cf/meta/llama-4-scout-17b-16e-instruct"
)

# Base system instruction supplied by user (kept verbatim, lightly cleaned for formatting)
SYSTEM_PROMPT = (
    "ROLE: You are an expert financial analyst assistant focused on U.S. SEC corporate filings (10-K, 10-Q, 8-K, S-1, etc.).\n"
    "DATA SOURCE: You ONLY use the provided retrieved context chunks from the vector store. You must NOT hallucinate information not present in that context.\n"
    "Do not mention about provided context, chunks, the vector store"
)


_EMBED_MODEL_CACHE: Dict[str, SentenceTransformer] = {}

# If the best collection similarity (cosine) is below this, we fallback to searching all collections
COLLECTION_SIM_THRESHOLD = 0.60


def _get_embed_model(model_name: str) -> SentenceTransformer:
    if model_name not in _EMBED_MODEL_CACHE:
        _EMBED_MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _EMBED_MODEL_CACHE[model_name]


def _call_cloudflare_ai(messages: List[Dict]) -> str:
    token = "CLOUDFLARE_TOKEM"
    if not token:
        return "[ERROR] CLOUDFLARE_API_TOKEN env var not set. Cannot call model."
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(CLOUDFLARE_ENDPOINT, headers=headers, json={"messages": messages}, timeout=60)
    except Exception as e:
        return f"[ERROR] Request failed: {e}"
    if resp.status_code != 200:
        return f"[ERROR] HTTP {resp.status_code}: {resp.text[:500]}"
    try:
        data = resp.json()
    except ValueError:
        return f"[ERROR] Non-JSON response: {resp.text[:500]}"
    # Cloudflare's response shape may contain result(s); fallback to raw
    # Attempt common fields
    result = data.get("result") or data.get("results") or data
    # If model returns message-like format, normalize
    if isinstance(result, dict):
        # Look for output text
        for key in ("response", "output", "text"):
            if key in result and isinstance(result[key], str):
                return result[key]
        # Look for message array
        if "messages" in result and isinstance(result["messages"], list):
            last = result["messages"][-1]
            if isinstance(last, dict) and isinstance(last.get("content"), str):
                return last["content"]
    return json.dumps(result)[:2000]


def _build_context_block(results: List[Dict], max_chars_per_chunk: int = 700) -> str:
    lines = []
    for r in results:
        meta = r.get("metadata", {})
        txt = r.get("document", "")
        if len(txt) > max_chars_per_chunk:
            txt = txt[:max_chars_per_chunk] + "…"
        lines.append(
            f"[ticker={meta.get('ticker')} company={meta.get('company_name')} year={meta.get('year')} chunk={meta.get('chunk_index')}]\n{txt}"\
        )
    return "\n\n".join(lines)


def _make_messages(user_query: str, context_block: str, history: List[Tuple[str, str]], max_history: int = 3) -> List[Dict]:
    # history: list of (user, assistant) pairs
    recent = history[-max_history:]
    msgs: List[Dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for u, a in recent:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    augmented_user = textwrap.dedent(
        f"""User Query: {user_query}\n\nRetrieved Context (RAG):\n{context_block}\n\n"""
    )
    msgs.append({"role": "user", "content": augmented_user})
    return msgs


def chat_loop(persist_dir: str, model_name: str, k: int = 5, max_history: int = 3):
    print("Enter 'exit' to quit chat.")
    client = get_chroma_client(persist_dir)
    # We'll pick a collection dynamically each query
    model = _get_embed_model(model_name)
    history: List[Tuple[str, str]] = []  # (user_query, model_answer)
    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue
        # Determine collection via embedding similarity of query vs collection labels
        col_names = list_collections(client)
        print("Available collections:", col_names)
        if not col_names:
            print("(no collections found; ingest data first)")
            continue
        # First attempt: ask LLM to pick relevant collections (may return multiple)
        selected_collections = _select_collections_via_llm(q, col_names)
        if selected_collections:
            print("LLM selected collections:", selected_collections)
        else:
            print("LLM collection selection failed or returned empty; falling back to embedding similarity.")

        q_emb_vec = model.encode(q, show_progress_bar=False)
        q_emb = q_emb_vec.tolist()

        results: List[Dict] = []
        if selected_collections:
            # Retrieve from each selected collection and merge
            merged: List[Dict] = []
            for cname in selected_collections:
                part = similarity_search(client, cname, q_emb, k=k)
                for r in part:
                    if 'collection' not in r['metadata']:
                        r['metadata']['collection'] = cname
                merged.extend(part)
            # Rank globally by distance
            merged.sort(key=lambda x: x.get('distance', 1e9))
            results = merged[:k]
        else:
            # Fallback: previous single/broadcast approach
            chosen, best_sim, scores = _select_collection_for_query_embedded(q_emb_vec, col_names, model)
            if not chosen:
                print("Could not map query to a collection; available:", col_names)
                continue
            multi_mode = best_sim is not None and best_sim < COLLECTION_SIM_THRESHOLD
            if multi_mode:
                print(f"Best similarity {best_sim:.4f} < {COLLECTION_SIM_THRESHOLD:.2f}; using ALL collections for merged retrieval.")
                merged = []
                for cname in col_names:
                    part = similarity_search(client, cname, q_emb, k=k)
                    for r in part:
                        if 'collection' not in r['metadata']:
                            r['metadata']['collection'] = cname
                    merged.extend(part)
                merged.sort(key=lambda x: x.get('distance', 1e9))
                results = merged[:k]
            else:
                results = similarity_search(client, chosen, q_emb, k=k)
        if not results:
            print("(no retrieval results)")
            continue
        context_block = _build_context_block(results)
        messages = _make_messages(q, context_block, history, max_history=max_history)
        print("→ Calling Cloudflare AI...")
        answer = _call_cloudflare_ai(messages)
        print("\n=== Model Answer ===")
        print(answer)
        print("\n=== Retrieved Chunks ===")
        for i, r in enumerate(results, start=1):
            meta = r["metadata"]
            print(f"[{i}] {meta.get('ticker')} ({meta.get('company_name')}) {meta.get('year')} chunk {meta.get('chunk_index')} dist={r['distance']:.4f} collection={meta.get('collection')}")
        print()
        # Full chunk texts used for the answer
        print("=== Retrieved Chunk Texts (Full) ===")
        for i, r in enumerate(results, start=1):
            meta = r["metadata"]
            header = f"[Chunk {i}] {meta.get('ticker')} {meta.get('year')} chunk={meta.get('chunk_index')} id={r.get('id','')}"
            print(header)
            print(r.get("document", "").strip())
            print("---")
        print()
        # store in history
        history.append((q, answer))


def _select_collection_for_query_embedded(query_emb: np.ndarray, collections: List[str], model: SentenceTransformer):
    """Return (best_collection, best_similarity, scores).

    scores is a list of tuples (name, similarity, distance).
    """
    if len(collections) == 1:
        print(f"Only one collection '{collections[0]}' available -> selected")
        return collections[0], 1.0, [(collections[0], 1.0, 0.0)]
    scores = []
    q_norm = np.linalg.norm(query_emb) + 1e-12
    for name in collections:
        label = name
        label_emb = model.encode(label, show_progress_bar=False)
        sim = float(np.dot(query_emb, label_emb) / (q_norm * (np.linalg.norm(label_emb) + 1e-12)))
        dist = 1.0 - sim
        scores.append((name, sim, dist))
    print("Collection similarity (higher similarity is better):")
    for name, sim, dist in sorted(scores, key=lambda x: x[1], reverse=True):
        print(f"  {name}: similarity={sim:.4f} distance={dist:.4f}")
    best = max(scores, key=lambda x: x[1]) if scores else None
    if not best:
        return None, None, scores
    print(f"→ Best collection candidate: {best[0]} (similarity={best[1]:.4f})")
    return best[0], best[1], scores


def _select_collections_via_llm(user_query: str, collections: List[str]) -> List[str]:
    """Use the LLM to select one or more relevant collections.

    Returns a list of collection names (may be empty if parsing fails).
    """
    if not collections:
        return []
    if len(collections) == 1:
        return collections
    system = (
        "You are a routing assistant. You receive a user financial / SEC filings query and a list of collections, "
        "each collection corresponds to a ticker or dataset bucket. Return ONLY a JSON array of the collection names "
        "that should be searched (no extra text). If unsure, include multiple likely relevant collections."
    )
    user = (
        "User Query: " + user_query + "\n\nAvailable Collections (strings):\n" +
        "\n".join(f"- {c}" for c in collections) +
        "\n\nRespond with a JSON array, e.g.: [\"google-2023\", \"nvda-2024\"]."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    raw = _call_cloudflare_ai(messages)
    # Attempt direct JSON parse first
    try:
        full = json.loads(raw)
        # If the model returned an object with a 'response' field containing list
        if isinstance(full, dict) and isinstance(full.get('response'), list):
            candidate = full.get('response')
            if all(isinstance(x, str) for x in candidate):
                seen = set()
                cleaned = []
                for item in candidate:
                    if item in collections and item not in seen:
                        cleaned.append(item)
                        seen.add(item)
                if cleaned:
                    return cleaned
        # If the root itself is a list
        if isinstance(full, list) and all(isinstance(x, str) for x in full):
            seen = set()
            cleaned = []
            for item in full:
                if item in collections and item not in seen:
                    cleaned.append(item)
                    seen.add(item)
            if cleaned:
                return cleaned
    except Exception:
        # Fallback to bracket extraction below
        pass

    # Fallback: scan for first JSON-like array substring
    try:
        start = raw.find('[')
        end = raw.rfind(']')
        if start != -1 and end != -1 and end > start:
            snippet = raw[start:end+1]
            parsed = json.loads(snippet)
            if isinstance(parsed, list):
                seen = set()
                cleaned = []
                for item in parsed:
                    if isinstance(item, str) and item in collections and item not in seen:
                        cleaned.append(item)
                        seen.add(item)
                if cleaned:
                    return cleaned
    except Exception:
        pass
    return []


__all__ = ["chat_loop"]
