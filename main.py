from pathlib import Path
import os
import json
from typing import Dict
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from build_embeddings import find_raw_text_files, read_files, stream_item_embeddings
from vector_store import get_chroma_client, upsert_single, similarity_search, get_or_create_collection
from ai_chat import chat_loop

MODEL_NAME = "google/embeddinggemma-300m"
CHUNK_SIZE = 400
OVERLAP = 100
PERSIST_DIR = "chroma_store"
MANIFEST_FILE = Path("data/processed_files.json")
MAX_SNIPPET_CHARS = 400  # how many characters of each chunk to show in chat output
HF_TOKEN="HuggingFace_Token"

def ensure_hf_login():
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN",HF_TOKEN)
    if token:
        try:
            login(token=token)
            print("🔐 Hugging Face login successful (token detected).")
        except Exception as e:
            print(f"⚠️ HF login failed (continuing if model is public): {e}")
    else:
        print("ℹ️ No HUGGINGFACEHUB_API_TOKEN env var; proceeding without explicit login.")


def build_and_store():
    ensure_hf_login()
    raw_paths = find_raw_text_files(Path("data/raw"))
    print(f"Found {len(raw_paths)} raw text files.")

    manifest = _load_manifest()
    items = read_files(raw_paths)
    if not items:
        print("No valid items found. Exiting.")
        return
    print(f"Processing {len(items)} filings with streaming embedding...")
    model = SentenceTransformer(MODEL_NAME)
    client = get_chroma_client(PERSIST_DIR)
    total = 0
    skipped = 0
    for item_idx, item in enumerate(items, start=1):
        path = item["path"]
        fp = _file_fingerprint(Path(path))
        filename_stem = Path(path).stem  # collection name equals filename (without extension)
        if manifest.get(path) == fp:
            print(f"⏭️  [{item_idx}/{len(items)}] {filename_stem} already processed (skip)")
            skipped += 1
            continue
        coll_name = filename_stem
        cname = item.get('company_name') or item['ticker']
        print(f"→ [{item_idx}/{len(items)}] file='{filename_stem}' ticker={item['ticker']} ({cname}) year={item['year']} -> collection '{coll_name}'")
        chunk_count = 0
        for emb_vec, meta in stream_item_embeddings(item, model, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
            # attach source path & filename for traceability
            meta = dict(meta)
            meta["source_path"] = path
            meta["collection"] = coll_name
            upsert_single(client, coll_name, emb_vec, meta)
            chunk_count += 1
            total += 1
            print(f"   {chunk_count} chunks for {filename_stem} (global {total})")
                
        print(f"✔ Stored {chunk_count} chunks for {filename_stem}")
        # persist manifest update after each file to be crash-resistant
        manifest[path] = fp
        _save_manifest(manifest)
    print(f"✅ Completed embedding. Total chunks: {total}. Skipped files: {skipped}.")


def _load_manifest() -> Dict[str, str]:
    if MANIFEST_FILE.exists():
        try:
            return json.loads(MANIFEST_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_manifest(data: Dict[str, str]):
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_FILE.write_text(json.dumps(data, indent=2))


def _file_fingerprint(p: Path) -> str:
    try:
        st = p.stat()
        return f"{st.st_size}-{int(st.st_mtime)}"
    except FileNotFoundError:
        return "missing"


def chat():  # legacy wrapper to keep menu unchanged
    ensure_hf_login()
    chat_loop(PERSIST_DIR, MODEL_NAME)


def main():
    while True:
        print("\nRAG Pipeline Menu:")
        print("1. Store new files to vector db")
        print("2. Chat")
        print("3. Exit")
        choice = input("Select option: ").strip()
        if choice == "1":
            build_and_store()
        elif choice == "2":
            chat()
        elif choice == "3":
            print("Bye.")
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()