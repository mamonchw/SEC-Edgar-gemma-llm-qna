from __future__ import annotations

from pathlib import Path
from typing import Generator, Iterable, List, Dict, Tuple
import re


RAW_EXTENSIONS = {".txt", ".md"}


def find_raw_text_files(base: Path) -> List[Path]:
    files: List[Path] = []
    if not base.exists():
        return files
    for p in base.iterdir():
        if p.is_file() and p.suffix.lower() in RAW_EXTENSIONS:
            files.append(p)
    return sorted(files)


FILENAME_RE = re.compile(r"^(?P<ticker>[A-Za-z0-9_-]+)[-_](?P<year>\d{4})\.(?:txt|md)$")

# Extendable mapping of ticker -> pretty company name
TICKER_NAME_OVERRIDES = {
    "GOOGL": "Alphabet Inc.",
    "GOOG": "Alphabet Inc.",
    "GOOGLE": "Alphabet Inc.",
}


def read_files(paths: Iterable[Path]) -> List[Dict]:
    items: List[Dict] = []
    for p in paths:
        m = FILENAME_RE.match(p.name)
        if not m:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        ticker = m.group("ticker").upper()
        company_name = TICKER_NAME_OVERRIDES.get(ticker, ticker.title())
        items.append({
            "path": str(p),
            "ticker": ticker,
            "company_name": company_name,
            "year": m.group("year"),
            "text": text,
        })
    return items


def _split_on_paragraphs(text: str) -> List[str]:
    # Normalize newlines, split on blank lines
    paras = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paras if p.strip()]


def chunk_text(text: str, chunk_size: int, overlap: int) -> Generator[str, None, None]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    # --- Sentence segmentation helpers ---
    sentence_split_re = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9])')

    def split_sentences(paragraph: str) -> List[str]:
        paragraph = paragraph.strip()
        if not paragraph:
            return []
        # First collapse excessive whitespace
        paragraph = re.sub(r'\s+', ' ', paragraph)
        parts = sentence_split_re.split(paragraph)
        # Keep punctuation; regex splits after punctuation boundaries
        return [p.strip() for p in parts if p.strip()]

    paragraphs = _split_on_paragraphs(text)
    sentences: List[str] = []
    for para in paragraphs:
        sentences.extend(split_sentences(para))

    current_tokens: List[str] = []
    current_sentences: List[str] = []
    current_len = 0

    def emit_chunk():
        if not current_sentences:
            return None
        chunk_text_out = " ".join(current_sentences)
        return chunk_text_out

    i = 0
    while i < len(sentences):
        sent = sentences[i]
        sent_tokens = sent.split()
        sent_len = len(sent_tokens)

        # If a single sentence is extremely long, we may need to hard-split it.
        if sent_len > chunk_size * 1.5:
            # Hard split inside sentence at chunk_size boundaries but try to break at commas/semicolons first.
            tokens = sent_tokens
            start = 0
            while start < len(tokens):
                end = min(start + chunk_size, len(tokens))
                piece = tokens[start:end]
                yield " ".join(piece)
                if overlap > 0 and end < len(tokens):
                    # Overlap based on tokens for intra-sentence splits
                    ov_slice = tokens[end - overlap:end] if end - overlap > start else tokens[start:end]
                start = end
            i += 1
            continue

        if current_len + sent_len > chunk_size and current_sentences:
            # Emit current full-sentence chunk
            emitted = emit_chunk()
            if emitted:
                yield emitted
            # Prepare next chunk with sentence-level overlap
            if overlap > 0 and current_tokens:
                # Reconstruct overlap sentences from tail tokens
                # We'll include as many whole sentences from the end whose cumulative tokens <= overlap
                overlap_sents: List[str] = []
                running = 0
                for s in reversed(current_sentences):
                    slen = len(s.split())
                    if running + slen <= overlap:
                        overlap_sents.append(s)
                        running += slen
                    else:
                        break
                overlap_sents.reverse()
                current_sentences = overlap_sents + [sent]
                current_tokens = []
                for s in current_sentences:
                    current_tokens.extend(s.split())
                current_len = len(current_tokens)
            else:
                current_sentences = [sent]
                current_tokens = sent_tokens[:]
                current_len = sent_len
        else:
            current_sentences.append(sent)
            current_tokens.extend(sent_tokens)
            current_len += sent_len
        i += 1

    if current_sentences:
        emitted = emit_chunk()
        if emitted:
            yield emitted


def stream_item_embeddings(item: Dict, model, chunk_size: int, overlap: int) -> Generator[Tuple[List[float], Dict], None, None]:
    for idx, chunk in enumerate(chunk_text(item["text"], chunk_size=chunk_size, overlap=overlap)):
        emb = model.encode(chunk, show_progress_bar=False).tolist()
        meta = {
            "ticker": item["ticker"],
            "company_name": item.get("company_name"),
            "year": item["year"],
            "chunk_index": idx,
            "text": chunk,
        }
        yield emb, meta


__all__ = [
    "find_raw_text_files",
    "read_files",
    "stream_item_embeddings",
    "chunk_text",
]
