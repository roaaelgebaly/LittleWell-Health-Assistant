import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import google.generativeai as genai
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_TEMPERATURE,
    LLM_BATCH_MAX_PARAGRAPHS,
    SKIP_SECTION_TITLES,
    REPEATED_LINE_MIN_OCCURRENCES,
    REPEATED_LINE_MAX_CHARS,
    PAGE_NUMBER_LINE_PATTERN,
)

EXTRACTED_DIR = Path("./extracted")
OUTPUT_DIR = Path("./chunks")
OUTPUT_DIR.mkdir(exist_ok=True)

CHUNK_SIZE = 3000  # target chunk size in characters; actual chunks may be smaller or larger
CHUNK_OVERLAP = 500  # only used by the character-based fallback splitter

HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

PAGE_MARKER_RE = re.compile(r"<!--\s*page:(\d+)\s*-->")

# Matches runs of dot-leaders / replacement characters produced when a PDF's
# table of contents uses a font whose "." glyph maps outside standard
# unicode (very common in WHO/UN-style reports). e.g. "�����������v"
NOISE_RUN_RE = re.compile(r"[\ufffd\u2022\u00b7.]{4,}")

PAGE_NUMBER_LINE_RE = re.compile(PAGE_NUMBER_LINE_PATTERN, re.IGNORECASE)

# Clinical/dosing keywords that should trigger safety redirection.
# Catches: dosing terms (mg/kg, ml/kg, dose), routes of administration (IV, IM, SC),
# medical procedures (lumbar puncture, intubation), and other clinical interventions.
# Used as a safety net at both ingestion (chunking) and generation (nodes) time.
CLINICAL_KEYWORD_RE = re.compile(
    r"\b(?:"
    r"mg/kg|ml/kg|mcg/kg|ug/kg|"  # dosing patterns
    r"dose|dosage|doses|dosing|"  # dosing terms
    r"IV|IM|SC|IV infusion|intravenous|intramuscular|subcutaneous|"  # routes
    r"injection|injections|inject|injected|"  # injection terms
    r"lumbar puncture|spinal tap|"  # invasive procedures
    r"intubation|intubate|intubated|"  # airway procedures
    r"catheter|catheterize|catheterization|"  # catheter procedures
    r"cannula|cannulation|"  # cannulation
    r"dialysis|hemodialysis|peritoneal dialysis|"  # dialysis
    r"bypass|transfusion|"  # major interventions
    r"surgery|surgical|"  # surgery
    r"medication|medicines?|drug therapy|pharmacological|"  # medication terms
    r"therapeutic|treatment regimen|"  # treatment terms
    r"antibiotic|antiviral|antifungal|corticosteroid|steroid"  # specific drugs
    r")\b",
    re.IGNORECASE
)

MIN_CHUNK_CHARS = 40

# If more than this fraction of a chunk's characters are non-alphanumeric
# (punctuation, digits, replacement chars) it's almost certainly a
# table-of-contents / index / boilerplate line rather than real content.
MAX_NON_ALNUM_RATIO = 0.35

# ---------------------------------------------------------------------------
# LLM-based chunking config (Gemini)
# ---------------------------------------------------------------------------
# Instead of blindly cutting oversized sections at a character count, we ask
# an LLM to group paragraphs into topically-coherent chunks. The LLM only
# ever returns paragraph INDICES (never rewritten text), so the actual chunk
# text is always reconstructed verbatim from the source document -- no risk
# of hallucinated or altered content. Oversized sections are processed in
# batches of LLM_BATCH_MAX_PARAGRAPHS so a huge, badly-subheaded section
# (e.g. one section spanning 100+ pages) can't blow up a single prompt.
LLM_CHUNKING_ENABLED = True
LLM_MAX_RETRIES = 2
LLM_PARAGRAPH_PREVIEW_CHARS = 600  # only the preview is sent to the LLM, not full text
LLM_OVERSIZED_GROUP_MULTIPLIER = 1.5  # safety-net threshold vs CHUNK_SIZE

if LLM_CHUNKING_ENABLED:
    genai.configure(api_key=GEMINI_API_KEY)
    _client = genai.GenerativeModel(GEMINI_MODEL)
else:
    _client = None

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=HEADERS_TO_SPLIT_ON,
    strip_headers=False,
)


def extract_page_number(text: str) -> Optional[int]:
    """Pull the first page marker found in a piece of text, if any."""
    match = PAGE_MARKER_RE.search(text)
    return int(match.group(1)) if match else None


def strip_repeated_lines(md_text: str) -> str:
    """Remove lines that repeat verbatim many times across the document --
    these are almost always running headers/footers (org name, report title,
    page numbers) injected by the PDF extractor on every page, not content.
    Headings (lines starting with '#') are never touched so document
    structure is preserved."""
    lines = md_text.split("\n")

    counts = Counter(
        line.strip() for line in lines
        if line.strip()
        and not line.strip().startswith("#")
        and len(line.strip()) <= REPEATED_LINE_MAX_CHARS
    )

    noisy_lines = {
        line for line, count in counts.items()
        if count >= REPEATED_LINE_MIN_OCCURRENCES
    }

    return "\n".join(line for line in lines if line.strip() not in noisy_lines)


def clean_text(text: str) -> str:
    """Remove page markers, dot-leader/replacement-char noise, standalone
    page-number lines, and collapse the resulting extra whitespace."""
    text = PAGE_MARKER_RE.sub("", text)
    text = NOISE_RUN_RE.sub(" ", text)
    text = "\n".join(
        line for line in text.split("\n")
        if not PAGE_NUMBER_LINE_RE.match(line.strip())
    )
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_noise(text: str, header_path: str) -> bool:
    """Heuristic filter for chunks that are table-of-contents entries,
    indexes, references, or other non-content noise that shouldn't be
    embedded."""
    if len(text) < MIN_CHUNK_CHARS:
        return True

    last_header = header_path.split(">")[-1].strip().lower() if header_path else ""
    if last_header in SKIP_SECTION_TITLES:
        return True

    alnum = sum(c.isalnum() and not c.isdigit() for c in text)
    non_alnum = len(text) - alnum
    if len(text) and (non_alnum / len(text)) > MAX_NON_ALNUM_RATIO:
        return True

    return False


# ---------------------------------------------------------------------------
# LLM-based topic grouping
# ---------------------------------------------------------------------------

def split_into_paragraphs(text: str) -> list[str]:
    """Split section text into paragraphs on blank lines."""
    paragraphs = re.split(r"\n\s*\n", text)
    return [p for p in paragraphs if p.strip()]


def _parse_llm_groups(raw: str) -> Optional[list]:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def llm_group_paragraphs(paragraphs: list[str], header_path: str) -> Optional[list[list[int]]]:
    """Ask the LLM to group paragraph indices into topically-coherent chunks.

    `paragraphs` is expected to already be at most LLM_BATCH_MAX_PARAGRAPHS
    long (batching happens in the caller). Returns None on failure (bad
    JSON, missing/duplicate/reordered indices, API error) so the caller
    falls back to character-based splitting. The LLM never sees or returns
    chunk text -- only indices -- so the source text can't be altered or
    hallucinated.
    """
    if not paragraphs:
        return None

    numbered = "\n\n".join(
        f"[{i}] {p[:LLM_PARAGRAPH_PREVIEW_CHARS]}" for i, p in enumerate(paragraphs)
    )

    prompt = f"""You are grouping paragraphs from a document section into topically coherent chunks for a RAG pipeline.

Section heading path: {header_path or '(none)'}

Paragraphs (index: preview text, some previews may be truncated):
{numbered}

Rules:
- Group CONSECUTIVE paragraph indices only (never reorder, never skip a paragraph out of order).
- Each group should cover one coherent sub-topic or a natural continuation of thought -- don't split a sub-topic across two groups if you can avoid it.
- Don't target {CHUNK_SIZE} characters per group based on paragraph lengths, prioritize topical coherence over hitting the exact size.
- Every paragraph index from 0 to {len(paragraphs) - 1} must appear in exactly one group, in ascending order, with no gaps or duplicates.

Respond with ONLY a JSON array of arrays of integers, e.g. [[0,1,2],[3,4],[5,6,7]]. No prose, no markdown fences, no explanation."""

    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            resp = _client.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
                    temperature=GEMINI_TEMPERATURE,
                ),
            )
            raw_text = resp.text
            groups = _parse_llm_groups(raw_text)

            if groups is not None:
                flat = [i for group in groups for i in group]
                if flat == list(range(len(paragraphs))):
                    return groups
                print(f"      [llm-chunk] index validation failed (attempt {attempt + 1}/{LLM_MAX_RETRIES + 1})")
            else:
                print(f"      [llm-chunk] JSON parse failed (attempt {attempt + 1}/{LLM_MAX_RETRIES + 1})")
        except Exception as e:
            print(f"      [llm-chunk] API error (attempt {attempt + 1}/{LLM_MAX_RETRIES + 1}): {e}")
            time.sleep(1)

    return None


def batch_paragraphs(paragraphs: list[str], batch_size: int) -> list[list[str]]:
    """Split paragraphs into fixed-size batches so a huge, badly-subheaded
    section never produces an oversized LLM prompt regardless of document
    length (e.g. a 500-page PDF with one giant un-subheaded section)."""
    return [paragraphs[i:i + batch_size] for i in range(0, len(paragraphs), batch_size)]


def llm_split_section(section_text: str, header_path: str) -> list[str]:
    """Split an oversized section into topic-coherent chunks using the LLM to
    decide paragraph groupings, processing the section in fixed-size batches.
    Falls back to the character-based splitter per-batch if a batch can't be
    usefully paragraph-split, or if the LLM call fails/returns invalid
    output -- so the pipeline never breaks and cost/latency stay bounded no
    matter how large the source document is."""
    paragraphs = split_into_paragraphs(section_text)

    if len(paragraphs) < 2:
        return recursive_splitter.split_text(section_text)

    batches = batch_paragraphs(paragraphs, LLM_BATCH_MAX_PARAGRAPHS)
    all_chunks: list[str] = []

    for batch_idx, batch in enumerate(batches):
        if len(batch) < 2:
            all_chunks.extend(recursive_splitter.split_text("\n\n".join(batch)))
            continue

        groups = llm_group_paragraphs(batch, header_path)
        if groups is None:
            print(f"    [llm-chunk] batch {batch_idx + 1}/{len(batches)} failed, falling back to character split")
            all_chunks.extend(recursive_splitter.split_text("\n\n".join(batch)))
            continue

        for group in groups:
            chunk = "\n\n".join(batch[i] for i in group)
            # Safety net: if the LLM produced a group that's still oversized
            # (e.g. a few very long paragraphs), character-split it further.
            if len(chunk) > CHUNK_SIZE * LLM_OVERSIZED_GROUP_MULTIPLIER:
                all_chunks.extend(recursive_splitter.split_text(chunk))
            else:
                all_chunks.append(chunk)

    return all_chunks


def chunk_document(md_text: str, source_name: str) -> list[dict]:
    """Apply the hybrid chunking strategy to one document's markdown text:
    strip repeated headers/footers, split on headers, then split any
    oversized section by topic (via LLM, in batches) rather than by raw
    character count."""
    md_text = strip_repeated_lines(md_text)
    sections = markdown_splitter.split_text(md_text)

    final_chunks = []
    last_known_page: Optional[int] = None  # carried forward across sections

    for section in sections:
        page_num = extract_page_number(section.page_content) or last_known_page
        header_path = " > ".join(
            v for k, v in section.metadata.items() if k in ("h1", "h2", "h3")
        )

        # Skip known-noise sections outright, before spending any LLM calls on them.
        last_header = header_path.split(">")[-1].strip().lower() if header_path else ""
        if last_header in SKIP_SECTION_TITLES:
            continue

        if len(section.page_content) > CHUNK_SIZE:
            if LLM_CHUNKING_ENABLED:
                sub_chunks = llm_split_section(section.page_content, header_path)
            else:
                sub_chunks = recursive_splitter.split_text(section.page_content)
        else:
            sub_chunks = [section.page_content]

        for i, sub in enumerate(sub_chunks):
            sub_page = extract_page_number(sub) or page_num
            if sub_page is not None:
                last_known_page = sub_page

            cleaned = clean_text(sub)
            if not cleaned or is_noise(cleaned, header_path):
                continue

            final_chunks.append(
                {
                    "text": cleaned,
                    "metadata": {
                        "source": source_name,
                        "section": header_path or None,
                        "page": sub_page,
                        "chunk_index": i,
                    },
                }
            )

    return final_chunks


def chunk_all_documents():
    md_files = sorted(EXTRACTED_DIR.glob("*.md"))
    if not md_files:
        print(f"No extracted .md files found in {EXTRACTED_DIR.resolve()}")
        print("Run extract_text_from_pdfs.py first.")
        return

    all_chunks = []
    for md_path in md_files:
        print(f"Chunking: {md_path.name}")
        md_text = md_path.read_text(encoding="utf-8")
        doc_chunks = chunk_document(md_text, source_name=md_path.stem)
        print(f"  -> {len(doc_chunks)} chunks")
        all_chunks.extend(doc_chunks)

    out_path = OUTPUT_DIR / "all_chunks.json"
    out_path.write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nTotal chunks across all documents: {len(all_chunks)}")
    print(f"Saved to {out_path.resolve()}")


if __name__ == "__main__":
    chunk_all_documents()
