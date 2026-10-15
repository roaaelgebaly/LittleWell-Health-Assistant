import json
from pathlib import Path

import pymupdf

PDF_DIR = Path("D:\projects\Med_RAG\Data")

OUTPUT_DIR = Path("./extracted")
OUTPUT_DIR.mkdir(exist_ok=True)


def get_body_font_size(page_dict: dict) -> float:
    """Estimate the most common (body text) font size on a page,
    weighted by character count."""
    sizes: dict[float, int] = {}
    for block in page_dict.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span["text"].strip()
                if not text:
                    continue
                size = round(span["size"], 1)
                sizes[size] = sizes.get(size, 0) + len(text)
    if not sizes:
        return 10.0
    return max(sizes, key=sizes.get)


def detect_heading_level(font_size: float, body_size: float) -> int:
    """Classify a text block as heading level 1-3, or 0 for normal text,
    based on how much larger its font size is than the body text size."""
    ratio = font_size / body_size if body_size else 1
    if ratio >= 1.6:
        return 1
    elif ratio >= 1.35:
        return 2
    elif ratio >= 1.15:
        return 3
    return 0


def page_to_markdown(page, page_num: int) -> str:
    """Convert a single PDF page to a markdown-ish string, promoting
    large-font lines to '#' headings so the structure survives extraction."""
    page_dict = page.get_text("dict")
    body_size = get_body_font_size(page_dict)

    lines_out = [f"<!-- page:{page_num} -->"]
    for block in page_dict.get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            text = "".join(s["text"] for s in spans).strip()
            if not text:
                continue
            avg_size = sum(s["size"] for s in spans) / len(spans)
            level = detect_heading_level(avg_size, body_size)
            if level:
                lines_out.append(f"\n{'#' * level} {text}\n")
            else:
                lines_out.append(text)

    return "\n".join(lines_out)


def extract_pdf(pdf_path: Path) -> str:
    """Extract all pages of one PDF into a single markdown string."""
    doc = pymupdf.open(pdf_path)
    pages_md = [page_to_markdown(doc[i], i + 1) for i in range(doc.page_count)]
    doc.close()
    return "\n\n".join(pages_md)


def extract_all_pdfs():
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {PDF_DIR.resolve()}")
        print("Set PDF_DIR at the top of this file to your folder of 6 PDFs.")
        return

    manifest = []
    for pdf_path in pdf_files:
        print(f"Extracting: {pdf_path.name}")
        md_text = extract_pdf(pdf_path)

        out_path = OUTPUT_DIR / f"{pdf_path.stem}.md"
        out_path.write_text(md_text, encoding="utf-8")

        manifest.append(
            {
                "source_pdf": pdf_path.name,
                "extracted_md": out_path.name,
                "chars": len(md_text),
            }
        )
        print(f"  -> {out_path} ({len(md_text)} chars)")

    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nDone. Extracted {len(pdf_files)} PDFs into {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    extract_all_pdfs()
