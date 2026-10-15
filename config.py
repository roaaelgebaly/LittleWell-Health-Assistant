import os
from pathlib import Path
from typing import Optional

import groq
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")  
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "med_rag_chunks"
EMBED_MODEL_PATH = "Qwen/Qwen3-Embedding-0.6B"
TOP_K = 10
GENERATION_CONTEXT_MAX_CHARS = 8000
# --- models ---
GROQ_CLASSIFY_MODEL = "allam-2-7b"
GROQ_GENERATE_MODEL = "allam-2-7b"
GEMINI_MODEL ="gemini-2.5-flash"
GEMINI_MAX_OUTPUT_TOKENS = 1024 
GEMINI_TEMPERATURE = 0  

# --- shared clients (lazy singletons) ---
_groq_client: Optional[groq.Groq] = None
_embed_model: Optional[SentenceTransformer] = None
_collection = None


def get_groq_client() -> groq.Groq:
    """Only raises if something that actually needs Groq (classify/generate)
    calls this -- retrieval-only scripts never touch it, so they don't need
    GROQ_API_KEY set at all."""
    global _groq_client
    if _groq_client is None:
        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set.\n"
                "  .env file:   put GROQ_API_KEY=your-key-here in the project .env file\n"
                "  PowerShell:  $env:GROQ_API_KEY=\"your-key-here\"\n"
                "  bash/zsh:    export GROQ_API_KEY=your-key-here"
            )
        _groq_client = groq.Groq(api_key=GROQ_API_KEY)
    return _groq_client

SKIP_SECTION_TITLES = {
    "contents", "table of contents", "list of figures", "list of tables",
    "index", "references", "bibliography", "acknowledgements", "acknowledgments",
    "abbreviations", "list of abbreviations", "glossary", "foreword", "preface",
    "copyright", "disclaimer", "about this document", "annex",
}

REPEATED_LINE_MIN_OCCURRENCES = 5
REPEATED_LINE_MAX_CHARS = 80  
PAGE_NUMBER_LINE_PATTERN = r"^\s*[-–]?\s*(?:page\s*)?\d{1,4}\s*(?:of\s*\d{1,4})?\s*[-–]?\s*$"
ENABLE_CITATION_LINE_FILTER = True
CITATION_LINE_MAX_CHARS = 400
CITATION_LINE_PATTERN = (
    r"^\s*(?:"
    r"(?:\[\d{1,3}\]|\d{1,3}\.)\s+[A-Z].{0,250}?(?:19|20)\d{2}[a-z]?\)?[\.,;].{0,60}"
    r"|"
    r"[A-Z][A-Za-z\-']+,\s*[A-Z]\.(?:\s?[A-Z]\.)?"
    r"(?:,\s*(?:&|and)?\s*[A-Z][A-Za-z\-']+,\s*[A-Z]\.(?:\s?[A-Z]\.)?)*"
    r"(?:,?\s*et al\.?)?"
    r"\s*\(?(?:19|20)\d{2}[a-z]?\)?[\.,;]\s*.{0,180}"
    r")\s*$"
)
LLM_BATCH_MAX_PARAGRAPHS = 60


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_PATH, trust_remote_code=True)
    return _embed_model


def get_collection():
    global _collection
    if _collection is None:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = chroma_client.get_collection(COLLECTION_NAME)
    return _collection
