import os
import hashlib

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from conf import get_project_root, get_agent_config
from utils.logger_handler import get_logger

"""
Vector store module.
Handles vectorizing and searching user-uploaded documents (recipes, nutrition guides, etc.).
Supports .txt / .pdf / .docx formats.
Note: files in local_privacy/ are NOT indexed here — they're accessed on demand via MCP.
"""

logger = get_logger("ai_chef.rag")

# Data paths
PROJECT_ROOT = get_project_root()
CHROMA_PERSIST_DIR = os.path.join(PROJECT_ROOT, "data", "chroma_db")
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "data", "uploads")

# Track which documents have been indexed to prevent duplicates
_INGESTED_HASHES_FILE = os.path.join(CHROMA_PERSIST_DIR, ".ingested_hashes")


def _load_ingested_hashes() -> set:
    """Load the set of hashes for already-indexed documents."""
    if os.path.exists(_INGESTED_HASHES_FILE):
        with open(_INGESTED_HASHES_FILE, "r") as f:
            return set(f.read().splitlines())
    return set()


def _save_ingested_hash(file_hash: str):
    """Save the hash of a newly indexed document."""
    os.makedirs(os.path.dirname(_INGESTED_HASHES_FILE), exist_ok=True)
    with open(_INGESTED_HASHES_FILE, "a") as f:
        f.write(file_hash + "\n")


def _file_hash(file_path: str) -> str:
    """Compute an MD5 hash of the file contents."""
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def is_document_ingested(file_path: str) -> bool:
    """
    Check if a file has already been indexed (for the UI to show a warning before re-uploading).
    Uses MD5 hash of the file content — same content = already indexed.
    """
    if not os.path.exists(file_path):
        return False
    return _file_hash(file_path) in _load_ingested_hashes()


def get_embeddings_model():
    """Initialize the text embedding model (DashScope / Qwen)."""
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY is missing from environment variables!")

    config = get_agent_config()
    return DashScopeEmbeddings(
        dashscope_api_key=api_key,
        model=config["llm"]["embedding_model"]
    )


def get_vector_store():
    """Get (or initialize) the Chroma vector database instance."""
    config = get_agent_config()
    embeddings = get_embeddings_model()
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=config["rag"]["collection_name"]
    )


def _get_text_splitter():
    """Get a text splitter configured from agent_config.yaml."""
    config = get_agent_config()
    return RecursiveCharacterTextSplitter(
        chunk_size=config["rag"]["chunk_size"],
        chunk_overlap=config["rag"]["chunk_overlap"],
        separators=["\n\n", "\n", "。", "！", "？", "，", "、", " "]
    )


def search_knowledge(query: str, k: int = None):
    """Search interface for the RAG core or agent to call."""
    if k is None:
        config = get_agent_config()
        k = config["rag"]["retrieval_top_k"]

    vector_store = get_vector_store()
    return vector_store.similarity_search(query, k=k)


def ingest_single_document(file_path: str) -> bool:
    """
    Accept a single file (txt / pdf / docx), split it, and add it to the vector database.
    Has built-in deduplication: same-content files won't be indexed twice.

    Returns:
        True: indexed successfully (or already existed, skipped)
        False: indexing failed
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False

    # Dedup check
    fhash = _file_hash(file_path)
    if fhash in _load_ingested_hashes():
        logger.info(f"File already indexed, skipping: {os.path.basename(file_path)}")
        return True

    logger.info(f"Parsing and indexing: {os.path.basename(file_path)}")

    try:
        # Pick the right loader based on file extension
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == '.txt':
            loader = TextLoader(file_path, autodetect_encoding=True)
        elif ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext == '.docx':
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            logger.error(f"Unsupported file format: {ext}")
            return False

        documents = loader.load()
        split_docs = _get_text_splitter().split_documents(documents)

        vector_store = get_vector_store()
        vector_store.add_documents(split_docs)

        _save_ingested_hash(fhash)
        logger.info(f"'{os.path.basename(file_path)}' indexed successfully! Generated {len(split_docs)} chunks.")
        return True

    except Exception as e:
        logger.error(f"Indexing failed: {str(e)}")
        return False


if __name__ == "__main__":
    print("=== Vector store test ===")

    test_query = "soups good for rainy weather"
    print(f"Searching: '{test_query}'")
    docs = search_knowledge(test_query, k=1)
    if docs:
        print(f"Found (source: {docs[0].metadata.get('source')}):")
        print(docs[0].page_content)
    else:
        print("No relevant content found.")
