"""
Phase 2 - Knowledge base ingestion pipeline.
Step 2.3: load the Markdown files into LangChain Document objects.
Step 2.4: split them into topic-sized chunks with context headers.
Step 2.5: embed the chunks and store them in a Chroma vector database.

Run this file whenever the knowledge base documents change.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

load_dotenv()

# ---------- Settings ----------
KB_DIR = Path(__file__).parent / "knowledge_base"
HEADERS = [("#", "title"), ("##", "section")]

# Vector database location: from .env, falling back to a project folder
CHROMA_DIR = os.getenv("CHROMA_DIR", str(Path(__file__).parent / "chroma_db"))
COLLECTION_NAME = "it_helpdesk_kb"
EMBEDDING_MODEL = "text-embedding-3-small"   # must be the SAME for indexing and searching


# ---------- Step 2.3: load ----------
def load_documents(folder: Path = KB_DIR) -> list[Document]:
    """Read every .md file in the folder into a Document with its file name as metadata."""
    docs = []
    for path in sorted(folder.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        docs.append(Document(page_content=text, metadata={"source": path.name}))
    return docs


# ---------- Step 2.4: chunk ----------
def split_documents(docs: list[Document]) -> list[Document]:
    """Split each document at its headings, then cap chunk size. Adds a context header to every chunk."""
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS, strip_headers=True)
    size_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    chunks = []
    for doc in docs:
        for section in header_splitter.split_text(doc.page_content):
            if section.page_content.startswith("> SAMPLE DOCUMENT"):
                continue

            title = section.metadata.get("title", "")
            heading = section.metadata.get("section", "")
            context = f"{title} > {heading}" if heading else title

            for piece in size_splitter.split_text(section.page_content):
                chunks.append(
                    Document(
                        page_content=f"{context}\n\n{piece}",
                        metadata={"source": doc.metadata["source"], "title": title, "section": heading},
                    )
                )
    return chunks


# ---------- Step 2.5: embed + store ----------
def get_vector_store() -> Chroma:
    """Open the Chroma collection (used by both ingest.py and the agent)."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=CHROMA_DIR,
    )


def build_vector_store(chunks: list[Document]) -> Chroma:
    """Rebuild the collection from scratch so re-running never creates duplicates."""
    store = get_vector_store()
    store.reset_collection()   # wipe old vectors first (idempotent rebuild)
    ids = [f"{c.metadata['source']}::{i}" for i, c in enumerate(chunks)]
    store.add_documents(chunks, ids=ids)   # embeds every chunk via OpenAI, then saves
    return store


if __name__ == "__main__":
    documents = load_documents()
    chunks = split_documents(documents)
    print(f"{len(documents)} documents -> {len(chunks)} chunks")

    store = build_vector_store(chunks)
    stored = len(store.get()["ids"])
    print(f"Stored {stored} chunks in collection '{COLLECTION_NAME}'")
    print(f"Database location: {CHROMA_DIR}")