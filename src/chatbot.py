"""
Step 8: GenAI Chatbot - RAG over bank PDFs using Groq + FAISS.
Embeddings run locally (sentence-transformers); generation via Groq API.
"""
import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq

DOCS_DIR = "data/bank_docs"
INDEX_DIR = "models/faiss_index"

_embeddings = None
_vectorstore = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _embeddings


def build_index():
    """Load all PDFs, chunk, embed, and save a FAISS index."""
    pdfs = glob.glob(f"{DOCS_DIR}/*.pdf")
    if not pdfs:
        raise FileNotFoundError(
            f"No PDFs found in {DOCS_DIR}. Add at least one bank PDF.")
    docs = []
    for path in pdfs:
        loaded = PyPDFLoader(path).load()
        # tag each chunk with its source filename for nicer citations
        for d in loaded:
            d.metadata["source_file"] = os.path.basename(path)
        docs.extend(loaded)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    if not chunks:
        raise ValueError(
            "PDFs loaded but no extractable text found. "
            "The PDF may be scanned/image-based (needs OCR).")

    vs = FAISS.from_documents(chunks, get_embeddings())
    vs.save_local(INDEX_DIR)
    # reset cached store so a rebuild is picked up
    global _vectorstore
    _vectorstore = None
    return len(pdfs), len(chunks)


def index_exists():
    return os.path.exists(os.path.join(INDEX_DIR, "index.faiss"))


def load_index():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = FAISS.load_local(
            INDEX_DIR, get_embeddings(),
            allow_dangerous_deserialization=True)
    return _vectorstore


def list_groq_models(api_key):
    """Fetch currently available Groq chat models (avoids hardcoding stale names)."""
    try:
        client = Groq(api_key=api_key)
        models = client.models.list()
        names = [m.id for m in models.data
                 if not any(x in m.id.lower()
                            for x in ["whisper", "tts", "guard"])]
        return sorted(names) if names else ["llama-3.3-70b-versatile"]
    except Exception:
        return ["llama-3.3-70b-versatile"]


def answer(question, api_key, model="llama-3.3-70b-versatile", k=4,
           history=None):
    """Retrieve relevant chunks and generate a grounded answer.

    history: optional list of {"role","content"} dicts for multi-turn chat.
    """
    vs = load_index()
    hits = vs.similarity_search(question, k=k)
    context = "\n\n".join(
        f"[{h.metadata.get('source_file','doc')} p.{h.metadata.get('page','?')}]\n"
        f"{h.page_content}" for h in hits)

    system = (
        "You are a helpful, polite bank customer-service assistant for "
        "Meridian Bank. Answer the customer's question using ONLY the context "
        "from the bank's documents provided below. If the answer is not in the "
        "context, clearly say you don't have that information in the provided "
        "documents and suggest contacting customer care. Be concise and "
        "accurate. Do not invent figures."
    )
    user_msg = f"Context from bank documents:\n{context}\n\nQuestion: {question}"

    messages = [{"role": "system", "content": system}]
    if history:
        messages.extend(history[-6:])  # keep last few turns for context
    messages.append({"role": "user", "content": user_msg})

    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=700,
    )
    return resp.choices[0].message.content, hits