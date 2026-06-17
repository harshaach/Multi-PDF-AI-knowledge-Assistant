import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from google import genai
import faiss
import numpy as np
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-PDF AI Knowledge Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Multi-PDF AI Knowledge Assistant")
st.caption("Upload PDFs and ask questions in plain English. Powered by Google Gemini + FAISS.")

# ── Constants ─────────────────────────────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 3
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL = "models/gemini-2.5-flash"

# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_text_from_pdfs(pdf_files) -> str:
    """Extract and concatenate text from a list of uploaded PDF files."""
    all_text = ""
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                all_text += page_text + "\n"
    return all_text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i : i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks


@st.cache_resource
def load_embed_model():
    return SentenceTransformer(EMBED_MODEL_NAME)


def build_faiss_index(chunks: list[str], embed_model) -> tuple:
    """Embed chunks and build a FAISS index."""
    embeddings = embed_model.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings


def retrieve_chunks(question: str, index, chunks: list[str], embed_model, k: int = TOP_K) -> list[str]:
    """Retrieve the top-k most relevant chunks for a question."""
    question_embedding = embed_model.encode([question])
    question_embedding = np.array(question_embedding).astype("float32")
    _, indices = index.search(question_embedding, k=k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]


def ask_gemini(question: str, context_chunks: list[str], api_key: str) -> str:
    """Send question + context to Gemini and return the answer."""
    client = genai.Client(api_key=api_key)
    context = "\n\n".join(context_chunks)
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context provided below.
If the answer is not found in the context, say "I couldn't find relevant information in the uploaded documents."

Context:
{context}

Question:
{question}

Answer:"""
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return response.text


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Setup")

    api_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        placeholder="Paste your API key here",
        help="Get a free key at https://aistudio.google.com/app/apikey",
    )

    st.divider()
    st.header("📄 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    process_btn = st.button("🚀 Process PDFs", use_container_width=True, type="primary")

    st.divider()
    st.markdown("**Built by** [Harshavardhan Chilukuri](https://github.com/harshaach)")
    st.markdown("Stack: Python · Gemini · FAISS · Sentence Transformers · Streamlit")


# ── Session state ─────────────────────────────────────────────────────────────
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "docs_processed" not in st.session_state:
    st.session_state.docs_processed = False


# ── Process PDFs ──────────────────────────────────────────────────────────────
if process_btn:
    if not api_key:
        st.sidebar.error("Please enter your Gemini API key.")
    elif not uploaded_files:
        st.sidebar.error("Please upload at least one PDF.")
    else:
        with st.spinner("Reading and indexing your PDFs…"):
            embed_model = load_embed_model()
            raw_text = extract_text_from_pdfs(uploaded_files)
            if not raw_text.strip():
                st.error("Could not extract text from the uploaded PDFs. Please check if they contain selectable text.")
            else:
                chunks = chunk_text(raw_text)
                faiss_index, _ = build_faiss_index(chunks, embed_model)
                st.session_state.faiss_index = faiss_index
                st.session_state.chunks = chunks
                st.session_state.chat_history = []
                st.session_state.docs_processed = True
                st.sidebar.success(f"✅ Indexed {len(chunks)} chunks from {len(uploaded_files)} PDF(s).")


# ── Chat interface ────────────────────────────────────────────────────────────
if st.session_state.docs_processed:
    st.subheader("💬 Ask anything about your documents")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📎 View source chunks"):
                    for i, src in enumerate(message["sources"], 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.caption(src[:400] + "…" if len(src) > 400 else src)

    # User input
    question = st.chat_input("Type your question here…")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                embed_model = load_embed_model()
                relevant_chunks = retrieve_chunks(
                    question,
                    st.session_state.faiss_index,
                    st.session_state.chunks,
                    embed_model,
                )
                answer = ask_gemini(question, relevant_chunks, api_key)
                st.markdown(answer)
                with st.expander("📎 View source chunks"):
                    for i, src in enumerate(relevant_chunks, 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.caption(src[:400] + "…" if len(src) > 400 else src)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": relevant_chunks,
        })

else:
    # Placeholder when no docs processed yet
    st.info("👈 Upload your PDFs and click **Process PDFs** to get started.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Embedding Model", "all-MiniLM-L6-v2")
    with col2:
        st.metric("LLM", "Gemini 2.5 Flash")
    with col3:
        st.metric("Vector Search", "FAISS")
