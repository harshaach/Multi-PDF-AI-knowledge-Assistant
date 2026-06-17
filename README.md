# Multi-PDF-AI-knowledge-Assistant
A production-ready RAG (Retrieval-Augmented Generation) application that lets you upload multiple PDFs and ask questions in plain English — powered by Google Gemini 2.5 Flash and FAISS vector search.
---
## ✨ Features
- 📄 Upload and query **multiple PDFs** simultaneously
- 🔍 **Semantic search** using FAISS + Sentence Transformers
- 🤖 **Context-aware answers** via Google Gemini 2.5 Flash
- 💬 **Chat interface** with full conversation history
- 📎 **Source citations** — see exactly which chunk the answer came from
- ⚡ Fast retrieval with overlapping chunking strategy
---
## 🏗️ Architecture
---
PDF Upload → Text Extraction (pypdf)
          → Chunking (1000 chars, 200 overlap)
          → Embeddings (all-MiniLM-L6-v2)
          → FAISS Index
          → Query → Top-3 Chunks → Gemini 2.5 Flash → Answer

---
## 🚀 Run Locally
```bash
git clone https://github.com/harshaach/multi-pdf-ai-assistant
cd multi-pdf-ai-assistant
pip install -r requirements.txt
streamlit run app.py
```

Then open your browser at `http://localhost:8501`

> Get a free Gemini API key at [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

---

## 🛠️ Tech Stack
| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector Search | FAISS |
| LLM | Google Gemini 2.5 Flash |
| PDF Parsing | pypdf |

---

## 📁 Project Structure

```
multi-pdf-ai-assistant/
│
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── .gitignore           # Files to ignore in git
```
---
## 🙋 How to Use

1. Open the live app
2. Paste your **Gemini API key** in the sidebar
3. Upload one or more **PDF files**
4. Click **Process PDFs**
5. Ask any question in the chat box
