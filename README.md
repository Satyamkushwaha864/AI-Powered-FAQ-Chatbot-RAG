# AI-Powered FAQ Chatbot using RAG

A Streamlit chatbot that answers questions grounded in your uploaded documents using **Retrieval-Augmented Generation (RAG)**: local Sentence Transformers embeddings, a FAISS vector store, and the Google Gemini API for answer generation.

## Architecture

```
app.py (Streamlit orchestration)
   │
   ├──▶ ingestion/loader.py      (extract text from PDF/TXT)
   │      └──▶ ingestion/chunker.py  (split into overlapping chunks)
   ├──▶ embeddings/embedder.py   (Sentence Transformers vectors)
   ├──▶ vectorstore/faiss_store.py (FAISS index + metadata store)
   ├──▶ llm/prompt_builder.py    (assemble grounded prompt)
   │      └──▶ llm/gemini_client.py  (Gemini API with retry/backoff)
   └──▶ config.py                (env-based configuration)
```

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your API key
copy .env.example .env       # then edit .env and add your GEMINI_API_KEY
```

## Run

```bash
streamlit run app.py
```

Then open http://localhost:8501, upload PDF/TXT documents in the sidebar, and ask questions.

## Tests

```bash
pytest tests/
```

## Project Structure

| Path | Responsibility |
|---|---|
| [app.py](app.py) | Streamlit entry point; wires modules and session state |
| [config.py](config.py) | Environment/config loading and validation |
| [ingestion/](ingestion/) | Document loading and text chunking |
| [embeddings/](embeddings/) | Sentence Transformers embedding generation |
| [vectorstore/](vectorstore/) | FAISS index + metadata store |
| [llm/](llm/) | Prompt templates, prompt builder, Gemini client |
| [ui/](ui/) | Sidebar (upload/KB status) and chat view |
| [utils/](utils/) | Logging and custom exception hierarchy |
| [tests/](tests/) | Unit tests for core modules |
