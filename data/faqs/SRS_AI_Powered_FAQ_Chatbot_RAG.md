# Software Requirements Specification (SRS)

## AI-Powered FAQ Chatbot using RAG (Retrieval-Augmented Generation)

**Version:** 1.0
**Date:** August 2026
**Prepared For:** Phase 2 — Requirements Analysis
**Document Status:** Draft

---

## 1. Introduction

### 1.1 Purpose
This Software Requirements Specification (SRS) document describes the functional and non-functional requirements for the **AI-Powered FAQ Chatbot using RAG**, a web-based conversational AI application. The system allows users to upload documents (PDFs, FAQs, text files) and interact with an AI assistant that retrieves relevant information from those documents and generates accurate, context-aware answers using a Large Language Model (LLM).

This document serves as a reference for developers, testers, and project stakeholders during the design, development, and testing phases of the project.

### 1.2 Scope
The system is a **Retrieval-Augmented Generation (RAG)** based chatbot that:
- Accepts document uploads (PDF, TXT, FAQ files) from users.
- Processes and chunks document content.
- Converts text chunks into vector embeddings using Sentence Transformers.
- Stores and retrieves embeddings using a FAISS vector database.
- Uses the Google Gemini API (via LangChain) to generate natural language answers grounded in retrieved context.
- Presents an interactive chat interface built with Streamlit.

The system is intended for students, customers, employees, and organizations who need quick, accurate answers from large or complex sets of documents without manually searching through them.

### 1.3 Intended Audience
- Development Team (Backend, AI/ML Engineers)
- Project Guide / Reviewer / Evaluator
- QA / Testing Team
- Future maintainers of the system

### 1.4 Definitions, Acronyms, and Abbreviations

| Term | Description |
|---|---|
| RAG | Retrieval-Augmented Generation |
| LLM | Large Language Model |
| API | Application Programming Interface |
| FAISS | Facebook AI Similarity Search (vector database) |
| Embedding | Numerical vector representation of text |
| Chunking | Splitting large text into smaller segments |
| SRS | Software Requirements Specification |
| UI/UX | User Interface / User Experience |

### 1.5 References
- Product Requirements Document (PRD) — Phase 1
- Project Proposal — Phase 1
- LangChain Documentation
- Google Gemini API Documentation
- FAISS Documentation
- Sentence Transformers Documentation

---

## 2. Overall Description

### 2.1 Product Perspective
The chatbot is a standalone web application built using Streamlit for the frontend and Python for backend logic. It is not part of a larger system but is designed to be modular so it can later be integrated into existing customer support portals, learning management systems, or internal knowledge bases.

### 2.2 Product Functions (Summary)
- Document upload and ingestion (PDF/TXT/FAQ)
- Text extraction and chunking
- Embedding generation and vector storage (FAISS)
- Semantic search / retrieval of relevant chunks
- Context-aware answer generation via Gemini LLM
- Conversational chat interface with history
- Source/citation display for transparency
- Session-based or persistent knowledge base management

### 2.3 User Classes and Characteristics

| User Class | Description | Technical Expertise |
|---|---|---|
| Students | Ask questions from study material/notes | Low |
| Customers | Ask product/service-related queries | Low |
| Employees | Query internal documents/policies | Low–Medium |
| Organizations/Admins | Upload and manage knowledge base documents | Medium |

### 2.4 Operating Environment
- **Client Side:** Any modern web browser (Chrome, Edge, Firefox)
- **Server Side:** Python 3.x runtime environment
- **Deployment:** Local machine, cloud VM, or platforms like Streamlit Cloud / Render / AWS / GCP

### 2.5 Design and Implementation Constraints
- Dependent on third-party Gemini API availability, rate limits, and pricing.
- Retrieval quality is limited by the embedding model's capability.
- FAISS index is primarily in-memory; persistence must be explicitly managed.
- Requires an active internet connection for LLM API calls.

### 2.6 Assumptions and Dependencies
- Users have access to a valid Google Gemini API key.
- Uploaded documents are in a readable, non-corrupted format.
- The system assumes English-language documents by default (multilingual support is a future enhancement).
- Third-party libraries (LangChain, FAISS, Sentence Transformers) remain compatible with the chosen Python version.

---

## 3. Functional Requirements

### FR-1: Document Upload
The system shall allow users to upload one or more documents in PDF or TXT format through the Streamlit interface.

### FR-2: Text Extraction
The system shall extract raw text content from uploaded PDFs using `pypdf`.

### FR-3: Text Chunking
The system shall split extracted text into smaller overlapping chunks suitable for embedding, using a configurable chunk size and overlap.

### FR-4: Embedding Generation
The system shall convert each text chunk into a vector embedding using a Sentence Transformers model.

### FR-5: Vector Storage
The system shall store generated embeddings, along with metadata (source document, chunk text), in a FAISS vector index.

### FR-6: User Query Input
The system shall provide a chat input box where users can type natural language questions.

### FR-7: Semantic Retrieval
The system shall retrieve the top-k most relevant chunks from the FAISS index based on the semantic similarity between the user query and stored embeddings.

### FR-8: Context-Aware Answer Generation
The system shall pass the retrieved chunks along with the user query to the Gemini LLM (via LangChain) to generate a grounded, natural language response.

### FR-9: Chat History Display
The system shall display the ongoing conversation (user queries and bot responses) in a scrollable chat window.

### FR-10: Source Attribution (Optional/Enhanced)
The system shall optionally display the source document/chunk used to generate each answer, for transparency.

### FR-11: Error Handling
The system shall display a user-friendly error message if:
- Document upload fails or file format is unsupported.
- The Gemini API call fails or times out.
- No relevant context is found for a query.

### FR-12: Session Management
The system shall maintain the current session's chat history and uploaded knowledge base until the session ends or is reset.

### FR-13: Reset/Clear Knowledge Base
The system shall allow users to clear the uploaded documents and start a new session.

### FR-14: Configuration via Environment Variables
The system shall load sensitive configuration (e.g., API keys) from a `.env` file using `python-dotenv`, and shall not hard-code credentials.

---

## 4. Non-Functional Requirements

### NFR-1: Performance
- The system should return an answer within **5–10 seconds** under normal load (dependent on Gemini API latency).
- Document ingestion and embedding generation for a standard-sized PDF (under 20 pages) should complete within **30 seconds**.

### NFR-2: Usability
- The chat interface shall be simple and intuitive, requiring no technical training for end users.
- The system shall provide visual feedback (loading spinners) during document processing and answer generation.

### NFR-3: Reliability
- The system shall handle malformed or unreadable documents gracefully without crashing.
- The system shall retry or gracefully fail on transient API errors.

### NFR-4: Scalability
- The architecture shall support scaling the knowledge base to handle multiple documents without major redesign (e.g., increasing FAISS index size).

### NFR-5: Security
- API keys and sensitive credentials shall not be exposed in source code or client-side logs.
- Uploaded documents shall be handled securely and not persisted beyond the intended session unless explicitly configured.

### NFR-6: Maintainability
- The codebase shall follow a modular structure (separate modules for ingestion, embedding, retrieval, and generation) to ease future updates.

### NFR-7: Portability
- The application shall run on any OS (Windows, macOS, Linux) supporting Python 3.x and the required dependencies.

### NFR-8: Availability
- The system shall be available whenever the hosting environment and Gemini API service are operational (subject to third-party SLA).

---

## 5. User Requirements

- Users must be able to upload documents without technical assistance.
- Users must receive clear, accurate answers grounded in the uploaded content (not hallucinated).
- Users must be able to see previous questions and answers in the same session.
- Users must be informed when the system cannot find a relevant answer, rather than receiving a fabricated response.
- Organizations/admins must be able to manage (add/remove) documents in the knowledge base.

---

## 6. Hardware Requirements

| Component | Minimum Requirement | Recommended |
|---|---|---|
| Processor | Dual-core 2 GHz | Quad-core 2.5 GHz+ |
| RAM | 4 GB | 8 GB or higher |
| Storage | 1 GB free space | 5 GB+ (for larger document sets) |
| Internet Connection | Required (for API calls) | Stable broadband connection |
| GPU | Not required | Optional, for faster local embedding generation |

---

## 7. Software Requirements

| Component | Requirement |
|---|---|
| Operating System | Windows 10+/macOS/Linux |
| Programming Language | Python 3.9 or higher |
| Frontend Framework | Streamlit |
| LLM Provider | Google Gemini API |
| Embedding Library | Sentence Transformers |
| Vector Database | FAISS |
| PDF Processing | pypdf |
| Orchestration Framework | LangChain |
| Environment Config | python-dotenv |
| Version Control | Git / GitHub |
| IDE (Development) | VS Code / PyCharm (or equivalent) |

---

## 8. Constraints

- The system relies on the Google Gemini API; usage costs and rate limits apply.
- Answer quality is directly dependent on the quality and coverage of uploaded documents.
- FAISS, by default, does not persist data automatically across restarts unless explicitly saved to disk.
- The project timeline (Agile, intermediate level) limits scope to core RAG functionality; advanced features (multi-user auth, analytics dashboards) are out of MVP scope.

## 9. Assumptions

- Users will upload documents relevant to the questions they intend to ask.
- A single Gemini API key will be used per deployment instance (no per-user billing in MVP).
- The system will primarily be used for English-language content in the initial version.
- Internet connectivity will be available at all times during system usage.

---

## 10. Appendix

### 10.1 Traceability to PRD
This SRS directly expands upon the objectives, features, and MVP scope defined in the Phase 1 PRD and Project Proposal, translating business/product-level goals into specific, testable functional and non-functional requirements.

### 10.2 Next Steps (Phase 3)
Following approval of this SRS, the project shall proceed to Phase 3 — System Design, including:
- System Architecture Document
- Software Design Document (SDD)
- Database / Knowledge Base Design
- API Design Document
- Prompt Engineering Document
- UI/UX Design Document

---
*End of Software Requirements Specification (SRS)*
