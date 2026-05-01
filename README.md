# 📄 SmartDoc AI — Privacy-First AI Document Intelligence System

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co)
[![LangChain](https://img.shields.io/badge/LangChain-0.1%2B-1C3C3C)](https://langchain.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20DB-0052CC)](https://faiss.ai)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Privacy](https://img.shields.io/badge/100%25%20Local-Zero%20API%20Calls-green)](.)
[![UK GDPR](https://img.shields.io/badge/UK%20GDPR-Privacy%20by%20Design-00A86B)](https://ico.org.uk)
[![Licence: MIT](https://img.shields.io/badge/Licence-MIT-yellow.svg)](LICENSE)

---

## 📌 What This Project Does — In Plain English

Every day, organisations deal with enormous quantities of documents: reports, contracts, policies, research papers, meeting minutes, and correspondence. Finding information in these documents — or getting quick answers from them — is time-consuming.

**SmartDoc AI** is an intelligent document assistant that:

1. **Accepts any text document** (PDF, TXT, or demo documents)
2. **Summarises it automatically** — giving you the key points in seconds
3. **Answers your questions** — you type a question, the AI finds and returns the relevant answer from the document
4. **Extracts key terms** — automatically identifies the most important topics
5. **Analyses sentiment** — tells you whether the document's tone is positive, negative, or neutral
6. **Does all of this locally** — your documents never leave your computer. No data is sent to any external server.

The "100% local" aspect is the critical differentiator: most AI document tools (ChatGPT, Claude, Gemini) send your document to a cloud server. This application uses locally downloaded models that run entirely within your own machine — meaning confidential business documents, legal files, or sensitive reports can be processed safely.

---

## 🎯 Why This Project Matters for a Career in AI/ML

| Skill Demonstrated | Why It Matters |
|---|---|
| RAG (Retrieval-Augmented Generation) | The most in-demand LLM engineering skill in UK enterprise AI |
| Transformer models (BART, BERT, DistilBERT) | Shows you understand modern NLP architecture, not just API calls |
| Vector databases (FAISS) | Core infrastructure for all production RAG systems |
| LangChain framework | Used in 60%+ of enterprise LLM applications |
| Privacy-by-design | Critical differentiator — most candidates ignore this |
| Extractive + abstractive summarisation | Shows depth — understanding both approaches, not just one |
| Local LLM deployment | Growing skill requirement as enterprises move away from cloud-only AI |

---

## 🏗️ System Architecture — How It All Works Together

Understanding this architecture is essential for interviews. Here is every step explained:

```
┌─────────────────────────────────────────────────────────────┐
│                    SMARTDOC AI ARCHITECTURE                   │
└─────────────────────────────────────────────────────────────┘

STAGE 1: DOCUMENT INGESTION
User uploads a PDF or text file
          │
          ▼
Text extraction:
  • PDF → PyPDF2 extracts raw text from each page
  • TXT → Read directly
  • Demo documents → Pre-loaded sample texts
          │
          ▼
STAGE 2: CHUNKING
Split document into overlapping segments
  • Chunk size: 512 tokens (~380 words)
  • Overlap: 64 tokens (prevents losing context at boundaries)
  • Why overlap? If an answer spans two chunks, the overlap
    ensures it's captured in at least one chunk's context
          │
          ▼
STAGE 3: EMBEDDING GENERATION
Convert each text chunk into a mathematical vector
  • Model: all-MiniLM-L6-v2 (22M parameters, 384 dimensions)
  • What is an embedding? A list of 384 numbers that represents
    the meaning of the text — similar texts have similar vectors
  • This runs locally — no text sent externally
          │
          ▼
STAGE 4: VECTOR INDEX (FAISS)
Store all chunk embeddings in a searchable index
  • FAISS (Facebook AI Similarity Search)
  • Enables extremely fast similarity search across all chunks
  • Index lives in RAM — cleared when session ends
          │
          ─────────── INDEXING COMPLETE ───────────
          
STAGE 5: QUESTION ANSWERING (at query time)
User types: "What were the Q3 revenue figures?"
          │
          ▼
Embed the question using the same all-MiniLM-L6-v2 model
          │
          ▼
Search FAISS index for top-5 most similar chunks
(Cosine similarity between question vector and chunk vectors)
          │
          ▼
Retrieve the 5 most relevant chunks from the document
          │
          ▼
Feed chunks + question to the answering model
(deepset/roberta-base-squad2 for extractive Q&A)
          │
          ▼
Return answer with source context shown
          │
          ▼
STAGE 6: SUMMARISATION (separate pipeline)
  • Extractive: Score sentences by TF-IDF importance, return top sentences
  • Abstractive: Pass document to facebook/bart-large-cnn, generate new text
          │
          ▼
OUTPUT: Answer, summary, keywords, sentiment — all displayed in Streamlit
```

---

## 🧠 Models Used — Explained Simply

### 1. all-MiniLM-L6-v2 (Sentence Transformers)

**What it is:** A compact version of the BERT architecture, trained specifically to produce high-quality sentence embeddings.

**Why it was chosen:**
- Only 22 million parameters (tiny by modern standards)
- Produces 384-dimensional embeddings — small enough to store efficiently
- Excellent retrieval performance on semantic similarity benchmarks
- Fast enough to embed hundreds of chunks in seconds on a standard laptop CPU
- Free, open-source, runs locally

**What it does in this project:** Converts every chunk of text and every user question into a vector of 384 numbers. The key insight: two pieces of text that mean the same thing will have similar vectors, even if they use different words. This allows the system to find relevant chunks even when the user doesn't use the exact words in the document.

**Example:** The chunk says "quarterly earnings increased by 12%" and the user asks "did revenue go up?". Even though "earnings" and "revenue" are different words, their embeddings are close — so this chunk will be retrieved.

### 2. facebook/bart-large-cnn (Abstractive Summarisation)

**What it is:** BART (Bidirectional and Auto-Regressive Transformer) — an encoder-decoder model pre-trained to reconstruct corrupted text. Fine-tuned on the CNN/DailyMail news summarisation dataset.

**Why it was chosen:** BART-large-cnn is the most widely used model for abstractive summarisation in English. It generates fluent, concise summaries that read naturally — not disconnected bullet points.

**What "abstractive" means:** Unlike extractive summarisation (which copies sentences directly from the document), abstractive summarisation generates new text that captures the meaning. Like a human summarising a document in their own words, it can combine ideas from different sentences and rephrase for clarity.

**Trade-off:** More fluent output, but slight risk of hallucination — the model may occasionally generate text that sounds right but isn't in the document. For critical use cases (legal, medical), extractive summarisation is safer.

### 3. deepset/roberta-base-squad2 (Extractive Q&A)

**What it is:** RoBERTa (Robustly Optimised BERT) fine-tuned on SQuAD2.0 — a dataset of 150,000 question-answer pairs from Wikipedia.

**What it does:** Given a passage of text and a question, it identifies the span of text within the passage that best answers the question. It returns an exact quote from the source text — guaranteed to be factually grounded.

**Why this matters for GDPR/trust:** Extractive Q&A is inherently auditable. The answer is a direct quote from a specific location in the document. The user can verify it themselves.

### 4. distilbert-base-uncased-finetuned-sst-2-english (Sentiment)

**What it is:** DistilBERT — a compressed version of BERT (40% smaller, 97% of the performance) fine-tuned on SST-2 (Stanford Sentiment Treebank).

**What it does:** Classifies text as POSITIVE or NEGATIVE with a confidence score. Applied sentence-by-sentence to the document, then aggregated for document-level sentiment.

**Limitation:** Trained on consumer reviews and social media — may not perform optimally on formal business or technical documents. For production, a domain-fine-tuned sentiment model would be more appropriate.

---

## 🔒 Privacy Architecture — Why 100% Local Matters

### The Problem with Cloud AI

When you paste a document into ChatGPT, Google Gemini, or any cloud-based AI:
- Your document text travels across the internet to a data centre
- It is stored (potentially temporarily) on servers you don't control
- The AI provider's terms of service govern how it can be used
- Under UK GDPR, the AI provider becomes a **data processor** — requiring a Data Processing Agreement
- If the document contains personal data, this transfer must be documented and justified

For a law firm, bank, NHS trust, or any business handling confidential information, this is a significant problem.

### How SmartDoc Avoids This

| Cloud AI Approach | SmartDoc AI Approach |
|---|---|
| Document sent to API server | Document stays in RAM, never leaves device |
| API key required | No API key — models run locally |
| Data processed in US/EU data centres | Data processed on your hardware |
| Provider subject to data requests | No provider — no third party |
| DPA required under GDPR | No DPA required |
| Potential data retention by provider | Session ends → data cleared from RAM |

---

## 🔒 UK GDPR Compliance Documentation

### Privacy by Design (Article 25)

This system implements the strongest possible privacy controls at the architecture level:

| Privacy Principle | Technical Implementation |
|---|---|
| **Data minimisation** | Only the document provided is processed; no metadata collected |
| **Purpose limitation** | Processing occurs solely for the user-requested analysis task |
| **Storage limitation** | All data cleared from RAM when the browser session ends |
| **Integrity & confidentiality** | No external transmission; no persistent storage |
| **Accountability** | Processing occurs entirely under the user's control |

### When GDPR Applies to SmartDoc Usage

**GDPR does NOT apply when:**
- Analysing publicly available documents (reports, papers, policies)
- Analysing synthetic or anonymised data
- Personal use for non-commercial document processing

**GDPR DOES apply when:**
- Analysing documents containing personal data (employee records, customer correspondence, patient notes)
- Operating as part of an organisational system affecting employees or customers
- Processing special category data (health, legal, financial)

### For Enterprise Deployment

When deployed in an organisational context:

1. **Records of Processing Activities (Article 30):** Document this system as a processing activity in your organisation's ROPA
2. **Privacy Notice (Article 13/14):** If employees' documents will be analysed, inform them in the privacy notice
3. **DPIA (Article 35):** Required if the system will process large volumes of personal data or special category data
4. **Data Controller Responsibility:** The organisation deploying this tool is the data controller — responsible for all GDPR obligations
5. **Access Controls:** Implement role-based access — not all employees should be able to upload all document types

---

## 🚀 Quick Start — Run Locally

### System Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.11 |
| RAM | 8 GB | 16 GB |
| Storage | 5 GB free | 10 GB free |
| Internet (first run) | Required to download models | Not needed after first run |

The first time you run the app, it will automatically download the four HuggingFace models (~2–3 GB total). After that, models are cached locally and no internet is needed.

### Step-by-Step Setup

```bash
# 1. Clone the project
git clone https://github.com/[your-username]/smartdoc-ai.git
cd smartdoc-ai

# 2. Create virtual environment
python -m venv smartdoc_env

# Windows activation:
smartdoc_env\Scripts\activate

# Mac/Linux activation:
source smartdoc_env/bin/activate

# 3. Install dependencies (will take 5-10 minutes first time)
pip install -r requirements_smartdoc.txt

# 4. Run the app
streamlit run smartdoc_app.py
```

The app opens at `http://localhost:8501`.

**First-time note:** When you first click "Summarise" or "Ask a Question", the app will download the required models. You will see a progress message — this can take 5–15 minutes depending on your internet speed. After that, models are cached and load instantly.

---

## ☁️ Deployment — Make It Live

### Option A: Streamlit Community Cloud (Free)

**Important note on resources:** The SmartDoc AI models require significant RAM (~4 GB). Streamlit Community Cloud's free tier provides limited resources, which may cause the app to crash when loading large transformer models.

**Lightweight deployment option:** Deploy with only TF-IDF features enabled (extractive summarisation, keyword extraction, basic Q&A) and add a note that transformers are available for local deployment. This uses no HuggingFace models and runs on minimal resources.

```bash
# Push to GitHub first
git init
git add smartdoc_app.py requirements_smartdoc.txt README_SmartDoc.md .gitignore
git commit -m "Initial commit: SmartDoc AI document intelligence"
git remote add origin https://github.com/[your-username]/smartdoc-ai.git
git push -u origin main

# Then deploy at share.streamlit.io
```

### Option B: Hugging Face Spaces (Recommended — More Memory)

Hugging Face Spaces provides more RAM and is better suited for transformer-based apps.

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Choose **Streamlit** as the SDK
3. Rename files: `smartdoc_app.py` → `app.py`, `requirements_smartdoc.txt` → `requirements.txt`
4. Upload all files
5. The build takes 10–15 minutes (downloading and caching models)

### Option C: Docker with Persistent Model Cache

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements_smartdoc.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models during build (so users don't wait)
RUN python -c "
from sentence_transformers import SentenceTransformer
from transformers import pipeline
SentenceTransformer('all-MiniLM-L6-v2')
pipeline('summarization', model='facebook/bart-large-cnn')
"

COPY smartdoc_app.py app.py

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

This pre-downloads models during the Docker build, so users get instant startup.

```bash
docker build -t smartdoc-ai .
docker run -p 8501:8501 smartdoc-ai
```

---

## 📁 File Structure

```
project3_smartdoc_ai/
│
├── smartdoc_app.py               ← Main Streamlit application
│   ├── Document upload and text extraction
│   ├── TF-IDF extractive summarisation
│   ├── BART abstractive summarisation
│   ├── RAG Q&A pipeline (FAISS + RoBERTa)
│   ├── Keyword extraction
│   ├── Sentence-level sentiment analysis
│   ├── Document analytics dashboard
│   ├── Model deep-dive explanations
│   └── Privacy & GDPR page
│
├── requirements_smartdoc.txt     ← All dependencies
├── README_SmartDoc.md            ← This file
├── DEPLOY_SmartDoc.md            ← Extended deployment guide
├── GDPR_SmartDoc.md              ← Standalone GDPR document
└── .gitignore                    ← Git exclusions
```

---

## 🧪 Interview Preparation for This Project

**Q: "What is RAG and why did you use it instead of fine-tuning a model on your documents?"**

> "RAG — Retrieval-Augmented Generation — combines a retrieval system with a language model. Instead of training the model to memorise the document contents, we retrieve relevant passages at query time and feed them as context to the model. I chose RAG over fine-tuning for three reasons. First, documents change — a fine-tuned model would need to be retrained every time the knowledge base updates, whereas RAG just needs the index updated. Second, RAG provides source attribution — you can show which part of the document the answer came from, which is critical for trust and auditability. Third, fine-tuning requires labelled question-answer pairs, which are expensive to create; RAG works without any labelled data."

**Q: "What is the difference between extractive and abstractive summarisation?"**

> "Extractive summarisation selects and returns the most important existing sentences from the document, without changing any words. The output is always factually accurate because it's a direct quote from the source. Abstractive summarisation generates new text that captures the meaning — like a human paraphrasing. It produces more fluent, readable output but carries a small risk of hallucination — introducing information that wasn't in the original document. In SmartDoc, I implemented both: extractive as the default for accuracy, with abstractive available via BART when readability is the priority."

**Q: "You said the app is 100% local — how does that actually work technically?"**

> "The HuggingFace Transformers library allows you to download and cache model weights locally. The first time a user runs a model, it downloads the weights from HuggingFace's servers and caches them in a local directory. Every subsequent use loads the weights from disk and runs inference entirely on the local CPU or GPU — no internet connection required and no data transmitted. FAISS, the vector database, operates entirely in memory — it builds the index from the document's embeddings and stores it in RAM for the duration of the session. When the session ends, everything is cleared."

---

## 📜 Licence

MIT Licence — free to use, modify, and distribute with attribution.

*All HuggingFace models used are subject to their respective licences: BART (MIT), MiniLM (Apache 2.0), RoBERTa (MIT), DistilBERT (Apache 2.0). All are free for commercial use.*
