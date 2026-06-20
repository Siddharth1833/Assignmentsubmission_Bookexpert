# Document Intelligence Hub (DocMind AI)

A professional RAG (Retrieval-Augmented Generation) chatbot powered by advanced LLMs that answers questions directly from your PDF documents. Upload your documents and get intelligent, context-aware answers with source citations in seconds.

## Tech Stack

| Technology | Version | Purpose |
|---|---|---|
| **Streamlit** | 1.28.1 | Web UI framework for interactive dashboard |
| **Python** | 3.8+ | Core programming language |
| **Groq API** | 1.4.0 | LLM inference API (Llama 3.3 70B model) |
| **ChromaDB** | 1.5.9 | Vector database for storing and retrieving embeddings |
| **Sentence-Transformers** | 5.6.0 | Embedding model (all-MiniLM-L6-v2) |
| **python-dotenv** | 1.2.2 | Environment variable management |
| **PyPDF** | (via pypdf) | PDF text extraction |

## Architecture Overview

### RAG Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│                      DOCUMENT PIPELINE                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. INGESTION          2. CHUNKING       3. EMBEDDING       │
│  ┌──────────────┐      ┌──────────────┐  ┌──────────────┐   │
│  │ Load PDFs    │─────>│ Split into   │─>│ Generate     │   │
│  │ from data/   │      │ chunks       │  │ embeddings   │   │
│  │ folder       │      │ (500 char)   │  │ using        │   │
│  └──────────────┘      └──────────────┘  │ all-MiniLM   │   │
│                                           │ L6-v2        │   │
│                                           └──────────────┘   │
│                                                   │          │
│                            ┌──────────────────────┘          │
│                            │                                 │
│                            v                                 │
│                    ┌──────────────────┐                      │
│                    │   ChromaDB       │                      │
│                    │  (Vector Store)  │                      │
│                    └──────────────────┘                      │
│                            ^                                 │
└─────────────────────────────┼──────────────────────────────┘
                              │
┌─────────────────────────────┼──────────────────────────────┐
│                   QUERY PIPELINE                           │
├─────────────────────────────┼──────────────────────────────┤
│                             │                              │
│  1. USER QUERY      2. EMBEDDING     3. RETRIEVAL          │
│  ┌──────────────┐   ┌──────────────┐ ┌──────────────┐      │
│  │ Enter        │──>│ Convert to   │>│ Query        │      │
│  │ Question     │   │ embedding    │ │ ChromaDB &   │      │
│  │             │   │ using same   │ │ get similar  │      │
│  └──────────────┘   │ model        │ │ chunks       │      │
│                     └──────────────┘ └──────────────┘      │
│                                              │              │
│                            ┌─────────────────┘              │
│                            │                               │
│                            v                               │
│  4. GENERATION       5. ANSWER + SOURCES                   │
│  ┌──────────────┐    ┌──────────────────┐                 │
│  │ Send context │───>│ Display to user  │                 │
│  │ + question   │    │ with citations   │                 │
│  │ to Llama 3.3 │    └──────────────────┘                 │
│  └──────────────┘                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Key Components:**

- **Ingestion Module** (`ingest.py`): Extracts text from PDF files
- **Indexing Module** (`index.py`): Creates embeddings and stores them in ChromaDB
- **Query Module** (`query.py`): CLI interface for querying the RAG system
- **LLM Module** (`llm.py`): Handles generation using Groq API
- **Web UI** (`app.py`): Streamlit-based interactive dashboard
- **Configuration** (`config.py`): Centralized settings and hyperparameters

## Chunking Strategy

**Strategy Used:** Sliding Window with Overlap

**Parameters:**
- **Chunk Size:** 500 characters
- **Overlap:** 100 characters (20% overlap)

**Why This Strategy?**

1. **Overlap Benefits:** Ensures that important information spanning chunk boundaries isn't lost
2. **Context Preservation:** The 20% overlap maintains semantic continuity between chunks
3. **Performance Balance:** 500 characters is a good middle ground between:
   - Too small: Would create too many chunks, increasing retrieval time
   - Too large: Would lose granularity and include irrelevant information
4. **Document Diversity:** Works well for PDFs with mixed content (tables, text, etc.)

The sliding window approach ensures queries find relevant information even when it spans across natural boundaries in the original document.

## Embedding Model and Vector Database

### Embedding Model: all-MiniLM-L6-v2

**Why This Model?**

- **Lightweight:** Only 22M parameters, runs efficiently on CPU
- **Fast:** Generates embeddings quickly without GPU requirement
- **Accurate:** Despite its size, achieves strong semantic understanding
- **Production-Ready:** Extensively tested and widely used in industry
- **Low Latency:** Real-time query embedding generation
- **Embedding Dimension:** 384 dimensions (compact representation)

**Trade-offs:**
- Slightly less accurate than larger models (384M embeddings)
- Well-suited for most enterprise document Q&A scenarios

### Vector Database: ChromaDB

**Why ChromaDB?**

- **Persistent Storage:** Stores embeddings permanently using SQLite backend
- **Simple API:** Minimal configuration, developer-friendly
- **Fast Retrieval:** Optimized similarity search on embeddings
- **Metadata Support:** Stores document source and page numbers for citation
- **Zero Setup:** Doesn't require external database server
- **Production Ready:** Used in real-world applications

**Configuration:**
- **Path:** `chroma_db/` (persistent on disk)
- **Collection:** "documents"
- **Retrieval:** Top-8 similar chunks with distance scores

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- A Groq API key (free tier available at [groq.com](https://groq.com))

### Step-by-Step Setup

#### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd siddharth
```

#### 2. Create Virtual Environment (Recommended)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Set Up Environment Variables
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your Groq API key
# Open .env in your text editor and replace:
# GROQ_API_KEY=your_groq_api_key_here
```

#### 5. Add Your PDF Documents
```bash
# Create a data folder if it doesn't exist
mkdir data

# Add your PDF files to the data/ folder
# Example:
# - data/document1.pdf
# - data/document2.pdf
# - data/your_research_paper.pdf
```

#### 6. Index Your Documents
```bash
# Run the indexing script to create embeddings
python src/index.py

# You should see output like:
# Loaded X pages
# Created Y chunks
# Generating embeddings...
# Embeddings generated
# Storing in ChromaDB...
# Indexing Complete
```

#### 7. Run the Application

**Option A: Web UI (Recommended)**
```bash
streamlit run src/app.py

# The app will open at http://localhost:8501
```

**Option B: CLI Interface**
```bash
python src/query.py

# Type your questions and press Enter
# Type "exit" to quit
```

## Environment Variables

### Required Variables

| Variable | Description | Example |
|---|---|---|
| `GROQ_API_KEY` | Your Groq API key for LLM access | `gsk_xxxxxxxxxxx...` |

**Important Security Notes:**
- ⚠️ **Never commit** `.env` file to Git
- `.env` is already in `.gitignore` to prevent accidental commits
- Always use `.env.example` as a template
- Rotate API keys regularly in production

### How to Get a Groq API Key

1. Visit [groq.com](https://groq.com)
2. Sign up for a free account
3. Navigate to API keys section
4. Create a new API key
5. Copy the key and add it to your `.env` file

## Example Queries

Here are 5+ sample questions you can ask the RAG bot:

### 1. Definitional Questions
**Query:** "What is machine learning?"
**Expected Answer Theme:** A clear, educational definition of machine learning, its types, and basic concepts covered in your documents

### 2. Explanation Questions
**Query:** "Explain how neural networks work in the context of the document."
**Expected Answer Theme:** Step-by-step explanation of neural network architecture, forward/backward propagation, or specific details from your documents

### 3. Comparative Questions
**Query:** "Compare supervised and unsupervised learning."
**Expected Answer Theme:** Structured comparison of both approaches with their differences, advantages, and use cases from your source material

### 4. Detailed Information Questions
**Query:** "What are the key benefits of using this technology?"
**Expected Answer Theme:** Detailed list of benefits with explanations and examples from the document context

### 5. Specific Details Questions
**Query:** "What parameters or settings were used in the experiment?"
**Expected Answer Theme:** Specific numerical values, configuration details, or technical specifications mentioned in your documents

### 6. Summary Questions
**Query:** "Summarize the main findings from this research."
**Expected Answer Theme:** Concise summary of key findings, results, and conclusions from your source documents

**Note:** The bot will respond with "I could not find this information in the provided documents" for questions that aren't covered in your uploaded PDFs. This is by design to ensure accuracy and prevent hallucination.

## Known Limitations

### 1. **Document-Only Knowledge**
   - **Limitation:** The bot cannot use external knowledge or the internet
   - **Why:** By design, to ensure accuracy and prevent hallucination
   - **Workaround:** Add any reference documents you want the bot to know about to the `data/` folder

### 2. **Context Window Constraints**
   - **Limitation:** Very large documents (100+ pages) may be chunked into small pieces, potentially losing some context
   - **Why:** Balancing retrieval speed with contextual accuracy
   - **Workaround:** Organize related documents together or use more specific queries

### 3. **Similarity Threshold**
   - **Limitation:** Questions very different from document content will be rejected (threshold: 1.2 distance score)
   - **Why:** Prevents the bot from making up answers for unrelated questions
   - **Workaround:** Rephrase your question using terminology from the documents

### 4. **PDF Format Limitations**
   - **Limitation:** Complex PDFs with images, tables, or special formatting may not extract text perfectly
   - **Why:** Using basic text extraction without OCR or advanced layout analysis
   - **Workaround:** Ensure PDFs are text-based; use OCR preprocessing for scanned documents

### 5. **Language Support**
   - **Limitation:** Currently optimized for English-language documents
   - **Why:** Embedding model and LLM are primarily trained on English text
   - **Workaround:** For other languages, consider using different embedding models

### 6. **Real-Time Updates**
   - **Limitation:** Adding new documents requires re-running the indexing script
   - **Why:** Embeddings are pre-computed for performance
   - **Workaround:** Run `python src/index.py` after adding new PDFs

### 7. **Multi-Document Retrieval**
   - **Limitation:** The bot retrieves top-8 chunks but may struggle with questions requiring synthesis across many documents
   - **Why:** Limited context window to keep generation prompt concise
   - **Workaround:** Ask more specific follow-up questions or combine related documents

### 8. **Model Limitations**
   - **Limitation:** The Llama 3.3 70B model has a training cutoff date and may not know about very recent events
   - **Why:** LLMs have knowledge cutoffs based on training data
   - **Workaround:** Include recent information in your document uploads

---

## Getting Started

1. **First Time?** Follow the [Setup Instructions](#setup-instructions) above
2. **Have Questions?** Check the [Example Queries](#example-queries) section
3. **Something Wrong?** Review the [Known Limitations](#known-limitations) section

## License

[Add your license here - e.g., MIT, Apache 2.0]

## Support

For issues or questions:
- Check the README carefully
- Review the [Known Limitations](#known-limitations) section
- Verify your Groq API key is valid
- Ensure PDFs are properly formatted text documents
