# RAG Streamlit App

A Retrieval-Augmented Generation (RAG) application using Streamlit.

## Setup Instructions

1. **Install Python** (3.8 or higher)

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment:**
   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **Mac/Linux:**
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up environment variables:**
   - Copy `.env.example` to `.env`
   - Add your GROQ API key to `.env`
   ```bash
   cp .env.example .env
   ```
   - Get your free API key from: https://console.groq.com/keys

6. **Run the Streamlit app:**
   ```bash
   streamlit run src/app.py
   ```

7. **Open in browser:**
   The app will be available at `http://localhost:8501`

## Project Structure
- `src/` - Python source code
- `data/` - Data files
- `chroma_db/` - Vector database
- `requirements.txt` - Python dependencies

## Requirements
- Python 3.8+
- GROQ API key (free)
