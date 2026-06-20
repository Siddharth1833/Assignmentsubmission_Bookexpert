from groq import Groq
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()

# Try to get API key from environment or Streamlit secrets
try:
    # First try environment variable (for local development)
    api_key = os.getenv("GROQ_API_KEY")
    
    # If not found, try Streamlit secrets (for Streamlit Cloud)
    if not api_key and hasattr(st, 'secrets'):
        try:
            api_key = st.secrets.get("GROQ_API_KEY")
        except:
            api_key = None
    
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Please set it as an environment variable or in Streamlit secrets."
        )
    
    client = Groq(api_key=api_key)
except Exception as e:
    raise RuntimeError(f"Failed to initialize Groq client: {str(e)}")

def generate_answer(question, context):

    prompt = f"""
You are DocMind AI, a professional AI Document Intelligence Assistant.

Your job is to answer questions ONLY using the provided document context.

IMPORTANT RULES:

1. Use ONLY information present in the context.
2. Never use outside knowledge.
3. Never invent facts.
4. If the answer is not clearly present in the context, respond EXACTLY:

I could not find this information in the provided documents.

5. When enough information is available:
   - Provide detailed answers.
   - Explain concepts thoroughly.
   - Use paragraphs.
   - Use bullet points when appropriate.
   - Include definitions.
   - Include important features.
   - Include benefits and examples if found in context.
   - Make the answer educational and easy to understand.

6. For questions starting with:
   - "What is" → give a detailed explanation.
   - "Explain" → provide a comprehensive explanation.
   - "Tell me about" → provide an in-depth explanation.
   - "Compare" → provide a structured comparison.

7. Structure answers professionally.

DOCUMENT CONTEXT:

{context}

QUESTION:

{question}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content