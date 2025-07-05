# utils/qa_engine.py

from sentence_transformers import SentenceTransformer, util
import openai
import faiss
import numpy as np
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

model = SentenceTransformer("all-MiniLM-L6-v2")


class SemanticQASystem:
    def __init__(self):
        self.index = None
        self.text_chunks = []
        self.embeddings = None

    def build_index(self, chunks):
        self.text_chunks = chunks
        self.embeddings = model.encode(chunks, convert_to_numpy=True)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

    def ask(self, question, top_k=5):
        question_embedding = model.encode([question], convert_to_numpy=True)
        D, I = self.index.search(question_embedding, top_k)

        retrieved_chunks = [self.text_chunks[idx] for idx in I[0]]
        context = "\n".join(retrieved_chunks)

        prompt = f"""
        Based on the following document excerpts, answer the user's question.

        Document:
        {context}

        Question:
        {question}

        Answer in clear, concise, professional language.
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a smart document assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=300
            )
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            return f"Error during Q&A: {str(e)}"
