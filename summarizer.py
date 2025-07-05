# utils/summarizer.py

import openai
import os

# Set your OpenAI API key via environment variable or directly
openai.api_key = os.getenv("OPENAI_API_KEY")  # or assign manually


def summarize_differences(differences, model="gpt-3.5-turbo"):
    """
    Takes a list of (label, text, score) tuples and returns a natural language summary using GPT.
    """
    if not differences:
        return "No significant semantic differences found between the documents."

    bullet_points = "\n".join([
        f"- {label}: {text}" for label, text, _ in differences[:20]  # limit to top 20 for clarity
    ])

    prompt = f"""
    The following are changes detected between two versions of a document.
    Each line starts with ADDED or REMOVED followed by the affected text.

    {bullet_points}

    Summarize these changes in clear, human-friendly language. Focus on what was added, removed, or changed overall.
    """

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes document changes."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=300
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error during summarization: {str(e)}"
